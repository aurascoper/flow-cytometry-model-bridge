using Test
using Random
using Statistics
using Printf

include("FlowCytometryObservationModel.jl")
include("MicrofluidicClassificationEngine.jl")

using .FlowCytometryObservationModel
using .MicrofluidicClassificationEngine

@testset "MicrofluidicClassificationEngine complete verification suite" begin

    # =======================================================================
    # 1. Module 1 (FCSIO): ISAC FCS 3.1 binary read/write, endianness,
    #    data types, metadata, SHA-256 provenance, and sensor CSV IO
    # =======================================================================
    @testset "Module 1 (FCSIO): ISAC FCS 3.1 binary IO and provenance" begin
        synth = generate_synthetic_nanobio_fcs_dataset(
            n_events = 300,
            sample_id = "NBL-CHIP01-RUN1",
            chip_id = "CHIP-01",
            run_id = "RUN-01",
            biological_replicate = "REP-A",
            seed = 20261029
        )
        ds = synth.dataset
        @test n_events(ds) == 300
        @test n_channels(ds) == 7
        @test channel_names(ds) == ["FSC-A", "FSC-H", "SSC-A", "FL1-SYBR-A", "FL2-PI-A", "FL3-Cy5-A", "TIME"]
        @test occursin(r"^[0-9a-f]{64}$", ds.provenance.sha256)

        # Write permanent representative .fcs file in code/
        rep_fcs_path = joinpath(@__DIR__, "sample_nanobio_chip01.fcs")
        written_ds = write_fcs_file(rep_fcs_path, ds; endian=:little, datatype=:F)
        @test isfile(rep_fcs_path)
        @test written_ds.provenance.byte_size > 58

        loaded_ds = read_fcs_file(rep_fcs_path)
        @test loaded_ds.provenance.sha256 == written_ds.provenance.sha256
        @test loaded_ds.provenance.fcs_version == "FCS3.1"
        @test loaded_ds.provenance.sample_id == "NBL-CHIP01-RUN1"
        @test loaded_ds.provenance.chip_id == "CHIP-01"
        @test loaded_ds.provenance.run_id == "RUN-01"
        @test loaded_ds.provenance.biological_replicate == "REP-A"
        @test n_events(loaded_ds) == 300
        @test n_channels(loaded_ds) == 7
        @test channel_names(loaded_ds) == channel_names(ds)
        @test isapprox(loaded_ds.events, ds.events; rtol=1e-5, atol=1e-2)

        # Verify big-endian Float64 (:D) and integer (:I) FCS 3.1 serialization
        bytes_d_big = serialize_fcs_bytes(ds; endian=:big, datatype=:D)
        parsed_d_big = parse_fcs_bytes(bytes_d_big)
        @test isapprox(parsed_d_big.events, ds.events; rtol=1e-12, atol=1e-12)

        bytes_i_lit = serialize_fcs_bytes(ds; endian=:little, datatype=:I)
        parsed_i_lit = parse_fcs_bytes(bytes_i_lit)
        @test isapprox(parsed_i_lit.events, round.(ds.events); atol=1.0)

        # Verify 16-bit integer (:I) channel round-trip and truncated DATA guard
        ch16 = [FCSChannelMetadata(1, "FSC-A"; range=65535.0, bits=16),
                FCSChannelMetadata(2, "SSC-A"; range=65535.0, bits=16)]
        ds16 = FCSDataset(ds.provenance, ch16, Dict{String,String}(), [1200.0 3400.0; 5600.0 7800.0])
        bytes16 = serialize_fcs_bytes(ds16; endian=:big, datatype=:I)
        parsed16 = parse_fcs_bytes(bytes16)
        @test parsed16.channels[1].bits == 16
        @test parsed16.events == [1200.0 3400.0; 5600.0 7800.0]
        @test_throws ArgumentError parse_fcs_bytes(bytes16[1:(end - 2)])

        # Subset events preserves channel metadata and records parent SHA-256
        mask = BitVector([i <= 120 for i in 1:300])
        sub_ds = subset_events(loaded_ds, mask; suffix="first120")
        @test n_events(sub_ds) == 120
        @test sub_ds.keywords["\$PARENT_SHA256"] == loaded_ds.provenance.sha256

        # Zero-event gated subset round-trips cleanly through binary FCS 3.1
        empty_ds = subset_events(loaded_ds, falses(300); suffix="empty")
        @test n_events(empty_ds) == 0
        empty_bytes = serialize_fcs_bytes(empty_ds; endian=:little, datatype=:F)
        parsed_empty = parse_fcs_bytes(empty_bytes)
        @test n_events(parsed_empty) == 0
        @test n_channels(parsed_empty) == 7

        # Sensor trajectory CSV round-trip
        mktempdir() do tmpdir
            sensor_csv = joinpath(tmpdir, "trajectories.csv")
            w_info = write_microfluidic_sensor_csv(sensor_csv, synth.sensor_rows)
            @test w_info.count == 300
            l_info = load_microfluidic_sensor_csv(sensor_csv)
            @test l_info.sha256 == w_info.sha256
            @test length(l_info.rows) == 300
            @test isapprox(l_info.rows[1].transit_time_ms, synth.sensor_rows[1].transit_time_ms; atol=1e-5)
            @test l_info.rows[1].expert_label == synth.sensor_rows[1].expert_label
        end

        # Edge cases and error paths
        @test_throws ArgumentError get_channel_vector(ds, "NON_EXISTENT_CHANNEL")
        @test_throws ArgumentError FCSChannelMetadata(0, "FSC-A")
        @test_throws ArgumentError FCSChannelMetadata(1, "")
        @test_throws ArgumentError FCSFileProvenance(source_path="x", sha256="not_a_hash")
        @test_throws ArgumentError parse_fcs_bytes(UInt8[1, 2, 3])
    end

    # =======================================================================
    # 2. Module 2 (CytometryAnalysis): Transforms, compensation, QC,
    #    all 5 gate geometries, hierarchical gating, and JSON serialization
    # =======================================================================
    @testset "Module 2 (CytometryAnalysis): Transforms, compensation, QC, and gating" begin
        synth = generate_synthetic_nanobio_fcs_dataset(n_events=300, seed=20261029)
        ds = synth.dataset

        # Transform invertibility across positive, zero, and extreme negative values
        t_lin = LinearTransform(2.5, -10.0)
        t_log = Log10Transform(10.0)
        t_ash = ArcsinhTransform(150.0)
        t_lgc = BiexponentialLogicleTransform(150.0)

        for x in (-1.0e6, -2.5e4, -120.0, 0.0, 50.0, 15_000.0)
            @test isapprox(inverse_transform(t_lin, apply_transform(t_lin, x)), x; rtol=1e-9, atol=1e-8)
            @test isapprox(inverse_transform(t_ash, apply_transform(t_ash, x)), x; rtol=1e-9, atol=1e-8)
            @test isapprox(inverse_transform(t_lgc, apply_transform(t_lgc, x)), x; rtol=1e-9, atol=1e-8)
        end
        for x in (0.0, 25.0, 10_000.0)
            @test isapprox(inverse_transform(t_log, apply_transform(t_log, x)), x; atol=1e-9)
        end

        # Near-singular spillover matrix rejection
        @test_throws ArgumentError SpilloverMatrix(["FL1", "FL2"], [1.0 0.9999; 0.9999 1.0])

        # Spillover parsing, formatting, and exact unmixing recovery
        spill = parse_fcs_spillover(ds.keywords["\$SPILLOVER"])
        @test spill.channels == ["FL1-SYBR-A", "FL2-PI-A", "FL3-Cy5-A"]
        reparsed = parse_fcs_spillover(format_fcs_spillover(spill))
        @test isapprox(reparsed.matrix, spill.matrix; atol=1e-5)

        comp_ds = apply_compensation(ds, spill)
        @test comp_ds.keywords["COMPENSATED"] == "TRUE"
        # Re-applying forward spillover matrix recovers original raw fluorescence
        fl_cols = [ds.channel_index[c] for c in spill.channels]
        recovered_raw = comp_ds.events[:, fl_cols] * spill.matrix
        @test isapprox(recovered_raw, ds.events[:, fl_cols]; rtol=1e-10, atol=1e-8)

        # Quality control check
        qc = run_quality_control(comp_ds)
        @test qc.total_count == 300
        @test 0.50 < qc.pass_fraction <= 1.0
        @test count(qc.clean_mask) == qc.pass_count

        # Gate types and hierarchical tree evaluation
        h = GatingHierarchy()
        g_singlet = PolygonGate(
            "singlet_poly", "Single Cells", "FSC-A", "FSC-H",
            [(10_000.0, 8_000.0), (250_000.0, 200_000.0), (250_000.0, 250_000.0), (10_000.0, 12_000.0)]
        )
        g_live_ell = EllipseGate(
            "live_ellipse", "Live Intact Ellipse", "FL1-SYBR-A", "FL2-PI-A",
            apply_transform(t_lgc, 14_500.0), apply_transform(t_lgc, 110.0),
            0.95, 0.85, 0.05;
            transform_x=t_lgc, transform_y=t_lgc
        )
        g_comp_rect = RectangleGate(
            "compromised_rect", "Membrane Compromised Rect", "FL1-SYBR-A", "FL2-PI-A",
            apply_transform(t_lgc, 1_200.0), apply_transform(t_lgc, 60_000.0),
            apply_transform(t_lgc, 600.0), apply_transform(t_lgc, 50_000.0);
            transform_x=t_lgc, transform_y=t_lgc
        )
        g_dead_range = RangeGate(
            "dead_pi_range", "High PI Dead Range", "FL2-PI-A",
            apply_transform(t_lgc, 3_200.0), apply_transform(t_lgc, 260_000.0);
            transform=t_lgc
        )
        g_uptake_quad = QuadGate(
            "uptake_quad_q1", "High Nanocarrier Uptake Q1", "FL1-SYBR-A", "FL3-Cy5-A",
            apply_transform(t_lgc, 2_500.0), apply_transform(t_lgc, 2_500.0);
            quadrant=:Q1_UR, transform_x=t_lgc, transform_y=t_lgc
        )

        add_gate!(h, g_singlet; parent_id="root")
        add_gate!(h, g_live_ell; parent_id="singlet_poly")
        add_gate!(h, g_comp_rect; parent_id="singlet_poly")
        add_gate!(h, g_dead_range; parent_id="root")
        add_gate!(h, g_uptake_quad; parent_id="live_ellipse")

        summaries = evaluate_hierarchy(h, comp_ds)
        @test summaries["singlet_poly"].parent_count == 300
        @test summaries["singlet_poly"].count > 180
        @test summaries["live_ellipse"].parent_count == summaries["singlet_poly"].count
        @test summaries["live_ellipse"].count > 120
        @test summaries["uptake_quad_q1"].parent_count == summaries["live_ellipse"].count
        @test summaries["uptake_quad_q1"].count > 80

        # Verify all 4 quadrants sum to total events
        quad_all = evaluate_quad_gate_all(g_uptake_quad, comp_ds)
        @test sum(count(v) for v in values(quad_all)) == 300

        # JSON / GatingML-compatible round-trip serialization
        json_str = export_gating_hierarchy_json(h)
        h_reloaded = import_gating_hierarchy_json(json_str)
        summaries_reloaded = evaluate_hierarchy(h_reloaded, comp_ds)
        for gid in keys(summaries)
            @test summaries_reloaded[gid].count == summaries[gid].count
            @test isapprox(summaries_reloaded[gid].fraction_of_parent, summaries[gid].fraction_of_parent; atol=1e-12)
            @test isapprox(summaries_reloaded[gid].fraction_of_total, summaries[gid].fraction_of_total; atol=1e-12)
        end

        # Edit and remove gate operations
        g_comp_rect_edited = RectangleGate(
            "compromised_rect", "Membrane Compromised Rect Edited", "FL1-SYBR-A", "FL2-PI-A",
            apply_transform(t_lgc, 1_000.0), apply_transform(t_lgc, 80_000.0),
            apply_transform(t_lgc, 500.0), apply_transform(t_lgc, 60_000.0);
            transform_x=t_lgc, transform_y=t_lgc
        )
        update_gate!(h_reloaded, g_comp_rect_edited)
        @test evaluate_hierarchy(h_reloaded, comp_ds)["compromised_rect"].count >= summaries["compromised_rect"].count
        remove_gate!(h_reloaded, "live_ellipse")
        @test !haskey(h_reloaded.nodes, "live_ellipse")
        @test !haskey(h_reloaded.nodes, "uptake_quad_q1") # descendant removed automatically

        # Point on polygon vertex and edge boundary inclusion
        tri = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
        @test point_in_polygon(0.0, 0.0, tri)
        @test point_in_polygon(5.0, 0.0, tri)
        @test point_in_polygon(2.0, 2.0, tri)
        @test !point_in_polygon(8.0, 8.0, tri)

        @test_throws ArgumentError RectangleGate("bad", "Bad", "FSC-A", "SSC-A", 10.0, 5.0, 0.0, 10.0)
        @test_throws ArgumentError PolygonGate("bad", "Bad", "FSC-A", "SSC-A", [(0.0, 0.0), (1.0, 1.0)])
        @test_throws ArgumentError SpilloverMatrix(["A", "B"], [1.0 0.5; 0.5 0.9])
    end

    # =======================================================================
    # 3. Module 3 (CellClassifier): All 4 Nano-Bio Lab target tasks,
    #    group-aware train/test splitting, confidence abstention, and CV
    # =======================================================================
    @testset "Module 3 (CellClassifier): 4 Nano-Bio targets, group splitting, and deferral" begin
        # Generate 3 independent microfluidic chips to test strict group-aware validation
        chip1 = generate_synthetic_nanobio_fcs_dataset(
            n_events=180, sample_id="S1", chip_id="CHIP-01", run_id="RUN-1", biological_replicate="REP-1", seed=101)
        chip2 = generate_synthetic_nanobio_fcs_dataset(
            n_events=180, sample_id="S2", chip_id="CHIP-02", run_id="RUN-2", biological_replicate="REP-2", seed=202)
        chip3 = generate_synthetic_nanobio_fcs_dataset(
            n_events=180, sample_id="S3", chip_id="CHIP-03", run_id="RUN-3", biological_replicate="REP-3", seed=303)

        f1 = extract_single_cell_features(chip1.dataset; transit_rows=chip1.sensor_rows)
        f2 = extract_single_cell_features(chip2.dataset; transit_rows=chip2.sensor_rows)
        f3 = extract_single_cell_features(chip3.dataset; transit_rows=chip3.sensor_rows)

        X_all = vcat(f1.features, f2.features, f3.features)
        feat_names = f1.feature_names
        chip_groups = vcat(
            fill("CHIP-01", 180),
            fill("CHIP-02", 180),
            fill("CHIP-03", 180)
        )

        # Group-aware split rejects random event-level splitting and group leakage
        @test_throws ArgumentError group_aware_split(chip_groups; holdout_groups=["CHIP-03"], split_by=:random_event)
        @test_throws ArgumentError verify_no_group_leakage([1, 2], [2, 3], chip_groups)
        @test_throws ArgumentError verify_no_group_leakage([1, 2], [3, 4], chip_groups) # both in CHIP-01

        sp = group_aware_split(chip_groups; holdout_groups=["CHIP-03"], split_by=:chip_id)
        @test sp.train_groups == ["CHIP-01", "CHIP-02"]
        @test sp.test_groups == ["CHIP-03"]
        @test verify_no_group_leakage(sp.train_indices, sp.test_indices, chip_groups)

        # Verify all 4 candidate Nano-Bio Lab classification tasks achieve high out-of-chip accuracy
        for target in VALID_CLASSIFICATION_TARGETS
            y_all = vcat(chip1.labels[target], chip2.labels[target], chip3.labels[target])
            cv_res = group_cross_validate(
                X_all, y_all, chip_groups;
                feature_names = feat_names,
                target_task = target,
                split_by = :chip_id,
                confidence_threshold = 0.75
            )
            rep = cv_res.report
            @test rep.target_task == target
            @test rep.total_events == 540
            @test rep.overall_accuracy > 0.92
            @test rep.accepted_accuracy >= rep.overall_accuracy
            @test rep.macro_f1 > 0.92
            @test rep.brier_score < 0.15
            @test 0.0 <= rep.expected_calibration_error < 0.15
        end

        # Verify confidence thresholding and :defer_for_review abstention behavior on boundary events
        y_stress = vcat(chip1.labels[:cell_stress_damage], chip2.labels[:cell_stress_damage], chip3.labels[:cell_stress_damage])
        clf = train_cell_classifier(
            X_all[sp.train_indices, :], y_stress[sp.train_indices];
            feature_names = feat_names,
            target_task = :cell_stress_damage,
            group_ids = chip_groups[sp.train_indices],
            split_group_key = :chip_id,
            confidence_threshold = 0.80
        )
        # Create a synthetic ambiguous midpoint cell between population 1 and population 2
        ambiguous_x = 0.5 .* X_all[1:1, :] .+ 0.5 .* X_all[130:130, :]
        preds_high_thresh = predict_cells(clf, vcat(X_all[sp.test_indices, :], ambiguous_x); confidence_threshold=0.9999)
        @test any(p -> p.deferred && p.final_decision == :defer_for_review, preds_high_thresh)
        for p in preds_high_thresh
            @test isapprox(sum(p.probabilities), 1.0; atol=1e-9)
            @test (p.confidence < 0.9999) == p.deferred
        end
    end

    # =======================================================================
    # 4. Module 4 (MicrofluidicBridge) & MVP Acceptance End-to-End Pipeline
    #    + Cross-Language Python Reference Verification
    # =======================================================================
    @testset "Module 4 (MicrofluidicBridge) and Python Reference Verification" begin
        rep_fcs_path = joinpath(@__DIR__, "sample_nanobio_chip01.fcs")
        ds_raw = read_fcs_file(rep_fcs_path)
        spill = parse_fcs_spillover(ds_raw.keywords["\$SPILLOVER"])
        ds_comp = apply_compensation(ds_raw, spill)

        synth_ref = generate_synthetic_nanobio_fcs_dataset(
            n_events = 300,
            sample_id = "NBL-CHIP01-RUN1",
            chip_id = "CHIP-01",
            run_id = "RUN-01",
            biological_replicate = "REP-A",
            seed = 20261029
        )
        chip2 = generate_synthetic_nanobio_fcs_dataset(
            n_events = 300,
            sample_id = "NBL-CHIP02-RUN1",
            chip_id = "CHIP-02",
            run_id = "RUN-02",
            biological_replicate = "REP-B",
            seed = 20261030
        )

        # Build gating hierarchy on ds_comp
        t_lgc = BiexponentialLogicleTransform(150.0)
        h = GatingHierarchy()
        add_gate!(h, PolygonGate(
            "singlet_poly", "Single Cells", "FSC-A", "FSC-H",
            [(10_000.0, 8_000.0), (250_000.0, 200_000.0), (250_000.0, 250_000.0), (10_000.0, 12_000.0)]
        ); parent_id="root")
        add_gate!(h, EllipseGate(
            "live_ellipse", "Live Intact Ellipse", "FL1-SYBR-A", "FL2-PI-A",
            apply_transform(t_lgc, 14_500.0), apply_transform(t_lgc, 110.0),
            0.95, 0.85, 0.05; transform_x=t_lgc, transform_y=t_lgc
        ); parent_id="singlet_poly")
        add_gate!(h, RectangleGate(
            "compromised_rect", "Compromised Rect", "FL1-SYBR-A", "FL2-PI-A",
            apply_transform(t_lgc, 1_200.0), apply_transform(t_lgc, 60_000.0),
            apply_transform(t_lgc, 600.0), apply_transform(t_lgc, 50_000.0);
            transform_x=t_lgc, transform_y=t_lgc
        ); parent_id="singlet_poly")
        add_gate!(h, RangeGate(
            "dead_pi_range", "Dead High PI", "FL2-PI-A",
            apply_transform(t_lgc, 3_200.0), apply_transform(t_lgc, 260_000.0);
            transform=t_lgc
        ); parent_id="root")
        add_gate!(h, QuadGate(
            "uptake_quad_q1", "High Uptake Q1", "FL1-SYBR-A", "FL3-Cy5-A",
            apply_transform(t_lgc, 2_500.0), apply_transform(t_lgc, 2_500.0);
            quadrant=:Q1_UR, transform_x=t_lgc, transform_y=t_lgc
        ); parent_id="live_ellipse")

        h_summaries = evaluate_hierarchy(h, ds_comp)

        # Train classifier on CHIP-02 and evaluate on CHIP-01 (loaded from binary FCS)
        f_chip1 = extract_single_cell_features(ds_comp; transit_rows=synth_ref.sensor_rows)
        ds_chip2_comp = apply_compensation(chip2.dataset, spill)
        f_chip2 = extract_single_cell_features(ds_chip2_comp; transit_rows=chip2.sensor_rows)

        X_comb = vcat(f_chip1.features, f_chip2.features)
        y_comb = vcat(synth_ref.labels[:cell_stress_damage], chip2.labels[:cell_stress_damage])
        g_comb = vcat(fill("CHIP-01", 300), fill("CHIP-02", 300))

        sp = group_aware_split(g_comb; holdout_groups=["CHIP-01"], split_by=:chip_id)
        clf = train_cell_classifier(
            X_comb[sp.train_indices, :],
            y_comb[sp.train_indices];
            feature_names = f_chip1.feature_names,
            target_task = :cell_stress_damage,
            group_ids = g_comb[sp.train_indices],
            split_group_key = :chip_id,
            confidence_threshold = 0.75
        )
        preds_chip1 = predict_cells(clf, f_chip1.features; confidence_threshold=0.75)
        val_report = evaluate_classifier_predictions(
            preds_chip1, synth_ref.labels[:cell_stress_damage],
            default_classes_for_target(:cell_stress_damage);
            target_task = :cell_stress_damage,
            group_ids = fill("CHIP-01", 300)
        )
        @test val_report.overall_accuracy > 0.93

        # Join cytometry events, classifier predictions, and micropore transit measurements
        cond = MicrofluidicAssayCondition(
            chip_id = "CHIP-01",
            run_id = "RUN-01",
            sample_id = "NBL-CHIP01-RUN1",
            biological_replicate = "REP-A",
            channel_width_um = 25.0,
            pore_constriction_um = 7.5,
            flow_rate_ul_min = 6.0,
            wall_shear_pa = 1.45,
            actuation_voltage_vpp = 4.0,
            actuation_frequency_khz = 120.0,
            temperature_c = 25.0
        )
        transit_events = sensor_rows_to_transit_events(synth_ref.sensor_rows)
        joined = join_cytometry_and_microfluidics(
            ds_comp, transit_events, cond;
            predictions = preds_chip1,
            gate_summaries = h_summaries
        )
        @test length(joined) == 300

        # Bridge classified population into FlowCytometryObservationModel
        cal = GateCalibration([0.94 0.04 0.01;
                               0.05 0.91 0.06;
                               0.01 0.05 0.93]; detection_efficiency=0.88)
        fcm_conv = summarize_to_fcm_sample(
            preds_chip1;
            bead_events = 120,
            beads_added = 15_000.0,
            sample_volume_ml = 0.10,
            dilution_factor = 5.0,
            extraction_recovery = 0.90,
            detection_efficiency = cal.detection_efficiency,
            sample_origin = :effluent,
            gate_id = "nbl-chip01-classifier"
        )
        t_snap = OpenFOAMTransportSnap("CHIP-01", 1.0, :effluent;
            velocity_m_s=0.018, wall_shear_pa=1.45, residence_time_s=0.42)
        c_snap = PHREEQCSpeciationSnap("CHIP-01", 1.0;
            pH=7.2, pe=4.5, ionic_strength_M=0.15)

        bridge_rep = bridge_to_observation_model(
            joined, fcm_conv.sample_summary, cal;
            state_prediction = fcm_conv.state_prediction,
            transport_snap = t_snap,
            chemistry_snap = c_snap,
            vessel_id = "CHIP-01",
            time_h = 1.0,
            track = :track1_nanobio,
            input_sha256 = ds_raw.provenance.sha256,
            heldout_validated = true
        )
        @test bridge_rep.likelihood_consistent
        @test isfinite(bridge_rep.joint_loglikelihood)
        @test bridge_rep.concentration_uncertainty.ci_95_lower < bridge_rep.absolute_concentration_cells_per_ml < bridge_rep.concentration_uncertainty.ci_95_upper
        @test bridge_rep.audit_record.consistent_likelihood
        @test bridge_rep.audit_record.identifiable_gates
        @test bridge_rep.audit_record.heldout_validated

        # Physical micropore transit stratification checks:
        # Intact deformable cells transit faster and with higher deformability index than compromised/dead cells
        ts_intact = bridge_rep.class_transit_statistics[:membrane_intact]
        ts_comp   = bridge_rep.class_transit_statistics[:membrane_compromised]
        ts_dead   = bridge_rep.class_transit_statistics[:dead_lysed]
        @test ts_intact.mean_transit_time_ms < ts_comp.mean_transit_time_ms < ts_dead.mean_transit_time_ms
        @test ts_intact.mean_deformability_index > ts_comp.mean_deformability_index > ts_dead.mean_deformability_index

        # Also verify hierarchy-to-FCM sample summary helper
        hier_fcm = summarize_hierarchy_to_fcm_sample(
            h_summaries, ("live_ellipse", "compromised_rect", "dead_pi_range");
            bead_events = 120, beads_added = 15_000.0, sample_volume_ml = 0.10
        )
        @test sum(hier_fcm.sample_summary.gate_counts) > 200

        # Export bundle and run independent Python reference verification
        mktempdir() do tmpdir
            cp(rep_fcs_path, joinpath(tmpdir, "sample_nanobio_chip01.fcs"); force=true)
            exp_info = export_reproducible_analysis_bundle(
                tmpdir;
                dataset = ds_raw,
                hierarchy = h,
                hierarchy_summaries = h_summaries,
                classifier_report = val_report,
                predictions = preds_chip1,
                bridge_report = bridge_rep
            )
            @test isfile(exp_info.gates_json_path)
            @test isfile(exp_info.events_csv_path)
            @test isfile(exp_info.summary_json_path)

            # Write python_verification_payload.json for cross-language numerical verification
            payload_path = joinpath(tmpdir, "python_verification_payload.json")
            open(payload_path, "w") do io
                println(io, "{")
                println(io, "  \"hierarchy_on_compensated\": true,")
                println(io, "  \"confidence_threshold\": 0.75,")
                println(io, "  \"classes\": [\"membrane_intact\", \"membrane_compromised\", \"dead_lysed\"],")
                println(io, "  \"split_train_groups\": [\"CHIP-02\"],")
                println(io, "  \"split_test_groups\": [\"CHIP-01\"],")
                println(io, "  \"compensated_head_events\": [")
                for r in 1:5
                    row_str = join([@sprintf("%.8f", ds_comp.events[r, c]) for c in 1:n_channels(ds_comp)], ", ")
                    comma = r < 5 ? "," : ""
                    println(io, "    [$row_str]$comma")
                end
                println(io, "  ],")
                println(io, "  \"cv_true_labels\": [", join(["\"$(String(l))\"" for l in synth_ref.labels[:cell_stress_damage]], ", "), "],")
                println(io, "  \"cv_predictions\": [")
                for (idx, p) in enumerate(preds_chip1)
                    p_str = join([@sprintf("%.10f", x) for x in p.probabilities], ", ")
                    comma = idx < length(preds_chip1) ? "," : ""
                    @printf(io, "    {\"predicted_class\": \"%s\", \"deferred\": %s, \"probabilities\": [%s]}%s\n",
                        String(p.predicted_class), p.deferred ? "true" : "false", p_str, comma)
                end
                println(io, "  ],")
                gc = bridge_rep.sample_summary.gate_counts
                gp = bridge_rep.predicted_gate_probs
                println(io, "  \"bridge_inputs\": {")
                @printf(io, "    \"gate_counts\": [%d, %d, %d],\n", gc[1], gc[2], gc[3])
                @printf(io, "    \"bead_events\": %d,\n", bridge_rep.sample_summary.bead_events)
                @printf(io, "    \"beads_added\": %.8f,\n", bridge_rep.sample_summary.beads_added)
                @printf(io, "    \"sample_volume_ml\": %.8f,\n", bridge_rep.sample_summary.sample_volume_ml)
                @printf(io, "    \"dilution_factor\": %.8f,\n", bridge_rep.sample_summary.dilution_factor)
                @printf(io, "    \"extraction_recovery\": %.8f,\n", bridge_rep.sample_summary.extraction_recovery)
                @printf(io, "    \"detection_efficiency\": %.8f,\n", cal.detection_efficiency)
                @printf(io, "    \"state_total_cells_per_ml\": %.8f,\n", bridge_rep.latent_state_mle.total_cells_per_ml)
                @printf(io, "    \"predicted_gate_probs\": [%.10f, %.10f, %.10f]\n", gp[1], gp[2], gp[3])
                println(io, "  }")
                println(io, "}")
            end

            py_script = joinpath(@__DIR__, "verify_against_python_reference.py")
            py_out = read(`python3 $py_script $tmpdir`, String)
            @test occursin("PYTHON_REFERENCE_VERIFICATION_PASSED", py_out)
        end
    end

    # =======================================================================
    # 5. Module 5 (SharedVisualizationPhysics): Ghosh Core-Multishell RD
    #    (R0, R1, R2) and Jack Interface-Resolved Emulsion Droplet Dynamics
    # =======================================================================
    @testset "Module 5 (SharedVisualizationPhysics): Ghosh core-multishell & Jack emulsion" begin
        # 5A. Ghosh Track 1 Core-Multishell Reaction-Diffusion (R0 < R1 < R2)
        cms_spec = CoreMultishellSpec(
            radii_um = [0.35, 0.70, 1.00],
            diffusivities_um2_s = [0.012, 0.028, 0.045],
            decay_rates_s = [0.001, 0.002, 0.002],
            initial_concentrations = [0.030, 0.018, 0.006],
            membrane_permeability_um_s = 0.08,
            channel_flow_velocity_mm_s = 15.0,
            track = :track1_nanobio,
            epistemic_status = :predicted
        )
        cms_sol = solve_core_multishell_rd(cms_spec; times_s=collect(0.0:2.0:20.0), n_radial_nodes=48)
        @test length(cms_sol.snapshots) == 11
        @test cms_sol.snapshots[1].cumulative_release_fraction == 0.0
        @test cms_sol.snapshots[end].cumulative_release_fraction > 0.35
        # Release fraction is monotonically non-decreasing over time
        for k in 2:length(cms_sol.snapshots)
            @test cms_sol.snapshots[k].cumulative_release_fraction >= cms_sol.snapshots[k - 1].cumulative_release_fraction
            @test isapprox(sum(cms_sol.snapshots[k].predicted_uptake_state.fractions), 1.0; atol=1e-6)
        end
        # High-uptake cell fraction increases as nanocarrier release progresses in the channel
        @test cms_sol.snapshots[end].predicted_uptake_state.fractions[1] > cms_sol.snapshots[1].predicted_uptake_state.fractions[1]

        # Stress-test highly discontinuous diffusivity ratio (D_inner / D_outer > 10^4) and thin stratum at n_radial_nodes = 12
        cms_discont = CoreMultishellSpec(
            radii_um = [0.35, 0.37, 1.00],
            diffusivities_um2_s = [25.0, 0.001, 0.0005],
            decay_rates_s = [0.05, 0.20, 0.50],
            initial_concentrations = [0.030, 0.020, 0.010],
            membrane_permeability_um_s = 2.5,
            channel_flow_velocity_mm_s = 15.0
        )
        sol_discont = solve_core_multishell_rd(cms_discont; times_s=[0.0, 1.0, 5.0], n_radial_nodes=12)
        @test all(x -> x > 0.0, sol_discont.snapshots[1].stratum_mean_concentrations)
        @test 0.0 <= sol_discont.snapshots[2].cumulative_release_fraction <= sol_discont.snapshots[3].cumulative_release_fraction <= 1.0

        # Enforce strict track and epistemic boundaries on CoreMultishellSpec
        @test_throws ArgumentError CoreMultishellSpec(track=:track3_bitumen_rheology)
        @test_throws ArgumentError CoreMultishellSpec(epistemic_status=:observed)
        @test_throws ArgumentError CoreMultishellSpec(radii_um=[0.8, 0.5, 1.0])

        # 5B. Jack Track 3 Interface-Resolved Emulsion Droplet Dynamics (t/tau_eta = 1..15)
        em_spec = EmulsionDropletSpec(
            droplet_radius_um = 10.0,
            inner_core_radius_um = 4.2,
            shear_rate_s = 120.0,
            temperature_c = 45.0,
            continuous_viscosity_pa_s = 0.08,
            dispersed_viscosity_pa_s = 1.60,
            interfacial_tension_n_m = 0.0045,
            interfacial_shear_viscosity_pa_s_m = 1.2e-5,
            interfacial_dilatational_viscosity_pa_s_m = 2.0e-5,
            track = :track3_bitumen_rheology,
            has_biological_release_shells = false
        )
        em_sol = solve_emulsion_interfacial_droplet(em_spec; dimensionless_times=collect(1.0:1.0:15.0))
        @test length(em_sol.snapshots) == 15
        @test isapprox(em_sol.capillary_number_ca, (0.08 * 120.0 * 10.0e-6) / 0.0045; rtol=1e-10)
        @test em_sol.effective_viscosity_ratio > 10.0

        # Also verify high-shear breakup regime (shear_rate_s = 1800 s^-1 => Ca = 0.32)
        em_high_shear = solve_emulsion_interfacial_droplet(
            EmulsionDropletSpec(shear_rate_s=1800.0);
            dimensionless_times=collect(1.0:1.0:15.0)
        )
        @test isapprox(em_high_shear.capillary_number_ca, 0.32; rtol=1e-10)

        for snap in em_sol.snapshots
            L, B, W = snap.semi_axes_um
            # Volume conservation: L * B * W == a0^3
            @test isapprox(L * B * W, em_spec.droplet_radius_um^3; rtol=1e-6)
            # Taylor deformation definition D = (L - B) / (L + B)
            @test isapprox(snap.taylor_deformation_D, (L - B) / (L + B); atol=1e-6)
            # Hydrodynamic pressure bounds P/Pa
            @test snap.max_pressure_pa > 0.5
            @test snap.min_pressure_pa < -0.1
            @test snap.vortex_circulation_strength > 0.0
        end
        @test em_sol.snapshots[end].taylor_deformation_D > em_sol.snapshots[1].taylor_deformation_D
        @test em_sol.snapshots[end].necking_ratio < em_sol.snapshots[1].necking_ratio

        # Enforce strict track and physical boundaries on EmulsionDropletSpec
        @test_throws ArgumentError EmulsionDropletSpec(track=:track1_nanobio)
        @test_throws ArgumentError EmulsionDropletSpec(has_biological_release_shells=true)
    end
end
