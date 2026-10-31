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

        # 5C. Ghosh2010_B12_PEGMA_315nm Public-Record Instance
        ghosh_b12 = Ghosh2010_B12_PEGMA_315nm()
        @test ghosh_b12.radii_um == [0.080, 0.125, 0.1575]
        @test ghosh_b12.material_names == ["Fe3O4_core", "PEGMA_intermediate_shell", "B12_targeting_corona"]
        @test ghosh_b12.partition_coefficients == [3.2, 1.5, 1.0]
        @test ghosh_b12.track == :track1_nanobio
        @test ghosh_b12.epistemic_status == :predicted
        b12_sol = solve_core_multishell_rd(ghosh_b12; times_s=[0.0, 5.0, 10.0], n_radial_nodes=24)
        @test length(b12_sol.snapshots) == 3
        @test b12_sol.snapshots[1].cumulative_release_fraction == 0.0
        @test b12_sol.snapshots[3].cumulative_release_fraction > b12_sol.snapshots[2].cumulative_release_fraction

        # 5D. GhoshPatent_Cisplatin_DoubleShell_191nm Public-Record Instance
        ghosh_cis = GhoshPatent_Cisplatin_DoubleShell_191nm()
        @test ghosh_cis.radii_um == [0.045, 0.075, 0.0955]
        @test ghosh_cis.material_names == ["Cisplatin_depot_core", "PLGA_diffusion_barrier_shell", "PEG_stealth_outer_shell"]
        @test ghosh_cis.partition_coefficients == [4.5, 1.8, 1.0]
        @test ghosh_cis.track == :track1_nanobio
        @test ghosh_cis.epistemic_status == :predicted
        cis_sol = solve_core_multishell_rd(ghosh_cis; times_s=[0.0, 5.0, 10.0], n_radial_nodes=24)
        @test length(cis_sol.snapshots) == 3
        @test cis_sol.snapshots[3].cumulative_release_fraction > 0.0

        # 5E. fit_core_multishell_release_curve Parameter Recovery and Goodness-of-Fit
        fit_times = collect(0.0:2.0:16.0)
        true_b12_sol = solve_core_multishell_rd(ghosh_b12; times_s=fit_times, n_radial_nodes=24)
        synth_release = [s.cumulative_release_fraction for s in true_b12_sol.snapshots]
        fit_res = fit_core_multishell_release_curve(fit_times, synth_release; base_spec=ghosh_b12, n_radial_nodes=24)
        @test fit_res.r_squared > 0.98
        @test fit_res.rmse < 0.05
        @test isapprox(fit_res.fitted_permeability_um_s, ghosh_b12.membrane_permeability_um_s; rtol=0.25)
        @test fit_res.epistemic_status == :predicted
        @test_throws ArgumentError fit_core_multishell_release_curve([1.0, 2.0], [0.1])
        @test_throws ArgumentError fit_core_multishell_release_curve([2.0, 1.0], [0.1, 0.2])

        # 5F. simulate_pulsed_duty_cycle_release (Field-Gated vs Thermal-Lag Dynamics)
        pulsed = simulate_pulsed_duty_cycle_release(ghosh_b12; total_time_s=30.0, period_s=10.0, duty_cycle=0.25, active_permeability_multiplier=15.0, thermal_relaxation_time_s=1.5, n_radial_nodes=24, n_output_points=31)
        @test length(pulsed.times_s) == 31
        @test pulsed.epistemic_status == :predicted
        @test pulsed.epistemic_label == "MODEL PREDICTION - illustrative parameters - no lab data"
        # Invariant 1: Step field achieves higher on-phase release concentration selectivity than lagged thermal dissipation
        @test pulsed.field_on_release_fraction_step >= pulsed.field_on_release_fraction_thermal
        # Invariant 2: Thermal leakback fraction is non-negative
        @test pulsed.thermal_leakback_fraction >= 0.0
        # Invariant 3: Cumulative release is strictly monotonic
        for k in 2:length(pulsed.times_s)
            @test pulsed.field_gated_release_fraction[k] >= pulsed.field_gated_release_fraction[k - 1]
            @test pulsed.thermal_lag_release_fraction[k] >= pulsed.thermal_lag_release_fraction[k - 1]
        end
        # Invariant 4: Peak thermal permeability lags behind step peak
        @test maximum(pulsed.field_gated_permeability_um_s) >= maximum(pulsed.thermal_lag_permeability_um_s)

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

    # =======================================================================
    # 6. Module 6 (BiofilmMultiphysics): Milestones M3 & M4 constitutive,
    #    transport, quorum-sensing, EV, and subpopulation dynamics
    # =======================================================================
    @testset "Module 6 (BiofilmMultiphysics): Milestones M3 and M4" begin
        # 6A. M3: BiofilmMatrixComposition & PNAS Oct 2026 PGA swelling vs curli/EPS wrinkling
        comp_wt = BiofilmMatrixComposition(
            pga_kg_m3 = 18.0, eps_kg_m3 = 28.0, curli_kg_m3 = 10.0,
            cellulose_kg_m3 = 6.0, porosity = 0.78, epistemic_status = :assumed
        )
        comp_no_pga = BiofilmMatrixComposition(
            pga_kg_m3 = 0.0, eps_kg_m3 = 28.0, curli_kg_m3 = 10.0,
            cellulose_kg_m3 = 6.0, porosity = 0.78, epistemic_status = :assumed
        )
        comp_high_pga_soft = BiofilmMatrixComposition(
            pga_kg_m3 = 35.0, eps_kg_m3 = 12.0, curli_kg_m3 = 2.0,
            cellulose_kg_m3 = 1.0, porosity = 0.85, epistemic_status = :assumed
        )
        @test isapprox(total_matrix_biopolymer_kg_m3(comp_wt), 62.0; atol=1e-12)

        mech_spec = ThinFilmSwellingMechanicsSpec(
            film_thickness_um = 45.0,
            substrate_modulus_pa = 600.0
        )
        m_wt = evaluate_biofilm_swelling_mechanics(comp_wt, mech_spec)
        m_no_pga = evaluate_biofilm_swelling_mechanics(comp_no_pga, mech_spec)
        m_soft = evaluate_biofilm_swelling_mechanics(comp_high_pga_soft, mech_spec)

        # PGA drives hydration and swelling pressure; deleting PGA collapses swelling pressure
        @test m_wt.hydration_fraction > m_no_pga.hydration_fraction
        @test m_wt.swelling_pressure_pa > 4.0 * m_no_pga.swelling_pressure_pa
        @test m_no_pga.morphological_phase == :rigid_compact_mat
        # Curli and EPS increase composite elastic modulus
        @test m_wt.composite_modulus_pa > m_soft.composite_modulus_pa
        # High PGA osmotic swelling on compliant substrate triggers thin-film wrinkling
        @test m_soft.wrinkling_index > 1.0
        @test m_soft.wrinkled
        @test m_soft.morphological_phase == :wrinkled_buckled_colony
        @test m_soft.wrinkle_wavelength_um > 50.0

        # Switching off PGA swelling hypothesis cleanly disables swelling pressure
        sw_no_swell = BiofilmHypothesisSwitches(enable_pga_swelling_mechanics = false)
        m_disabled = evaluate_biofilm_swelling_mechanics(comp_wt, mech_spec; switches=sw_no_swell)
        @test m_disabled.swelling_pressure_pa == 0.0
        @test m_disabled.morphological_phase == :flat_unswollen

        # 6B. M3: UPEC npj Biofilms Oct 2026 spatially heterogeneous diffusivity D_eff(x,t; M, r, q)
        diff_small_neutral = HeterogeneousDiffusivitySpec(
            aqueous_diffusivity_m2_s = 6.0e-10, probe_radius_nm = 1.0,
            probe_charge_valence = 0.0, ionic_strength_mol_l = 0.15
        )
        diff_large_cationic = HeterogeneousDiffusivitySpec(
            aqueous_diffusivity_m2_s = 6.0e-10, probe_radius_nm = 4.5,
            probe_charge_valence = 2.0, ionic_strength_mol_l = 0.05
        )
        d_sn = compute_effective_diffusivity(comp_wt, diff_small_neutral)
        d_lc = compute_effective_diffusivity(comp_wt, diff_large_cationic)
        @test d_sn.d_eff_m2_s > 3.0 * d_lc.d_eff_m2_s
        @test d_lc.curli_sieving_factor < d_sn.curli_sieving_factor
        @test d_lc.electrostatic_factor < 1.0
        @test isapprox(d_sn.electrostatic_factor, 1.0; atol=1e-12)

        # Switching off heterogeneous diffusion returns porosity-scaled homogeneous diffusivity
        sw_hom_diff = BiofilmHypothesisSwitches(enable_heterogeneous_matrix_diffusion = false)
        d_hom = compute_effective_diffusivity(comp_wt, diff_large_cationic; switches=sw_hom_diff)
        @test isapprox(d_hom.d_eff_m2_s, 6.0e-10 * comp_wt.porosity; rtol=1e-12)

        # Spatial diffusivity field across 1D depth profile (surface porous -> deep dense curli)
        n_cells = 16
        x_grid = collect(range(0.0, 150.0e-6; length=n_cells))
        comps_1d = [
            BiofilmMatrixComposition(
                pga_kg_m3 = 10.0 + 10.0 * (i / n_cells),
                eps_kg_m3 = 15.0 + 25.0 * (i / n_cells),
                curli_kg_m3 = 2.0 + 18.0 * (i / n_cells),
                cellulose_kg_m3 = 4.0,
                porosity = 0.86 - 0.20 * (i / n_cells)
            )
            for i in 1:n_cells
        ]
        dfield = compute_spatial_diffusivity_field(comps_1d, diff_large_cationic)
        @test dfield.heterogeneity_ratio > 2.5
        @test dfield.d_eff_m2_s[1] > dfield.d_eff_m2_s[end]

        # 6C. M3: Localized bioelectronic delivery source S_i(x,t) and 1D conservative FV transport
        src_spec = LocalizedDeliverySourceSpec(
            actuator_position_m = 50.0e-6,
            spatial_spread_m = 15.0e-6,
            peak_release_rate_mol_m3_s = 0.40,
            pulse_start_s = 0.0,
            pulse_duration_s = 10.0,
            duty_cycle = 0.80,
            carrier_type = :bioelectronic_ion_pump
        )
        @test evaluate_localized_source_term(50.0e-6, 5.0, src_spec) > 0.30
        @test evaluate_localized_source_term(50.0e-6, 25.0, src_spec) == 0.0
        @test evaluate_localized_source_term(
            50.0e-6, 5.0, src_spec;
            switches=BiofilmHypothesisSwitches(enable_localized_bioelectronic_delivery=false)
        ) == 0.0

        rt_sol = solve_1d_reactive_transport_biofilm(
            x_grid_m = x_grid,
            times_s = [0.0, 2.0, 6.0, 12.0],
            compositions = comps_1d,
            diff_spec = diff_large_cationic,
            source_spec = src_spec,
            initial_concentration_mol_m3 = fill(0.10, n_cells),
            first_order_uptake_s = 0.02,
            boundary_left_flux_mol_m2_s = 0.0,
            boundary_right_flux_mol_m2_s = 0.0
        )
        @test rt_sol.mass_conserved
        @test rt_sol.relative_mass_balance_error < 1e-12
        @test rt_sol.cumulative_source_mol_m2 > 0.0
        @test rt_sol.cumulative_uptake_mol_m2 > 0.0
        @test maximum(rt_sol.profiles_mol_m3[3]) > 0.10

        # Boundary extraction outflow and sub-nanometer grid remain strictly mass-conserving
        rt_outflow = solve_1d_reactive_transport_biofilm(
            x_grid_m = collect(range(0.0, 7.0e-9; length=8)),
            times_s = [0.0, 5.0, 10.0],
            compositions = comps_1d[1:8],
            diff_spec = diff_large_cationic,
            initial_concentration_mol_m3 = fill(0.001, 8),
            first_order_uptake_s = 0.05,
            boundary_right_flux_mol_m2_s = 1.0e-4
        )
        @test rt_outflow.mass_conserved
        @test rt_outflow.relative_mass_balance_error < 1e-8

        # 6D. M4: Electrokinetic / ROS AHL degradation in supernatant vs intact biofilm (Oct 17, 2026)
        ek_spec = ElectrokineticQSSpec(
            electric_field_v_m = 2000.0,
            ros_generation_factor = 1.2,
            k_electric_ros_s = 0.020,
            k_enzymatic_qq_s = 0.003,
            matrix_shielding_coeff = 0.06
        )
        deg_supernatant = evaluate_qs_signal_degradation(2.0, ek_spec; intact_biofilm_matrix=nothing)
        deg_biofilm = evaluate_qs_signal_degradation(2.0, ek_spec; intact_biofilm_matrix=comp_wt)
        @test deg_supernatant.matrix_transmission_factor == 1.0
        @test deg_biofilm.matrix_transmission_factor < 0.30
        @test deg_supernatant.removal_flux_um_s > 3.0 * deg_biofilm.removal_flux_um_s

        # 6E. M4: Three-mode polymicrobial QS crosstalk (Critical Reviews in Microbiology Oct 2026)
        ct_spec = PolymicrobialCrosstalkSpec(
            direct_receptor_competition_coeff = 0.40,
            chemical_effector_inhibition_coeff = 0.50,
            alkaline_lactonolysis_coeff = 0.30,
            redox_modulation_coeff = 0.15
        )
        ct_neutral = evaluate_polymicrobial_crosstalk(
            ct_spec; cognate_signal_um=1.0, non_cognate_signal_um=0.0,
            chemical_effector_um=0.0, local_ph=7.0, local_eh_mv=100.0
        )
        ct_competed = evaluate_polymicrobial_crosstalk(
            ct_spec; cognate_signal_um=1.0, non_cognate_signal_um=2.5,
            chemical_effector_um=1.8, local_ph=8.3, local_eh_mv=-80.0
        )
        @test ct_neutral.direct_signal_perception_factor == 1.0
        @test ct_neutral.chemical_effector_factor == 1.0
        @test ct_neutral.environment_lactonolysis_rate_h == 0.0
        @test ct_competed.direct_signal_perception_factor < 0.60
        @test ct_competed.chemical_effector_factor < 0.60
        @test ct_competed.environment_lactonolysis_rate_h > 0.50
        @test ct_competed.net_receptor_drive_multiplier < 0.35

        # 6F. M4: Coupled Droop quota, RNAIII vs alpha-PSM EV biogenesis (mBio Oct 2026),
        #         and S. aureus fruA/fruK vancomycin-tolerant heterogeneity (ISME J Oct 2026)
        init_reg = BiofilmRegulatoryState(
            time_h = 0.0,
            extracellular_nutrient_mm = 4.0,
            intracellular_quota_q = 0.85,
            autoinducer_um = 0.12,
            receptor_activity_r = 0.08,
            rnaiii_activity = 0.08,
            alpha_psm_activity = 0.08,
            ev_particles_per_ml = 1.0e6,
            frac_basal_planktonic = 0.82,
            frac_fructose_hirt_tolerant = 0.06,
            frac_metabolically_reduced = 0.06,
            frac_oxidative_stress_repair = 0.06
        )
        reg_params = BiofilmRegulatoryParams(
            topology = QSRegulatoryTopology(topology_id=:staphylococcus_agr_rnaiii_psm, rnaiii_psm_decoupled=true),
            oxidative_stress_index = 1.2,
            fructose_flux_index = 1.8,
            vancomycin_exposure_ug_ml = 4.0
        )
        traj = simulate_biofilm_regulatory_trajectory(
            init_reg, reg_params;
            times_h = collect(0.0:2.0:12.0),
            electrokinetic_spec = ek_spec,
            intact_matrix = comp_wt,
            crosstalk_spec = ct_spec,
            non_cognate_signal_um = 0.2,
            chemical_effector_um = 0.1,
            local_ph = 7.4
        )
        @test length(traj) == 7
        final_reg = traj[end]
        @test final_reg.extracellular_nutrient_mm < init_reg.extracellular_nutrient_mm
        @test final_reg.ev_particles_per_ml > init_reg.ev_particles_per_ml
        # Under oxidative + nutrient stress, alpha-PSM activity decouples from RNAIII activity
        @test final_reg.alpha_psm_activity > final_reg.rnaiii_activity
        # Under fructose flux + vancomycin exposure, fruA/fruK tolerant and stress subpopulations expand
        @test final_reg.frac_fructose_hirt_tolerant > init_reg.frac_fructose_hirt_tolerant
        @test final_reg.frac_oxidative_stress_repair > init_reg.frac_oxidative_stress_repair
        @test isapprox(
            final_reg.frac_basal_planktonic + final_reg.frac_fructose_hirt_tolerant +
            final_reg.frac_metabolically_reduced + final_reg.frac_oxidative_stress_repair,
            1.0; atol=1e-9
        )

        # Project physiological subpopulations into 3-state CellStatePrediction
        fcm_viab = subpopulations_to_fcm_state(final_reg; total_cells_per_ml=2.5e7, collapse_mode=:viability_3state)
        fcm_tol  = subpopulations_to_fcm_state(final_reg; total_cells_per_ml=2.5e7, collapse_mode=:tolerance_3state)
        @test isapprox(sum(fcm_viab.fractions), 1.0; atol=1e-12)
        @test isapprox(sum(fcm_tol.fractions), 1.0; atol=1e-12)
        @test isapprox(fcm_tol.fractions[1], final_reg.frac_fructose_hirt_tolerant; atol=1e-12)
        @test_throws ArgumentError subpopulations_to_fcm_state(final_reg; collapse_mode=:unknown_mode)
    end

    # =======================================================================
    # 7. Milestone M5 & Multiphysics Bridge Integration:
    #    Domain-shift PSI detection, synthetic-vs-empirical generalization
    #    auditing, and OpenFOAMMeshExchange + PHREEQCSpeciesAdapter bridge
    # =======================================================================
    @testset "Milestone M5: Domain shift, empirical generalization, and mesh/species bridge" begin
        chip1 = generate_synthetic_nanobio_fcs_dataset(
            n_events=150, sample_id="S1", chip_id="CHIP-01", run_id="RUN-1", biological_replicate="REP-1", seed=501)
        chip2 = generate_synthetic_nanobio_fcs_dataset(
            n_events=150, sample_id="S2", chip_id="CHIP-02", run_id="RUN-2", biological_replicate="REP-2", seed=502)

        f1 = extract_single_cell_features(chip1.dataset; transit_rows=chip1.sensor_rows)
        f2 = extract_single_cell_features(chip2.dataset; transit_rows=chip2.sensor_rows)

        # In-distribution chips exhibit low PSI and z-shift
        shift_in_dist = detect_feature_domain_shift(
            f1.features, f2.features;
            feature_names = f1.feature_names,
            psi_threshold = 0.35,
            z_threshold = 1.5
        )
        @test !shift_in_dist.shifted
        @test shift_in_dist.max_abs_z_shift < 1.0

        # Non-finite values in X_train or X_eval are rejected, and constant/tied features tolerate machine-epsilon noise
        X_nan = copy(f2.features)
        X_nan[1, 1] = NaN
        @test_throws ArgumentError detect_feature_domain_shift(f1.features, X_nan; feature_names=f1.feature_names)
        X_const_tr = zeros(Float64, 20, 2)
        X_const_ev = fill(1.0e-12, 10, 2)
        shift_const = detect_feature_domain_shift(X_const_tr, X_const_ev; feature_names=["c1", "c2"])
        @test !shift_const.shifted

        # Perturbed out-of-distribution evaluation matrix triggers domain-shift alert
        X_shifted = copy(f2.features)
        X_shifted[:, 1] .+= 4.5 .* std(f1.features[:, 1])
        shift_ood = detect_feature_domain_shift(
            f1.features, X_shifted;
            feature_names = f1.feature_names,
            psi_threshold = 0.25,
            z_threshold = 1.5
        )
        @test shift_ood.shifted
        @test f1.feature_names[1] in shift_ood.shifted_features
        @test shift_ood.max_psi > 0.25
        @test shift_ood.max_abs_z_shift > 3.0

        # Train classifier on CHIP-01 and evaluate on CHIP-02 for :nanocarrier_uptake
        clf_up = train_cell_classifier(
            f1.features, chip1.labels[:nanocarrier_uptake];
            feature_names = f1.feature_names,
            target_task = :nanocarrier_uptake,
            group_ids = fill("CHIP-01", 150),
            split_group_key = :chip_id
        )
        preds_up = predict_cells(clf_up, f2.features)
        rep_up = evaluate_classifier_predictions(
            preds_up, chip2.labels[:nanocarrier_uptake],
            default_classes_for_target(:nanocarrier_uptake);
            target_task = :nanocarrier_uptake,
            group_ids = fill("CHIP-02", 150)
        )
        @test rep_up.accepted_accuracy > 0.90

        # Synthetic benchmark protocol must block empirical generalization claims despite >90% accuracy
        synth_proto = EmpiricalValidationProtocol(
            protocol_id = "SYNTH-UPTAKE-01",
            track = :track1_nanobio,
            data_provenance = :synthetic_simulation,
            independent_reference_labels = false,
            counting_beads_verified = true,
            internalization_imaging_verified = false,
            n_independent_runs = 2,
            n_heldout_runs = 1,
            external_solvers_coupled_in_production = false
        )
        audit_synth = audit_classifier_empirical_generalization(
            rep_up, synth_proto; feature_psi_drift = shift_in_dist.max_psi
        )
        @test !audit_synth.empirically_generalizable
        @test audit_synth.synthetic_benchmark_only
        @test audit_synth.claim_level == :synthetic_software_benchmark_only
        @test audit_synth.internalization_surface_warning
        @test "nanocarrier_uptake_surface_vs_internalization_unverified" in audit_synth.blocking_reasons

        # Full empirical protocol with quenched internalization control passes readiness check
        emp_proto = EmpiricalValidationProtocol(
            protocol_id = "EMP-UPTAKE-02",
            track = :track1_nanobio,
            data_provenance = :deidentified_experimental_fcs,
            independent_reference_labels = true,
            counting_beads_verified = true,
            internalization_imaging_verified = true,
            n_independent_runs = 3,
            n_heldout_runs = 1,
            external_solvers_coupled_in_production = false
        )
        audit_emp = audit_classifier_empirical_generalization(
            rep_up, emp_proto; feature_psi_drift = shift_in_dist.max_psi
        )
        @test audit_emp.empirically_generalizable
        @test !audit_emp.synthetic_benchmark_only
        @test audit_emp.claim_level == :empirical_heldout_generalization_verified
        @test isempty(audit_emp.blocking_reasons)

        # Bridge with full OpenFOAMMeshExchange and PHREEQCSpeciesAdapter
        cond = MicrofluidicAssayCondition(chip_id="CHIP-02", run_id="RUN-2", sample_id="S2", biological_replicate="REP-2")
        joined = join_cytometry_and_microfluidics(
            chip2.dataset, sensor_rows_to_transit_events(chip2.sensor_rows), cond;
            predictions = preds_up
        )
        fcm_up = summarize_to_fcm_sample(
            preds_up;
            bead_events = 100, beads_added = 12_000.0, sample_volume_ml = 0.10,
            sample_origin = :effluent,
            state_order = (:high_uptake_active, :low_uptake_passive, :unbound_negative)
        )
        cal = GateCalibration([0.93 0.04 0.01; 0.05 0.91 0.05; 0.02 0.05 0.94]; detection_efficiency=0.90)
        mesh_ex = OpenFOAMMeshExchange(
            vessel_id = "CHIP-02", time_h = 2.0, timestep_s = 0.02,
            compartment = :effluent,
            cell_centers_m = [0.0 0.0 0.0; 1.0e-4 0.0 0.0],
            cell_volumes_m3 = [1.0e-10, 1.0e-10],
            internal_field = [2.5, 2.1]
        )
        ph_ad = PHREEQCSpeciesAdapter(
            vessel_id = "CHIP-02", time_h = 2.0, compartment = :effluent,
            pH = 7.2, pe = 4.0, ionic_strength_M = 0.10,
            species_molalities_mol_kgw = Dict("Na+" => 0.10, "Cl-" => 0.10),
            species_charges = Dict("Na+" => 1, "Cl-" => -1)
        )
        bridge_mp = bridge_to_observation_model(
            joined, fcm_up.sample_summary, cal;
            state_prediction = fcm_up.state_prediction,
            mesh_transport = mesh_ex,
            species_chemistry = ph_ad,
            empirical_protocol = synth_proto,
            vessel_id = "CHIP-02",
            time_h = 2.0,
            heldout_validated = true
        )
        # Because synth_proto contains synthetic events, audit_record.heldout_validated is downgraded to false
        @test !bridge_mp.audit_record.heldout_validated
        @test any(q -> startswith(q.name, "openfoam_mesh_exchange_"), bridge_mp.audit_record.quantities)
        @test any(q -> q.name == "phreeqc_species_ph", bridge_mp.audit_record.quantities)

        # Experimental protocol without quenched internalization imaging on :nanocarrier_uptake also downgrades heldout_validated
        emp_no_img_proto = EmpiricalValidationProtocol(
            protocol_id = "EMP-UPTAKE-NO-IMG",
            track = :track1_nanobio,
            data_provenance = :deidentified_experimental_fcs,
            independent_reference_labels = true,
            counting_beads_verified = true,
            internalization_imaging_verified = false,
            n_independent_runs = 3,
            n_heldout_runs = 1
        )
        bridge_no_img = bridge_to_observation_model(
            joined, fcm_up.sample_summary, cal;
            state_prediction = fcm_up.state_prediction,
            empirical_protocol = emp_no_img_proto,
            vessel_id = "CHIP-02",
            time_h = 2.0,
            heldout_validated = true
        )
        @test !bridge_no_img.audit_record.heldout_validated
    end

    # =======================================================================
    # 8. Milestone M6.1 (Gate G0): Default Biofilm Parameter Provenance Registry
    # =======================================================================
    @testset "Milestone M6.1: Default biofilm parameter provenance registry and Gate G0 audit" begin
        @test ENGINE_VERSION == "0.4.0"

        default_reg = build_default_biofilm_provenance_registry()
        @test length(default_reg.entries) == 54
        @test all(e -> e.physical_role != :unspecified, values(default_reg.entries))

        # Verify every numeric parameter across all 6 biofilm structs + QSRegulatoryTopology is registered
        required_symbols = [
            :pga_kg_m3, :eps_kg_m3, :curli_kg_m3, :cellulose_kg_m3, :porosity,
            :film_thickness_um, :film_thickness_m, :substrate_modulus_pa, :poisson_ratio,
            :external_osmotic_pressure_pa, :pga_osmotic_coeff_pa_m3_kg, :eps_osmotic_coeff_pa_m3_kg,
            :basal_gel_modulus_pa, :eps_modulus_coeff_pa_m3_kg, :curli_modulus_coeff_pa_m3_kg,
            :cellulose_modulus_coeff_pa_m3_kg, :poroelastic_biot_modulus_pa, :viscoelastic_relaxation_time_s,
            :aqueous_diffusivity_m2_s, :probe_radius_nm, :probe_charge_valence, :ionic_strength_mol_l,
            :reference_mesh_size_nm, :reference_mesh_size_m, :curli_sieving_coeff, :charge_binding_coeff,
            :electric_field_v_m, :ros_generation_factor, :k_electric_ros_s, :k_enzymatic_qq_s,
            :matrix_shielding_coeff, :direct_electrochemical_fraction,
            :direct_receptor_competition_coeff, :chemical_effector_inhibition_coeff,
            :alkaline_lactonolysis_coeff, :alkaline_lactonolysis_coeff_s, :redox_modulation_coeff,
            :activation_threshold_um, :hill_coefficient, :pqse_chaperone_coupling,
            :nutrient_uptake_rate_mm_h, :nutrient_half_sat_mm, :min_droop_quota,
            :quota_replenish_rate_h, :quota_consumption_rate_h, :ai_basal_synth_um_h,
            :ai_induced_synth_um_h, :ai_natural_decay_h, :ev_basal_release_per_ml_h,
            :ev_psm_induced_release_per_ml_h, :ev_psm_induced_release_per_m3_s,
            :oxidative_stress_index, :fructose_flux_index, :vancomycin_exposure_ug_ml
        ]
        @test length(required_symbols) == 54
        for sym in required_symbols
            @test haskey(default_reg.entries, sym)
        end

        g0_audit = audit_parameter_provenance_registry(default_reg; require_all_calibrated=false)
        @test g0_audit.gate_g0_dimensional_passed
        @test g0_audit.audit_passed
        @test g0_audit.dimensional_audit.equation_homogeneity_ok
        @test isempty(g0_audit.dimensional_audit.mismatched_quantities)
        # Because default registry includes assumed/uncalibrated entries (e.g. ev_psm_induced_release_per_m3_s),
        # it must not claim all_parameters_calibrated = true
        @test !g0_audit.all_parameters_calibrated
        for uncal_sym in (:pga_kg_m3, :eps_kg_m3, :curli_kg_m3, :cellulose_kg_m3, :reference_mesh_size_nm, :reference_mesh_size_m, :curli_sieving_coeff, :pga_osmotic_coeff_pa_m3_kg, :ev_psm_induced_release_per_m3_s)
            @test uncal_sym in g0_audit.uncalibrated_parameters
            @test default_reg.entries[uncal_sym].source_status == :assumed
            @test !default_reg.entries[uncal_sym].calibrated
        end
    end

    # =======================================================================
    # 9. Milestone M6.3 (Gate G3): Constitutive Maturity, Multi-Probe Diffusivity,
    #    Non-Uniform 1D FV Reactive Transport, and Spatial vs 0D AHL Shielding
    # =======================================================================
    @testset "Milestone M6.3: Constitutive maturity levels, multi-probe diffusivity calibration, non-uniform 1D FV transport, and spatial AHL shielding" begin
        comp_wt = BiofilmMatrixComposition(pga_kg_m3=18.0, eps_kg_m3=28.0, curli_kg_m3=12.0, cellulose_kg_m3=6.0, porosity=0.76)

        # 1. Constitutive maturity level tagging and poroelastic/viscoelastic timescales
        spec_l1 = ThinFilmSwellingMechanicsSpec(constitutive_maturity_level=:level1_phenomenological_hypothesis)
        @test spec_l1.constitutive_maturity_level isa ConstitutiveMaturityLevel
        spec_l3 = ThinFilmSwellingMechanicsSpec(
            constitutive_maturity_level=:level3_coupled_poroelastic_viscoelastic,
            poroelastic_biot_modulus_pa=3200.0,
            viscoelastic_relaxation_time_s=240.0
        )
        @test_throws ArgumentError ThinFilmSwellingMechanicsSpec(constitutive_maturity_level=:invalid_level)

        mech_l1 = evaluate_biofilm_swelling_mechanics(comp_wt, spec_l1)
        mech_l3 = evaluate_biofilm_swelling_mechanics(comp_wt, spec_l3)
        @test mech_l1.constitutive_maturity_level == :level1_phenomenological_hypothesis
        @test mech_l3.constitutive_maturity_level == :level3_coupled_poroelastic_viscoelastic
        @test mech_l3.viscoelastic_relaxed_modulus_pa < mech_l3.composite_modulus_pa
        @test mech_l3.poroelastic_consolidation_time_s > 0.0

        # 2. Multi-probe diffusivity calibration and physical monotonicity invariants
        cal_default = calibrate_and_audit_effective_diffusivity(comp_wt)
        @test cal_default.size_exclusion_monotonic
        @test cal_default.curli_sieving_monotonic
        @test cal_default.cationic_retardation_verified
        @test cal_default.debye_screening_monotonic
        @test cal_default.all_physical_invariants_satisfied
        @test cal_default.n_calibration_points == 12
        @test cal_default.calibration_rmse <= 1e-12

        # Perturbed experimental probe calibration points report positive RMSE
        pt_pert = ProbeDiffusivityCalibrationPoint(
            probe_name = "FITC-dextran-4kDa",
            probe_radius_nm = 1.4,
            probe_charge_valence = 0.0,
            ionic_strength_mol_l = 0.15,
            measured_relative_diffusivity = 0.45,
            composition = comp_wt
        )
        cal_custom = calibrate_and_audit_effective_diffusivity(comp_wt; calibration_points=[pt_pert])
        @test cal_custom.all_physical_invariants_satisfied
        @test cal_custom.n_calibration_points == 1
        @test isfinite(cal_custom.calibration_rmse) && cal_custom.calibration_rmse >= 0.0
        @test_throws ArgumentError ProbeDiffusivityCalibrationPoint(
            probe_name="", probe_radius_nm=1.0, probe_charge_valence=0.0,
            ionic_strength_mol_l=0.1, measured_relative_diffusivity=0.5
        )

        # 3. Non-uniform 1D finite-volume reactive transport (refined near boundary layer)
        x_nonunif = [0.0, 2.0e-6, 5.0e-6, 10.0e-6, 20.0e-6, 40.0e-6, 70.0e-6, 110.0e-6]
        n_nu = length(x_nonunif)
        comps_nu = [
            BiofilmMatrixComposition(
                pga_kg_m3 = 10.0 + 1.5 * i,
                eps_kg_m3 = 20.0 + 2.0 * i,
                curli_kg_m3 = 4.0 + 1.5 * i,
                cellulose_kg_m3 = 5.0,
                porosity = 0.82 - 0.02 * i
            )
            for i in 1:n_nu
        ]
        sol_nu = solve_1d_reactive_transport_biofilm(
            x_grid_m = x_nonunif,
            times_s = [0.0, 5.0, 15.0, 30.0],
            compositions = comps_nu,
            initial_concentration_mol_m3 = fill(0.40, n_nu),
            first_order_uptake_s = 0.015,
            boundary_left_flux_mol_m2_s = 1.0e-6,
            boundary_right_flux_mol_m2_s = 0.5e-6
        )
        @test sol_nu.nonuniform_grid
        @test sol_nu.mass_conserved
        @test sol_nu.relative_mass_balance_error <= 1e-6
        @test all(all(c .>= 0.0) for c in sol_nu.profiles_mol_m3)

        # Explicit cell_widths_m vector validation
        @test_throws ArgumentError solve_1d_reactive_transport_biofilm(
            x_grid_m = x_nonunif,
            times_s = [0.0, 5.0],
            compositions = comps_nu,
            cell_widths_m = [1.0e-6, 2.0e-6] # wrong length
        )

        # 4. Direct electrochemical vs ROS-mediated vs enzymatic quorum-quenching separation
        ek_spec = ElectrokineticQSSpec(
            electric_field_v_m = 2000.0,
            ros_generation_factor = 1.2,
            k_electric_ros_s = 0.020,
            k_enzymatic_qq_s = 0.005,
            matrix_shielding_coeff = 0.07,
            direct_electrochemical_fraction = 0.35
        )
        deg_res = evaluate_qs_signal_degradation(1.5, ek_spec; intact_biofilm_matrix=comp_wt)
        @test isapprox(
            deg_res.direct_electrochemical_rate_s + deg_res.ros_mediated_degradation_rate_s,
            deg_res.abiotic_electric_ros_rate_s; rtol=1e-12
        )
        @test isapprox(
            deg_res.direct_electrochemical_rate_s / deg_res.abiotic_electric_ros_rate_s,
            0.35; rtol=1e-12
        )
        @test deg_res.enzymatic_quorum_quenching_rate_s == deg_res.biotic_quorum_quenching_rate_s

        # 5. Spatial 1D reaction-diffusion vs 0D multiplicative AHL shielding comparator
        shield_cmp = compare_spatial_vs_multiplicative_ahl_shielding(
            comp_wt, ek_spec;
            initial_signal_um = 2.0,
            film_thickness_um = 120.0,
            exposure_duration_s = 45.0,
            n_cells = 32
        )
        @test shield_cmp.spatial_basal_signal_um > shield_cmp.spatial_surface_signal_um
        @test shield_cmp.thiele_modulus > 0.0
        @test shield_cmp.penetration_depth_um > 0.0
        @test shield_cmp.relative_discrepancy > 0.0
    end

    # =======================================================================
    # 10. Milestone M6.4 (Gate G4): Auditable Regulatory Graph G=(V,E),
    #     Species Separation, and Droop Quota Conservation
    # =======================================================================
    @testset "Milestone M6.4: RegulatoryNetworkGraph species separation and Droop quota mass conservation" begin
        # 1. Monoculture S. aureus vs P. aeruginosa vs polymicrobial co-culture regulatory graphs
        g_sa = build_species_regulatory_network(:staphylococcus_aureus)
        g_pa = build_species_regulatory_network(:pseudomonas_aeruginosa)
        g_poly = build_species_regulatory_network(:polymicrobial_co_culture)

        @test g_sa.primary_species == :staphylococcus_aureus
        @test !g_sa.allow_cross_species_edges
        @test "RNAIII" in g_sa.nodes && "alpha_PSM" in g_sa.nodes
        @test !("PqsE_Chaperone" in g_sa.nodes)
        @test all(e -> !isempty(e.signal_molecule) && !isempty(e.receptor_target) && !isempty(e.environmental_conditions), g_sa.edges)

        @test g_pa.primary_species == :pseudomonas_aeruginosa
        @test !g_pa.allow_cross_species_edges
        @test "PqsE_Chaperone" in g_pa.nodes && "RhlR_Regulon" in g_pa.nodes

        @test g_poly.primary_species == :polymicrobial_co_culture
        @test g_poly.allow_cross_species_edges
        @test length(g_poly.edges) > length(g_sa.edges) + length(g_pa.edges)
        @test any(e -> e.sign == :inhibition, g_poly.edges)

        # Explicit RegulatoryInteractionEdge construction with sign = :inhibition and full M6.4 metadata
        inhib_edge = RegulatoryInteractionEdge(
            source_node = "HQNO_Siderophore",
            target_node = "AgrC_AgrA",
            species_scope = :polymicrobial_shared,
            signal_producer = :pseudomonas_aeruginosa,
            signal_molecule = "HQNO + pyoverdine",
            receiving_species = :staphylococcus_aureus,
            receptor_target = "Cytochrome b / AgrCA respiratory regulon",
            sign = :inhibition,
            mechanism_class = :chemical_effector,
            environmental_conditions = "Polymicrobial co-culture, pH 7.2, microaerobic",
            parameter_uncertainty = (0.05, 0.45),
            evidence_status = :calibrated,
            citation = "Hotterbeekx et al. (2017) Front. Cell. Infect. Microbiol. 7:106"
        )
        @test inhib_edge.sign == :inhibition
        @test inhib_edge.signal_producer == :pseudomonas_aeruginosa
        @test inhib_edge.receiving_species == :staphylococcus_aureus
        @test inhib_edge.parameter_uncertainty == (0.05, 0.45)

        # 2. Validate species applicability contract and catch cross-species Chimera errors
        params_sa_ok = BiofilmRegulatoryParams(
            topology = QSRegulatoryTopology(topology_id=:staphylococcus_agr_rnaiii_psm, pqse_chaperone_coupling=0.0),
            vancomycin_exposure_ug_ml = 4.0
        )
        app_sa_ok = validate_regulatory_network_applicability(g_sa, params_sa_ok)
        @test app_sa_ok.valid
        @test app_sa_ok.species_separation_enforced
        @test isempty(app_sa_ok.blocking_violations)

        # Applying P. aeruginosa PqsE chaperone coupling > 0 to S. aureus monoculture is rejected
        params_chimera = BiofilmRegulatoryParams(
            topology = QSRegulatoryTopology(topology_id=:staphylococcus_agr_rnaiii_psm, pqse_chaperone_coupling=0.85)
        )
        app_chimera = validate_regulatory_network_applicability(g_sa, params_chimera)
        @test !app_chimera.valid
        @test "pseudomonas_pqse_chaperone_coupling_nonzero_in_staphylococcus" in app_chimera.blocking_violations

        # Applying S. aureus Agr topology or vancomycin fruA/fruK tolerance to P. aeruginosa monoculture is rejected
        app_pa_bad = validate_regulatory_network_applicability(g_pa, params_sa_ok)
        @test !app_pa_bad.valid
        @test "staphylococcus_agr_topology_applied_to_pseudomonas_monoculture" in app_pa_bad.blocking_violations
        @test "staphylococcus_vancomycin_fruA_fruK_tolerance_applied_to_pseudomonas" in app_pa_bad.blocking_violations

        # Calibrated edge without citation or edge referencing unknown node throws ArgumentError
        @test_throws ArgumentError RegulatoryInteractionEdge(
            source_node="A", target_node="B", evidence_status=:calibrated, citation=""
        )
        @test_throws ArgumentError build_species_regulatory_network(:unknown_bacterium)

        # 3. Closed-system Droop intracellular nutrient quota mass conservation: N(t) + Q(t)*X(t) = const
        droop_audit = audit_droop_quota_conservation(
            initial_nutrient_mm = 6.0,
            initial_quota_mmol_gdw = 0.65,
            initial_biomass_gdw_l = 0.40,
            min_quota_mmol_gdw = 0.22,
            max_quota_mmol_gdw = 1.25,
            max_uptake_rate_mmol_gdw_h = 0.50,
            half_saturation_mm = 0.75,
            max_growth_rate_h = 0.38,
            times_h = collect(0.0:1.0:16.0)
        )
        @test droop_audit.mass_conserved
        @test droop_audit.quota_bounds_respected
        @test droop_audit.max_relative_conservation_error <= 1e-13
        @test droop_audit.extracellular_nutrient_mm[end] < droop_audit.extracellular_nutrient_mm[1]
        @test droop_audit.biomass_gdw_l[end] > droop_audit.biomass_gdw_l[1]
        @test isapprox(droop_audit.initial_inventory_mm, droop_audit.final_inventory_mm; rtol=1e-13)
    end

    # =======================================================================
    # 11. Milestone M6.5 (Gate G5): Controlled Ablation Matrix (R0..R5 + Pairwise),
    #     Validation Ladder (G0..G5), Multivariate Mahalanobis OOD, and Stratified Gates
    # =======================================================================
    @testset "Milestone M6.5: Controlled ablation matrix R0-R5, ValidationLadderReport G0-G5, multivariate Mahalanobis OOD, and stratified gates" begin

        # -------------------------------------------------------------------
        # 1. Controlled ablation matrix: all 17 configurations evaluated (Knock-in mode)
        # -------------------------------------------------------------------
        ablation = run_biofilm_ablation_matrix()
        @test ablation.n_configurations_evaluated == 17 # R0 + 5 single + 10 pairwise + R_full
        @test haskey(ablation.runs, :R0)
        @test haskey(ablation.runs, :R_full)

        # 5 individual knock-in configurations
        for i in 1:5
            @test haskey(ablation.runs, Symbol("R$i"))
            @test haskey(ablation.main_effects, Symbol("R$i"))
            @test haskey(ablation.first_order_sensitivity_indices, Symbol("R$i"))
            @test haskey(ablation.total_order_sensitivity_indices, Symbol("R$i"))
        end

        # 10 pairwise interaction configurations (C(5,2) = 10)
        pairwise_keys = [
            :R12, :R13, :R14, :R15,
            :R23, :R24, :R25,
            :R34, :R35,
            :R45
        ]
        @test length(pairwise_keys) == 10
        @test length(ablation.pairwise_interactions) == 10
        @test length(ablation.pairwise_sensitivity_indices) == 10
        for k in pairwise_keys
            @test haskey(ablation.runs, k)
        end

        # Verify physical output observables across all 17 configurations
        for (cfg_name, cfg_res) in ablation.runs
            @test isfinite(cfg_res.wrinkling_index) && cfg_res.wrinkling_index >= 0.0
            @test isfinite(cfg_res.relative_diffusivity) && (0.0 <= cfg_res.relative_diffusivity <= 1.0)
            @test isfinite(cfg_res.final_autoinducer_um) && cfg_res.final_autoinducer_um >= 0.0
            @test isfinite(cfg_res.final_ev_particles_per_ml) && cfg_res.final_ev_particles_per_ml >= 0.0
            @test isfinite(cfg_res.final_tolerant_fraction) && (0.0 <= cfg_res.final_tolerant_fraction <= 1.0)
            @test isfinite(cfg_res.fcm_intact_fraction) && (0.0 <= cfg_res.fcm_intact_fraction <= 1.0)
            @test isfinite(cfg_res.composite_response) && cfg_res.composite_response > 0.0
        end

        # Hypothesis-specific physical responses
        # R0: homogeneous matrix, no PGA swelling
        @test ablation.runs[:R0].wrinkling_index == 0.0
        # R1: PGA swelling induces wrinkling instability
        @test ablation.runs[:R1].wrinkling_index > 0.0
        # R2: curli/matrix tortuosity reduces probe diffusivity vs homogeneous R0
        @test ablation.runs[:R2].relative_diffusivity < ablation.runs[:R0].relative_diffusivity
        # R4: nutrient/oxidative stress EV biogenesis maintains/increases EV particles
        @test ablation.runs[:R4].final_ev_particles_per_ml >= ablation.runs[:R0].final_ev_particles_per_ml
        # R5: electrokinetic ROS/AHL degradation reduces final autoinducer vs R0
        @test ablation.runs[:R5].final_autoinducer_um < ablation.runs[:R0].final_autoinducer_um
        # R_full: combined mechanisms active
        @test ablation.runs[:R_full].wrinkling_index > 0.0
        @test isfinite(ablation.runs[:R_full].composite_response)

        # -------------------------------------------------------------------
        # 2. Main effect and pairwise epistatic interaction effect estimators
        # -------------------------------------------------------------------
        y0 = ablation.runs[:R0].composite_response

        # Main effect definition: Delta_i = Y_i - Y_0
        for i in 1:5
            ri = Symbol("R$i")
            expected_delta = ablation.runs[ri].composite_response - y0
            @test isapprox(ablation.main_effects[ri], expected_delta; atol=1e-12)
        end

        # Pairwise interaction effect definition: Delta_ij = Y_ij - Y_i - Y_j + Y_0
        for i in 1:4, j in (i + 1):5
            ri = Symbol("R$i")
            rj = Symbol("R$j")
            rij = Symbol("R$(i)$(j)")
            pair_key = (ri, rj)
            @test haskey(ablation.pairwise_interactions, pair_key)
            expected_interaction = ablation.runs[rij].composite_response -
                ablation.runs[ri].composite_response -
                ablation.runs[rj].composite_response + y0
            @test isapprox(ablation.pairwise_interactions[pair_key], expected_interaction; atol=1e-12)
        end

        # First-order sensitivity indices: S_i in [0, 1] and sum(S_i) <= 1.0
        sum_sens = 0.0
        for i in 1:5
            ri = Symbol("R$i")
            s_i = ablation.first_order_sensitivity_indices[ri]
            s_ti = ablation.total_order_sensitivity_indices[ri]
            @test 0.0 <= s_i <= 1.0
            @test 0.0 <= s_ti <= 1.0
            @test s_ti >= s_i # Total order sensitivity must be >= first order
            sum_sens += s_i
        end
        @test sum_sens <= 1.0 + 1e-12

        # -------------------------------------------------------------------
        # 3. Knock-out mode ablation matrix: 17 configurations
        # -------------------------------------------------------------------
        ablation_ko = run_biofilm_ablation_matrix(mode=:knock_out)
        @test ablation_ko.n_configurations_evaluated == 17
        @test haskey(ablation_ko.runs, :R0)
        @test haskey(ablation_ko.runs, :R_null)
        for i in 1:5
            @test haskey(ablation_ko.runs, Symbol("R$i"))
            @test haskey(ablation_ko.main_effects, Symbol("R$i"))
            @test haskey(ablation_ko.first_order_sensitivity_indices, Symbol("R$i"))
            @test haskey(ablation_ko.total_order_sensitivity_indices, Symbol("R$i"))
            @test ablation_ko.total_order_sensitivity_indices[Symbol("R$i")] >= ablation_ko.first_order_sensitivity_indices[Symbol("R$i")]
        end
        y0_ko = ablation_ko.runs[:R0].composite_response
        for i in 1:5
            ri = Symbol("R$i")
            expected_loss = y0_ko - ablation_ko.runs[ri].composite_response
            @test isapprox(ablation_ko.main_effects[ri], expected_loss; atol=1e-12)
        end
        for i in 1:4, j in (i + 1):5
            ri = Symbol("R$i")
            rj = Symbol("R$j")
            rij = Symbol("R$(i)$(j)")
            pair_key = (ri, rj)
            @test haskey(ablation_ko.pairwise_interactions, pair_key)
            expected_ko_interaction = ablation_ko.runs[ri].composite_response +
                ablation_ko.runs[rj].composite_response -
                ablation_ko.runs[rij].composite_response - y0_ko
            @test isapprox(ablation_ko.pairwise_interactions[pair_key], expected_ko_interaction; atol=1e-12)
        end

        # -------------------------------------------------------------------
        # 4. Multivariate Mahalanobis distance & covariance Frobenius shift
        # -------------------------------------------------------------------
        rng_mv = MersenneTwister(20261031)
        n_tr = 120
        z1 = randn(rng_mv, n_tr)
        z2 = 0.92 .* z1 .+ 0.39 .* randn(rng_mv, n_tr) # strongly positively correlated in training
        X_corr_tr = hcat(z1, z2)

        e1 = randn(rng_mv, 60)
        e2 = -0.92 .* e1 .+ 0.39 .* randn(rng_mv, 60) # correlation flipped (-0.92) in evaluation
        X_corr_ev = hcat(e1, e2)

        shift_mv = detect_feature_domain_shift(
            X_corr_tr, X_corr_ev;
            feature_names = ["f_corr_1", "f_corr_2"],
            psi_threshold = 0.50,
            z_threshold = 2.5,
            covariance_shift_threshold = 0.80,
            ood_fraction_threshold = 0.15
        )
        @test shift_mv.max_abs_z_shift < 1.0 # univariate means match
        @test shift_mv.covariance_frobenius_rel_shift > 0.80 # correlation flip caught by covariance Frobenius shift
        @test shift_mv.multivariate_shift_detected
        @test shift_mv.shifted
        @test shift_mv.recommended_abstention_policy in (:abstain_and_recalibrate, :elevated_deferral_threshold)

        # High-dimensional d >> n regime (d = 40, n_tr = 12, n_ev = 6)
        X_hd_tr = randn(rng_mv, 12, 40)
        X_hd_ev_in = randn(rng_mv, 6, 40)
        X_hd_ev_ood = randn(rng_mv, 6, 40) .+ 3.5
        shift_hd_in = detect_feature_domain_shift(
            X_hd_tr, X_hd_ev_in;
            feature_names = ["f_$j" for j in 1:40]
        )
        shift_hd_ood = detect_feature_domain_shift(
            X_hd_tr, X_hd_ev_ood;
            feature_names = ["f_$j" for j in 1:40]
        )
        @test isfinite(shift_hd_in.multivariate_mahalanobis_mean)
        @test shift_hd_ood.multivariate_mahalanobis_mean > shift_hd_in.multivariate_mahalanobis_mean
        @test shift_hd_ood.ood_event_fraction > 0.50
        @test shift_hd_ood.shifted

        # -------------------------------------------------------------------
        # 5. Stratified classifier empirical audit & OOD blocking
        # -------------------------------------------------------------------
        chip_tr = generate_synthetic_nanobio_fcs_dataset(
            n_events=180, sample_id="S-TR", chip_id="CHIP-TR", run_id="RUN-TR", biological_replicate="REP-1", seed=601)
        chip_ev = generate_synthetic_nanobio_fcs_dataset(
            n_events=180, sample_id="S-EV", chip_id="CHIP-EV", run_id="RUN-EV", biological_replicate="REP-2", seed=602)
        f_tr = extract_single_cell_features(chip_tr.dataset; transit_rows=chip_tr.sensor_rows)
        f_ev = extract_single_cell_features(chip_ev.dataset; transit_rows=chip_ev.sensor_rows)

        clf_m6 = train_cell_classifier(
            f_tr.features, chip_tr.labels[:cell_stress_damage];
            feature_names = f_tr.feature_names,
            target_task = :cell_stress_damage,
            group_ids = fill("CHIP-TR", 180),
            split_group_key = :chip_id
        )
        preds_m6 = predict_cells(clf_m6, f_ev.features)
        rep_groups = vcat(fill("REP-A", 90), fill("REP-B", 90))
        rep_m6 = evaluate_classifier_predictions(
            preds_m6, chip_ev.labels[:cell_stress_damage],
            default_classes_for_target(:cell_stress_damage);
            target_task = :cell_stress_damage,
            group_ids = rep_groups
        )
        @test haskey(rep_m6.group_ece, "REP-A") && haskey(rep_m6.group_ece, "REP-B")
        @test 0.0 <= rep_m6.min_class_recall <= 1.0
        @test 0.0 <= rep_m6.min_group_accuracy <= rep_m6.accepted_accuracy

        emp_proto_stress = EmpiricalValidationProtocol(
            protocol_id = "EMP-STRESS-M6",
            track = :track1_nanobio,
            data_provenance = :deidentified_experimental_fcs,
            independent_reference_labels = true,
            counting_beads_verified = true,
            n_independent_runs = 3,
            n_heldout_runs = 1
        )
        audit_with_mv_shift = audit_classifier_empirical_generalization(
            rep_m6, emp_proto_stress;
            feature_psi_drift = shift_mv.max_psi,
            feature_z_shift = shift_mv.max_abs_z_shift,
            ood_event_fraction = shift_mv.ood_event_fraction,
            max_ood_fraction = 0.15
        )
        @test !audit_with_mv_shift.empirically_generalizable
        @test audit_with_mv_shift.recommended_decision == :abstain_or_defer_domain_shift
        @test !isempty(audit_with_mv_shift.blocking_reasons)

        # -------------------------------------------------------------------
        # 6. Validation Ladder (G0..G5) Parameter Pass-Through & Fail Modes
        # -------------------------------------------------------------------
        # 6.1 Baseline synthetic evaluation: G5 strictly false
        ladder_default = evaluate_validation_ladder()
        @test ladder_default.gate_passed[:G0_dimensional_provenance]
        @test ladder_default.gate_passed[:G1_mesh_conservation_cfl]
        @test ladder_default.gate_passed[:G2_phreeqc_thermodynamic_benchmark]
        @test ladder_default.gate_passed[:G3_constitutive_transport_verification]
        @test ladder_default.gate_passed[:G4_regulatory_graph_quota_audit]
        @test !ladder_default.gate_passed[:G5_empirical_generalization_gate]
        @test ladder_default.highest_passed_gate == :G4_regulatory_graph_quota_audit
        @test ladder_default.numerical_verification_complete
        @test !ladder_default.empirical_generalization_established
        @test ladder_default.claim_boundary == :synthetic_numerical_verification_only

        # 6.2 Clean empirical pass with parameter pass-through
        ladder_emp = evaluate_validation_ladder(
            empirical_protocol = emp_proto_stress,
            heldout_run_accuracy = 0.91,
            feature_psi_drift = 0.04,
            feature_z_shift = 0.20,
            ood_fraction = 0.05,
            minority_class_recall = 0.85,
            replicate_min_accuracy = 0.82
        )
        @test ladder_emp.numerical_verification_complete
        @test ladder_emp.empirical_generalization_established
        @test ladder_emp.gate_passed[:G5_empirical_generalization_gate]
        @test ladder_emp.highest_passed_gate == :G5_empirical_generalization_gate
        @test ladder_emp.claim_boundary == :empirical_generalization_verified

        # 6.3 OOD domain shift fail mode under parameter pass-through (ood_fraction > 0.15)
        ladder_ood_fail = evaluate_validation_ladder(
            empirical_protocol = emp_proto_stress,
            heldout_run_accuracy = 0.91,
            ood_fraction = 0.25,
            max_allowed_ood_fraction = 0.15
        )
        @test !ladder_ood_fail.gate_passed[:G5_empirical_generalization_gate]
        @test !ladder_ood_fail.empirical_generalization_established
        @test ladder_ood_fail.highest_passed_gate == :G4_regulatory_graph_quota_audit
        @test ladder_ood_fail.claim_boundary == :synthetic_numerical_verification_only

        # 6.4 Minority class recall failure under parameter pass-through (recall < 0.75)
        ladder_min_fail = evaluate_validation_ladder(
            empirical_protocol = emp_proto_stress,
            heldout_run_accuracy = 0.91,
            minority_class_recall = 0.60,
            min_minority_class_recall = 0.75
        )
        @test !ladder_min_fail.gate_passed[:G5_empirical_generalization_gate]
        @test !ladder_min_fail.empirical_generalization_established
        @test ladder_min_fail.highest_passed_gate == :G4_regulatory_graph_quota_audit

        # 6.5 Biological replicate stratification failure (replicate accuracy < 0.75)
        ladder_rep_fail = evaluate_validation_ladder(
            empirical_protocol = emp_proto_stress,
            heldout_run_accuracy = 0.91,
            replicate_min_accuracy = 0.65,
            min_replicate_accuracy = 0.75
        )
        @test !ladder_rep_fail.gate_passed[:G5_empirical_generalization_gate]
        @test !ladder_rep_fail.empirical_generalization_established
        @test ladder_rep_fail.highest_passed_gate == :G4_regulatory_graph_quota_audit

        # 6.6 Nanocarrier uptake without internalization imaging fails Gate G5
        proto_uptake_no_img = EmpiricalValidationProtocol(
            protocol_id = "EMP-UPTAKE-NOIMG",
            track = :track1_nanobio,
            data_provenance = :deidentified_experimental_fcs,
            independent_reference_labels = true,
            counting_beads_verified = true,
            internalization_imaging_verified = false,
            n_independent_runs = 3,
            n_heldout_runs = 1
        )
        ladder_uptake_no_img = evaluate_validation_ladder(
            empirical_protocol = proto_uptake_no_img,
            target_task = :nanocarrier_uptake,
            heldout_run_accuracy = 0.92
        )
        @test !ladder_uptake_no_img.gate_passed[:G5_empirical_generalization_gate]
        @test !ladder_uptake_no_img.empirical_generalization_established

        # 6.7 Nanocarrier uptake with internalization imaging passes Gate G5
        proto_uptake_img = EmpiricalValidationProtocol(
            protocol_id = "EMP-UPTAKE-IMG",
            track = :track1_nanobio,
            data_provenance = :deidentified_experimental_fcs,
            independent_reference_labels = true,
            counting_beads_verified = true,
            internalization_imaging_verified = true,
            n_independent_runs = 3,
            n_heldout_runs = 1
        )
        ladder_uptake_img = evaluate_validation_ladder(
            empirical_protocol = proto_uptake_img,
            target_task = :nanocarrier_uptake,
            heldout_run_accuracy = 0.92
        )
        @test ladder_uptake_img.gate_passed[:G5_empirical_generalization_gate]
        @test ladder_uptake_img.empirical_generalization_established

        # -------------------------------------------------------------------
        # 7. Formal 5-Tier Status Flags for Gates G0 through G5 (Control 6)
        # -------------------------------------------------------------------
        # Synthetic default evaluation: experimentally_calibrated & empirically_validated MUST be false
        @test hasfield(typeof(ladder_default), :gate_status_flags)
        for g in VALID_VALIDATION_LADDER_GATES
            @test haskey(ladder_default.gate_status_flags, g)
            st = ladder_default.gate_status_flags[g]
            @test st.implemented == true
            @test st.software_verified == true
            @test st.numerically_benchmarked == true
            @test st.experimentally_calibrated == false
            @test st.empirically_validated == false
        end

        # Test property accessors
        @test length(ladder_default.gate_status) == 6
        @test length(ladder_default.gate_status_5tier) == 6
        @test ladder_default.implemented == true
        @test ladder_default.software_verified == true
        @test ladder_default.numerically_benchmarked == true
        @test ladder_default.experimentally_calibrated == false
        @test ladder_default.empirically_validated == false

        # Empirical evaluation: G5 transitions to experimentally_calibrated=true and empirically_validated=true
        @test haskey(ladder_emp.gate_status_flags, :G5_empirical_generalization_gate)
        g5_st = ladder_emp.gate_status_flags[:G5_empirical_generalization_gate]
        @test g5_st.implemented == true
        @test g5_st.software_verified == true
        @test g5_st.numerically_benchmarked == true
        @test g5_st.experimentally_calibrated == true
        @test g5_st.empirically_validated == true
        @test ladder_emp.empirically_validated == true

        # Failed empirical run (OOD fail) has empirically_validated = false
        @test ladder_ood_fail.gate_status_flags[:G5_empirical_generalization_gate].empirically_validated == false
    end

    # -----------------------------------------------------------------------
    # 13. Milestone M2 (R3.5): Microfluidic Experimental Characterization Contracts & Control 3
    # -----------------------------------------------------------------------
    @testset "Milestone M2: Microfluidic characterization contracts (tapered constriction, developing flow, serpentine focusing)" begin
        # 1. HydrogelBiofilmShearModulusContract
        shear_ctr = evaluate_hydrogel_shear_modulus()
        @test shear_ctr.contract_verified
        @test shear_ctr.constriction_ratio ≈ 50.0 / 15.0 atol=1e-6
        @test shear_ctr.apparent_strain ≈ 1.0 / 3.0 atol=1e-4
        @test shear_ctr.taylor_deformation_parameter ≈ 1.0 / 3.0 atol=1e-4
        @test shear_ctr.strain_hardening_factor > shear_ctr.apparent_strain
        @test shear_ctr.transit_shear_strain > 0.0
        @test shear_ctr.throat_reynolds_number < 10.0
        @test 50.0 < shear_ctr.shear_modulus_pa < 500.0 # Physiological soft hydrogel / biofilm range

        # Linear elastic limit (beta = 0.0)
        shear_linear = evaluate_hydrogel_shear_modulus(strain_hardening_coeff=0.0)
        @test shear_linear.strain_hardening_factor ≈ shear_linear.apparent_strain atol=1e-9
        @test shear_linear.shear_modulus_pa ≈ shear_linear.applied_pressure_drop_pa / (shear_linear.transit_shear_strain * shear_linear.apparent_strain) atol=1e-6

        # Mechanistic status tagging for Contract 1
        shear_tags = categorize_mechanistic_quantities(shear_ctr)
        @test :shear_modulus_pa in shear_tags[:calibrated_constitutive_model]
        @test :entry_width_m in shear_tags[:experimental_contract]
        @test :applied_pressure_drop_pa in shear_tags[:experimental_contract]
        @test :taylor_deformation_parameter in shear_tags[:analytical_solution]
        @test :throat_wall_shear_rate_s in shear_tags[:analytical_solution]
        @test :transit_shear_strain in shear_tags[:numerical_prediction]
        @test :throat_reynolds_number in shear_tags[:numerical_prediction]

        # Contract 1 validation errors
        @test_throws ArgumentError evaluate_hydrogel_shear_modulus(entry_width_m=10.0e-6, throat_width_m=20.0e-6)
        @test_throws ArgumentError evaluate_hydrogel_shear_modulus(applied_pressure_drop_pa=-500.0)

        # 2. DevelopingFlowShearContract and Shah & London (1978) series
        # Circular limit: f * Re = 64
        circ_ctr = evaluate_developing_flow_shear(channel_geometry=:circular, width_m=100.0e-6, height_m=100.0e-6)
        @test circ_ctr.contract_verified
        @test circ_ctr.poiseuille_friction_factor_re ≈ 64.0 atol=1e-9
        @test circ_ctr.hydrodynamic_entrance_length_m ≈ 0.056 * circ_ctr.reynolds_number * circ_ctr.hydraulic_diameter_m atol=1e-9

        # Square limit: f * Re = 56.91
        sq_ctr = evaluate_developing_flow_shear(channel_geometry=:square, width_m=100.0e-6, height_m=100.0e-6)
        @test sq_ctr.contract_verified
        @test sq_ctr.aspect_ratio ≈ 1.0 atol=1e-9
        @test 56.91 <= sq_ctr.poiseuille_friction_factor_re <= 56.92
        @test sq_ctr.poiseuille_friction_factor_re ≈ shah_london_friction_factor_re(1.0) atol=1e-9

        # Rectangular ducts across aspect ratios alpha in [0, 1]
        f_re_plates = shah_london_friction_factor_re(0.0)
        @test f_re_plates ≈ 96.0 atol=1e-9

        # Shah & London monotonicity across alpha: 0.0 -> 0.2 -> 0.5 -> 0.8 -> 1.0
        alphas = [0.0, 0.2, 0.5, 0.8, 1.0]
        f_res = [shah_london_friction_factor_re(a) for a in alphas]
        for i in 1:(length(f_res) - 1)
            @test f_res[i] > f_res[i + 1]
        end
        @test_throws ArgumentError shah_london_friction_factor_re(-0.1)
        @test_throws ArgumentError shah_london_friction_factor_re(1.1)

        # Developing wall shear stress profile tau_w(x)
        rect_ctr = evaluate_developing_flow_shear(channel_geometry=:rectangular, width_m=150.0e-6, height_m=50.0e-6)
        @test rect_ctr.contract_verified
        @test rect_ctr.aspect_ratio ≈ 50.0 / 150.0 atol=1e-6
        @test rect_ctr.shear_monotonic_decay_verified
        @test rect_ctr.peak_inlet_shear_pa > rect_ctr.fully_developed_wall_shear_pa
        @test rect_ctr.developing_wall_shear_pa[end] ≈ rect_ctr.fully_developed_wall_shear_pa rtol=0.20

        # Mechanistic status tagging for Contract 2
        flow_tags = categorize_mechanistic_quantities(rect_ctr)
        @test :poiseuille_friction_factor_re in flow_tags[:analytical_solution]
        @test :hydrodynamic_entrance_length_m in flow_tags[:analytical_solution]
        @test :developing_wall_shear_pa in flow_tags[:numerical_prediction]
        @test :peak_inlet_shear_pa in flow_tags[:numerical_prediction]
        @test :channel_geometry in flow_tags[:experimental_contract]
        @test :channel_length_m in flow_tags[:experimental_contract]
        @test :fluid_viscosity_pa_s in flow_tags[:calibrated_constitutive_model]

        # 3. SerpentineViscoelasticFocusingContract
        visco_ctr = evaluate_serpentine_viscoelastic_focusing()
        @test visco_ctr.contract_verified
        @test visco_ctr.hydraulic_diameter_m ≈ 50.0e-6 atol=1e-9
        @test visco_ctr.dean_number ≈ visco_ctr.reynolds_number * sqrt(visco_ctr.hydraulic_diameter_m / (2.0 * visco_ctr.radius_of_curvature_m)) atol=1e-9
        @test visco_ctr.weissenberg_number ≈ visco_ctr.relaxation_time_s * visco_ctr.characteristic_shear_rate_s atol=1e-9
        @test visco_ctr.focusing_regime_parameter ≈ visco_ctr.weissenberg_number / sqrt(visco_ctr.dean_number) atol=1e-6
        @test visco_ctr.focusing_regime_parameter >= 1.0
        @test visco_ctr.single_line_focusing_predicted
        @test visco_ctr.dean_drag_force_n > 0.0
        @test visco_ctr.elastic_lift_force_n > 0.0
        @test visco_ctr.force_ratio_fe_over_fd > 1.0 # Elastic lift dominates secondary Dean drag

        # Sub-threshold regime (weak elasticity, Wi * De^(-1/2) < 1.0)
        visco_sub = evaluate_serpentine_viscoelastic_focusing(relaxation_time_s=0.00005)
        @test visco_sub.contract_verified
        @test visco_sub.focusing_regime_parameter < 1.0
        @test !visco_sub.single_line_focusing_predicted

        # Mechanistic status tagging for Contract 3
        visco_tags = categorize_mechanistic_quantities(visco_ctr)
        @test :dean_number in visco_tags[:analytical_solution]
        @test :weissenberg_number in visco_tags[:analytical_solution]
        @test :focusing_regime_parameter in visco_tags[:analytical_solution]
        @test :dean_drag_force_n in visco_tags[:numerical_prediction]
        @test :elastic_lift_force_n in visco_tags[:numerical_prediction]
        @test :force_ratio_fe_over_fd in visco_tags[:numerical_prediction]
        @test :particle_diameter_m in visco_tags[:experimental_contract]
        @test :radius_of_curvature_m in visco_tags[:experimental_contract]
        @test :relaxation_time_s in visco_tags[:calibrated_constitutive_model]
        @test :first_normal_stress_difference_pa in visco_tags[:calibrated_constitutive_model]

        # 4. Extended Gate G3 in evaluate_validation_ladder
        ladder = evaluate_validation_ladder()
        @test ladder.gate_passed[:G3_constitutive_transport_verification]
        @test occursin("G=", ladder.gate_summaries[:G3_constitutive_transport_verification])
        @test occursin("fRe=", ladder.gate_summaries[:G3_constitutive_transport_verification])
        @test occursin("Wi*De^(-0.5)=", ladder.gate_summaries[:G3_constitutive_transport_verification])
    end

    # =======================================================================
    # 14. Milestone M3 (Requirements R4.1-R4.4): Species Regulatory Networks,
    #     Chimera Rejection, Droop Mass Balance & Gate G4 Multi-Species Audit
    # =======================================================================
    @testset "Milestone M3 (R4.1-R4.4): Species Regulatory Networks, Chimera Rejection, Droop Mass Balance & Gate G4 Audit" begin
        # 1. S. aureus USA300 monoculture regulatory network (R4.1)
        g_sa_strain = build_species_regulatory_network(:Staphylococcus_aureus_USA300)
        @test g_sa_strain.primary_species in (:staphylococcus_aureus, :Staphylococcus_aureus_USA300)
        @test g_sa_strain.organism_strain == :Staphylococcus_aureus_USA300
        @test !g_sa_strain.allow_cross_species_edges
        @test length(g_sa_strain.nodes) >= 10
        @test "RNAIII" in g_sa_strain.nodes
        @test "alpha_PSM" in g_sa_strain.nodes
        @test "AgrC_AgrA" in g_sa_strain.nodes
        @test g_sa_strain.node_species["RNAIII"] == :Staphylococcus_aureus_USA300
        @test g_sa_strain.node_species["alpha_PSM"] == :Staphylococcus_aureus_USA300
        @test all(e -> e.organism_strain == :Staphylococcus_aureus_USA300, g_sa_strain.edges)
        @test all(e -> !e.is_cross_species_bridge, g_sa_strain.edges)
        @test all(e -> e.interaction_type == :intraspecies_regulation, g_sa_strain.edges)

        # Audit S. aureus monoculture with valid parameters
        sa_valid_params = BiofilmRegulatoryParams(
            topology = QSRegulatoryTopology(topology_id=:staphylococcus_agr_rnaiii_psm, pqse_chaperone_coupling=0.0),
            vancomycin_exposure_ug_ml = 5.0
        )
        sa_audit = validate_regulatory_network_applicability(g_sa_strain, sa_valid_params; requested_species=:Staphylococcus_aureus_USA300)
        @test sa_audit.valid
        @test sa_audit.species_separation_enforced
        @test isempty(sa_audit.blocking_violations)
        @test sa_audit.cross_species_edges_count == 0

        # 2. P. aeruginosa PA14 monoculture regulatory network with PqsE-RhlR chaperone coupling (R4.1)
        g_pa_strain = build_species_regulatory_network(:Pseudomonas_aeruginosa_PA14)
        @test g_pa_strain.primary_species in (:pseudomonas_aeruginosa, :Pseudomonas_aeruginosa_PA14)
        @test g_pa_strain.organism_strain == :Pseudomonas_aeruginosa_PA14
        @test !g_pa_strain.allow_cross_species_edges
        @test "PqsE_Chaperone" in g_pa_strain.nodes
        @test "RhlR_Regulon" in g_pa_strain.nodes
        @test "3OC12_HSL" in g_pa_strain.nodes
        @test "Pyocyanin_HQNO_Effector" in g_pa_strain.nodes
        @test g_pa_strain.node_species["PqsE_Chaperone"] == :Pseudomonas_aeruginosa_PA14
        @test g_pa_strain.node_species["RhlR_Regulon"] == :Pseudomonas_aeruginosa_PA14

        # Specific chaperone-allosteric edge verification
        pqse_edge = only(filter(e -> e.source_node == "PqsE_Chaperone" && e.target_node == "RhlR_Regulon", g_pa_strain.edges))
        @test pqse_edge.mechanism_class == :chaperone_allosteric
        @test pqse_edge.sign == :activation
        @test pqse_edge.strain_scope == "PA14"
        @test pqse_edge.organism_strain == :Pseudomonas_aeruginosa_PA14
        @test !pqse_edge.is_cross_species_bridge
        @test pqse_edge.interaction_type == :intraspecies_regulation
        @test pqse_edge.evidence_status == :observed

        pa_valid_params = BiofilmRegulatoryParams(
            topology = QSRegulatoryTopology(topology_id=:pseudomonas_pqse_rhlr, pqse_chaperone_coupling=0.80),
            vancomycin_exposure_ug_ml = 0.0
        )
        pa_audit = validate_regulatory_network_applicability(g_pa_strain, pa_valid_params; requested_species=:Pseudomonas_aeruginosa_PA14)
        @test pa_audit.valid
        @test pa_audit.species_separation_enforced
        @test isempty(pa_audit.blocking_violations)
        @test pa_audit.cross_species_edges_count == 0

        # 3. Negative Chimera test: structural rejection of cross-wiring and invalid parameterizations (R4.1, R4.2)
        # 3a. Target-node Chimera rejection: foreign target nodes rejected in monoculture graphs
        @test_throws ArgumentError begin
            e_tgt_sa = RegulatoryInteractionEdge(
                source_node = "AgrC_AgrA",
                target_node = "PqsE_Chaperone",
                species_scope = :staphylococcus_aureus,
                citation = "Negative target-node test"
            )
            RegulatoryNetworkGraph(
                graph_id = "CHIMERA-TARGET-SA",
                primary_species = :staphylococcus_aureus,
                nodes = ["AgrC_AgrA", "PqsE_Chaperone"],
                edges = [e_tgt_sa],
                allow_cross_species_edges = false
            )
        end
        @test_throws ArgumentError begin
            e_tgt_pa = RegulatoryInteractionEdge(
                source_node = "LasI_LasR",
                target_node = "alpha_PSM",
                species_scope = :pseudomonas_aeruginosa,
                citation = "Negative target-node test"
            )
            RegulatoryNetworkGraph(
                graph_id = "CHIMERA-TARGET-PA",
                primary_species = :pseudomonas_aeruginosa,
                nodes = ["LasI_LasR", "alpha_PSM"],
                edges = [e_tgt_pa],
                allow_cross_species_edges = false
            )
        end

        # 3b. Foreign nodes in monoculture node list rejected with ArgumentError
        valid_sa_edge = only(filter(e -> e.source_node == "Extracellular_Nutrient_N", g_sa_strain.edges))
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "CHIMERA-NODELIST-SA",
            primary_species = :staphylococcus_aureus,
            nodes = ["Extracellular_Nutrient_N", "Intracellular_Quota_Q", "PqsE_Chaperone"],
            edges = [valid_sa_edge],
            allow_cross_species_edges = false
        )
        valid_pa_edge = only(filter(e -> e.source_node == "3OC12_HSL", g_pa_strain.edges))
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "CHIMERA-NODELIST-PA",
            primary_species = :pseudomonas_aeruginosa,
            nodes = ["Extracellular_Nutrient_N", "Intracellular_Quota_Q", "3OC12_HSL", "LasI_LasR", "AgrC_AgrA"],
            edges = [valid_pa_edge],
            allow_cross_species_edges = false
        )

        # 3c. Foreign node_species & applicability audit
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "CHIMERA-NODESPECIES-SA",
            primary_species = :staphylococcus_aureus,
            nodes = ["Extracellular_Nutrient_N", "Intracellular_Quota_Q"],
            edges = [valid_sa_edge],
            allow_cross_species_edges = false,
            node_species = Dict(
                "Extracellular_Nutrient_N" => :Staphylococcus_aureus_USA300,
                "Intracellular_Quota_Q" => :Pseudomonas_aeruginosa_PA14
            )
        )
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "CHIMERA-NODESPECIES-PA",
            primary_species = :pseudomonas_aeruginosa,
            nodes = ["3OC12_HSL", "LasI_LasR"],
            edges = [valid_pa_edge],
            allow_cross_species_edges = false,
            node_species = Dict(
                "3OC12_HSL" => :Pseudomonas_aeruginosa_PA14,
                "LasI_LasR" => :Staphylococcus_aureus_USA300
            )
        )

        # 3d. Strain validation and biological consistency in RegulatoryInteractionEdge
        @test_throws ArgumentError RegulatoryInteractionEdge(
            source_node = "Extracellular_Nutrient_N",
            target_node = "Intracellular_Quota_Q",
            species_scope = :staphylococcus_aureus,
            organism_strain = :Pseudomonas_aeruginosa_PA14,
            citation = "Strain mismatch negative test"
        )
        @test_throws ArgumentError RegulatoryInteractionEdge(
            source_node = "3OC12_HSL",
            target_node = "LasI_LasR",
            species_scope = :pseudomonas_aeruginosa,
            organism_strain = :Staphylococcus_aureus_USA300,
            citation = "Strain mismatch negative test"
        )
        @test_throws ArgumentError RegulatoryInteractionEdge(
            source_node = "Extracellular_Nutrient_N",
            target_node = "Intracellular_Quota_Q",
            species_scope = :staphylococcus_aureus,
            organism_strain = :completely_bogus_strain,
            citation = "Invalid strain negative test"
        )
        @test_throws ArgumentError RegulatoryInteractionEdge(
            source_node = "Extracellular_Nutrient_N",
            target_node = "Intracellular_Quota_Q",
            species_scope = :staphylococcus_aureus,
            organism_strain = :polymicrobial_consortium,
            citation = "Consortium strain in intraspecies edge negative test"
        )

        # Monoculture graph rejects foreign organism_strain parameter
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "CHIMERA-STRAIN-PARAM-SA",
            primary_species = :staphylococcus_aureus,
            nodes = ["Extracellular_Nutrient_N", "Intracellular_Quota_Q"],
            edges = [valid_sa_edge],
            allow_cross_species_edges = false,
            organism_strain = :Pseudomonas_aeruginosa_PA14
        )
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "CHIMERA-STRAIN-PARAM-PA",
            primary_species = :pseudomonas_aeruginosa,
            nodes = ["3OC12_HSL", "LasI_LasR"],
            edges = [valid_pa_edge],
            allow_cross_species_edges = false,
            organism_strain = :Staphylococcus_aureus_USA300
        )

        # Attempting to wire PqsE chaperone coupling edge directly into S. aureus monoculture graph throws ArgumentError
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "CHIMERA-SA-PA-CHAPERONE",
            primary_species = :staphylococcus_aureus,
            nodes = ["Extracellular_Nutrient_N", "Intracellular_Quota_Q", "PqsE_Chaperone", "RhlR_Regulon"],
            edges = [pqse_edge],
            allow_cross_species_edges = false
        )

        # Attempting to wire S. aureus RNAIII edge directly into P. aeruginosa PA14 monoculture graph throws ArgumentError
        rnaiii_edge = only(filter(e -> e.target_node == "RNAIII", g_sa_strain.edges))
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "CHIMERA-PA-SA-RNAIII",
            primary_species = :pseudomonas_aeruginosa,
            nodes = ["Extracellular_Nutrient_N", "Intracellular_Quota_Q", "AgrC_AgrA", "RNAIII"],
            edges = [rnaiii_edge],
            allow_cross_species_edges = false
        )

        # Untyped cross-species interaction throws ArgumentError at edge construction
        @test_throws ArgumentError RegulatoryInteractionEdge(
            source_node = "HQNO_Toxin",
            target_node = "AgrC_AgrA",
            species_scope = :staphylococcus_aureus,
            signal_producer = :pseudomonas_aeruginosa,
            receiving_species = :staphylococcus_aureus,
            interaction_type = :intraspecies_regulation,
            is_cross_species_bridge = false,
            citation = "Illegal cross-talk without bridge typing"
        )

        # Monoculture graph attempting allow_cross_species_edges=true without polymicrobial consortium throws ArgumentError
        @test_throws ArgumentError RegulatoryNetworkGraph(
            graph_id = "INVALID-MONO-BRIDGE",
            primary_species = :staphylococcus_aureus,
            participating_species = [:staphylococcus_aureus],
            nodes = g_sa_strain.nodes,
            edges = g_sa_strain.edges,
            allow_cross_species_edges = true
        )

        # Chimera parameter audit: foreign PA chaperone topology applied to S. aureus monoculture
        chimera_coupling_params = BiofilmRegulatoryParams(
            topology = QSRegulatoryTopology(topology_id=:pseudomonas_pqse_rhlr, pqse_chaperone_coupling=1.20),
            vancomycin_exposure_ug_ml = 2.0
        )
        chimera_audit = validate_regulatory_network_applicability(g_sa_strain, chimera_coupling_params; requested_species=:staphylococcus_aureus)
        @test !chimera_audit.valid
        @test !chimera_audit.species_separation_enforced
        @test "pseudomonas_topology_applied_to_staphylococcus_monoculture" in chimera_audit.blocking_violations
        @test "pseudomonas_pqse_chaperone_coupling_nonzero_in_staphylococcus" in chimera_audit.blocking_violations

        # Audit catches foreign nodes and foreign node_species in applicability audit
        poly_as_sa_audit = validate_regulatory_network_applicability(
            build_species_regulatory_network(:polymicrobial_consortium; include_polymicrobial_crosstalk=true),
            BiofilmRegulatoryParams(topology=QSRegulatoryTopology(topology_id=:staphylococcus_agr_rnaiii_psm));
            requested_species = :staphylococcus_aureus
        )
        @test !poly_as_sa_audit.valid
        @test !poly_as_sa_audit.species_separation_enforced
        @test any(v -> startswith(v, "foreign_pseudomonas_node_in_staphylococcus_graph"), poly_as_sa_audit.blocking_violations)
        @test any(v -> startswith(v, "foreign_pseudomonas_node_species_in_staphylococcus_graph"), poly_as_sa_audit.blocking_violations)

        poly_as_pa_audit = validate_regulatory_network_applicability(
            build_species_regulatory_network(:polymicrobial_consortium; include_polymicrobial_crosstalk=true),
            BiofilmRegulatoryParams(topology=QSRegulatoryTopology(topology_id=:pseudomonas_pqse_rhlr));
            requested_species = :pseudomonas_aeruginosa
        )
        @test !poly_as_pa_audit.valid
        @test !poly_as_pa_audit.species_separation_enforced
        @test any(v -> startswith(v, "foreign_staphylococcus_node_in_pseudomonas_graph"), poly_as_pa_audit.blocking_violations)
        @test any(v -> startswith(v, "foreign_staphylococcus_node_species_in_pseudomonas_graph"), poly_as_pa_audit.blocking_violations)

        # 4. Polymicrobial regulatory bridge valid execution (R4.1, R4.2)
        g_poly = build_species_regulatory_network(:polymicrobial_consortium; include_polymicrobial_crosstalk=true)
        @test g_poly.primary_species in (:polymicrobial_co_culture, :polymicrobial_consortium)
        @test g_poly.organism_strain == :polymicrobial_consortium
        @test g_poly.allow_cross_species_edges
        @test length(g_poly.participating_species) == 2
        @test :staphylococcus_aureus in g_poly.participating_species
        @test :pseudomonas_aeruginosa in g_poly.participating_species

        # Verify bridge edges
        bridge_edges = filter(e -> e.is_cross_species_bridge, g_poly.edges)
        @test length(bridge_edges) >= 2
        @test all(e -> e.interaction_type == :polymicrobial_cross_talk, bridge_edges)

        hqno_cross = only(filter(e -> e.source_node == "Pyocyanin_HQNO_Effector" && e.target_node == "AgrC_AgrA", g_poly.edges))
        @test hqno_cross.sign == :inhibition
        @test hqno_cross.signal_producer == :pseudomonas_aeruginosa
        @test hqno_cross.receiving_species == :staphylococcus_aureus
        @test hqno_cross.is_cross_species_bridge
        @test hqno_cross.interaction_type == :polymicrobial_cross_talk

        poly_audit = validate_regulatory_network_applicability(g_poly, BiofilmRegulatoryParams(); requested_species=:polymicrobial_co_culture)
        @test poly_audit.valid
        @test poly_audit.species_separation_enforced
        @test poly_audit.cross_species_edges_count >= 2

        # 5. Droop intracellular nutrient quota mass balance and growth-dilution conservation (R4.2)
        droop_model = DroopCellQuotaModel(
            model_id = "DROOP-SA-USA300",
            organism_strain = :Staphylococcus_aureus_USA300,
            min_quota_mmol_gdw = 0.22,
            max_quota_mmol_gdw = 1.25,
            max_uptake_rate_mmol_gdw_h = 0.48,
            half_saturation_mm = 0.75,
            max_growth_rate_h = 0.36,
            yield_biomass_per_quota = 1.0
        )
        @test droop_model.organism_strain == :Staphylococcus_aureus_USA300
        @test droop_model.min_quota_mmol_gdw == 0.22
        @test droop_model.max_growth_rate_h == 0.36

        # Negative DroopCellQuotaModel tests
        @test_throws ArgumentError DroopCellQuotaModel(min_quota_mmol_gdw=1.5, max_quota_mmol_gdw=1.0)
        @test_throws ArgumentError DroopCellQuotaModel(organism_strain=:invalid_organism_strain)
        @test_throws ArgumentError DroopCellQuotaModel(max_growth_rate_h=-0.1)

        # Transient integration over 24 hours
        t_span = collect(0.0:0.5:24.0)
        sim_res = simulate_droop_quota_dynamics(
            droop_model;
            initial_nutrient_mm = 8.0,
            initial_quota_mmol_gdw = 0.65,
            initial_biomass_gdw_l = 0.35,
            times_h = t_span,
            dt_substep_h = 0.005
        )
        @test sim_res.mass_conserved
        @test sim_res.quota_bounds_respected
        @test sim_res.max_relative_conservation_error < 1e-12
        @test sim_res.max_relative_conservation_error < 1e-13
        # Inventory invariant: N(t) + Q(t)*X(t) == const to machine precision
        inv_0 = sim_res.initial_inventory_mm
        @test isapprox(sim_res.final_inventory_mm, inv_0; rtol=1e-13)
        for inv_t in sim_res.total_nutrient_inventory_mm
            @test abs(inv_t - inv_0) / inv_0 < 1e-12
        end

        # Dynamics behavior: nutrient consumed, biomass grows
        @test sim_res.extracellular_nutrient_mm[end] < sim_res.extracellular_nutrient_mm[1]
        @test sim_res.biomass_gdw_l[end] > sim_res.biomass_gdw_l[1]

        # Growth dilution rate -mu(Q)*Q(t) is strictly <= 0 and uptake rate rho(S) is >= 0
        @test length(sim_res.growth_dilution_rates) == length(t_span)
        @test all(r -> r <= 0.0, sim_res.growth_dilution_rates)
        @test all(r -> r >= 0.0, sim_res.uptake_rates)
        @test all(r -> r >= 0.0, sim_res.growth_rates)

        # Audit function audit_droop_mass_conservation verification
        mass_audit = audit_droop_mass_conservation(
            model = droop_model,
            initial_nutrient_mm = 6.0,
            initial_quota_mmol_gdw = 0.60,
            initial_biomass_gdw_l = 0.40,
            times_h = collect(0.0:1.0:16.0),
            tolerance = 1e-12
        )
        @test mass_audit.mass_conserved
        @test mass_audit.quota_bounds_respected
        @test mass_audit.max_relative_conservation_error < 1e-12
        @test length(mass_audit.growth_dilution_rates) == 17

        # 6. Extended Gate G4 multi-species evaluation in ValidationLadderReport (R4.3)
        ladder = evaluate_validation_ladder()
        @test ladder.gate_passed[:G4_regulatory_graph_quota_audit]
        g4_summary = ladder.gate_summaries[:G4_regulatory_graph_quota_audit]
        @test occursin("SA=true", g4_summary)
        @test occursin("PA14=true", g4_summary)
        @test occursin("poly_bridge=true", g4_summary)
        @test occursin("chimera_rejection=true", g4_summary)
        @test occursin("droop_err=", g4_summary)
    end
end


