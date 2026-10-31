using Test
using Random
include("FlowCytometryObservationModel.jl")
using .FlowCytometryObservationModel

@testset "Flow cytometry observation adapter" begin
    # -----------------------------------------------------------------------
    # 0. Original baseline assertions (preserved intact)
    # -----------------------------------------------------------------------
    state = CellStatePrediction(1.0e6, (0.80, 0.15, 0.05))
    identity_gates = GateCalibration([1 0 0; 0 1 0; 0 0 1])
    @test predicted_gate_probabilities(state, identity_gates) ≈ [0.80, 0.15, 0.05]

    sample = FCMSampleSummary((80, 15, 5);
        bead_events=90, beads_added=10_000, sample_volume_ml=0.1,
        dilution_factor=10, extraction_recovery=0.9,
        sample_origin=:detached_biofilm, gate_id="pilot-v1")
    expected = (100 / 90) * (10_000 / 0.1) * 10 / 0.9
    @test absolute_cells_per_ml(sample) ≈ expected
    @test isfinite(count_loglikelihood(sample, state, identity_gates))
    @test isfinite(gate_loglikelihood(sample, state, identity_gates))
    @test isfinite(joint_loglikelihood(sample, state, identity_gates))
    @test_throws ArgumentError CellStatePrediction(10.0, (0.5, 0.3, 0.1))
    @test_throws ArgumentError GateCalibration([0.5 0 0; 0.5 0 0; 0 0 0])
    @test_throws ArgumentError FCMSampleSummary((1,2,3); bead_events=10, beads_added=1000,
        sample_volume_ml=0.1, sample_origin=:in_situ)

    # -----------------------------------------------------------------------
    # 1. Priority 1 & Section 2: Volume conventions, likelihood-concentration
    #    consistency, zero-event and low-count edge cases, overdispersion
    # -----------------------------------------------------------------------
    @testset "Priority 1: Volume conventions and MLE consistency" begin
        cal_eff = GateCalibration([0.92 0.05 0.01; 0.06 0.88 0.07; 0.02 0.07 0.92];
                                  detection_efficiency=0.85)
        @test effective_sample_volume_ml(sample) ≈ 0.01

        sample_undiluted = FCMSampleSummary((80, 15, 5);
            bead_events=90, beads_added=10_000, sample_volume_ml=0.01,
            dilution_factor=10, extraction_recovery=0.9,
            sample_origin=:detached_biofilm, gate_id="pilot-v1",
            volume_convention=:undiluted_equivalent)
        @test effective_sample_volume_ml(sample_undiluted) ≈ 0.01
        @test absolute_cells_per_ml(sample, cal_eff) ≈ absolute_cells_per_ml(sample_undiluted, cal_eff)

        sample_codiluted = FCMSampleSummary((80, 15, 5);
            bead_events=90, beads_added=10_000, sample_volume_ml=0.01,
            dilution_factor=10, extraction_recovery=0.9,
            sample_origin=:detached_biofilm, gate_id="pilot-v1",
            volume_convention=:beads_co_diluted)
        @test absolute_cells_per_ml(sample, cal_eff) ≈ absolute_cells_per_ml(sample_codiluted, cal_eff)

        for s in (sample, sample_undiluted, sample_codiluted)
            chk = verify_concentration_likelihood_consistency(s, cal_eff)
            @test chk.consistent
            @test chk.expected_events_at_mle ≈ 100.0
            @test chk.observed_events == 100
            @test abs(chk.score_at_mle) <= 1e-12
        end

        # Zero-event and low-count edge cases
        zero_sample = FCMSampleSummary((0, 0, 0);
            bead_events=120, beads_added=10_000, sample_volume_ml=0.1,
            dilution_factor=1.0, extraction_recovery=0.9, sample_origin=:effluent)
        @test absolute_cells_per_ml(zero_sample, cal_eff) == 0.0
        zero_chk = verify_concentration_likelihood_consistency(zero_sample, cal_eff)
        @test zero_chk.consistent
        @test gate_loglikelihood(zero_sample, state, cal_eff) == 0.0
        @test count_loglikelihood(zero_sample, CellStatePrediction(0.0, (1.0, 0.0, 0.0)), cal_eff) == 0.0
        @test count_loglikelihood(sample, CellStatePrediction(0.0, (1.0, 0.0, 0.0)), cal_eff) == -Inf

        low_sample = FCMSampleSummary((1, 0, 0);
            bead_events=50, beads_added=10_000, sample_volume_ml=0.1,
            dilution_factor=1.0, extraction_recovery=1.0, sample_origin=:planktonic)
        low_chk = verify_concentration_likelihood_consistency(low_sample, cal_eff)
        @test low_chk.consistent
        @test gate_loglikelihood(low_sample, CellStatePrediction(1e4, (0.0, 1.0, 0.0)), identity_gates) == -Inf

        # Overdispersed count and gate likelihoods converge to Poisson and Multinomial
        nb_ll_large_k = negative_binomial_count_loglikelihood(sample, state, cal_eff; dispersion_k=1.0e8)
        pois_ll = count_loglikelihood(sample, state, cal_eff)
        @test isapprox(nb_ll_large_k, pois_ll; rtol=1e-4)

        dm_ll_large_a = dirichlet_multinomial_gate_loglikelihood(sample, state, cal_eff; concentration_alpha0=1.0e7)
        mult_ll = gate_loglikelihood(sample, state, cal_eff)
        @test isapprox(dm_ll_large_a, mult_ll; rtol=1e-4)

        @test_throws ArgumentError FCMSampleSummary((1, 1, 1);
            bead_events=10, beads_added=1000, sample_volume_ml=0.1,
            volume_convention=:invalid_convention)

        # CSV template schema validation
        csv_path = joinpath(@__DIR__, "flow_cytometry_samples_template.csv")
        header_line = first(eachline(csv_path))
        @test validate_sample_csv_header(header_line)
        @test_throws ArgumentError validate_sample_csv_header("sample_id,vessel_id")
    end

    # -----------------------------------------------------------------------
    # 2. Priority 2: Synthetic-data simulation and parameter recovery
    # -----------------------------------------------------------------------
    @testset "Priority 2: Synthetic-data parameter recovery" begin
        rng = MersenneTwister(20261029)
        true_state = CellStatePrediction(2.5e6, (0.72, 0.18, 0.10))
        cal = GateCalibration([0.90 0.06 0.02;
                               0.08 0.86 0.08;
                               0.02 0.08 0.90]; detection_efficiency=0.88)
        replicates = [
            simulate_fcm_sample(true_state, cal;
                beads_added=20_000, sample_volume_ml=0.2,
                dilution_factor=5.0, extraction_recovery=0.85,
                aspirated_fraction=0.02, sample_origin=:detached_biofilm,
                gate_id="synth-rep-$k", rng=rng)
            for k in 1:40
        ]
        recovered = recover_latent_state_mle(replicates, cal)
        @test isapprox(recovered.total_cells_per_ml, true_state.total_cells_per_ml; rtol=0.05)
        @test isapprox(recovered.fractions[1], true_state.fractions[1]; atol=0.03)
        @test isapprox(recovered.fractions[2], true_state.fractions[2]; atol=0.03)
        @test isapprox(recovered.fractions[3], true_state.fractions[3]; atol=0.03)

        # Recovered MLE achieves at least as high joint log-likelihood as true state on the finite sample
        ll_rec = sum(joint_loglikelihood(s, recovered, cal) for s in replicates)
        ll_true = sum(joint_loglikelihood(s, true_state, cal) for s in replicates)
        @test ll_rec >= ll_true - 1e-6

        # Boundary recovery when all events fall in gate 1 under non-identity confusion
        boundary_sample = [FCMSampleSummary((200, 0, 0);
            bead_events=100, beads_added=10_000, sample_volume_ml=0.1,
            dilution_factor=1.0, extraction_recovery=1.0, sample_origin=:effluent)]
        rec_bound = recover_latent_state_mle(boundary_sample, cal)
        @test isapprox(sum(rec_bound.fractions), 1.0; atol=1e-6)
        @test all(x -> 0.0 <= x <= 1.0, rec_bound.fractions)
        @test rec_bound.fractions[1] > 0.95

        # Pooling across distinct sample_origin compartments is rejected
        mixed_origins = [replicates[1], boundary_sample[1]]
        @test_throws ArgumentError recover_latent_state_mle(mixed_origins, cal)

        # Rank-deficient (singular) confusion matrix does not throw SingularException in EM recovery
        singular_cal = GateCalibration([0.5 0.5 0.0;
                                        0.5 0.5 0.0;
                                        0.0 0.0 1.0]; detection_efficiency=0.90)
        rec_sing = recover_latent_state_mle([sample], singular_cal)
        @test isfinite(rec_sing.total_cells_per_ml)
        @test isapprox(sum(rec_sing.fractions), 1.0; atol=1e-6)
        @test isapprox(rec_sing.fractions[3], 0.05; atol=1e-4)

        # Exact Atkinson Poisson sampling at high event counts (lambda > 10,000)
        high_state = CellStatePrediction(2.0e8, (0.60, 0.25, 0.15))
        high_sample = simulate_fcm_sample(high_state, cal;
            beads_added=50_000, sample_volume_ml=0.5,
            dilution_factor=1.0, extraction_recovery=0.95,
            aspirated_fraction=0.5, sample_origin=:effluent, rng=rng)
        @test sum(high_sample.gate_counts) > 10_000
    end

    # -----------------------------------------------------------------------
    # 3. Priority 3: Calibration protocol (MESF, bead size, and gate controls)
    # -----------------------------------------------------------------------
    @testset "Priority 3: Calibration bead regressions and control matrix estimation" begin
        # FL1 MESF calibration matching reference figure: log10(MESF) = 0.94 * log10(FL1) + 1.63
        fl1_beads = [25.0, 110.0, 380.0, 1300.0, 4300.0, 14000.0]
        mesf_fl1 = [10.0^(0.94 * log10(x) + 1.63) for x in fl1_beads]
        curve_fl1 = fit_mesf_calibration(fl1_beads, mesf_fl1; channel=:FL1)
        @test curve_fl1.channel == :FL1
        @test isapprox(curve_fl1.slope, 0.94; atol=1e-6)
        @test isapprox(curve_fl1.intercept, 1.63; atol=1e-6)
        @test curve_fl1.r_squared > 0.999
        @test isapprox(inverse_mesf(curve_fl1, predict_mesf(curve_fl1, 1000.0)), 1000.0; rtol=1e-9)

        # FL2 MESF calibration matching reference figure: log10(MESF) = 0.90 * log10(FL2) + 1.71
        fl2_beads = [30.0, 105.0, 360.0, 1250.0, 4100.0, 11000.0]
        mesf_fl2 = [10.0^(0.90 * log10(x) + 1.71) for x in fl2_beads]
        curve_fl2 = fit_mesf_calibration(fl2_beads, mesf_fl2; channel=:FL2)
        @test isapprox(curve_fl2.slope, 0.90; atol=1e-6)
        @test isapprox(curve_fl2.intercept, 1.71; atol=1e-6)
        @test curve_fl2.r_squared > 0.997

        # Bead size calibration matching reference figure: pi * (Diameter/2)^2 = 1.26 * Area - 5.67
        areas_um2 = [7.0, 12.0, 21.5, 41.5, 68.0]
        diameters_um = [2.0 * sqrt((1.26 * a - 5.67) / pi) for a in areas_um2]
        size_cal = fit_bead_size_calibration(areas_um2, diameters_um)
        @test isapprox(size_cal.slope, 1.26; atol=1e-6)
        @test isapprox(size_cal.intercept, -5.67; atol=1e-6)
        @test size_cal.r_squared > 0.999
        @test isapprox(predict_equivalent_diameter_um(size_cal, 41.5), diameters_um[4]; rtol=1e-8)

        # Gate calibration from pure single-state controls with clay-only blank subtraction
        raw_controls = [925 65 25;
                        65 870 80;
                        40 95 925]
        clay_blank = (5, 10, 15)
        est_cal = estimate_gate_calibration(raw_controls;
            clay_blank_counts=clay_blank,
            spiked_cells=[1000.0, 1000.0, 1000.0],
            measured_control_cells=[880.0, 860.0, 870.0])
        @test isapprox(est_cal.detection_efficiency, 2610.0 / 3000.0; atol=1e-6)
        @test all(isapprox.(vec(sum(est_cal.confusion; dims=1)), 1.0; atol=1e-9))
        @test isapprox(est_cal.confusion[1, 1], 920.0 / 1000.0; atol=1e-6)

        @test_throws ArgumentError estimate_gate_calibration(raw_controls;
            spiked_cells=[1000.0, 1000.0, 1000.0],
            measured_control_cells=[1200.0, 1000.0, 1000.0])

        # Per-control recovery > 100% is rejected even when sum(measured) <= sum(spiked)
        @test_throws ArgumentError estimate_gate_calibration(raw_controls;
            spiked_cells=[1000.0, 1000.0, 1000.0],
            measured_control_cells=[1200.0, 800.0, 800.0])

        # Control column with zero net events after blank subtraction is rejected even when pseudocount > 0
        blank_dominated = [2 900 10;
                           3 50 20;
                           1 50 970]
        @test_throws ArgumentError estimate_gate_calibration(blank_dominated;
            clay_blank_counts=(10, 10, 10), pseudocount=0.0)
        @test_throws ArgumentError estimate_gate_calibration(blank_dominated;
            clay_blank_counts=(10, 10, 10), pseudocount=0.5)
    end

    # -----------------------------------------------------------------------
    # 4. Priority 4: Uncertainty propagation and identifiability / equifinality
    # -----------------------------------------------------------------------
    @testset "Priority 4: Uncertainty propagation and identifiability" begin
        unc = concentration_uncertainty(sample;
            detection_efficiency=0.9,
            rel_sd_recovery=0.05,
            rel_sd_efficiency=0.04,
            rel_sd_beads_added=0.02)
        @test unc.estimate_cells_per_ml ≈ absolute_cells_per_ml(sample; detection_efficiency=0.9)
        @test unc.ci_95_lower < unc.estimate_cells_per_ml < unc.ci_95_upper
        @test unc.relative_sd > sqrt(1/100 + 1/90)

        # GateCalibration overload matches keyword detection_efficiency
        cal_90 = GateCalibration([1 0 0; 0 1 0; 0 0 1]; detection_efficiency=0.9)
        unc_cal = concentration_uncertainty(sample, cal_90;
            rel_sd_recovery=0.05, rel_sd_efficiency=0.04, rel_sd_beads_added=0.02)
        @test unc_cal.estimate_cells_per_ml ≈ unc.estimate_cells_per_ml
        @test unc_cal.ci_95_lower ≈ unc.ci_95_lower
        @test unc_cal.ci_95_upper ≈ unc.ci_95_upper

        zero_sample = FCMSampleSummary((0, 0, 0);
            bead_events=100, beads_added=10_000, sample_volume_ml=0.1)
        unc_zero = concentration_uncertainty(zero_sample)
        @test unc_zero.estimate_cells_per_ml == 0.0
        @test unc_zero.ci_95_lower == 0.0
        @test unc_zero.ci_95_upper > 0.0

        # Single-event sample (n_cells = 1) covers the exact Garwood 95% Poisson interval
        one_sample = FCMSampleSummary((1, 0, 0);
            bead_events=10_000, beads_added=10_000, sample_volume_ml=1.0)
        unc_one = concentration_uncertainty(one_sample)
        @test unc_one.ci_95_lower <= -log(0.975)
        @test unc_one.ci_95_upper >= 5.571643390992898

        # Identifiability check on well-conditioned vs degenerate confusion matrices
        id_good = check_parameter_identifiability(identity_gates)
        @test id_good.gate_fractions_identifiable
        @test id_good.scale_parameters_collinear_without_controls
        @test id_good.confusion_determinant ≈ 1.0

        degenerate_cal = GateCalibration([0.34 0.33 0.33;
                                          0.33 0.34 0.33;
                                          0.33 0.33 0.34])
        id_bad = check_parameter_identifiability(degenerate_cal; condition_threshold=50.0)
        @test !id_bad.gate_fractions_identifiable
        @test id_bad.confusion_condition_number > 50.0

        # Equifinality test: two distinct latent states under degenerate gating vs sharp gating
        state_a = CellStatePrediction(1.0e6, (0.75, 0.15, 0.10))
        state_b = CellStatePrediction(1.0e6, (0.20, 0.70, 0.10))
        eq_deg = evaluate_equifinality([sample], state_a, state_b, degenerate_cal; loglik_threshold=2.0)
        @test eq_deg.equifinal
        @test eq_deg.joint_kl_divergence < 0.05
        eq_sharp = evaluate_equifinality([sample], state_a, state_b, identity_gates; loglik_threshold=2.0)
        @test !eq_sharp.equifinal
        @test eq_sharp.abs_delta_loglikelihood > 20.0
        @test eq_sharp.joint_kl_divergence > 10.0

        # Opposite states tying in log-likelihood on a symmetric sample are not falsely marked equifinal
        sym_sample = FCMSampleSummary((50, 50, 0);
            bead_events=100, beads_added=10_000, sample_volume_ml=0.1)
        st_sym_a = CellStatePrediction(1.0e6, (0.90, 0.10, 0.00))
        st_sym_b = CellStatePrediction(1.0e6, (0.10, 0.90, 0.00))
        eq_sym = evaluate_equifinality([sym_sample], st_sym_a, st_sym_b, identity_gates)
        @test eq_sym.abs_delta_loglikelihood == 0.0
        @test eq_sym.joint_kl_divergence > 100.0
        @test !eq_sym.equifinal

        # Two impossible states (-Inf log-likelihood) do not produce NaN abs_delta_loglikelihood
        st_imp_a = CellStatePrediction(1.0e6, (1.0, 0.0, 0.0))
        st_imp_b = CellStatePrediction(1.0e6, (0.0, 1.0, 0.0))
        eq_imp = evaluate_equifinality([sample], st_imp_a, st_imp_b, identity_gates)
        @test !isnan(eq_imp.abs_delta_loglikelihood)
        @test !eq_imp.equifinal
    end

    # -----------------------------------------------------------------------
    # 5. Priority 5: Held-out validation and posterior-predictive checks
    # -----------------------------------------------------------------------
    @testset "Priority 5: Held-out split and posterior-predictive checks" begin
        rng = MersenneTwister(77)
        true_state = CellStatePrediction(1.5e6, (0.70, 0.20, 0.10))
        cal = GateCalibration([0.92 0.05 0.02;
                               0.06 0.89 0.06;
                               0.02 0.06 0.92]; detection_efficiency=0.90)
        all_samples = [
            simulate_fcm_sample(true_state, cal;
                beads_added=15_000, sample_volume_ml=0.1,
                dilution_factor=2.0, extraction_recovery=0.90,
                aspirated_fraction=0.02, sample_origin=:effluent,
                gate_id="vessel-$k", rng=rng)
            for k in 1:24
        ]
        split = split_train_holdout(all_samples, [19, 20, 21, 22, 23, 24])
        @test length(split.train) == 18
        @test length(split.holdout) == 6
        @test_throws ArgumentError split_train_holdout(all_samples, [19, 19, 20])

        fitted_state = recover_latent_state_mle(split.train, cal)
        pred_states = fill(fitted_state, length(split.holdout))
        ppc_good = posterior_predictive_check(split.holdout, pred_states, cal; n_draws=400, seed=101)
        @test isfinite(ppc_good.total_heldout_loglikelihood)
        @test ppc_good.empirical_coverage_95 >= 0.65
        @test 0.02 <= ppc_good.posterior_predictive_p_value <= 0.98

        # Misspecified model on held-out vessels is flagged by PPC
        wrong_state = CellStatePrediction(5.0e6, (0.10, 0.20, 0.70))
        ppc_bad = posterior_predictive_check(split.holdout, fill(wrong_state, 6), cal; n_draws=400, seed=101)
        @test ppc_bad.total_heldout_loglikelihood < ppc_good.total_heldout_loglikelihood - 100.0
        @test ppc_bad.posterior_predictive_p_value < 0.01

        # Zero-expectation gate with positive observed events yields Inf Pearson chi-square
        ppc_zero_gate = posterior_predictive_check([sample], [CellStatePrediction(1.0e6, (1.0, 0.0, 0.0))], identity_gates; n_draws=50)
        @test ppc_zero_gate.gate_chi2_statistic == Inf
    end

    # -----------------------------------------------------------------------
    # 6. Priority 6: Solver interfaces and stakeholder track boundaries
    # -----------------------------------------------------------------------
    @testset "Priority 6: Solver contracts and stakeholder track separation" begin
        t1 = StakeholderTrackContract(:track1_nanobio;
            requires_fcm_observer=true, requires_openfoam=false, requires_phreeqc=false,
            primary_endpoint="validated_single_cell_uptake_and_state_fraction")
        @test t1.track == :track1_nanobio

        t2 = StakeholderTrackContract(:track2_clay_corrosion;
            requires_fcm_observer=true, requires_openfoam=true, requires_phreeqc=true,
            primary_endpoint="maximum_pit_depth_per_coupon")
        @test t2.primary_endpoint == "maximum_pit_depth_per_coupon"

        # Displacing maximum pit depth on Track 2 is rejected
        @test_throws ArgumentError StakeholderTrackContract(:track2_clay_corrosion;
            requires_fcm_observer=true, requires_openfoam=true, requires_phreeqc=true,
            primary_endpoint="flow_cytometry_cell_count")

        t3 = StakeholderTrackContract(:track3_bitumen_rheology;
            requires_fcm_observer=false, requires_openfoam=true, requires_phreeqc=false,
            primary_endpoint="dsr_complex_modulus_and_moisture_susceptibility")
        @test !t3.requires_fcm_observer

        # Forcing flow cytometry or PHREEQC onto Track 3 (bitumen emulsion) is rejected
        @test_throws ArgumentError StakeholderTrackContract(:track3_bitumen_rheology;
            requires_fcm_observer=true, requires_openfoam=true, requires_phreeqc=false,
            primary_endpoint="dsr_complex_modulus")

        of_snap = OpenFOAMTransportSnap("V07", 168.0, :detached_biofilm;
            velocity_m_s=0.012, wall_shear_pa=0.45, residence_time_s=1800.0,
            detachment_flux_cells_m2_s=1.2e4)
        ph_snap = PHREEQCSpeciationSnap("V07", 168.0;
            pH=7.1, pe=-3.8, ionic_strength_M=0.08,
            ca_mol_L=0.015, mg_mol_L=0.005, fe2_mol_L=1.2e-4, sulfide_mol_L=4.5e-4)

        @test validate_solver_observation_alignment(sample, "V07", 168.0;
            transport=of_snap, chemistry=ph_snap)

        of_wrong_comp = OpenFOAMTransportSnap("V07", 168.0, :effluent;
            velocity_m_s=0.012, wall_shear_pa=0.45, residence_time_s=1800.0)
        @test_throws ArgumentError validate_solver_observation_alignment(sample, "V07", 168.0;
            transport=of_wrong_comp, chemistry=ph_snap)

        # Invalid compartment symbol in OpenFOAMTransportSnap is rejected at construction
        @test_throws ArgumentError OpenFOAMTransportSnap("V07", 168.0, :in_situ;
            velocity_m_s=0.012, wall_shear_pa=0.45, residence_time_s=1800.0)
    end

    # -----------------------------------------------------------------------
    # 7. Priority 7: Production acceptance and epistemic status accounting
    # -----------------------------------------------------------------------
    @testset "Priority 7: Production audit record and epistemic separation" begin
        quantities = [
            EpistemicQuantity("gate_counts", sample.gate_counts, :observed, "FCS gate summary pilot-v1"),
            EpistemicQuantity("detection_efficiency", 1.0, :calibrated, "Independent bead/cell control"),
            EpistemicQuantity("poisson_shot_noise_model", "Poisson-Multinomial", :assumed, "Conditional count factorization"),
            EpistemicQuantity("predicted_cells_per_ml", state.total_cells_per_ml, :predicted, "Julia compartment state"),
        ]
        sha = "a25e3941b44aded0488384e2859e17b80c5b302cd2b03d1c644b3c7819936459"
        rec = build_production_audit_record(;
            track=:track2_clay_corrosion,
            input_sha256=sha,
            quantities=quantities,
            sample=sample,
            calibration=identity_gates,
            heldout_validated=true)
        @test rec.model_version == MODEL_ADAPTER_VERSION
        @test rec.consistent_likelihood
        @test rec.identifiable_gates
        @test rec.heldout_validated

        # Omitting an epistemic category (e.g. :assumed) is rejected
        incomplete_q = quantities[1:3]
        @test_throws ArgumentError build_production_audit_record(;
            track=:track2_clay_corrosion,
            input_sha256=sha,
            quantities=incomplete_q,
            sample=sample,
            calibration=identity_gates)

        # Attaching a flow-cytometry observation sample to Track 3 (bitumen rheology) is rejected
        @test_throws ArgumentError build_production_audit_record(;
            track=:track3_bitumen_rheology,
            input_sha256=sha,
            quantities=quantities,
            sample=sample,
            calibration=identity_gates)

        # Duplicate quantity names with conflicting epistemic statuses are rejected
        dup_q = vcat(quantities, [EpistemicQuantity("detection_efficiency", 1.0, :assumed, "Contradictory tag")])
        @test_throws ArgumentError build_production_audit_record(;
            track=:track2_clay_corrosion,
            input_sha256=sha,
            quantities=dup_q,
            sample=sample,
            calibration=identity_gates)
    end

    # -----------------------------------------------------------------------
    # 8. Milestone M1: Mesh-aware OpenFOAM spatial field exchange and conservation
    # -----------------------------------------------------------------------
    @testset "Milestone M1: OpenFOAMMeshExchange, ASCII IO, and conservative mapping" begin
        centers = [
            0.001 0.000 0.000;
            0.003 0.000 0.000;
            0.005 0.000 0.000;
            0.007 0.000 0.000;
            0.009 0.000 0.000;
            0.011 0.000 0.000
        ]
        vols = [1.0e-9, 1.5e-9, 2.0e-9, 2.0e-9, 1.5e-9, 1.0e-9]
        conc = [12.5, 10.2, 8.4, 6.1, 4.3, 2.8]
        vels = [0.004, 0.005, 0.006, 0.006, 0.005, 0.004]
        diffs = fill(1.2e-9, 6)
        patches = [
            OpenFOAMBoundaryPatch("inlet", :fixedValue, [14.0, 14.0]; flux_mol_s=2.5e-8),
            OpenFOAMBoundaryPatch("outlet", :zeroGradient, Float64[]; flux_mol_s=-2.1e-8),
            OpenFOAMBoundaryPatch("biofilm_wall", :fixedFlux, [2.8]; flux_mol_s=-0.4e-8)
        ]

        ex = OpenFOAMMeshExchange(
            vessel_id = "V07",
            time_h = 168.0,
            timestep_s = 0.05,
            compartment = :detached_biofilm,
            mesh_id = "coupon-pore-mesh-6cell",
            solver_version = "OpenFOAM-v2406",
            field_name = "C_fructose",
            field_class = :volScalarField,
            dimensions = (0, -3, 0, 0, 1, 0, 0),
            cell_centers_m = centers,
            cell_volumes_m3 = vols,
            internal_field = conc,
            boundary_patches = patches,
            velocity_magnitudes_m_s = vels,
            diffusivities_m2_s = diffs,
            epistemic_status = :predicted
        )
        @test occursin(r"^[0-9a-f]{64}$", ex.source_sha256)

        # ASCII serialization and parsing round-trip
        ascii_str = serialize_openfoam_ascii_field(ex)
        @test occursin("FoamFile", ascii_str)
        @test occursin("nonuniform List<scalar>", ascii_str)
        ex_parsed = parse_openfoam_ascii_field(
            ascii_str;
            vessel_id = "V07",
            time_h = 168.0,
            timestep_s = 0.05,
            compartment = :detached_biofilm,
            mesh_id = "coupon-pore-mesh-6cell",
            cell_centers_m = centers,
            cell_volumes_m3 = vols,
            velocity_magnitudes_m_s = vels,
            diffusivities_m2_s = diffs
        )
        @test isapprox(vec(ex_parsed.internal_field), conc; rtol=1e-10)
        @test ex_parsed.dimensions == (0, -3, 0, 0, 1, 0, 0)
        @test length(ex_parsed.boundary_patches) == 3
        @test isapprox(ex_parsed.boundary_patches[1].flux_mol_s, 2.5e-8; rtol=1e-10)

        # Vector field (:volVectorField) and uniform internalField round-trip
        vec_fld = [
            0.004 0.0001 0.0;
            0.005 0.0002 0.0;
            0.006 0.0001 0.0;
            0.006 -0.0001 0.0;
            0.005 -0.0002 0.0;
            0.004 0.0000 0.0
        ]
        ex_vec = OpenFOAMMeshExchange(
            vessel_id = "V07",
            time_h = 168.0,
            timestep_s = 0.05,
            compartment = :detached_biofilm,
            field_name = "U",
            field_class = :volVectorField,
            dimensions = (0, 1, -1, 0, 0, 0, 0),
            cell_centers_m = centers,
            cell_volumes_m3 = vols,
            internal_field = vec_fld
        )
        vec_ascii = serialize_openfoam_ascii_field(ex_vec)
        vec_parsed = parse_openfoam_ascii_field(
            vec_ascii;
            vessel_id = "V07",
            time_h = 168.0,
            timestep_s = 0.05,
            compartment = :detached_biofilm,
            cell_centers_m = centers,
            cell_volumes_m3 = vols
        )
        @test vec_parsed.field_class == :volVectorField
        @test isapprox(vec_parsed.internal_field, vec_fld; rtol=1e-10)

        # Uniform scalar field parsing
        unif_ascii = """
        FoamFile
        {
            version 2.0;
            format ascii;
            class volScalarField;
            object C_uniform;
        }
        dimensions [0 -3 0 0 1 0 0];
        internalField uniform 5.25;
        boundaryField
        {
            walls { type zeroGradient; }
        }
        """
        unif_parsed = parse_openfoam_ascii_field(
            unif_ascii;
            vessel_id = "V07",
            time_h = 168.0,
            timestep_s = 0.05,
            compartment = :detached_biofilm,
            cell_centers_m = centers,
            cell_volumes_m3 = vols
        )
        @test all(x -> isapprox(x, 5.25; atol=1e-12), unif_parsed.internal_field)

        # Conservative volume-weighted mapping from 6 fine cells to 2 coarse compartments
        # Cell 3 (V = 2.0e-9) is split 40% into compartment 1 and 60% into compartment 2
        W = [
            1.0 1.0 0.4 0.0 0.0 0.0;
            0.0 0.0 0.6 1.0 1.0 1.0
        ]
        tgt_vols = vec(W * vols)
        mapped_res = map_openfoam_field_conservative(ex, tgt_vols, W)
        @test mapped_res.mass_conserved
        @test mapped_res.max_relative_mass_error <= 1e-14
        @test isapprox(mapped_res.source_totals[1], sum(conc .* vols); rtol=1e-14)
        @test isapprox(mapped_res.target_totals[1], mapped_res.source_totals[1]; rtol=1e-14)

        # Non-conservative overlap weights (column sum != 1) or mismatched target volumes are rejected
        W_bad = copy(W)
        W_bad[1, 3] = 0.5
        @test_throws ArgumentError map_openfoam_field_conservative(ex, tgt_vols, W_bad)
        @test_throws ArgumentError map_openfoam_field_conservative(ex, tgt_vols .* 1.05, W)

        # Courant / Peclet / nonnegativity numerical stability audit
        stab = audit_openfoam_transport_stability(ex; max_courant=1.0, max_peclet=10_000.0)
        @test stab.stable
        @test stab.max_courant_observed < 0.5
        @test isapprox(stab.total_integrated_amount, sum(conc .* vols); rtol=1e-14)
        @test isapprox(stab.net_boundary_flux_mol_s, 0.0; atol=1e-15)

        # Excessive Courant number or negative concentration fails stability check
        stab_strict = audit_openfoam_transport_stability(ex; max_courant=0.01, max_peclet=10_000.0)
        @test !stab_strict.stable

        # Summarize to scalar OpenFOAMTransportSnap and verify alignment
        snap_from_mesh = summarize_openfoam_mesh_to_snap(ex; wall_shear_pa=0.45, residence_time_s=1800.0)
        @test snap_from_mesh.vessel_id == "V07"
        @test validate_solver_observation_alignment(sample, "V07", 168.0; mesh_transport=ex)

        ex_wrong_vessel = OpenFOAMMeshExchange(
            vessel_id = "V99", time_h = 168.0, timestep_s = 0.05,
            compartment = :detached_biofilm, cell_centers_m = centers,
            cell_volumes_m3 = vols, internal_field = conc
        )
        @test_throws ArgumentError validate_solver_observation_alignment(sample, "V07", 168.0; mesh_transport=ex_wrong_vessel)
    end

    # -----------------------------------------------------------------------
    # 9. Milestone M2: Full-species PHREEQC snapshot adapter and balance audits
    # -----------------------------------------------------------------------
    @testset "Milestone M2: PHREEQCSpeciesAdapter, charge/element balance, and TSV IO" begin
        # Construct an electrically neutral, element-balanced aqueous solution at 25 C
        pH_val = 7.0
        pe_val = -3.5
        h_mol  = 1.0e-7
        oh_mol = 1.0e-7
        ca_free = 0.010
        cahco3  = 0.001
        mg_free = 0.004
        fe2_free = 0.0005
        na_free = 0.020
        hco3    = 0.008
        so4     = 0.010
        hs_minus = 0.001
        h2s_aq   = 0.0005
        # Total positive charge = 2*(0.010) + 1*(0.001) + 2*(0.004) + 2*(0.0005) + 1*(0.020) + 1e-7 = 0.050 + 1e-7
        # Negative charge without Cl- = 1*(0.008) + 2*(0.010) + 1*(0.001) + 1e-7 = 0.029 + 1e-7
        cl_minus = 0.021

        molalities = Dict{String,Float64}(
            "H+" => h_mol,
            "OH-" => oh_mol,
            "Ca+2" => ca_free,
            "CaHCO3+" => cahco3,
            "Mg+2" => mg_free,
            "Fe+2" => fe2_free,
            "Na+" => na_free,
            "HCO3-" => hco3,
            "SO4-2" => so4,
            "HS-" => hs_minus,
            "H2S" => h2s_aq,
            "Cl-" => cl_minus
        )
        charges = Dict{String,Int}(
            "H+" => 1, "OH-" => -1, "Ca+2" => 2, "CaHCO3+" => 1,
            "Mg+2" => 2, "Fe+2" => 2, "Na+" => 1, "HCO3-" => -1,
            "SO4-2" => -2, "HS-" => -1, "H2S" => 0, "Cl-" => -1
        )
        act_coeffs = Dict{String,Float64}(
            sp => (abs(z) == 2 ? 0.72 : (abs(z) == 1 ? 0.90 : 1.0))
            for (sp, z) in charges
        )
        act_coeffs["H+"] = 1.0 # so -log10(a_H+) == 7.0 == pH

        stoich = Dict{String,Dict{String,Float64}}(
            "Ca+2" => Dict("Ca" => 1.0),
            "CaHCO3+" => Dict("Ca" => 1.0, "C" => 1.0),
            "Mg+2" => Dict("Mg" => 1.0),
            "Fe+2" => Dict("Fe" => 1.0),
            "Na+" => Dict("Na" => 1.0),
            "HCO3-" => Dict("C" => 1.0),
            "SO4-2" => Dict("S" => 1.0),
            "HS-" => Dict("S" => 1.0),
            "H2S" => Dict("S" => 1.0),
            "Cl-" => Dict("Cl" => 1.0)
        )
        elem_totals = Dict{String,Float64}(
            "Ca" => ca_free + cahco3,
            "Mg" => mg_free,
            "Fe" => fe2_free,
            "Na" => na_free,
            "C"  => cahco3 + hco3,
            "S"  => so4 + hs_minus + h2s_aq,
            "Cl" => cl_minus
        )
        ionic_str = 0.5 * sum((charges[sp]^2) * m for (sp, m) in molalities)
        minerals = [
            PHREEQCMineralPhase("Calcite", "CaCO3"; saturation_index=0.18, moles_mol_L=0.002, delta_moles_mol_L=1.5e-5),
            PHREEQCMineralPhase("Mackinawite", "FeS"; saturation_index=0.65, moles_mol_L=0.0004, delta_moles_mol_L=3.0e-5),
            PHREEQCMineralPhase("Gypsum", "CaSO4:2H2O"; saturation_index=-1.12, moles_mol_L=0.0, delta_moles_mol_L=0.0)
        ]

        ph_adapter = PHREEQCSpeciesAdapter(
            vessel_id = "V07",
            time_h = 168.0,
            compartment = :detached_biofilm,
            pH = pH_val,
            pe = pe_val,
            ionic_strength_M = ionic_str,
            species_molalities_mol_kgw = molalities,
            species_charges = charges,
            species_activity_coeffs = act_coeffs,
            element_totals_mol_kgw = elem_totals,
            species_stoichiometry = stoich,
            mineral_phases = minerals
        )

        audit = audit_phreeqc_speciation_balance(ph_adapter)
        @test audit.consistent
        @test abs(audit.charge_imbalance_pct) < 1e-9
        @test audit.ionic_strength_consistent
        @test audit.element_balance_consistent
        @test audit.max_element_rel_error < 1e-12
        @test audit.nernst_eh_consistent
        @test audit.ph_activity_consistent
        @test "Calcite" in audit.supersaturated_minerals
        @test "Mackinawite" in audit.supersaturated_minerals
        @test "Gypsum" in audit.undersaturated_minerals

        # Selected-output TSV round-trip
        tsv_str = serialize_phreeqc_selected_output(ph_adapter)
        ph_parsed = parse_phreeqc_selected_output(
            tsv_str;
            species_charges = charges,
            species_activity_coeffs = act_coeffs,
            species_stoichiometry = stoich
        )
        audit_parsed = audit_phreeqc_speciation_balance(ph_parsed)
        @test audit_parsed.consistent
        @test isapprox(ph_parsed.species_molalities_mol_kgw["Fe+2"], fe2_free; rtol=1e-9)
        @test length(ph_parsed.mineral_phases) == 3

        # Summarize to scalar PHREEQCSpeciationSnap and verify observation alignment
        snap_chem = summarize_phreeqc_adapter_to_snap(ph_adapter)
        @test isapprox(snap_chem.ca_mol_L, ca_free + cahco3; rtol=1e-10)
        @test isapprox(snap_chem.sulfide_mol_L, hs_minus + h2s_aq; rtol=1e-10)
        @test validate_solver_observation_alignment(sample, "V07", 168.0; species_chemistry=ph_adapter)

        # Detect severe charge or element imbalance
        bad_elem_totals = copy(elem_totals)
        bad_elem_totals["Ca"] = 0.050 # 5x higher than species sum
        ph_bad_el = PHREEQCSpeciesAdapter(
            vessel_id = "V07", time_h = 168.0, compartment = :detached_biofilm,
            pH = pH_val, pe = pe_val, ionic_strength_M = ionic_str,
            species_molalities_mol_kgw = molalities, species_charges = charges,
            element_totals_mol_kgw = bad_elem_totals, species_stoichiometry = stoich
        )
        audit_bad = audit_phreeqc_speciation_balance(ph_bad_el)
        @test !audit_bad.consistent
        @test !audit_bad.element_balance_consistent
    end

    # -----------------------------------------------------------------------
    # 10. Milestone M5: Empirical validation protocol & generalization gate
    # -----------------------------------------------------------------------
    @testset "Milestone M5: EmpiricalValidationProtocol and generalization readiness" begin
        # 1. Synthetic simulation cannot claim empirical generalization even with 99% CV accuracy
        proto_synth = EmpiricalValidationProtocol(
            protocol_id = "SYNTH-01",
            track = :track1_nanobio,
            data_provenance = :synthetic_simulation,
            independent_reference_labels = false,
            n_independent_runs = 3,
            n_heldout_runs = 1
        )
        res_synth = evaluate_empirical_generalization_readiness(
            proto_synth;
            target_task = :cell_stress_damage,
            synthetic_accuracy = 0.99,
            heldout_run_accuracy = 0.98
        )
        @test res_synth.software_verification_passed
        @test !res_synth.empirical_generalization_established
        @test !isempty(res_synth.blocking_reasons)

        # Synthetic provenance claiming independent_reference_labels = true is rejected
        @test_throws ArgumentError EmpiricalValidationProtocol(
            data_provenance = :synthetic_simulation,
            independent_reference_labels = true
        )

        # 2. Nanocarrier uptake on experimental FCS without internalization imaging flags surface association warning
        proto_uptake_no_img = EmpiricalValidationProtocol(
            protocol_id = "EXP-GHOSH-01",
            track = :track1_nanobio,
            data_provenance = :deidentified_experimental_fcs,
            independent_reference_labels = true,
            counting_beads_verified = true,
            internalization_imaging_verified = false,
            n_independent_runs = 4,
            n_heldout_runs = 1
        )
        res_no_img = evaluate_empirical_generalization_readiness(
            proto_uptake_no_img;
            target_task = :nanocarrier_uptake,
            synthetic_accuracy = 0.96,
            heldout_run_accuracy = 0.91,
            feature_psi_drift = 0.08
        )
        @test res_no_img.software_verification_passed
        @test res_no_img.surface_association_only_warning
        @test !res_no_img.internalization_proven
        @test !res_no_img.empirical_generalization_established

        # 3. Full empirical readiness when imaging, controls, held-out run, and low PSI drift are satisfied
        proto_uptake_full = EmpiricalValidationProtocol(
            protocol_id = "EXP-GHOSH-02",
            track = :track1_nanobio,
            data_provenance = :deidentified_experimental_fcs,
            independent_reference_labels = true,
            counting_beads_verified = true,
            internalization_imaging_verified = true,
            n_independent_runs = 4,
            n_heldout_runs = 1
        )
        res_full = evaluate_empirical_generalization_readiness(
            proto_uptake_full;
            target_task = :nanocarrier_uptake,
            synthetic_accuracy = 0.96,
            heldout_run_accuracy = 0.91,
            feature_psi_drift = 0.08
        )
        @test res_full.software_verification_passed
        @test res_full.empirical_generalization_established
        @test res_full.internalization_proven
        @test !res_full.surface_association_only_warning
        @test !res_full.domain_shift_detected
        @test !res_full.external_solvers_coupled_in_production
        @test isempty(res_full.blocking_reasons)

        # 4. Feature domain shift (PSI > 0.25) blocks empirical generalization
        res_drift = evaluate_empirical_generalization_readiness(
            proto_uptake_full;
            target_task = :nanocarrier_uptake,
            synthetic_accuracy = 0.96,
            heldout_run_accuracy = 0.91,
            feature_psi_drift = 0.42
        )
        @test res_drift.domain_shift_detected
        @test !res_drift.empirical_generalization_established
    end
end
