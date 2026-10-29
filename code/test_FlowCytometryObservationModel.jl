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

        zero_sample = FCMSampleSummary((0, 0, 0);
            bead_events=100, beads_added=10_000, sample_volume_ml=0.1)
        unc_zero = concentration_uncertainty(zero_sample)
        @test unc_zero.estimate_cells_per_ml == 0.0
        @test unc_zero.ci_95_lower == 0.0
        @test unc_zero.ci_95_upper > 0.0

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
    end
end
