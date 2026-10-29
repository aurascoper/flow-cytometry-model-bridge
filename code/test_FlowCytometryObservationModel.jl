using Test
include("FlowCytometryObservationModel.jl")
using .FlowCytometryObservationModel

@testset "Flow cytometry observation adapter" begin
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
end
