module FlowCytometryObservationModel

"""
Observation adapter for flow-cytometry summaries of microbial systems.

This module does not solve flow, geochemistry, or biology. It maps a Julia
model's latent cell-state summary to a simple, calibratable cytometry
observation model. Callers must supply predictions for the same sample
compartment (effluent, planktonic suspension, or detached biofilm) that was
actually sampled.

State order used throughout:
  1. membrane-intact / live-like gate
  2. membrane-compromised gate
  3. dead-like gate

These are assay-defined gates, not universal biological truths. Validate the
stain and gates for the organism and matrix being measured.
"""

export CellStatePrediction, GateCalibration, FCMSampleSummary,
       predicted_gate_probabilities, absolute_cells_per_ml,
       count_loglikelihood, gate_loglikelihood, joint_loglikelihood

const VALID_SAMPLE_ORIGINS = (:effluent, :planktonic, :detached_biofilm)

struct CellStatePrediction
    total_cells_per_ml::Float64
    fractions::NTuple{3,Float64}

    function CellStatePrediction(total_cells_per_ml::Real,
                                 fractions::NTuple{3,<:Real})
        total = Float64(total_cells_per_ml)
        fs = (Float64(fractions[1]), Float64(fractions[2]), Float64(fractions[3]))
        isfinite(total) && total >= 0 || throw(ArgumentError("total_cells_per_ml must be finite and >= 0"))
        all(x -> isfinite(x) && 0 <= x <= 1, fs) ||
            throw(ArgumentError("cell-state fractions must be finite and in [0,1]"))
        isapprox(sum(fs), 1.0; atol=1e-6, rtol=0.0) ||
            throw(ArgumentError("cell-state fractions must sum to 1"))
        new(total, fs)
    end
end

"""
GateCalibration(confusion; detection_efficiency=1.0)

`confusion[i,j] = P(observed gate i | true latent state j, detected)`.
Columns must sum to 1. `detection_efficiency` is the separately validated
fraction of modelled cells expected to enter the gated event population.
It should not be guessed from the data being fitted.
"""
struct GateCalibration
    confusion::Matrix{Float64}
    detection_efficiency::Float64

    function GateCalibration(confusion::AbstractMatrix{<:Real}; detection_efficiency::Real=1.0)
        size(confusion) == (3,3) || throw(ArgumentError("confusion matrix must be 3x3"))
        c = Float64.(confusion)
        all(isfinite, c) && all(x -> 0 <= x <= 1, c) ||
            throw(ArgumentError("confusion entries must be finite probabilities"))
        colsums = vec(sum(c; dims=1))
        all(x -> isapprox(x, 1.0; atol=1e-6, rtol=0.0), colsums) ||
            throw(ArgumentError("each confusion-matrix column must sum to 1"))
        d = Float64(detection_efficiency)
        isfinite(d) && 0 < d <= 1 || throw(ArgumentError("detection_efficiency must be in (0,1]"))
        new(c, d)
    end
end

"""
FCMSampleSummary(gate_counts; ...)

Summary of already gated cell events. `beads_added` is the known number of
counting beads added to the represented aliquot; `sample_volume_ml` is the
original sample volume represented (before the recorded dilution); and
`extraction_recovery` is a validated recovery fraction (use 1.0 for a liquid
sample without an extraction step). Blank / clay-only gates and instrument
compensation are handled upstream and recorded alongside the raw FCS data.
"""
struct FCMSampleSummary
    gate_counts::NTuple{3,Int}
    bead_events::Int
    beads_added::Float64
    sample_volume_ml::Float64
    dilution_factor::Float64
    extraction_recovery::Float64
    sample_origin::Symbol
    gate_id::String

    function FCMSampleSummary(gate_counts::NTuple{3,<:Integer};
                              bead_events::Integer,
                              beads_added::Real,
                              sample_volume_ml::Real,
                              dilution_factor::Real=1.0,
                              extraction_recovery::Real=1.0,
                              sample_origin::Symbol=:effluent,
                              gate_id::AbstractString="unspecified")
        gc = (Int(gate_counts[1]), Int(gate_counts[2]), Int(gate_counts[3]))
        all(x -> x >= 0, gc) || throw(ArgumentError("gate counts must be nonnegative"))
        be = Int(bead_events)
        be > 0 || throw(ArgumentError("bead_events must be > 0 for bead-based absolute counts"))
        ba = Float64(beads_added)
        sv = Float64(sample_volume_ml)
        df = Float64(dilution_factor)
        er = Float64(extraction_recovery)
        isfinite(ba) && ba > 0 || throw(ArgumentError("beads_added must be finite and > 0"))
        isfinite(sv) && sv > 0 || throw(ArgumentError("sample_volume_ml must be finite and > 0"))
        isfinite(df) && df >= 1 || throw(ArgumentError("dilution_factor must be finite and >= 1"))
        isfinite(er) && 0 < er <= 1 || throw(ArgumentError("extraction_recovery must be in (0,1]"))
        sample_origin in VALID_SAMPLE_ORIGINS ||
            throw(ArgumentError("sample_origin must be one of $(VALID_SAMPLE_ORIGINS)"))
        new(gc, be, ba, sv, df, er, sample_origin, String(gate_id))
    end
end

"Return the predicted probability of each observed gate, after calibration."
function predicted_gate_probabilities(state::CellStatePrediction, calibration::GateCalibration)
    p = calibration.confusion * collect(state.fractions)
    total = sum(p)
    total > 0 || throw(ArgumentError("calibration produced zero total gate probability"))
    return p ./ total
end

"""
Estimate absolute gated-cell concentration from the counting-bead ratio.

Formula:
  cells/mL = (cell_events / bead_events) * (beads_added / sample_volume_ml)
             * dilution_factor / extraction_recovery / detection_efficiency

If no validated detection-efficiency correction is available, use the default
1.0 and report the result as a gated-event-derived concentration rather than
an exhaustive total-cell count.
"""
function absolute_cells_per_ml(sample::FCMSampleSummary; detection_efficiency::Real=1.0)
    de = Float64(detection_efficiency)
    isfinite(de) && 0 < de <= 1 || throw(ArgumentError("detection_efficiency must be in (0,1]"))
    cell_events = sum(sample.gate_counts)
    return (cell_events / sample.bead_events) *
           (sample.beads_added / sample.sample_volume_ml) *
           sample.dilution_factor / sample.extraction_recovery / de
end

function _logfactorial(n::Int)
    n >= 0 || throw(ArgumentError("n must be nonnegative"))
    result = 0.0
    for k in 2:n
        result += log(k)
    end
    return result
end

"Poisson log-likelihood for total gated events, conditional on bead events."
function count_loglikelihood(sample::FCMSampleSummary,
                             state::CellStatePrediction,
                             calibration::GateCalibration)
    observed = sum(sample.gate_counts)
    λ = state.total_cells_per_ml * calibration.detection_efficiency *
        sample.sample_volume_ml * sample.extraction_recovery * sample.bead_events /
        (sample.beads_added * sample.dilution_factor)
    if λ <= 0
        return observed == 0 ? 0.0 : -Inf
    end
    return observed * log(λ) - λ - _logfactorial(observed)
end

"Multinomial log-likelihood for gate composition, conditional on total events."
function gate_loglikelihood(sample::FCMSampleSummary,
                            state::CellStatePrediction,
                            calibration::GateCalibration)
    counts = sample.gate_counts
    n = sum(counts)
    n == 0 && return 0.0
    p = predicted_gate_probabilities(state, calibration)
    ll = _logfactorial(n)
    for i in 1:3
        ll -= _logfactorial(counts[i])
        if counts[i] > 0
            p[i] > 0 || return -Inf
            ll += counts[i] * log(p[i])
        end
    end
    return ll
end

"Joint likelihood factorizes total event count and gate composition (not duplicated totals)."
function joint_loglikelihood(sample::FCMSampleSummary,
                             state::CellStatePrediction,
                             calibration::GateCalibration)
    return count_loglikelihood(sample, state, calibration) +
           gate_loglikelihood(sample, state, calibration)
end

end # module
