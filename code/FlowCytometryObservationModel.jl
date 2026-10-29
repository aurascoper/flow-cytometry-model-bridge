module FlowCytometryObservationModel

using Random

"""
Observation adapter for flow-cytometry summaries of microbial systems.

This module does not solve fluid flow, aqueous geochemistry, or biological reaction
kinetics. It maps a Julia model's latent cell-state summary to a calibratable
cytometry observation model. Callers must supply predictions for the same sample
compartment (effluent, planktonic suspension, or detached biofilm) that was
actually sampled.

State order used throughout:
  1. membrane-intact / live-like gate
  2. membrane-compromised gate
  3. dead-like gate

These are assay-defined gates, not universal biological truths. Validate the
stain and gates for the organism and matrix being measured.
"""

export MODEL_ADAPTER_VERSION,
       VALID_SAMPLE_ORIGINS, VALID_VOLUME_CONVENTIONS,
       VALID_STAKEHOLDER_TRACKS, VALID_EPISTEMIC_STATUSES,
       REQUIRED_CSV_COLUMNS,
       CellStatePrediction, GateCalibration, FCMSampleSummary,
       MESFCalibrationCurve, BeadSizeCalibration,
       ConcentrationUncertainty,
       OpenFOAMTransportSnap, PHREEQCSpeciationSnap, StakeholderTrackContract,
       EpistemicQuantity, ProductionAuditRecord,
       effective_sample_volume_ml, expected_cell_events,
       predicted_gate_probabilities, absolute_cells_per_ml,
       verify_concentration_likelihood_consistency,
       count_loglikelihood, gate_loglikelihood, joint_loglikelihood,
       negative_binomial_count_loglikelihood, dirichlet_multinomial_gate_loglikelihood,
       simulate_fcm_sample, recover_latent_state_mle,
       fit_mesf_calibration, predict_mesf, inverse_mesf,
       fit_bead_size_calibration, predict_equivalent_diameter_um,
       estimate_gate_calibration,
       concentration_uncertainty, check_parameter_identifiability, evaluate_equifinality,
       split_train_holdout, posterior_predictive_check,
       validate_solver_observation_alignment, validate_sample_csv_header,
       build_production_audit_record

const MODEL_ADAPTER_VERSION = "0.2.0"
const VALID_SAMPLE_ORIGINS = (:effluent, :planktonic, :detached_biofilm)
const VALID_VOLUME_CONVENTIONS = (:aliquot_post_dilution, :undiluted_equivalent, :beads_co_diluted)
const VALID_STAKEHOLDER_TRACKS = (:track1_nanobio, :track2_clay_corrosion, :track3_bitumen_rheology)
const VALID_EPISTEMIC_STATUSES = (:observed, :calibrated, :assumed, :predicted)

const REQUIRED_CSV_COLUMNS = (
    "sample_id", "vessel_id", "arm", "time_h", "sample_origin", "gate_id",
    "gate_live", "gate_compromised", "gate_dead", "bead_events", "beads_added",
    "sample_volume_ml", "dilution_factor", "extraction_recovery",
    "detection_efficiency", "stain_panel", "clay_blank_id",
    "raw_fcs_path", "raw_fcs_sha256"
)

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
sample volume according to `volume_convention`:
- `:aliquot_post_dilution` (default): `sample_volume_ml` is the assayed aliquot
  volume taken from a `dilution_factor`-fold diluted suspension (or the initial
  sample volume before dividing by `dilution_factor`), with `beads_added` added
  to the diluted assay tube. Effective undiluted volume in the tube is
  `sample_volume_ml / dilution_factor`.
- `:undiluted_equivalent`: `sample_volume_ml` is already the net undiluted original
  sample volume present in the bead-spiked assay tube.
- `:beads_co_diluted`: `beads_added` were added to `sample_volume_ml` of undiluted
  extract prior to co-diluting both cells and beads by `dilution_factor`.

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
    volume_convention::Symbol

    function FCMSampleSummary(gate_counts::NTuple{3,<:Integer};
                              bead_events::Integer,
                              beads_added::Real,
                              sample_volume_ml::Real,
                              dilution_factor::Real=1.0,
                              extraction_recovery::Real=1.0,
                              sample_origin::Symbol=:effluent,
                              gate_id::AbstractString="unspecified",
                              volume_convention::Symbol=:aliquot_post_dilution)
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
        volume_convention in VALID_VOLUME_CONVENTIONS ||
            throw(ArgumentError("volume_convention must be one of $(VALID_VOLUME_CONVENTIONS)"))
        new(gc, be, ba, sv, df, er, sample_origin, String(gate_id), volume_convention)
    end
end

"""
Return the effective undiluted sample volume (mL) represented in the bead-spiked
measurement tube under `sample.volume_convention`.
"""
function effective_sample_volume_ml(sample::FCMSampleSummary)
    if sample.volume_convention == :aliquot_post_dilution
        return sample.sample_volume_ml / sample.dilution_factor
    else
        return sample.sample_volume_ml
    end
end

"""
Expected total gated cell events `lambda` for `sample` given `state` and `calibration`:
  lambda = C * eta * V_eff * r * (bead_events / beads_added)
where `V_eff = effective_sample_volume_ml(sample)`.
"""
function expected_cell_events(sample::FCMSampleSummary,
                              state::CellStatePrediction,
                              calibration::GateCalibration)
    v_eff = effective_sample_volume_ml(sample)
    return state.total_cells_per_ml * calibration.detection_efficiency *
           v_eff * sample.extraction_recovery *
           (sample.bead_events / sample.beads_added)
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

Under `:aliquot_post_dilution` (default):
  cells/mL = (cell_events / bead_events) * (beads_added / sample_volume_ml)
             * dilution_factor / extraction_recovery / detection_efficiency

More generally, using `V_eff = effective_sample_volume_ml(sample)`:
  cells/mL = (cell_events / bead_events) * (beads_added / V_eff)
             / extraction_recovery / detection_efficiency

If no validated detection-efficiency correction is available, use the default
1.0 and report the result as a gated-event-derived concentration rather than
an exhaustive total-cell count.
"""
function absolute_cells_per_ml(sample::FCMSampleSummary; detection_efficiency::Real=1.0)
    de = Float64(detection_efficiency)
    isfinite(de) && 0 < de <= 1 || throw(ArgumentError("detection_efficiency must be in (0,1]"))
    cell_events = sum(sample.gate_counts)
    v_eff = effective_sample_volume_ml(sample)
    return (cell_events / sample.bead_events) *
           (sample.beads_added / v_eff) /
           sample.extraction_recovery / de
end

function absolute_cells_per_ml(sample::FCMSampleSummary, calibration::GateCalibration)
    return absolute_cells_per_ml(sample; detection_efficiency=calibration.detection_efficiency)
end

function _logfactorial(n::Int)
    n >= 0 || throw(ArgumentError("n must be nonnegative"))
    result = 0.0
    for k in 2:n
        result += log(k)
    end
    return result
end

function _loggamma_positive(x::Float64)
    x > 0 || throw(ArgumentError("x must be > 0"))
    shift = 0.0
    while x < 8.0
        shift -= log(x)
        x += 1.0
    end
    invx = 1.0 / x
    invx2 = invx * invx
    stirling = (x - 0.5) * log(x) - x + 0.5 * log(2.0 * pi) +
               invx * (1.0 / 12.0 - invx2 * (1.0 / 360.0 - invx2 / 1260.0))
    return stirling + shift
end

"Poisson log-likelihood for total gated events, conditional on bead events."
function count_loglikelihood(sample::FCMSampleSummary,
                             state::CellStatePrediction,
                             calibration::GateCalibration)
    observed = sum(sample.gate_counts)
    λ = expected_cell_events(sample, state, calibration)
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

"""
Negative-binomial log-likelihood for overdispersed total gated cell counts,
parameterized by mean `lambda = expected_cell_events(sample, state, calibration)`
and dispersion `dispersion_k > 0` such that `Var(N) = lambda + lambda^2 / dispersion_k`.
"""
function negative_binomial_count_loglikelihood(sample::FCMSampleSummary,
                                               state::CellStatePrediction,
                                               calibration::GateCalibration;
                                               dispersion_k::Real)
    k = Float64(dispersion_k)
    isfinite(k) && k > 0 || throw(ArgumentError("dispersion_k must be finite and > 0"))
    observed = sum(sample.gate_counts)
    λ = expected_cell_events(sample, state, calibration)
    if λ <= 0
        return observed == 0 ? 0.0 : -Inf
    end
    return _loggamma_positive(observed + k) - _loggamma_positive(k) - _logfactorial(observed) +
           k * (log(k) - log(k + λ)) + observed * (log(λ) - log(k + λ))
end

"""
Dirichlet-multinomial log-likelihood for overdispersed gate composition,
parameterized by total concentration `concentration_alpha0 > 0` and mean gate
probabilities `p = predicted_gate_probabilities(state, calibration)`.
"""
function dirichlet_multinomial_gate_loglikelihood(sample::FCMSampleSummary,
                                                  state::CellStatePrediction,
                                                  calibration::GateCalibration;
                                                  concentration_alpha0::Real)
    alpha0 = Float64(concentration_alpha0)
    isfinite(alpha0) && alpha0 > 0 || throw(ArgumentError("concentration_alpha0 must be finite and > 0"))
    counts = sample.gate_counts
    n = sum(counts)
    n == 0 && return 0.0
    p = predicted_gate_probabilities(state, calibration)
    ll = _logfactorial(n) + _loggamma_positive(alpha0) - _loggamma_positive(n + alpha0)
    for i in 1:3
        ll -= _logfactorial(counts[i])
        if counts[i] > 0
            p[i] > 0 || return -Inf
            ai = alpha0 * p[i]
            ll += _loggamma_positive(counts[i] + ai) - _loggamma_positive(ai)
        end
    end
    return ll
end

"""
Verify that `absolute_cells_per_ml(sample, calibration)` is the exact maximum
likelihood estimator of `count_loglikelihood(sample, state, calibration)`.
"""
function verify_concentration_likelihood_consistency(sample::FCMSampleSummary,
                                                     calibration::GateCalibration;
                                                     atol::Real=1e-8)
    observed = sum(sample.gate_counts)
    c_hat = absolute_cells_per_ml(sample, calibration)
    state_mle = CellStatePrediction(c_hat, (1.0, 0.0, 0.0))
    lambda_mle = expected_cell_events(sample, state_mle, calibration)
    ll_mle = count_loglikelihood(sample, state_mle, calibration)
    v_eff = effective_sample_volume_ml(sample)
    kappa = calibration.detection_efficiency * v_eff *
            sample.extraction_recovery * (sample.bead_events / sample.beads_added)

    mean_matches = isapprox(lambda_mle, Float64(observed); atol=atol, rtol=1e-10)
    if observed == 0
        score_mle = 0.0
        state_pert = CellStatePrediction(1.0, (1.0, 0.0, 0.0))
        is_mode = ll_mle > count_loglikelihood(sample, state_pert, calibration)
    else
        score_mle = (observed / c_hat) - kappa
        state_low = CellStatePrediction(0.95 * c_hat, (1.0, 0.0, 0.0))
        state_high = CellStatePrediction(1.05 * c_hat, (1.0, 0.0, 0.0))
        is_mode = (ll_mle > count_loglikelihood(sample, state_low, calibration)) &&
                  (ll_mle > count_loglikelihood(sample, state_high, calibration)) &&
                  (abs(score_mle) <= atol)
    end
    return (
        consistent = mean_matches && is_mode,
        c_hat_cells_per_ml = c_hat,
        expected_events_at_mle = lambda_mle,
        observed_events = observed,
        ll_at_mle = ll_mle,
        score_at_mle = score_mle,
    )
end

# ---------------------------------------------------------------------------
# Priority 2: Synthetic-data generation and parameter recovery
# ---------------------------------------------------------------------------

function _rand_poisson_knuth(rng::AbstractRNG, λ::Float64)
    λ <= 0.0 && return 0
    L = exp(-λ)
    k = 0
    p = 1.0
    while p > L
        k += 1
        p *= rand(rng)
    end
    return k - 1
end

function _rand_poisson(rng::AbstractRNG, λ::Float64)
    λ <= 0.0 && return 0
    if λ <= 10_000.0
        # Exact Poisson draw via independent additive decomposition into chunks <= 12.0
        total = 0
        rem = λ
        while rem > 12.0
            total += _rand_poisson_knuth(rng, 12.0)
            rem -= 12.0
        end
        if rem > 0.0
            total += _rand_poisson_knuth(rng, rem)
        end
        return total
    else
        z = randn(rng)
        val = round(Int, λ + sqrt(λ) * z)
        return max(0, val)
    end
end

function _rand_multinomial3(rng::AbstractRNG, n::Int, p::AbstractVector{Float64})
    n == 0 && return (0, 0, 0)
    n1 = 0
    n2 = 0
    n3 = 0
    p1 = p[1]
    p12 = p[1] + p[2]
    for _ in 1:n
        u = rand(rng)
        if u < p1
            n1 += 1
        elseif u < p12
            n2 += 1
        else
            n3 += 1
        end
    end
    return (n1, n2, n3)
end

"""
Simulate a synthetic `FCMSampleSummary` from known latent `state` and `calibration`.
"""
function simulate_fcm_sample(state::CellStatePrediction,
                             calibration::GateCalibration;
                             beads_added::Real=10_000.0,
                             sample_volume_ml::Real=0.1,
                             dilution_factor::Real=1.0,
                             extraction_recovery::Real=1.0,
                             aspirated_fraction::Real=0.01,
                             sample_origin::Symbol=:effluent,
                             gate_id::AbstractString="synthetic-v1",
                             volume_convention::Symbol=:aliquot_post_dilution,
                             rng::AbstractRNG=Random.default_rng())
    af = Float64(aspirated_fraction)
    isfinite(af) && 0 < af <= 1 || throw(ArgumentError("aspirated_fraction must be in (0,1]"))
    ba = Float64(beads_added)
    sv = Float64(sample_volume_ml)
    df = Float64(dilution_factor)
    er = Float64(extraction_recovery)
    bead_events = max(1, _rand_poisson(rng, ba * af))

    dummy = FCMSampleSummary((0, 0, 0);
        bead_events=bead_events, beads_added=ba, sample_volume_ml=sv,
        dilution_factor=df, extraction_recovery=er,
        sample_origin=sample_origin, gate_id=gate_id,
        volume_convention=volume_convention)
    λ = expected_cell_events(dummy, state, calibration)
    total_events = _rand_poisson(rng, λ)
    gate_probs = predicted_gate_probabilities(state, calibration)
    counts = _rand_multinomial3(rng, total_events, gate_probs)

    return FCMSampleSummary(counts;
        bead_events=bead_events, beads_added=ba, sample_volume_ml=sv,
        dilution_factor=df, extraction_recovery=er,
        sample_origin=sample_origin, gate_id=gate_id,
        volume_convention=volume_convention)
end

function _project_simplex3(v::AbstractVector{Float64})
    u = sort(collect(v); rev=true)
    cssv = cumsum(u)
    rho = 1
    for j in 1:3
        if u[j] + (1.0 - cssv[j]) / j > 0.0
            rho = j
        end
    end
    theta = (cssv[rho] - 1.0) / rho
    w = [max(v[i] - theta, 0.0) for i in 1:3]
    s = sum(w)
    return s > 0 ? w ./ s : [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
end

"""
Recover the maximum-likelihood `CellStatePrediction` from one or more `FCMSampleSummary`
observations from the same `sample_origin` compartment given a known `GateCalibration`.
"""
function recover_latent_state_mle(samples::AbstractVector{FCMSampleSummary},
                                  calibration::GateCalibration)
    isempty(samples) && throw(ArgumentError("samples must not be empty"))
    origin = first(samples).sample_origin
    all(s -> s.sample_origin == origin, samples) ||
        throw(ArgumentError("cannot pool samples across different sample_origin compartments in recover_latent_state_mle"))

    total_counts = [0.0, 0.0, 0.0]
    total_exposure = 0.0
    for s in samples
        total_counts[1] += s.gate_counts[1]
        total_counts[2] += s.gate_counts[2]
        total_counts[3] += s.gate_counts[3]
        v_eff = effective_sample_volume_ml(s)
        total_exposure += calibration.detection_efficiency * v_eff *
                          s.extraction_recovery * (s.bead_events / s.beads_added)
    end
    n_sum = sum(total_counts)
    c_hat = total_exposure > 0 ? n_sum / total_exposure : 0.0

    if n_sum == 0
        return CellStatePrediction(0.0, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    end

    q_emp = total_counts ./ n_sum
    Cmat = calibration.confusion
    # Multiplicative EM / Richardson-Lucy updates on the 3-simplex, guaranteed to
    # monotonically increase the multinomial log-likelihood on the simplex.
    f = _project_simplex3(Cmat \ q_emp)
    f = clamp.(f, 1e-9, 1.0)
    f ./= sum(f)
    for _ in 1:400
        p = Cmat * f
        ratio = [q_emp[i] / max(p[i], 1e-15) for i in 1:3]
        f_next = f .* (transpose(Cmat) * ratio)
        s = sum(f_next)
        s > 0 || break
        f_next ./= s
        if maximum(abs.(f_next .- f)) < 1e-12
            f = f_next
            break
        end
        f = f_next
    end
    f = _project_simplex3(f)
    return CellStatePrediction(c_hat, (f[1], f[2], f[3]))
end

# ---------------------------------------------------------------------------
# Priority 3: Calibration protocol (MESF, bead size, and gate confusion)
# ---------------------------------------------------------------------------

"""
Log-linear fluorophore calibration curve from calibration beads:
  log10(MESF) = slope * log10(FL) + intercept
"""
struct MESFCalibrationCurve
    channel::Symbol
    slope::Float64
    intercept::Float64
    r_squared::Float64
    n_bead_populations::Int

    function MESFCalibrationCurve(channel::Symbol, slope::Real, intercept::Real,
                                  r_squared::Real, n_bead_populations::Integer=5)
        s = Float64(slope)
        ic = Float64(intercept)
        r2 = Float64(r_squared)
        nb = Int(n_bead_populations)
        isfinite(s) && s > 0 || throw(ArgumentError("MESF slope must be finite and > 0"))
        isfinite(ic) || throw(ArgumentError("MESF intercept must be finite"))
        isfinite(r2) && 0 <= r2 <= 1.000001 || throw(ArgumentError("r_squared must be in [0,1]"))
        nb >= 3 || throw(ArgumentError("at least 3 bead populations are required"))
        new(channel, s, ic, min(r2, 1.0), nb)
    end
end

"""
Fit `log10(MESF) = slope * log10(FL) + intercept` by ordinary least squares.
"""
function fit_mesf_calibration(fl_intensities::AbstractVector{<:Real},
                              mesf_values::AbstractVector{<:Real};
                              channel::Symbol=:FL1)
    n = length(fl_intensities)
    n == length(mesf_values) || throw(ArgumentError("fl_intensities and mesf_values must have equal length"))
    n >= 3 || throw(ArgumentError("at least 3 calibration bead populations are required"))
    all(x -> isfinite(x) && x > 0, fl_intensities) ||
        throw(ArgumentError("fl_intensities must be finite and > 0"))
    all(y -> isfinite(y) && y > 0, mesf_values) ||
        throw(ArgumentError("mesf_values must be finite and > 0"))

    x = log10.(Float64.(fl_intensities))
    y = log10.(Float64.(mesf_values))
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((x .- mx) .^ 2)
    syy = sum((y .- my) .^ 2)
    sxy = sum((x .- mx) .* (y .- my))
    sxx > 0 || throw(ArgumentError("fl_intensities must not all be identical"))
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = syy > 0 ? (sxy * sxy) / (sxx * syy) : 1.0
    return MESFCalibrationCurve(channel, slope, intercept, r2, n)
end

"Convert arbitrary fluorescence intensity `fl_intensity` to MESF units."
function predict_mesf(curve::MESFCalibrationCurve, fl_intensity::Real)
    fl = Float64(fl_intensity)
    isfinite(fl) && fl > 0 || throw(ArgumentError("fl_intensity must be finite and > 0"))
    return 10.0 ^ (curve.slope * log10(fl) + curve.intercept)
end

"Convert MESF value back to expected fluorescence intensity in channel units."
function inverse_mesf(curve::MESFCalibrationCurve, mesf::Real)
    m = Float64(mesf)
    isfinite(m) && m > 0 || throw(ArgumentError("mesf must be finite and > 0"))
    return 10.0 ^ ((log10(m) - curve.intercept) / curve.slope)
end

"""
Calibration curve relating measured optical/scatter area (`um^2`) to spherical
cross-sectional area (`pi * (Diameter / 2)^2`):
  pi * (Diameter / 2)^2 = slope * Area + intercept
"""
struct BeadSizeCalibration
    slope::Float64
    intercept::Float64
    r_squared::Float64
    n_bead_populations::Int

    function BeadSizeCalibration(slope::Real, intercept::Real,
                                 r_squared::Real, n_bead_populations::Integer=5)
        s = Float64(slope)
        ic = Float64(intercept)
        r2 = Float64(r_squared)
        nb = Int(n_bead_populations)
        isfinite(s) && s > 0 || throw(ArgumentError("size slope must be finite and > 0"))
        isfinite(ic) || throw(ArgumentError("size intercept must be finite"))
        isfinite(r2) && 0 <= r2 <= 1.000001 || throw(ArgumentError("r_squared must be in [0,1]"))
        nb >= 3 || throw(ArgumentError("at least 3 bead populations are required"))
        new(s, ic, min(r2, 1.0), nb)
    end
end

"""
Fit `pi * (Diameter / 2)^2 = slope * Area + intercept` from size-standard beads.
"""
function fit_bead_size_calibration(measured_areas_um2::AbstractVector{<:Real},
                                   true_diameters_um::AbstractVector{<:Real})
    n = length(measured_areas_um2)
    n == length(true_diameters_um) || throw(ArgumentError("areas and diameters must have equal length"))
    n >= 3 || throw(ArgumentError("at least 3 bead size standards are required"))
    all(a -> isfinite(a) && a > 0, measured_areas_um2) ||
        throw(ArgumentError("measured_areas_um2 must be finite and > 0"))
    all(d -> isfinite(d) && d > 0, true_diameters_um) ||
        throw(ArgumentError("true_diameters_um must be finite and > 0"))

    x = Float64.(measured_areas_um2)
    y = [pi * (Float64(d) / 2.0)^2 for d in true_diameters_um]
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((x .- mx) .^ 2)
    syy = sum((y .- my) .^ 2)
    sxy = sum((x .- mx) .* (y .- my))
    sxx > 0 || throw(ArgumentError("measured_areas_um2 must not all be identical"))
    slope = sxy / sxx
    intercept = my - slope * mx
    r2 = syy > 0 ? (sxy * sxy) / (sxx * syy) : 1.0
    return BeadSizeCalibration(slope, intercept, r2, n)
end

"Predict equivalent spherical diameter (um) from measured scatter/image area (`um^2`)."
function predict_equivalent_diameter_um(cal::BeadSizeCalibration, measured_area_um2::Real)
    a = Float64(measured_area_um2)
    isfinite(a) && a > 0 || throw(ArgumentError("measured_area_um2 must be finite and > 0"))
    cross_section = cal.slope * a + cal.intercept
    cross_section > 0 || throw(ArgumentError("measured_area_um2 is below calibrated positive cross-section threshold"))
    return 2.0 * sqrt(cross_section / pi)
end

"""
Estimate `GateCalibration` from independent single-state control experiments.

`control_counts[i, j]` is the number of events observed in gate `i` when running
the pure control for true state `j`. Optional `clay_blank_counts` are subtracted
prior to column normalization, and optional `spiked_cells` and `measured_control_cells`
estimate `detection_efficiency`.
"""
function estimate_gate_calibration(control_counts::AbstractMatrix{<:Real};
                                   clay_blank_counts::NTuple{3,<:Real}=(0.0, 0.0, 0.0),
                                   spiked_cells::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                   measured_control_cells::Union{Nothing,AbstractVector{<:Real}}=nothing,
                                   pseudocount::Real=0.0)
    size(control_counts) == (3, 3) || throw(ArgumentError("control_counts must be 3x3"))
    pc = Float64(pseudocount)
    isfinite(pc) && pc >= 0 || throw(ArgumentError("pseudocount must be finite and >= 0"))
    blank = [Float64(clay_blank_counts[1]), Float64(clay_blank_counts[2]), Float64(clay_blank_counts[3])]
    all(b -> isfinite(b) && b >= 0, blank) || throw(ArgumentError("clay_blank_counts must be >= 0"))

    corrected = zeros(Float64, 3, 3)
    for j in 1:3
        for i in 1:3
            val = Float64(control_counts[i, j])
            isfinite(val) && val >= 0 || throw(ArgumentError("control_counts entries must be >= 0"))
            corrected[i, j] = max(0.0, val - blank[i]) + pc
        end
        colsum = sum(@view corrected[:, j])
        colsum > 0 || throw(ArgumentError("control column $j has zero net events after blank subtraction"))
        corrected[:, j] ./= colsum
    end

    det_eff = 1.0
    if spiked_cells !== nothing || measured_control_cells !== nothing
        spiked_cells !== nothing && measured_control_cells !== nothing ||
            throw(ArgumentError("provide both spiked_cells and measured_control_cells or neither"))
        length(spiked_cells) == length(measured_control_cells) && !isempty(spiked_cells) ||
            throw(ArgumentError("spiked_cells and measured_control_cells must be non-empty and equal length"))
        s_vec = Float64.(spiked_cells)
        m_vec = Float64.(measured_control_cells)
        all(x -> isfinite(x) && x > 0, s_vec) || throw(ArgumentError("spiked_cells entries must be > 0"))
        all(x -> isfinite(x) && x > 0, m_vec) || throw(ArgumentError("measured_control_cells entries must be > 0"))
        s_sum = sum(s_vec)
        m_sum = sum(m_vec)
        m_sum <= s_sum || throw(ArgumentError("measured_control_cells sum ($m_sum) cannot exceed spiked_cells sum ($s_sum)"))
        det_eff = m_sum / s_sum
    end
    return GateCalibration(corrected; detection_efficiency=det_eff)
end

# ---------------------------------------------------------------------------
# Priority 4: Uncertainty propagation and identifiability / equifinality
# ---------------------------------------------------------------------------

struct ConcentrationUncertainty
    estimate_cells_per_ml::Float64
    standard_error::Float64
    relative_sd::Float64
    ci_95_lower::Float64
    ci_95_upper::Float64
end

"""
Propagate counting shot noise (cells and beads) plus relative uncertainties in
extraction recovery, detection efficiency, and bead count into a standard error
and 95% confidence interval for `absolute_cells_per_ml`.
"""
function concentration_uncertainty(sample::FCMSampleSummary;
                                   detection_efficiency::Real=1.0,
                                   rel_sd_recovery::Real=0.0,
                                   rel_sd_efficiency::Real=0.0,
                                   rel_sd_beads_added::Real=0.0)
    c_hat = absolute_cells_per_ml(sample; detection_efficiency=detection_efficiency)
    rr = Float64(rel_sd_recovery)
    re = Float64(rel_sd_efficiency)
    rb = Float64(rel_sd_beads_added)
    all(x -> isfinite(x) && x >= 0, (rr, re, rb)) ||
        throw(ArgumentError("relative standard deviations must be finite and >= 0"))

    n_cells = sum(sample.gate_counts)
    v_eff = effective_sample_volume_ml(sample)
    de = Float64(detection_efficiency)

    if n_cells == 0
        # Exact one-sided 97.5% Poisson upper limit (-log(0.025) = 3.6888794541139363 events)
        upper_events = 3.6888794541139363
        scale_factor = (sample.beads_added / (sample.bead_events * v_eff * sample.extraction_recovery * de))
        ci_upper = upper_events * scale_factor * sqrt(1.0 + rr^2 + re^2 + rb^2)
        return ConcentrationUncertainty(0.0, scale_factor, Inf, 0.0, ci_upper)
    end

    rel_var = (1.0 / n_cells) + (1.0 / sample.bead_events) + rr^2 + re^2 + rb^2
    rel_sd = sqrt(rel_var)
    se = c_hat * rel_sd
    sigma_log = sqrt(log(1.0 + rel_var))
    z95 = 1.959963984540054
    ci_lower = c_hat * exp(-z95 * sigma_log)
    ci_upper = c_hat * exp(z95 * sigma_log)
    return ConcentrationUncertainty(c_hat, se, rel_sd, ci_lower, ci_upper)
end

function _det3x3(m::AbstractMatrix{Float64})
    return m[1,1] * (m[2,2]*m[3,3] - m[2,3]*m[3,2]) -
           m[1,2] * (m[2,1]*m[3,3] - m[2,3]*m[3,1]) +
           m[1,3] * (m[2,1]*m[3,2] - m[2,2]*m[3,1])
end

function _frobenius_cond3x3(m::AbstractMatrix{Float64})
    d = _det3x3(m)
    abs(d) <= 1e-14 && return Inf
    inv_m = zeros(Float64, 3, 3)
    inv_m[1,1] =  (m[2,2]*m[3,3] - m[2,3]*m[3,2]) / d
    inv_m[1,2] = -(m[1,2]*m[3,3] - m[1,3]*m[3,2]) / d
    inv_m[1,3] =  (m[1,2]*m[2,3] - m[1,3]*m[2,2]) / d
    inv_m[2,1] = -(m[2,1]*m[3,3] - m[2,3]*m[3,1]) / d
    inv_m[2,2] =  (m[1,1]*m[3,3] - m[1,3]*m[3,1]) / d
    inv_m[2,3] = -(m[1,1]*m[2,3] - m[1,3]*m[2,1]) / d
    inv_m[3,1] =  (m[2,1]*m[3,2] - m[2,2]*m[3,1]) / d
    inv_m[3,2] = -(m[1,1]*m[3,2] - m[1,2]*m[3,1]) / d
    inv_m[3,3] =  (m[1,1]*m[2,2] - m[1,2]*m[2,1]) / d
    norm_m = sqrt(sum(m .^ 2))
    norm_inv = sqrt(sum(inv_m .^ 2))
    return norm_m * norm_inv
end

"""
Audit structural and numerical identifiability of the observation model.
Reports:
- whether the 3x3 gate confusion matrix is invertible and well-conditioned, and
- that `total_cells_per_ml`, `extraction_recovery`, and `detection_efficiency`
  are structurally collinear unless `extraction_recovery` and `detection_efficiency`
  are fixed from independent calibration controls.
"""
function check_parameter_identifiability(calibration::GateCalibration;
                                         condition_threshold::Real=50.0)
    det_val = _det3x3(calibration.confusion)
    cond_val = _frobenius_cond3x3(calibration.confusion)
    well_conditioned = isfinite(cond_val) && cond_val <= Float64(condition_threshold)
    return (
        gate_fractions_identifiable = well_conditioned,
        confusion_determinant = det_val,
        confusion_condition_number = cond_val,
        scale_parameters_collinear_without_controls = true,
        collinear_triad = (:total_cells_per_ml, :extraction_recovery, :detection_efficiency),
    )
end

"""
Test whether two candidate latent states / biological mechanisms (`state_a`, `state_b`)
are observationally equifinal on `samples` under `calibration`.
"""
function evaluate_equifinality(samples::AbstractVector{FCMSampleSummary},
                               state_a::CellStatePrediction,
                               state_b::CellStatePrediction,
                               calibration::GateCalibration;
                               loglik_threshold::Real=2.0)
    isempty(samples) && throw(ArgumentError("samples must not be empty"))
    ll_a = sum(joint_loglikelihood(s, state_a, calibration) for s in samples)
    ll_b = sum(joint_loglikelihood(s, state_b, calibration) for s in samples)
    delta_ll = abs(ll_a - ll_b)

    pa = predicted_gate_probabilities(state_a, calibration)
    pb = predicted_gate_probabilities(state_b, calibration)
    gate_kl = 0.0
    for i in 1:3
        if pa[i] > 0
            gate_kl += pa[i] * log(pa[i] / max(pb[i], 1e-15))
        end
    end

    joint_kl = 0.0
    for s in samples
        la = expected_cell_events(s, state_a, calibration)
        lb = expected_cell_events(s, state_b, calibration)
        count_kl = if la > 0
            la * log(la / max(lb, 1e-15)) - la + lb
        else
            lb
        end
        joint_kl += count_kl + la * gate_kl
    end

    return (
        loglikelihood_a = ll_a,
        loglikelihood_b = ll_b,
        abs_delta_loglikelihood = delta_ll,
        gate_kl_divergence = gate_kl,
        joint_kl_divergence = joint_kl,
        equifinal = delta_ll <= Float64(loglik_threshold),
    )
end

# ---------------------------------------------------------------------------
# Priority 5: Held-out validation and posterior-predictive checks
# ---------------------------------------------------------------------------

"""
Split `samples` into training and held-out validation subsets using a boolean mask
or integer holdout indices.
"""
function split_train_holdout(samples::AbstractVector{FCMSampleSummary},
                             holdout_mask::AbstractVector{Bool})
    length(samples) == length(holdout_mask) ||
        throw(ArgumentError("samples and holdout_mask must have the same length"))
    train = [samples[i] for i in eachindex(samples) if !holdout_mask[i]]
    holdout = [samples[i] for i in eachindex(samples) if holdout_mask[i]]
    !isempty(train) && !isempty(holdout) ||
        throw(ArgumentError("both train and holdout subsets must be non-empty"))
    return (train = train, holdout = holdout)
end

function split_train_holdout(samples::AbstractVector{FCMSampleSummary},
                             holdout_indices::AbstractVector{<:Integer})
    mask = falses(length(samples))
    for idx in holdout_indices
        1 <= idx <= length(samples) || throw(ArgumentError("holdout index $idx out of bounds"))
        mask[idx] = true
    end
    return split_train_holdout(samples, mask)
end

function _freeman_tukey_discrepancy(obs_counts::NTuple{3,Int}, exp_counts::NTuple{3,Float64})
    d = 0.0
    for i in 1:3
        d += (sqrt(Float64(obs_counts[i])) - sqrt(max(exp_counts[i], 0.0)))^2
    end
    return d
end

"""
Run out-of-sample posterior-predictive checks on held-out observations.
"""
function posterior_predictive_check(heldout_samples::AbstractVector{FCMSampleSummary},
                                    predicted_states::AbstractVector{CellStatePrediction},
                                    calibration::GateCalibration;
                                    n_draws::Int=500,
                                    seed::Int=42)
    m = length(heldout_samples)
    m > 0 && m == length(predicted_states) ||
        throw(ArgumentError("heldout_samples and predicted_states must be non-empty and equal length"))
    n_draws >= 50 || throw(ArgumentError("n_draws must be >= 50"))

    rng = MersenneTwister(seed)
    total_ll = 0.0
    z_sum = 0.0
    chi2_sum = 0.0
    covered_count = 0
    obs_ft_total = 0.0

    lambdas = Vector{Float64}(undef, m)
    probs = Vector{Vector{Float64}}(undef, m)

    for j in 1:m
        s = heldout_samples[j]
        st = predicted_states[j]
        total_ll += joint_loglikelihood(s, st, calibration)
        λ = expected_cell_events(s, st, calibration)
        p = predicted_gate_probabilities(st, calibration)
        lambdas[j] = λ
        probs[j] = p

        obs_n = sum(s.gate_counts)
        z_sum += λ > 0 ? abs(obs_n - λ) / sqrt(λ) : Float64(obs_n)
        exp_tuple = (λ * p[1], λ * p[2], λ * p[3])
        obs_ft_total += _freeman_tukey_discrepancy(s.gate_counts, exp_tuple)
        for i in 1:3
            if exp_tuple[i] > 0
                chi2_sum += (s.gate_counts[i] - exp_tuple[i])^2 / exp_tuple[i]
            end
        end

        sim_totals = Vector{Int}(undef, n_draws)
        for d in 1:n_draws
            sim_totals[d] = _rand_poisson(rng, λ)
        end
        sort!(sim_totals)
        lo_idx = max(1, floor(Int, 0.025 * n_draws))
        hi_idx = min(n_draws, ceil(Int, 0.975 * n_draws))
        if sim_totals[lo_idx] <= obs_n <= sim_totals[hi_idx]
            covered_count += 1
        end
    end

    exceed_count = 0
    for _ in 1:n_draws
        sim_ft_total = 0.0
        for j in 1:m
            λ = lambdas[j]
            p = probs[j]
            sim_n = _rand_poisson(rng, λ)
            sim_gates = _rand_multinomial3(rng, sim_n, p)
            exp_tuple = (λ * p[1], λ * p[2], λ * p[3])
            sim_ft_total += _freeman_tukey_discrepancy(sim_gates, exp_tuple)
        end
        if sim_ft_total >= obs_ft_total
            exceed_count += 1
        end
    end

    return (
        total_heldout_loglikelihood = total_ll,
        mean_count_abs_z = z_sum / m,
        gate_chi2_statistic = chi2_sum,
        empirical_coverage_95 = covered_count / m,
        posterior_predictive_p_value = exceed_count / n_draws,
    )
end

# ---------------------------------------------------------------------------
# Priority 6: Solver interfaces and stakeholder track data contracts
# ---------------------------------------------------------------------------

struct OpenFOAMTransportSnap
    vessel_id::String
    time_h::Float64
    compartment::Symbol
    velocity_m_s::Float64
    wall_shear_pa::Float64
    residence_time_s::Float64
    detachment_flux_cells_m2_s::Float64

    function OpenFOAMTransportSnap(vessel_id::AbstractString, time_h::Real, compartment::Symbol;
                                   velocity_m_s::Real, wall_shear_pa::Real,
                                   residence_time_s::Real, detachment_flux_cells_m2_s::Real=0.0)
        !isempty(strip(vessel_id)) || throw(ArgumentError("vessel_id must be non-empty"))
        th = Float64(time_h)
        vel = Float64(velocity_m_s)
        sh = Float64(wall_shear_pa)
        rt = Float64(residence_time_s)
        df = Float64(detachment_flux_cells_m2_s)
        isfinite(th) && th >= 0 || throw(ArgumentError("time_h must be finite and >= 0"))
        isfinite(vel) && vel >= 0 || throw(ArgumentError("velocity_m_s must be finite and >= 0"))
        isfinite(sh) && sh >= 0 || throw(ArgumentError("wall_shear_pa must be finite and >= 0"))
        isfinite(rt) && rt > 0 || throw(ArgumentError("residence_time_s must be finite and > 0"))
        isfinite(df) && df >= 0 || throw(ArgumentError("detachment_flux_cells_m2_s must be finite and >= 0"))
        new(String(vessel_id), th, compartment, vel, sh, rt, df)
    end
end

struct PHREEQCSpeciationSnap
    vessel_id::String
    time_h::Float64
    pH::Float64
    pe::Float64
    ionic_strength_M::Float64
    ca_mol_L::Float64
    mg_mol_L::Float64
    fe2_mol_L::Float64
    sulfide_mol_L::Float64

    function PHREEQCSpeciationSnap(vessel_id::AbstractString, time_h::Real;
                                   pH::Real, pe::Real, ionic_strength_M::Real,
                                   ca_mol_L::Real=0.0, mg_mol_L::Real=0.0,
                                   fe2_mol_L::Real=0.0, sulfide_mol_L::Real=0.0)
        !isempty(strip(vessel_id)) || throw(ArgumentError("vessel_id must be non-empty"))
        th = Float64(time_h)
        ph_v = Float64(pH)
        pe_v = Float64(pe)
        is_v = Float64(ionic_strength_M)
        ca = Float64(ca_mol_L)
        mg = Float64(mg_mol_L)
        fe2 = Float64(fe2_mol_L)
        s2 = Float64(sulfide_mol_L)
        isfinite(th) && th >= 0 || throw(ArgumentError("time_h must be finite and >= 0"))
        isfinite(ph_v) && 0 <= ph_v <= 14 || throw(ArgumentError("pH must be in [0,14]"))
        isfinite(pe_v) || throw(ArgumentError("pe must be finite"))
        isfinite(is_v) && is_v >= 0 || throw(ArgumentError("ionic_strength_M must be >= 0"))
        all(x -> isfinite(x) && x >= 0, (ca, mg, fe2, s2)) ||
            throw(ArgumentError("species concentrations must be finite and >= 0"))
        new(String(vessel_id), th, ph_v, pe_v, is_v, ca, mg, fe2, s2)
    end
end

"""
Explicit stakeholder track boundary contract preventing premature merging of
distinct physical systems.
"""
struct StakeholderTrackContract
    track::Symbol
    requires_fcm_observer::Bool
    requires_openfoam::Bool
    requires_phreeqc::Bool
    primary_endpoint::String

    function StakeholderTrackContract(track::Symbol;
                                      requires_fcm_observer::Bool,
                                      requires_openfoam::Bool,
                                      requires_phreeqc::Bool,
                                      primary_endpoint::AbstractString)
        track in VALID_STAKEHOLDER_TRACKS ||
            throw(ArgumentError("track must be one of $(VALID_STAKEHOLDER_TRACKS)"))
        pe = String(strip(primary_endpoint))
        !isempty(pe) || throw(ArgumentError("primary_endpoint must be non-empty"))

        if track == :track3_bitumen_rheology
            (!requires_fcm_observer && !requires_phreeqc) ||
                throw(ArgumentError("track3_bitumen_rheology must not introduce flow cytometry or clay-microbe PHREEQC models"))
        elseif track == :track2_clay_corrosion
            pe == "maximum_pit_depth_per_coupon" ||
                throw(ArgumentError("track2_clay_corrosion must preserve maximum_pit_depth_per_coupon as the primary endpoint"))
        end
        new(track, requires_fcm_observer, requires_openfoam, requires_phreeqc, pe)
    end
end

"""
Verify that OpenFOAM transport and PHREEQC speciation snapshots match the
cytometry sample's vessel, time point, and compartment before coupling.
"""
function validate_solver_observation_alignment(sample::FCMSampleSummary,
                                               sample_vessel_id::AbstractString,
                                               sample_time_h::Real;
                                               transport::Union{Nothing,OpenFOAMTransportSnap}=nothing,
                                               chemistry::Union{Nothing,PHREEQCSpeciationSnap}=nothing,
                                               atol_time_h::Real=1e-6)
    th = Float64(sample_time_h)
    isfinite(th) && th >= 0 || throw(ArgumentError("sample_time_h must be finite and >= 0"))
    if transport !== nothing
        transport.vessel_id == sample_vessel_id ||
            throw(ArgumentError("OpenFOAM vessel_id ($(transport.vessel_id)) does not match sample vessel_id ($sample_vessel_id)"))
        isapprox(transport.time_h, th; atol=atol_time_h, rtol=0.0) ||
            throw(ArgumentError("OpenFOAM time_h ($(transport.time_h)) does not match sample time_h ($th)"))
        transport.compartment == sample.sample_origin ||
            throw(ArgumentError("OpenFOAM compartment ($(transport.compartment)) does not match sample_origin ($(sample.sample_origin))"))
    end
    if chemistry !== nothing
        chemistry.vessel_id == sample_vessel_id ||
            throw(ArgumentError("PHREEQC vessel_id ($(chemistry.vessel_id)) does not match sample vessel_id ($sample_vessel_id)"))
        isapprox(chemistry.time_h, th; atol=atol_time_h, rtol=0.0) ||
            throw(ArgumentError("PHREEQC time_h ($(chemistry.time_h)) does not match sample time_h ($th)"))
    end
    return true
end

"""
Verify that a CSV header line contains all required columns defined in
`REQUIRED_CSV_COLUMNS`.
"""
function validate_sample_csv_header(header_line::AbstractString)
    cols = [strip(c) for c in split(chomp(header_line), ',')]
    missing_cols = [req for req in REQUIRED_CSV_COLUMNS if !(req in cols)]
    isempty(missing_cols) ||
        throw(ArgumentError("CSV header is missing required columns: $(join(missing_cols, ", "))"))
    return true
end

# ---------------------------------------------------------------------------
# Priority 7: Production acceptance and epistemic status accounting
# ---------------------------------------------------------------------------

struct EpistemicQuantity{T}
    name::String
    value::T
    status::Symbol
    source_note::String

    function EpistemicQuantity(name::AbstractString, value::T, status::Symbol,
                               source_note::AbstractString="") where {T}
        !isempty(strip(name)) || throw(ArgumentError("quantity name must be non-empty"))
        status in VALID_EPISTEMIC_STATUSES ||
            throw(ArgumentError("epistemic status must be one of $(VALID_EPISTEMIC_STATUSES)"))
        new{T}(String(name), value, status, String(source_note))
    end
end

struct ProductionAuditRecord
    model_version::String
    input_sha256::String
    track::Symbol
    quantities::Vector{EpistemicQuantity}
    consistent_likelihood::Bool
    identifiable_gates::Bool
    heldout_validated::Bool
end

"""
Construct a reproducible production acceptance audit record separating
observed, calibrated, assumed, and predicted quantities.
"""
function build_production_audit_record(; track::Symbol,
                                       input_sha256::AbstractString,
                                       quantities::AbstractVector{<:EpistemicQuantity},
                                       sample::FCMSampleSummary,
                                       calibration::GateCalibration,
                                       heldout_validated::Bool=false)
    track in VALID_STAKEHOLDER_TRACKS ||
        throw(ArgumentError("track must be one of $(VALID_STAKEHOLDER_TRACKS)"))
    digest = lowercase(strip(String(input_sha256)))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        throw(ArgumentError("input_sha256 must be a valid 64-character lowercase hex SHA-256 digest"))

    present_statuses = Set(q.status for q in quantities)
    for req in VALID_EPISTEMIC_STATUSES
        req in present_statuses ||
            throw(ArgumentError("production audit record must explicitly include at least one :$req quantity"))
    end

    consistency = verify_concentration_likelihood_consistency(sample, calibration)
    ident = check_parameter_identifiability(calibration)

    return ProductionAuditRecord(
        MODEL_ADAPTER_VERSION,
        digest,
        track,
        collect(EpistemicQuantity, quantities),
        consistency.consistent,
        ident.gate_fractions_identifiable,
        heldout_validated,
    )
end

end # module
