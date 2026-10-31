module FlowCytometryObservationModel

using Random
using Printf
import SHA

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
       VALID_OPENFOAM_FIELD_CLASSES, VALID_OPENFOAM_PATCH_TYPES,
       VALID_DATA_PROVENANCES,
       REQUIRED_CSV_COLUMNS,
       CellStatePrediction, GateCalibration, FCMSampleSummary,
       MESFCalibrationCurve, BeadSizeCalibration,
       ConcentrationUncertainty,
       OpenFOAMTransportSnap, PHREEQCSpeciationSnap, StakeholderTrackContract,
       OpenFOAMBoundaryPatch, OpenFOAMMeshExchange,
       PHREEQCMineralPhase, PHREEQCSpeciesAdapter,
       EmpiricalValidationProtocol,
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
       serialize_openfoam_ascii_field, parse_openfoam_ascii_field,
       map_openfoam_field_conservative, audit_openfoam_transport_stability,
       summarize_openfoam_mesh_to_snap,
       audit_phreeqc_speciation_balance,
       serialize_phreeqc_selected_output, parse_phreeqc_selected_output,
       summarize_phreeqc_adapter_to_snap,
       evaluate_empirical_generalization_readiness,
       validate_solver_observation_alignment, validate_sample_csv_header,
       build_production_audit_record

const MODEL_ADAPTER_VERSION = "0.3.0"
const VALID_SAMPLE_ORIGINS = (:effluent, :planktonic, :detached_biofilm)
const VALID_VOLUME_CONVENTIONS = (:aliquot_post_dilution, :undiluted_equivalent, :beads_co_diluted)
const VALID_STAKEHOLDER_TRACKS = (:track1_nanobio, :track2_clay_corrosion, :track3_bitumen_rheology)
const VALID_EPISTEMIC_STATUSES = (:observed, :calibrated, :assumed, :predicted)
const VALID_OPENFOAM_FIELD_CLASSES = (:volScalarField, :volVectorField)
const VALID_OPENFOAM_PATCH_TYPES = (:fixedValue, :zeroGradient, :fixedFlux, :empty, :wall, :inlet, :outlet)
const VALID_DATA_PROVENANCES = (:synthetic_simulation, :bench_control_only, :deidentified_experimental_fcs)

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

function _rand_poisson_atkinson(rng::AbstractRNG, λ::Float64)
    c = 0.767 - 3.36 / λ
    beta = pi / sqrt(3.0 * λ)
    alpha = beta * λ
    k = log(c) - λ - log(beta)
    log_lam = log(λ)
    while true
        u = rand(rng)
        (u <= 0.0 || u >= 1.0) && continue
        x = (alpha - log((1.0 - u) / u)) / beta
        x + 0.5 < 0.0 && continue
        n = floor(Int, x + 0.5)
        v = rand(rng)
        v <= 0.0 && continue
        y = alpha - beta * x
        lhs = y + log(v / (1.0 + exp(y))^2)
        rhs = k + n * log_lam - _loggamma_positive(Float64(n) + 1.0)
        if lhs <= rhs
            return n
        end
    end
end

function _rand_poisson(rng::AbstractRNG, λ::Float64)
    λ <= 0.0 && return 0
    if λ <= 64.0
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
        # Exact Poisson draw via Atkinson (1979) logistic rejection method
        return _rand_poisson_atkinson(rng, λ)
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

function _solve3x3_safe(m::AbstractMatrix{Float64}, rhs::AbstractVector{Float64})
    d = m[1,1] * (m[2,2]*m[3,3] - m[2,3]*m[3,2]) -
        m[1,2] * (m[2,1]*m[3,3] - m[2,3]*m[3,1]) +
        m[1,3] * (m[2,1]*m[3,2] - m[2,2]*m[3,1])
    abs(d) <= 1e-10 && return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    inv11 =  (m[2,2]*m[3,3] - m[2,3]*m[3,2]) / d
    inv12 = -(m[1,2]*m[3,3] - m[1,3]*m[3,2]) / d
    inv13 =  (m[1,2]*m[2,3] - m[1,3]*m[2,2]) / d
    inv21 = -(m[2,1]*m[3,3] - m[2,3]*m[3,1]) / d
    inv22 =  (m[1,1]*m[3,3] - m[1,3]*m[3,1]) / d
    inv23 = -(m[1,1]*m[2,3] - m[1,3]*m[2,1]) / d
    inv31 =  (m[2,1]*m[3,2] - m[2,2]*m[3,1]) / d
    inv32 = -(m[1,1]*m[3,2] - m[1,2]*m[3,1]) / d
    inv33 =  (m[1,1]*m[2,2] - m[1,2]*m[2,1]) / d
    return [
        inv11 * rhs[1] + inv12 * rhs[2] + inv13 * rhs[3],
        inv21 * rhs[1] + inv22 * rhs[2] + inv23 * rhs[3],
        inv31 * rhs[1] + inv32 * rhs[2] + inv33 * rhs[3],
    ]
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
    # monotonically increase the multinomial log-likelihood on the simplex and never
    # fail on rank-deficient confusion matrices.
    f_inv = _project_simplex3(_solve3x3_safe(Cmat, q_emp))
    f = 0.85 .* f_inv .+ 0.15 .* [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    f ./= sum(f)
    for _ in 1:600
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
        net_colsum = 0.0
        for i in 1:3
            val = Float64(control_counts[i, j])
            isfinite(val) && val >= 0 || throw(ArgumentError("control_counts entries must be >= 0"))
            net_val = max(0.0, val - blank[i])
            net_colsum += net_val
            corrected[i, j] = net_val + pc
        end
        net_colsum > 0 || throw(ArgumentError("control column $j has zero net events after blank subtraction"))
        colsum = sum(@view corrected[:, j])
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
        all(k -> m_vec[k] <= s_vec[k], eachindex(s_vec)) ||
            throw(ArgumentError("each measured_control_cells entry cannot exceed its corresponding spiked_cells entry"))
        s_sum = sum(s_vec)
        m_sum = sum(m_vec)
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

function _garwood_poisson_95_events(n::Int)
    if n == 0
        return (0.0, 3.6888794541139363)
    elseif n == 1
        return (0.025317807984289886, 5.571643390992898)
    elseif n == 2
        return (0.2422092777972947, 7.224687665243912)
    else
        z95 = 1.959963984540054
        # Exact Wilson-Hilferty chi-square quantile approximation for 2n and 2(n+1) d.o.f.
        k_lo = 2.0 * n
        lo = 0.5 * k_lo * max(0.0, 1.0 - 2.0 / (9.0 * k_lo) - z95 * sqrt(2.0 / (9.0 * k_lo)))^3
        k_hi = 2.0 * (n + 1)
        hi = 0.5 * k_hi * (1.0 - 2.0 / (9.0 * k_hi) + z95 * sqrt(2.0 / (9.0 * k_hi)))^3
        return (lo, hi)
    end
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
    scale_factor = sample.beads_added / (sample.bead_events * v_eff * sample.extraction_recovery * de)
    mult_rel_var = (1.0 / sample.bead_events) + rr^2 + re^2 + rb^2

    if n_cells == 0
        # Exact one-sided 97.5% Poisson upper limit (-log(0.025) = 3.6888794541139363 events)
        _, upper_events = _garwood_poisson_95_events(0)
        ci_upper = upper_events * scale_factor * sqrt(1.0 + mult_rel_var)
        return ConcentrationUncertainty(0.0, scale_factor, Inf, 0.0, ci_upper)
    end

    rel_var = (1.0 / n_cells) + mult_rel_var
    rel_sd = sqrt(rel_var)
    se = c_hat * rel_sd
    sigma_log = sqrt(log(1.0 + rel_var))
    z95 = 1.959963984540054
    lognorm_lower = c_hat * exp(-z95 * sigma_log)
    lognorm_upper = c_hat * exp(z95 * sigma_log)

    gar_lo, gar_hi = _garwood_poisson_95_events(n_cells)
    sigma_mult_log = sqrt(log(1.0 + mult_rel_var))
    garwood_lower = (gar_lo * scale_factor) * exp(-z95 * sigma_mult_log)
    garwood_upper = (gar_hi * scale_factor) * exp(z95 * sigma_mult_log)

    ci_lower = min(lognorm_lower, garwood_lower)
    ci_upper = max(lognorm_upper, garwood_upper)
    return ConcentrationUncertainty(c_hat, se, rel_sd, ci_lower, ci_upper)
end

function concentration_uncertainty(sample::FCMSampleSummary,
                                   calibration::GateCalibration;
                                   rel_sd_recovery::Real=0.0,
                                   rel_sd_efficiency::Real=0.0,
                                   rel_sd_beads_added::Real=0.0)
    return concentration_uncertainty(sample;
        detection_efficiency=calibration.detection_efficiency,
        rel_sd_recovery=rel_sd_recovery,
        rel_sd_efficiency=rel_sd_efficiency,
        rel_sd_beads_added=rel_sd_beads_added)
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
    ct = Float64(condition_threshold)
    isfinite(ct) && ct > 0 || throw(ArgumentError("condition_threshold must be finite and > 0"))
    det_val = _det3x3(calibration.confusion)
    cond_val = _frobenius_cond3x3(calibration.confusion)
    well_conditioned = isfinite(cond_val) && cond_val <= ct
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
                               loglik_threshold::Real=2.0,
                               kl_threshold::Real=1.0)
    isempty(samples) && throw(ArgumentError("samples must not be empty"))
    llt = Float64(loglik_threshold)
    klt = Float64(kl_threshold)
    isfinite(llt) && llt >= 0 || throw(ArgumentError("loglik_threshold must be finite and >= 0"))
    isfinite(klt) && klt >= 0 || throw(ArgumentError("kl_threshold must be finite and >= 0"))

    ll_a = sum(joint_loglikelihood(s, state_a, calibration) for s in samples)
    ll_b = sum(joint_loglikelihood(s, state_b, calibration) for s in samples)
    delta_ll = (ll_a == ll_b) ? 0.0 : abs(ll_a - ll_b)

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

    is_equifinal = isfinite(ll_a) && isfinite(ll_b) &&
                   (delta_ll <= llt) && (joint_kl <= klt)

    return (
        loglikelihood_a = ll_a,
        loglikelihood_b = ll_b,
        abs_delta_loglikelihood = delta_ll,
        gate_kl_divergence = gate_kl,
        joint_kl_divergence = joint_kl,
        equifinal = is_equifinal,
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
        !mask[idx] || throw(ArgumentError("duplicate holdout index $idx"))
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
    k_tail = floor(Int, 0.025 * n_draws)
    lo_idx = k_tail + 1
    hi_idx = n_draws - k_tail

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
            elseif s.gate_counts[i] > 0
                chi2_sum = Inf
            end
        end

        sim_totals = Vector{Int}(undef, n_draws)
        for d in 1:n_draws
            sim_totals[d] = _rand_poisson(rng, λ)
        end
        sort!(sim_totals)
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
        vid = String(strip(vessel_id))
        !isempty(vid) || throw(ArgumentError("vessel_id must be non-empty"))
        compartment in VALID_SAMPLE_ORIGINS ||
            throw(ArgumentError("compartment must be one of $(VALID_SAMPLE_ORIGINS)"))
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
        new(vid, th, compartment, vel, sh, rt, df)
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
        vid = String(strip(vessel_id))
        !isempty(vid) || throw(ArgumentError("vessel_id must be non-empty"))
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
        new(vid, th, ph_v, pe_v, is_v, ca, mg, fe2, s2)
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

# ---------------------------------------------------------------------------
# M1: Mesh-aware OpenFOAM spatial transport field exchange and conservation
# ---------------------------------------------------------------------------

struct OpenFOAMBoundaryPatch
    name::String
    patch_type::Symbol
    values::Vector{Float64}
    flux_mol_s::Float64

    function OpenFOAMBoundaryPatch(name::AbstractString,
                                   patch_type::Symbol,
                                   values::AbstractVector{<:Real}=Float64[];
                                   flux_mol_s::Real=0.0)
        nm = String(strip(name))
        !isempty(nm) || throw(ArgumentError("boundary patch name must be non-empty"))
        patch_type in VALID_OPENFOAM_PATCH_TYPES ||
            throw(ArgumentError("patch_type must be one of $(VALID_OPENFOAM_PATCH_TYPES)"))
        vals = Float64.(values)
        all(isfinite, vals) || throw(ArgumentError("boundary patch values must be finite"))
        flx = Float64(flux_mol_s)
        isfinite(flx) || throw(ArgumentError("flux_mol_s must be finite"))
        new(nm, patch_type, vals, flx)
    end
end

"""
Mesh-aware OpenFOAM spatial field snapshot (`OpenFOAMMeshExchange`) for offline
one-way and conservative two-way multiphysics exchange.

`dimensions` follows the 7-element OpenFOAM SI dimension set:
  `[kg, m, s, K, mol, A, cd]`
For example:
  - molar concentration (`mol/m^3`): `(0, -3, 0, 0, 1, 0, 0)`
  - velocity (`m/s`): `(0, 1, -1, 0, 0, 0, 0)`
  - diffusivity (`m^2/s`): `(0, 2, -1, 0, 0, 0, 0)`
  - pressure / shear stress (`Pa`): `(1, -1, -2, 0, 0, 0, 0)`
"""
struct OpenFOAMMeshExchange
    vessel_id::String
    time_h::Float64
    timestep_s::Float64
    compartment::Symbol
    mesh_id::String
    solver_version::String
    field_name::String
    field_class::Symbol
    dimensions::NTuple{7,Int}
    cell_centers_m::Matrix{Float64}
    cell_volumes_m3::Vector{Float64}
    internal_field::Matrix{Float64}
    boundary_patches::Vector{OpenFOAMBoundaryPatch}
    velocity_magnitudes_m_s::Vector{Float64}
    diffusivities_m2_s::Vector{Float64}
    epistemic_status::Symbol
    source_sha256::String

    function OpenFOAMMeshExchange(;
            vessel_id::AbstractString,
            time_h::Real,
            timestep_s::Real,
            compartment::Symbol,
            mesh_id::AbstractString="openfoam-polyMesh-v1",
            solver_version::AbstractString="OpenFOAM-v2406",
            field_name::AbstractString="C_substrate",
            field_class::Symbol=:volScalarField,
            dimensions::NTuple{7,<:Integer}=(0, -3, 0, 0, 1, 0, 0),
            cell_centers_m::AbstractMatrix{<:Real},
            cell_volumes_m3::AbstractVector{<:Real},
            internal_field::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
            boundary_patches::AbstractVector{OpenFOAMBoundaryPatch}=OpenFOAMBoundaryPatch[],
            velocity_magnitudes_m_s::AbstractVector{<:Real}=Float64[],
            diffusivities_m2_s::AbstractVector{<:Real}=Float64[],
            epistemic_status::Symbol=:predicted,
            source_sha256::AbstractString="")
        vid = String(strip(vessel_id))
        mid = String(strip(mesh_id))
        sver = String(strip(solver_version))
        fname = String(strip(field_name))
        !isempty(vid) && !isempty(mid) && !isempty(sver) && !isempty(fname) ||
            throw(ArgumentError("vessel_id, mesh_id, solver_version, and field_name must be non-empty"))
        compartment in VALID_SAMPLE_ORIGINS ||
            throw(ArgumentError("compartment must be one of $(VALID_SAMPLE_ORIGINS)"))
        field_class in VALID_OPENFOAM_FIELD_CLASSES ||
            throw(ArgumentError("field_class must be one of $(VALID_OPENFOAM_FIELD_CLASSES)"))
        epistemic_status in VALID_EPISTEMIC_STATUSES ||
            throw(ArgumentError("epistemic_status must be one of $(VALID_EPISTEMIC_STATUSES)"))

        th = Float64(time_h)
        dt = Float64(timestep_s)
        isfinite(th) && th >= 0 || throw(ArgumentError("time_h must be finite and >= 0"))
        isfinite(dt) && dt > 0 || throw(ArgumentError("timestep_s must be finite and > 0"))

        n_cells = length(cell_volumes_m3)
        n_cells >= 1 || throw(ArgumentError("mesh must contain at least 1 cell"))
        size(cell_centers_m) == (n_cells, 3) ||
            throw(ArgumentError("cell_centers_m must have dimensions (n_cells, 3)"))
        centers = Float64.(cell_centers_m)
        vols = Float64.(cell_volumes_m3)
        all(isfinite, centers) || throw(ArgumentError("cell_centers_m entries must be finite"))
        all(v -> isfinite(v) && v > 0, vols) ||
            throw(ArgumentError("cell_volumes_m3 entries must be finite and > 0"))

        fld = if internal_field isa AbstractVector
            reshape(Float64.(internal_field), :, 1)
        else
            Float64.(internal_field)
        end
        size(fld, 1) == n_cells ||
            throw(ArgumentError("internal_field row count ($(size(fld, 1))) must equal n_cells ($n_cells)"))
        expected_cols = field_class == :volScalarField ? 1 : 3
        size(fld, 2) == expected_cols ||
            throw(ArgumentError("internal_field column count ($(size(fld, 2))) must be $expected_cols for $field_class"))
        all(isfinite, fld) || throw(ArgumentError("internal_field entries must be finite"))

        vel_vec = isempty(velocity_magnitudes_m_s) ? zeros(Float64, n_cells) : Float64.(velocity_magnitudes_m_s)
        default_diff = field_class == :volVectorField ? 1.0e-6 : 1.0e-9
        diff_vec = isempty(diffusivities_m2_s) ? fill(default_diff, n_cells) : Float64.(diffusivities_m2_s)
        length(vel_vec) == n_cells ||
            throw(ArgumentError("velocity_magnitudes_m_s length must equal n_cells"))
        length(diff_vec) == n_cells ||
            throw(ArgumentError("diffusivities_m2_s length must equal n_cells"))
        all(x -> isfinite(x) && x >= 0, vel_vec) ||
            throw(ArgumentError("velocity_magnitudes_m_s must be finite and >= 0"))
        all(x -> isfinite(x) && x >= 0, diff_vec) ||
            throw(ArgumentError("diffusivities_m2_s must be finite and >= 0"))

        dims = NTuple{7,Int}(Int.(dimensions))
        patches = collect(OpenFOAMBoundaryPatch, boundary_patches)
        seen_patches = Set{String}()
        for p in patches
            !(p.name in seen_patches) ||
                throw(ArgumentError("duplicate boundary patch name '$(p.name)'"))
            push!(seen_patches, p.name)
        end

        digest = if isempty(strip(source_sha256))
            bytes2hex(SHA.sha256(string(vid, ":", th, ":", mid, ":", fname, ":", sum(vols), ":", sum(fld))))
        else
            d = lowercase(strip(String(source_sha256)))
            occursin(r"^[0-9a-f]{64}$", d) ||
                throw(ArgumentError("source_sha256 must be a valid 64-character lowercase hex SHA-256 digest"))
            d
        end

        new(vid, th, dt, compartment, mid, sver, fname, field_class, dims,
            centers, vols, fld, patches, vel_vec, diff_vec, epistemic_status, digest)
    end
end

"""
Serialize an `OpenFOAMMeshExchange` snapshot into an OpenFOAM v2.0 ASCII dictionary
with `FoamFile`, `dimensions`, `internalField`, and `boundaryField` blocks.
"""
function serialize_openfoam_ascii_field(ex::OpenFOAMMeshExchange)
    io = IOBuffer()
    println(io, "FoamFile")
    println(io, "{")
    println(io, "    version     2.0;")
    println(io, "    format      ascii;")
    println(io, "    class       $(ex.field_class);")
    println(io, "    location    \"$(ex.time_h)\";")
    println(io, "    object      $(ex.field_name);")
    println(io, "}")
    d = ex.dimensions
    println(io, "dimensions      [$(d[1]) $(d[2]) $(d[3]) $(d[4]) $(d[5]) $(d[6]) $(d[7])];")
    n = length(ex.cell_volumes_m3)
    if ex.field_class == :volScalarField
        println(io, "internalField   nonuniform List<scalar>")
        println(io, n)
        println(io, "(")
        for i in 1:n
            @printf(io, "%.12g\n", ex.internal_field[i, 1])
        end
        println(io, ");")
    else
        println(io, "internalField   nonuniform List<vector>")
        println(io, n)
        println(io, "(")
        for i in 1:n
            @printf(io, "(%.12g %.12g %.12g)\n",
                ex.internal_field[i, 1], ex.internal_field[i, 2], ex.internal_field[i, 3])
        end
        println(io, ");")
    end
    println(io, "boundaryField")
    println(io, "{")
    for p in ex.boundary_patches
        println(io, "    $(p.name)")
        println(io, "    {")
        println(io, "        type            $(p.patch_type);")
        @printf(io, "        flux_mol_s      %.12g;\n", p.flux_mol_s)
        if !isempty(p.values)
            println(io, "        value           nonuniform List<scalar> $(length(p.values)) ( ",
                join([@sprintf("%.12g", v) for v in p.values], " "), " );")
        end
        println(io, "    }")
    end
    println(io, "}")
    return String(take!(io))
end

"""
Strip C/C++ multi-line (`/* ... */`) and single-line (`// ...`) comments from an
OpenFOAM ASCII dictionary string before token/regex parsing.
"""
function _strip_openfoam_comments(txt::AbstractString)
    no_block = replace(String(txt), r"/\*.*?\*/"s => " ")
    return replace(no_block, r"//[^\n\r]*" => "")
end

const _OF_FLOAT_PATTERN = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"

"""
Parse an OpenFOAM v2.0 ASCII field dictionary (`volScalarField` or `volVectorField`)
into an `OpenFOAMMeshExchange` struct and record SHA-256 provenance of the raw text.
Supports standard OpenFOAM C/C++ header/footer comments, inline comments,
`uniform` and `nonuniform List<scalar|vector>` internal fields, and `uniform` or
`nonuniform` boundary patch values.
"""
function parse_openfoam_ascii_field(
        ascii_text::AbstractString;
        vessel_id::AbstractString,
        time_h::Real,
        timestep_s::Real,
        compartment::Symbol,
        mesh_id::AbstractString="openfoam-polyMesh-v1",
        solver_version::AbstractString="OpenFOAM-v2406",
        cell_centers_m::AbstractMatrix{<:Real},
        cell_volumes_m3::AbstractVector{<:Real},
        velocity_magnitudes_m_s::AbstractVector{<:Real}=Float64[],
        diffusivities_m2_s::AbstractVector{<:Real}=Float64[],
        epistemic_status::Symbol=:predicted)
    raw_txt = String(ascii_text)
    txt = _strip_openfoam_comments(raw_txt)
    occursin("FoamFile", txt) ||
        throw(ArgumentError("OpenFOAM dictionary is missing FoamFile header"))

    cls_m = match(r"class\s+(volScalarField|volVectorField)\s*;", txt)
    cls_m !== nothing || throw(ArgumentError("unsupported or missing OpenFOAM field class"))
    field_class = Symbol(cls_m.captures[1])

    obj_m = match(r"object\s+([A-Za-z0-9_.-]+)\s*;", txt)
    field_name = obj_m === nothing ? "C_substrate" : obj_m.captures[1]

    dim_m = match(r"dimensions\s*\[\s*([^\]]+)\s*\]\s*;", txt)
    dim_m !== nothing || throw(ArgumentError("OpenFOAM dictionary is missing dimensions [...]"))
    dim_tokens = split(strip(dim_m.captures[1]))
    length(dim_tokens) == 7 ||
        throw(ArgumentError("OpenFOAM dimensions must contain exactly 7 integers"))
    dims = NTuple{7,Int}(Tuple(parse(Int, t) for t in dim_tokens))

    n_cells = length(cell_volumes_m3)
    internal_mat = Matrix{Float64}(undef, n_cells, field_class == :volScalarField ? 1 : 3)

    if field_class == :volScalarField
        unif_m = match(r"internalField\s+uniform\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*;", txt)
        if unif_m !== nothing
            val = parse(Float64, unif_m.captures[1])
            fill!(internal_mat, val)
        else
            nonunif_m = match(r"internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\(\s*([^)]*)\)\s*;", txt)
            nonunif_m !== nothing ||
                throw(ArgumentError("unable to parse OpenFOAM scalar internalField"))
            n_declared = parse(Int, nonunif_m.captures[1])
            n_declared == n_cells ||
                throw(ArgumentError("OpenFOAM internalField count ($n_declared) does not match mesh cell count ($n_cells)"))
            val_tokens = split(strip(nonunif_m.captures[2]))
            length(val_tokens) == n_cells ||
                throw(ArgumentError("OpenFOAM scalar list length ($(length(val_tokens))) does not match $n_cells"))
            for i in 1:n_cells
                internal_mat[i, 1] = parse(Float64, val_tokens[i])
            end
        end
    else
        unif_v = match(r"internalField\s+uniform\s*\(\s*([^\)]+)\s*\)\s*;", txt)
        if unif_v !== nothing
            vparts = split(strip(unif_v.captures[1]))
            length(vparts) == 3 || throw(ArgumentError("uniform vector must have 3 components"))
            v1, v2, v3 = parse(Float64, vparts[1]), parse(Float64, vparts[2]), parse(Float64, vparts[3])
            for i in 1:n_cells
                internal_mat[i, 1] = v1
                internal_mat[i, 2] = v2
                internal_mat[i, 3] = v3
            end
        else
            nonunif_v = match(r"internalField\s+nonuniform\s+List<vector>\s+(\d+)\s*\(\s*(.*?)\s*\)\s*;"s, txt)
            nonunif_v !== nothing ||
                throw(ArgumentError("unable to parse OpenFOAM vector internalField"))
            n_declared = parse(Int, nonunif_v.captures[1])
            n_declared == n_cells ||
                throw(ArgumentError("OpenFOAM vector internalField count ($n_declared) does not match mesh cell count ($n_cells)"))
            vec_pat = r"\(\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*\)"
            vec_matches = collect(eachmatch(vec_pat, nonunif_v.captures[2]))
            length(vec_matches) == n_cells ||
                throw(ArgumentError("parsed $(length(vec_matches)) vectors, expected $n_cells"))
            for i in 1:n_cells
                internal_mat[i, 1] = parse(Float64, vec_matches[i].captures[1])
                internal_mat[i, 2] = parse(Float64, vec_matches[i].captures[2])
                internal_mat[i, 3] = parse(Float64, vec_matches[i].captures[3])
            end
        end
    end

    patches = OpenFOAMBoundaryPatch[]
    bf_m = match(r"boundaryField\s*\{(.*)\}\s*$"s, strip(txt))
    if bf_m !== nothing
        bf_body = bf_m.captures[1]
        for pm in eachmatch(r"([A-Za-z0-9_.-]+)\s*\{([^{}]*)\}", bf_body)
            pname = pm.captures[1]
            pbody = pm.captures[2]
            tm = match(r"type\s+([A-Za-z0-9_]+)\s*;", pbody)
            ptype = tm === nothing ? :zeroGradient : Symbol(tm.captures[1])
            fm = match(r"flux_mol_s\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*;", pbody)
            flx = fm === nothing ? 0.0 : parse(Float64, fm.captures[1])

            vm_list = match(r"value\s+nonuniform\s+List<scalar>\s+\d+\s*\(\s*([^)]*)\)\s*;", pbody)
            vm_unif_s = match(r"value\s+uniform\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*;", pbody)
            vm_unif_v = match(r"value\s+uniform\s*\(\s*([^\)]+)\s*\)\s*;", pbody)
            vals = if vm_list !== nothing
                [parse(Float64, s) for s in split(strip(vm_list.captures[1])) if !isempty(s)]
            elseif vm_unif_s !== nothing
                [parse(Float64, vm_unif_s.captures[1])]
            elseif vm_unif_v !== nothing
                [parse(Float64, s) for s in split(strip(vm_unif_v.captures[1])) if !isempty(s)]
            else
                Float64[]
            end
            push!(patches, OpenFOAMBoundaryPatch(pname, ptype, vals; flux_mol_s=flx))
        end
    end

    digest = bytes2hex(SHA.sha256(raw_txt))
    return OpenFOAMMeshExchange(
        vessel_id = vessel_id,
        time_h = time_h,
        timestep_s = timestep_s,
        compartment = compartment,
        mesh_id = mesh_id,
        solver_version = solver_version,
        field_name = field_name,
        field_class = field_class,
        dimensions = dims,
        cell_centers_m = cell_centers_m,
        cell_volumes_m3 = cell_volumes_m3,
        internal_field = internal_mat,
        boundary_patches = patches,
        velocity_magnitudes_m_s = velocity_magnitudes_m_s,
        diffusivities_m2_s = diffusivities_m2_s,
        epistemic_status = epistemic_status,
        source_sha256 = digest
    )
end

"""
Conservative volume-weighted field mapping from an OpenFOAM mesh (`source`) to
Julia biological compartments or coarse mesh cells with volumes `target_volumes_m3`.

`overlap_weights[j, i]` is the fraction of source cell `i`'s volume mapped into
target compartment `j`. Each column `i` must sum to 1 (partition of unity) and
`overlap_weights * source.cell_volumes_m3` must equal `target_volumes_m3`.
"""
function map_openfoam_field_conservative(
        source::OpenFOAMMeshExchange,
        target_volumes_m3::AbstractVector{<:Real},
        overlap_weights::AbstractMatrix{<:Real};
        atol::Real=1e-8)
    n_src = length(source.cell_volumes_m3)
    n_tgt = length(target_volumes_m3)
    n_tgt >= 1 || throw(ArgumentError("target_volumes_m3 must be non-empty"))
    size(overlap_weights) == (n_tgt, n_src) ||
        throw(ArgumentError("overlap_weights must have size (n_target, n_source) = ($n_tgt, $n_src)"))
    tol = Float64(atol)
    tol >= 0 || throw(ArgumentError("atol must be >= 0"))

    tgt_vols = Float64.(target_volumes_m3)
    all(v -> isfinite(v) && v > 0, tgt_vols) ||
        throw(ArgumentError("target_volumes_m3 entries must be finite and > 0"))
    W = Float64.(overlap_weights)
    all(w -> isfinite(w) && w >= -tol, W) ||
        throw(ArgumentError("overlap_weights must be finite and nonnegative"))

    for i in 1:n_src
        col_sum = sum(@view W[:, i])
        isapprox(col_sum, 1.0; atol=max(tol, 1e-7), rtol=1e-7) ||
            throw(ArgumentError("overlap_weights column $i sums to $col_sum != 1.0 (violates mass conservation)"))
    end

    implied_tgt_vols = W * source.cell_volumes_m3
    for j in 1:n_tgt
        isapprox(implied_tgt_vols[j], tgt_vols[j]; atol=1e-24, rtol=max(tol, 1e-6)) ||
            throw(ArgumentError("target volume $j ($(tgt_vols[j])) does not match overlapped source volume ($(implied_tgt_vols[j]))"))
    end

    n_comp = size(source.internal_field, 2)
    mapped = zeros(Float64, n_tgt, n_comp)
    src_totals = zeros(Float64, n_comp)
    tgt_totals = zeros(Float64, n_comp)

    for c in 1:n_comp
        for i in 1:n_src
            src_totals[c] += source.internal_field[i, c] * source.cell_volumes_m3[i]
        end
        for j in 1:n_tgt
            amt_j = 0.0
            for i in 1:n_src
                amt_j += W[j, i] * source.cell_volumes_m3[i] * source.internal_field[i, c]
            end
            mapped[j, c] = amt_j / tgt_vols[j]
            tgt_totals[c] += mapped[j, c] * tgt_vols[j]
        end
    end

    max_rel_err = maximum(
        abs(src_totals[c] - tgt_totals[c]) / max(abs(src_totals[c]), 1e-15)
        for c in 1:n_comp
    )
    conserved = max_rel_err <= max(tol, 1e-9)
    return (
        mapped_field = mapped,
        source_totals = src_totals,
        target_totals = tgt_totals,
        mass_conserved = conserved,
        max_relative_mass_error = max_rel_err
    )
end

"""
Audit OpenFOAM transport numerical stability, Courant-Friedrichs-Lewy (CFL)
number, grid Peclet number, nonnegativity, and integrated mass/flux balance.
For `:volScalarField`, nonnegativity is enforced by default (`require_nonnegative=true`);
for `:volVectorField`, signed vector components are permitted by default unless
`require_nonnegative=true` is explicitly requested.
"""
function audit_openfoam_transport_stability(
        ex::OpenFOAMMeshExchange;
        max_courant::Real=1.0,
        max_peclet::Real=50.0,
        require_nonnegative::Union{Nothing,Bool}=nothing)
    co_lim = Float64(max_courant)
    pe_lim = Float64(max_peclet)
    co_lim > 0 && pe_lim > 0 ||
        throw(ArgumentError("max_courant and max_peclet must be > 0"))

    n = length(ex.cell_volumes_m3)
    max_co = 0.0
    max_pe = 0.0
    for i in 1:n
        dx = cbrt(ex.cell_volumes_m3[i])
        u = ex.velocity_magnitudes_m_s[i]
        if u == 0.0 && ex.field_class == :volVectorField
            u = hypot(ex.internal_field[i, 1], ex.internal_field[i, 2], ex.internal_field[i, 3])
        end
        d = ex.diffusivities_m2_s[i]
        co_i = (u * ex.timestep_s) / dx
        pe_i = u > 0 ? (u * dx) / max(d, 1e-18) : 0.0
        if co_i > max_co
            max_co = co_i
        end
        if pe_i > max_pe
            max_pe = pe_i
        end
    end

    min_val = minimum(ex.internal_field)
    max_val = maximum(ex.internal_field)
    total_amt = if ex.field_class == :volScalarField
        sum(ex.internal_field[i, 1] * ex.cell_volumes_m3[i] for i in 1:n)
    else
        sum(hypot(ex.internal_field[i, 1], ex.internal_field[i, 2], ex.internal_field[i, 3]) * ex.cell_volumes_m3[i] for i in 1:n)
    end
    net_flux = isempty(ex.boundary_patches) ? 0.0 : sum(p.flux_mol_s for p in ex.boundary_patches)

    enforce_nonneg = require_nonnegative === nothing ? (ex.field_class == :volScalarField) : require_nonnegative
    nonneg_ok = (!enforce_nonneg) || (min_val >= -1e-12)
    stable = (max_co <= co_lim) && (max_pe <= pe_lim) && nonneg_ok
    return (
        stable = stable,
        max_courant_observed = max_co,
        max_peclet_observed = max_pe,
        min_field_value = min_val,
        max_field_value = max_val,
        total_integrated_amount = total_amt,
        net_boundary_flux_mol_s = net_flux
    )
end

"""
Summarize a spatial `OpenFOAMMeshExchange` snapshot into a volume-weighted
scalar `OpenFOAMTransportSnap` for compartment-level observation alignment.
"""
function summarize_openfoam_mesh_to_snap(
        ex::OpenFOAMMeshExchange;
        wall_shear_pa::Real=0.25,
        residence_time_s::Union{Nothing,Real}=nothing,
        detachment_flux_cells_m2_s::Real=0.0)
    tot_vol = sum(ex.cell_volumes_m3)
    mean_vel = sum(ex.velocity_magnitudes_m_s[i] * ex.cell_volumes_m3[i] for i in eachindex(ex.cell_volumes_m3)) / tot_vol
    char_len = cbrt(tot_vol)
    rt = residence_time_s === nothing ? (char_len / max(mean_vel, 1e-6)) : Float64(residence_time_s)
    return OpenFOAMTransportSnap(
        ex.vessel_id, ex.time_h, ex.compartment;
        velocity_m_s = mean_vel,
        wall_shear_pa = wall_shear_pa,
        residence_time_s = rt,
        detachment_flux_cells_m2_s = detachment_flux_cells_m2_s
    )
end

# ---------------------------------------------------------------------------
# M2: Full-species PHREEQC geochemical snapshot adapter and balance audits
# ---------------------------------------------------------------------------

struct PHREEQCMineralPhase
    name::String
    formula::String
    saturation_index::Float64
    moles_mol_L::Float64
    delta_moles_mol_L::Float64

    function PHREEQCMineralPhase(name::AbstractString,
                                 formula::AbstractString;
                                 saturation_index::Real,
                                 moles_mol_L::Real=0.0,
                                 delta_moles_mol_L::Real=0.0)
        nm = String(strip(name))
        fm = String(strip(formula))
        !isempty(nm) && !isempty(fm) ||
            throw(ArgumentError("mineral phase name and formula must be non-empty"))
        si = Float64(saturation_index)
        m = Float64(moles_mol_L)
        dm = Float64(delta_moles_mol_L)
        isfinite(si) && isfinite(m) && m >= 0 && isfinite(dm) ||
            throw(ArgumentError("invalid mineral phase numerical values"))
        new(nm, fm, si, m, dm)
    end
end

"""
Full-species PHREEQC aqueous speciation and mineral equilibrium snapshot
(`PHREEQCSpeciesAdapter`). PHREEQC owns aqueous equilibrium, activity
coefficients, charge balance, and saturation indices; it does not own
quorum-sensing regulation or biofilm mechanics.
"""
struct PHREEQCSpeciesAdapter
    vessel_id::String
    time_h::Float64
    compartment::Symbol
    solver_version::String
    database_name::String
    temperature_c::Float64
    pH::Float64
    pe::Float64
    eh_volts::Float64
    ionic_strength_M::Float64
    species_molalities_mol_kgw::Dict{String,Float64}
    species_charges::Dict{String,Int}
    species_activity_coeffs::Dict{String,Float64}
    element_totals_mol_kgw::Dict{String,Float64}
    species_stoichiometry::Dict{String,Dict{String,Float64}}
    mineral_phases::Vector{PHREEQCMineralPhase}
    epistemic_status::Symbol
    source_sha256::String

    function PHREEQCSpeciesAdapter(;
            vessel_id::AbstractString,
            time_h::Real,
            compartment::Symbol=:effluent,
            solver_version::AbstractString="PHREEQC-3.7.3",
            database_name::AbstractString="phreeqc.dat",
            temperature_c::Real=25.0,
            pH::Real,
            pe::Real,
            eh_volts::Union{Nothing,Real}=nothing,
            ionic_strength_M::Real,
            species_molalities_mol_kgw::Dict{String,<:Real},
            species_charges::Dict{String,<:Integer},
            species_activity_coeffs::Dict{String,<:Real}=Dict{String,Float64}(),
            element_totals_mol_kgw::Dict{String,<:Real}=Dict{String,Float64}(),
            species_stoichiometry::Dict{String,<:Dict{String,<:Real}}=Dict{String,Dict{String,Float64}}(),
            mineral_phases::AbstractVector{PHREEQCMineralPhase}=PHREEQCMineralPhase[],
            epistemic_status::Symbol=:predicted,
            source_sha256::AbstractString="")
        vid = String(strip(vessel_id))
        sver = String(strip(solver_version))
        db = String(strip(database_name))
        !isempty(vid) && !isempty(sver) && !isempty(db) ||
            throw(ArgumentError("vessel_id, solver_version, and database_name must be non-empty"))
        compartment in VALID_SAMPLE_ORIGINS ||
            throw(ArgumentError("compartment must be one of $(VALID_SAMPLE_ORIGINS)"))
        epistemic_status in VALID_EPISTEMIC_STATUSES ||
            throw(ArgumentError("epistemic_status must be one of $(VALID_EPISTEMIC_STATUSES)"))

        th = Float64(time_h)
        tc = Float64(temperature_c)
        ph_v = Float64(pH)
        pe_v = Float64(pe)
        is_v = Float64(ionic_strength_M)
        isfinite(th) && th >= 0 || throw(ArgumentError("time_h must be finite and >= 0"))
        isfinite(tc) && tc > -273.15 || throw(ArgumentError("temperature_c must be > -273.15"))
        isfinite(ph_v) && -2.0 <= ph_v <= 16.0 || throw(ArgumentError("pH must be in [-2.0, 16.0]"))
        isfinite(pe_v) || throw(ArgumentError("pe must be finite"))
        isfinite(is_v) && is_v >= 0 || throw(ArgumentError("ionic_strength_M must be finite and >= 0"))

        nernst_factor = (8.314462618 * (tc + 273.15) * log(10.0)) / 96485.33212
        eh_v = eh_volts === nothing ? (nernst_factor * pe_v) : Float64(eh_volts)
        isfinite(eh_v) || throw(ArgumentError("eh_volts must be finite"))

        !isempty(species_molalities_mol_kgw) ||
            throw(ArgumentError("species_molalities_mol_kgw must be non-empty"))
        mol_dict = Dict{String,Float64}()
        chg_dict = Dict{String,Int}()
        act_dict = Dict{String,Float64}()
        for (sp, m_raw) in species_molalities_mol_kgw
            m_val = Float64(m_raw)
            isfinite(m_val) && m_val >= 0 ||
                throw(ArgumentError("species '$sp' molality must be finite and >= 0"))
            haskey(species_charges, sp) ||
                throw(ArgumentError("missing ionic charge for species '$sp' in species_charges"))
            mol_dict[sp] = m_val
            chg_dict[sp] = Int(species_charges[sp])
            gam = Float64(get(species_activity_coeffs, sp, 1.0))
            isfinite(gam) && gam > 0 ||
                throw(ArgumentError("activity coefficient for '$sp' must be finite and > 0"))
            act_dict[sp] = gam
        end

        elem_dict = Dict{String,Float64}()
        for (el, tot_raw) in element_totals_mol_kgw
            t_val = Float64(tot_raw)
            isfinite(t_val) && t_val >= 0 ||
                throw(ArgumentError("element '$el' total molality must be finite and >= 0"))
            elem_dict[el] = t_val
        end

        stoich_dict = Dict{String,Dict{String,Float64}}()
        for (sp, el_map) in species_stoichiometry
            inner = Dict{String,Float64}()
            for (el, coeff) in el_map
                c_val = Float64(coeff)
                isfinite(c_val) && c_val >= 0 ||
                    throw(ArgumentError("stoichiometric coefficient for '$el' in '$sp' must be >= 0"))
                inner[el] = c_val
            end
            stoich_dict[sp] = inner
        end

        minerals = collect(PHREEQCMineralPhase, mineral_phases)
        digest = if isempty(strip(source_sha256))
            bytes2hex(SHA.sha256(string(vid, ":", th, ":", ph_v, ":", pe_v, ":", is_v, ":", length(mol_dict))))
        else
            d = lowercase(strip(String(source_sha256)))
            occursin(r"^[0-9a-f]{64}$", d) ||
                throw(ArgumentError("source_sha256 must be a valid 64-character lowercase hex SHA-256 digest"))
            d
        end

        new(vid, th, compartment, sver, db, tc, ph_v, pe_v, eh_v, is_v,
            mol_dict, chg_dict, act_dict, elem_dict, stoich_dict, minerals,
            epistemic_status, digest)
    end
end

"""
Audit charge balance, element conservation, ionic strength, Nernst Eh-pe
consistency, pH and water autoionization (`Kw`) activity consistency, and
mineral saturation states (`SI = log10(IAP / Ksp)` and thermodynamic
precipitation rules) for a `PHREEQCSpeciesAdapter`.
"""
function audit_phreeqc_speciation_balance(
        adapter::PHREEQCSpeciesAdapter;
        max_charge_imbalance_pct::Real=2.0,
        rtol_element::Real=1e-4,
        rtol_ionic_strength::Real=0.05,
        atol_eh_volts::Real=0.005,
        atol_si::Real=0.08,
        mineral_reactions::Dict{String,<:NamedTuple}=Dict{String,NamedTuple}())
    max_ci = Float64(max_charge_imbalance_pct)
    rtol_el = Float64(rtol_element)
    rtol_is = Float64(rtol_ionic_strength)
    atol_eh = Float64(atol_eh_volts)
    tol_si = Float64(atol_si)

    pos_eq = 0.0
    neg_eq = 0.0
    calc_is = 0.0
    activities = Dict{String,Float64}()

    for (sp, m) in adapter.species_molalities_mol_kgw
        z = adapter.species_charges[sp]
        gam = adapter.species_activity_coeffs[sp]
        activities[sp] = gam * m
        if z > 0
            pos_eq += z * m
        elseif z < 0
            neg_eq += abs(z) * m
        end
        calc_is += 0.5 * (z * z) * m
    end

    net_charge = pos_eq - neg_eq
    denom_charge = pos_eq + neg_eq
    charge_pct = denom_charge > 0 ? (100.0 * abs(net_charge) / denom_charge) : 0.0
    charge_ok = charge_pct <= max_ci

    is_ok = isapprox(calc_is, adapter.ionic_strength_M; rtol=rtol_is, atol=1e-5)

    max_el_rel_err = 0.0
    elem_ok = true
    if !isempty(adapter.element_totals_mol_kgw)
        if isempty(adapter.species_stoichiometry)
            elem_ok = any(v -> v > 0.0, values(adapter.element_totals_mol_kgw)) ? false : true
            max_el_rel_err = elem_ok ? 0.0 : 1.0
        else
            for (el, tot_expected) in adapter.element_totals_mol_kgw
                tot_calc = 0.0
                for (sp, m) in adapter.species_molalities_mol_kgw
                    if haskey(adapter.species_stoichiometry, sp) && haskey(adapter.species_stoichiometry[sp], el)
                        tot_calc += adapter.species_stoichiometry[sp][el] * m
                    end
                end
                rel_err = abs(tot_calc - tot_expected) / max(tot_expected, 1e-15)
                if rel_err > max_el_rel_err
                    max_el_rel_err = rel_err
                end
                if rel_err > rtol_el
                    elem_ok = false
                end
            end
        end
    end

    T_K = adapter.temperature_c + 273.15
    nernst_factor = (8.314462618 * T_K * log(10.0)) / 96485.33212
    expected_eh = nernst_factor * adapter.pe
    eh_ok = abs(adapter.eh_volts - expected_eh) <= atol_eh

    ph_ok = true
    if haskey(activities, "H+") && activities["H+"] > 0
        ph_from_act = -log10(activities["H+"])
        ph_ok = abs(ph_from_act - adapter.pH) <= 0.08
    end
    if ph_ok && haskey(activities, "H+") && haskey(activities, "OH-") &&
       activities["H+"] > 0 && activities["OH-"] > 0
        pkw_T = 4471.33 / T_K - 6.0846 + 0.017053 * T_K
        pk_obs = -log10(activities["H+"]) - log10(activities["OH-"])
        if abs(pk_obs - pkw_T) > 0.35
            ph_ok = false
        end
    end

    mineral_si_ok = true
    for mp in adapter.mineral_phases
        # An undersaturated mineral (SI < -tol_si) cannot precipitate (delta_moles > 0) at equilibrium
        if mp.saturation_index < -tol_si && mp.delta_moles_mol_L > 1e-12
            mineral_si_ok = false
        end
        if haskey(mineral_reactions, mp.name)
            rxn = mineral_reactions[mp.name]
            log_iap = 0.0
            valid_iap = true
            for (sp_name, nu_coeff) in rxn.stoichiometry
                if haskey(activities, sp_name) && activities[sp_name] > 0
                    log_iap += Float64(nu_coeff) * log10(activities[sp_name])
                else
                    valid_iap = false
                end
            end
            if valid_iap
                calc_si = log_iap - Float64(rxn.log10_ksp)
                if abs(calc_si - mp.saturation_index) > tol_si
                    mineral_si_ok = false
                end
            else
                mineral_si_ok = false
            end
        end
    end

    supersat = [m.name for m in adapter.mineral_phases if m.saturation_index > 0.0]
    undersat = [m.name for m in adapter.mineral_phases if m.saturation_index < 0.0]

    return (
        consistent = charge_ok && is_ok && elem_ok && eh_ok && ph_ok && mineral_si_ok,
        charge_balance_eq_kgw = net_charge,
        charge_imbalance_pct = charge_pct,
        charge_balance_ok = charge_ok,
        computed_ionic_strength_M = calc_is,
        ionic_strength_consistent = is_ok,
        high_ionic_strength_pitzer_recommended = adapter.ionic_strength_M > 1.0,
        element_balance_consistent = elem_ok,
        max_element_rel_error = max_el_rel_err,
        nernst_eh_consistent = eh_ok,
        ph_activity_consistent = ph_ok,
        mineral_si_consistent = mineral_si_ok,
        activities_mol_kgw = activities,
        supersaturated_minerals = supersat,
        undersaturated_minerals = undersat
    )
end

"""
Serialize a `PHREEQCSpeciesAdapter` into a machine-readable PHREEQC
selected-output key-value table.
"""
function serialize_phreeqc_selected_output(adapter::PHREEQCSpeciesAdapter)
    sp_names = sort!(collect(keys(adapter.species_molalities_mol_kgw)))
    el_names = sort!(collect(keys(adapter.element_totals_mol_kgw)))

    headers = String[
        "vessel_id", "time_h", "compartment", "solver_version", "database_name",
        "temperature_c", "pH", "pe", "eh_volts", "ionic_strength_M"
    ]
    vals = String[
        adapter.vessel_id,
        @sprintf("%.10g", adapter.time_h),
        String(adapter.compartment),
        adapter.solver_version,
        adapter.database_name,
        @sprintf("%.10g", adapter.temperature_c),
        @sprintf("%.10g", adapter.pH),
        @sprintf("%.10g", adapter.pe),
        @sprintf("%.10g", adapter.eh_volts),
        @sprintf("%.12g", adapter.ionic_strength_M)
    ]

    for sp in sp_names
        push!(headers, "m_$(sp)")
        push!(vals, @sprintf("%.12g", adapter.species_molalities_mol_kgw[sp]))
    end
    for el in el_names
        push!(headers, "tot_$(el)")
        push!(vals, @sprintf("%.12g", adapter.element_totals_mol_kgw[el]))
    end
    for mp in adapter.mineral_phases
        push!(headers, "si_$(mp.name)")
        push!(vals, @sprintf("%.10g", mp.saturation_index))
        push!(headers, "min_moles_$(mp.name)")
        push!(vals, @sprintf("%.10g", mp.moles_mol_L))
        push!(headers, "min_delta_$(mp.name)")
        push!(vals, @sprintf("%.10g", mp.delta_moles_mol_L))
        push!(headers, "min_formula_$(mp.name)")
        push!(vals, mp.formula)
    end

    return join(headers, "\t") * "\n" * join(vals, "\t") * "\n"
end

"""
Parse a PHREEQC selected-output TSV/CSV table into a `PHREEQCSpeciesAdapter`.
"""
function parse_phreeqc_selected_output(
        table_text::AbstractString;
        species_charges::Dict{String,<:Integer},
        species_activity_coeffs::Dict{String,<:Real}=Dict{String,Float64}(),
        species_stoichiometry::Dict{String,<:Dict{String,<:Real}}=Dict{String,Dict{String,Float64}}(),
        epistemic_status::Symbol=:predicted)
    lines = [strip(l) for l in split(chomp(String(table_text)), '\n') if !isempty(strip(l))]
    length(lines) >= 2 || throw(ArgumentError("PHREEQC selected-output table must have header and data row"))
    delim = occursin('\t', lines[1]) ? '\t' : ','
    hdrs = [strip(h) for h in split(lines[1], delim)]
    row  = [strip(v) for v in split(lines[2], delim)]
    length(hdrs) == length(row) ||
        throw(ArgumentError("PHREEQC selected-output header column count ($(length(hdrs))) != row count ($(length(row)))"))

    kv = Dict{String,String}(hdrs[i] => row[i] for i in eachindex(hdrs))
    vid = get(kv, "vessel_id", "UNSPECIFIED")
    th  = parse(Float64, get(kv, "time_h", "0.0"))
    comp = Symbol(get(kv, "compartment", "effluent"))
    sver = get(kv, "solver_version", "PHREEQC-3.7.3")
    db   = get(kv, "database_name", "phreeqc.dat")
    tc   = parse(Float64, get(kv, "temperature_c", "25.0"))
    ph_v = parse(Float64, kv["pH"])
    pe_v = parse(Float64, kv["pe"])
    eh_v = haskey(kv, "eh_volts") ? parse(Float64, kv["eh_volts"]) : nothing
    is_v = parse(Float64, kv["ionic_strength_M"])

    mol_dict = Dict{String,Float64}()
    elem_dict = Dict{String,Float64}()
    min_names = String[]

    for h in hdrs
        if startswith(h, "m_")
            sp = h[3:end]
            mol_dict[sp] = parse(Float64, kv[h])
        elseif startswith(h, "tot_")
            el = h[5:end]
            elem_dict[el] = parse(Float64, kv[h])
        elseif startswith(h, "si_")
            push!(min_names, h[4:end])
        end
    end

    minerals = PHREEQCMineralPhase[]
    for mname in min_names
        si = parse(Float64, kv["si_$(mname)"])
        moles = haskey(kv, "min_moles_$(mname)") ? parse(Float64, kv["min_moles_$(mname)"]) : 0.0
        dmoles = haskey(kv, "min_delta_$(mname)") ? parse(Float64, kv["min_delta_$(mname)"]) : 0.0
        form = get(kv, "min_formula_$(mname)", mname)
        push!(minerals, PHREEQCMineralPhase(mname, form;
            saturation_index=si, moles_mol_L=moles, delta_moles_mol_L=dmoles))
    end

    digest = bytes2hex(SHA.sha256(String(table_text)))
    return PHREEQCSpeciesAdapter(
        vessel_id = vid,
        time_h = th,
        compartment = comp,
        solver_version = sver,
        database_name = db,
        temperature_c = tc,
        pH = ph_v,
        pe = pe_v,
        eh_volts = eh_v,
        ionic_strength_M = is_v,
        species_molalities_mol_kgw = mol_dict,
        species_charges = Dict{String,Int}(k => Int(v) for (k, v) in species_charges),
        species_activity_coeffs = Dict{String,Float64}(k => Float64(v) for (k, v) in species_activity_coeffs),
        element_totals_mol_kgw = elem_dict,
        species_stoichiometry = Dict{String,Dict{String,Float64}}(
            sp => Dict{String,Float64}(el => Float64(c) for (el, c) in emap)
            for (sp, emap) in species_stoichiometry
        ),
        mineral_phases = minerals,
        epistemic_status = epistemic_status,
        source_sha256 = digest
    )
end

"""
Summarize a full-species `PHREEQCSpeciesAdapter` into a scalar `PHREEQCSpeciationSnap`
for backward-compatible observation model coupling.
"""
function summarize_phreeqc_adapter_to_snap(adapter::PHREEQCSpeciesAdapter)
    m = adapter.species_molalities_mol_kgw
    t = adapter.element_totals_mol_kgw
    ca = get(t, "Ca", get(m, "Ca+2", 0.0))
    mg = get(t, "Mg", get(m, "Mg+2", 0.0))
    fe2 = get(m, "Fe+2", get(t, "Fe", 0.0))
    sulf = get(m, "HS-", 0.0) + get(m, "H2S", 0.0) + get(m, "S-2", 0.0)
    return PHREEQCSpeciationSnap(
        adapter.vessel_id, adapter.time_h;
        pH = adapter.pH,
        pe = adapter.pe,
        ionic_strength_M = adapter.ionic_strength_M,
        ca_mol_L = ca,
        mg_mol_L = mg,
        fe2_mol_L = fe2,
        sulfide_mol_L = sulf
    )
end

"""
Verify that OpenFOAM transport and PHREEQC speciation snapshots (scalar or
mesh/species-level) match the cytometry sample's vessel, time point, and
compartment before coupling.
"""
function validate_solver_observation_alignment(sample::FCMSampleSummary,
                                               sample_vessel_id::AbstractString,
                                               sample_time_h::Real;
                                               transport::Union{Nothing,OpenFOAMTransportSnap}=nothing,
                                               chemistry::Union{Nothing,PHREEQCSpeciationSnap}=nothing,
                                               mesh_transport::Union{Nothing,OpenFOAMMeshExchange}=nothing,
                                               species_chemistry::Union{Nothing,PHREEQCSpeciesAdapter}=nothing,
                                               atol_time_h::Real=1e-6)
    vid = String(strip(sample_vessel_id))
    !isempty(vid) || throw(ArgumentError("sample_vessel_id must be non-empty"))
    th = Float64(sample_time_h)
    isfinite(th) && th >= 0 || throw(ArgumentError("sample_time_h must be finite and >= 0"))
    atol_t = Float64(atol_time_h)
    isfinite(atol_t) && atol_t >= 0 || throw(ArgumentError("atol_time_h must be finite and >= 0"))
    if transport !== nothing
        transport.vessel_id == vid ||
            throw(ArgumentError("OpenFOAM vessel_id ($(transport.vessel_id)) does not match sample vessel_id ($vid)"))
        isapprox(transport.time_h, th; atol=atol_t, rtol=0.0) ||
            throw(ArgumentError("OpenFOAM time_h ($(transport.time_h)) does not match sample time_h ($th)"))
        transport.compartment == sample.sample_origin ||
            throw(ArgumentError("OpenFOAM compartment ($(transport.compartment)) does not match sample_origin ($(sample.sample_origin))"))
    end
    if chemistry !== nothing
        chemistry.vessel_id == vid ||
            throw(ArgumentError("PHREEQC vessel_id ($(chemistry.vessel_id)) does not match sample vessel_id ($vid)"))
        isapprox(chemistry.time_h, th; atol=atol_t, rtol=0.0) ||
            throw(ArgumentError("PHREEQC time_h ($(chemistry.time_h)) does not match sample time_h ($th)"))
    end
    if mesh_transport !== nothing
        mesh_transport.vessel_id == vid ||
            throw(ArgumentError("OpenFOAMMeshExchange vessel_id ($(mesh_transport.vessel_id)) does not match sample vessel_id ($vid)"))
        isapprox(mesh_transport.time_h, th; atol=atol_t, rtol=0.0) ||
            throw(ArgumentError("OpenFOAMMeshExchange time_h ($(mesh_transport.time_h)) does not match sample time_h ($th)"))
        mesh_transport.compartment == sample.sample_origin ||
            throw(ArgumentError("OpenFOAMMeshExchange compartment ($(mesh_transport.compartment)) does not match sample_origin ($(sample.sample_origin))"))
    end
    if species_chemistry !== nothing
        species_chemistry.vessel_id == vid ||
            throw(ArgumentError("PHREEQCSpeciesAdapter vessel_id ($(species_chemistry.vessel_id)) does not match sample vessel_id ($vid)"))
        isapprox(species_chemistry.time_h, th; atol=atol_t, rtol=0.0) ||
            throw(ArgumentError("PHREEQCSpeciesAdapter time_h ($(species_chemistry.time_h)) does not match sample time_h ($th)"))
        species_chemistry.compartment == sample.sample_origin ||
            throw(ArgumentError("PHREEQCSpeciesAdapter compartment ($(species_chemistry.compartment)) does not match sample_origin ($(sample.sample_origin))"))
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
# M5: Empirical validation protocol separating synthetic software checks
#     from real experimental generalization and internalization claims
# ---------------------------------------------------------------------------

"""
Explicit empirical validation contract (`EmpiricalValidationProtocol`) that
separates internal software verification on synthetic data from empirical
generalization on de-identified, expert-reviewed experimental runs.
"""
struct EmpiricalValidationProtocol
    protocol_id::String
    track::Symbol
    data_provenance::Symbol
    independent_reference_labels::Bool
    clay_matrix_blank_verified::Bool
    counting_beads_verified::Bool
    extraction_recovery_verified::Bool
    internalization_imaging_verified::Bool
    n_independent_runs::Int
    n_heldout_runs::Int
    external_solvers_coupled_in_production::Bool

    function EmpiricalValidationProtocol(;
            protocol_id::AbstractString="EMP-VAL-001",
            track::Symbol=:track1_nanobio,
            data_provenance::Symbol=:synthetic_simulation,
            independent_reference_labels::Bool=false,
            clay_matrix_blank_verified::Bool=false,
            counting_beads_verified::Bool=true,
            extraction_recovery_verified::Bool=false,
            internalization_imaging_verified::Bool=false,
            n_independent_runs::Integer=0,
            n_heldout_runs::Integer=0,
            external_solvers_coupled_in_production::Bool=false)
        pid = String(strip(protocol_id))
        !isempty(pid) || throw(ArgumentError("protocol_id must be non-empty"))
        track in VALID_STAKEHOLDER_TRACKS ||
            throw(ArgumentError("track must be one of $(VALID_STAKEHOLDER_TRACKS)"))
        data_provenance in VALID_DATA_PROVENANCES ||
            throw(ArgumentError("data_provenance must be one of $(VALID_DATA_PROVENANCES)"))
        nir = Int(n_independent_runs)
        nhr = Int(n_heldout_runs)
        nir >= 0 && nhr >= 0 && nhr <= nir ||
            throw(ArgumentError("require 0 <= n_heldout_runs <= n_independent_runs"))
        if data_provenance == :synthetic_simulation && independent_reference_labels
            throw(ArgumentError("synthetic_simulation provenance cannot claim independent_reference_labels = true"))
        end
        new(pid, track, data_provenance, independent_reference_labels,
            clay_matrix_blank_verified, counting_beads_verified,
            extraction_recovery_verified, internalization_imaging_verified,
            nir, nhr, external_solvers_coupled_in_production)
    end
end

"""
Evaluate whether a model/classifier result establishes only software-level
consistency (`software_verification_passed`) or genuine empirical generalization
(`empirical_generalization_established`).

Enforces:
1. High synthetic cross-validation accuracy never implies empirical generalization.
2. For `:nanocarrier_uptake`, a fluorescence-positive cytometry gate alone
   cannot prove intracellular internalization without independent imaging/quenching
   (`surface_association_only_warning = true`, `internalization_proven = false`).
3. For `:track2_clay_corrosion`, clay-only matrix blanks, counting beads, and
   extraction-recovery controls are mandatory before empirical readiness.
4. Feature domain shift (`feature_psi_drift > max_allowed_psi`) blocks empirical
   generalization claims.
"""
function evaluate_empirical_generalization_readiness(
        protocol::EmpiricalValidationProtocol;
        target_task::Symbol=:cell_stress_damage,
        synthetic_accuracy::Real=0.95,
        heldout_run_accuracy::Union{Nothing,Real}=nothing,
        min_empirical_accuracy::Real=0.80,
        feature_psi_drift::Real=0.0,
        max_allowed_psi::Real=0.25)
    sa = Float64(synthetic_accuracy)
    0.0 <= sa <= 1.0 || throw(ArgumentError("synthetic_accuracy must be in [0, 1]"))
    min_ea = Float64(min_empirical_accuracy)
    0.0 < min_ea <= 1.0 || throw(ArgumentError("min_empirical_accuracy must be in (0, 1]"))
    psi = Float64(feature_psi_drift)
    max_psi = Float64(max_allowed_psi)
    psi >= 0.0 && max_psi > 0.0 ||
        throw(ArgumentError("feature_psi_drift must be >= 0 and max_allowed_psi > 0"))

    software_ok = sa >= 0.80
    domain_shift = psi > max_psi

    uptake_task = (target_task == :nanocarrier_uptake)
    surface_warning = uptake_task && !protocol.internalization_imaging_verified
    internalization_ok = (!uptake_task) || protocol.internalization_imaging_verified

    track2_controls_ok = if protocol.track == :track2_clay_corrosion
        protocol.clay_matrix_blank_verified &&
        protocol.counting_beads_verified &&
        protocol.extraction_recovery_verified
    else
        protocol.counting_beads_verified
    end

    heldout_acc_val = heldout_run_accuracy === nothing ? 0.0 : Float64(heldout_run_accuracy)
    if heldout_run_accuracy !== nothing
        0.0 <= heldout_acc_val <= 1.0 ||
            throw(ArgumentError("heldout_run_accuracy must be in [0, 1]"))
    end

    emp_ready = (protocol.data_provenance == :deidentified_experimental_fcs) &&
                protocol.independent_reference_labels &&
                (protocol.n_independent_runs >= 2) &&
                (protocol.n_heldout_runs >= 1) &&
                (heldout_run_accuracy !== nothing) &&
                (heldout_acc_val >= min_ea) &&
                (!domain_shift) &&
                internalization_ok &&
                track2_controls_ok

    blockers = String[]
    if protocol.data_provenance != :deidentified_experimental_fcs
        push!(blockers, "data_provenance is :$(protocol.data_provenance) rather than :deidentified_experimental_fcs")
    end
    if !protocol.independent_reference_labels
        push!(blockers, "independent assay reference labels have not been established")
    end
    if protocol.n_independent_runs < 2 || protocol.n_heldout_runs < 1
        push!(blockers, "requires >= 2 independent experimental runs and >= 1 held-out run")
    end
    if heldout_run_accuracy === nothing || heldout_acc_val < min_ea
        push!(blockers, "held-out experimental run accuracy is missing or below threshold ($min_ea)")
    end
    if domain_shift
        push!(blockers, "feature domain shift detected (PSI = $(round(psi; digits=4)) > $max_psi)")
    end
    if surface_warning
        push!(blockers, "fluorescence gate measures nanocarrier association/uptake; intracellular internalization is not proven without independent imaging")
    end
    if !track2_controls_ok
        push!(blockers, "missing mandatory assay controls (counting beads, clay blank, or extraction recovery)")
    end

    return (
        software_verification_passed = software_ok,
        empirical_generalization_established = emp_ready,
        domain_shift_detected = domain_shift,
        surface_association_only_warning = surface_warning,
        internalization_proven = uptake_task ? protocol.internalization_imaging_verified : false,
        track_controls_verified = track2_controls_ok,
        external_solvers_coupled_in_production = protocol.external_solvers_coupled_in_production,
        blocking_reasons = blockers
    )
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
        nm = String(strip(name))
        !isempty(nm) || throw(ArgumentError("quantity name must be non-empty"))
        status in VALID_EPISTEMIC_STATUSES ||
            throw(ArgumentError("epistemic status must be one of $(VALID_EPISTEMIC_STATUSES)"))
        new{T}(nm, value, status, String(source_note))
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
    track != :track3_bitumen_rheology ||
        throw(ArgumentError("track3_bitumen_rheology must not attach a flow-cytometry observation sample"))
    digest = lowercase(strip(String(input_sha256)))
    occursin(r"^[0-9a-f]{64}$", digest) ||
        throw(ArgumentError("input_sha256 must be a valid 64-character lowercase hex SHA-256 digest"))

    seen_names = Set{String}()
    present_statuses = Set{Symbol}()
    for q in quantities
        !(q.name in seen_names) ||
            throw(ArgumentError("duplicate quantity name '$(q.name)' in production audit record"))
        push!(seen_names, q.name)
        push!(present_statuses, q.status)
    end
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
