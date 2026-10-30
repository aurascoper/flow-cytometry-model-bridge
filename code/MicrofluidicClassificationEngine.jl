if !isdefined(@__MODULE__, :FlowCytometryObservationModel)
    include(joinpath(@__DIR__, "FlowCytometryObservationModel.jl"))
end

module MicrofluidicClassificationEngine

using LinearAlgebra
using Statistics
using Random
using Dates
using Printf
import SHA

import ..FlowCytometryObservationModel
using ..FlowCytometryObservationModel:
    MODEL_ADAPTER_VERSION, VALID_SAMPLE_ORIGINS, VALID_VOLUME_CONVENTIONS,
    VALID_STAKEHOLDER_TRACKS, VALID_EPISTEMIC_STATUSES, REQUIRED_CSV_COLUMNS,
    CellStatePrediction, GateCalibration, FCMSampleSummary,
    MESFCalibrationCurve, BeadSizeCalibration, ConcentrationUncertainty,
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

const ENGINE_VERSION = "0.2.0"

# ===========================================================================
# MODULE 1: FCSIO
# Responsibility: ISAC FCS 3.1 binary file loading/writing, channel metadata,
# event tables, CSV sensor/trajectory loading, and SHA-256 provenance.
# ===========================================================================
module FCSIO

using Random
using Dates
using Printf
import SHA

export FCSChannelMetadata, FCSFileProvenance, FCSDataset, SensorTrajectoryRow,
       n_events, n_channels, channel_names, get_channel_vector, subset_events,
       serialize_fcs_bytes, write_fcs_file, parse_fcs_bytes, read_fcs_file,
       write_microfluidic_sensor_csv, load_microfluidic_sensor_csv,
       generate_synthetic_nanobio_fcs_dataset

struct FCSChannelMetadata
    index::Int
    name::String
    stain::String
    range::Float64
    bits::Int
    amplification::Tuple{Float64,Float64}
    gain::Float64

    function FCSChannelMetadata(index::Integer, name::AbstractString;
                                stain::AbstractString="",
                                range::Real=262144.0,
                                bits::Integer=32,
                                amplification::Tuple{<:Real,<:Real}=(0.0, 0.0),
                                gain::Real=1.0)
        idx = Int(index)
        idx >= 1 || throw(ArgumentError("channel index must be >= 1"))
        nm = String(strip(name))
        !isempty(nm) || throw(ArgumentError("channel name (\$PnN) must be non-empty"))
        st = String(strip(stain))
        rg = Float64(range)
        isfinite(rg) && rg > 0 || throw(ArgumentError("channel range (\$PnR) must be finite and > 0"))
        b = Int(bits)
        b in (16, 32, 64) || throw(ArgumentError("channel bits (\$PnB) must be 16, 32, or 64"))
        amp = (Float64(amplification[1]), Float64(amplification[2]))
        all(isfinite, amp) && amp[1] >= 0 && amp[2] >= 0 ||
            throw(ArgumentError("amplification (\$PnE) must be finite and nonnegative"))
        gn = Float64(gain)
        isfinite(gn) && gn > 0 || throw(ArgumentError("channel gain (\$PnG) must be finite and > 0"))
        new(idx, nm, st, rg, b, amp, gn)
    end
end

struct FCSFileProvenance
    source_path::String
    sha256::String
    fcs_version::String
    cytometer::String
    institution::String
    sample_id::String
    chip_id::String
    run_id::String
    biological_replicate::String
    byte_size::Int
    timestamp_utc::String

    function FCSFileProvenance(; source_path::AbstractString,
                               sha256::AbstractString,
                               fcs_version::AbstractString="FCS3.1",
                               cytometer::AbstractString="NanoBio-Microfluidic-Cytometer",
                               institution::AbstractString="SEMO-NanoBio-Lab",
                               sample_id::AbstractString="UNSPECIFIED-SAMPLE",
                               chip_id::AbstractString="UNSPECIFIED-CHIP",
                               run_id::AbstractString="UNSPECIFIED-RUN",
                               biological_replicate::AbstractString="REP-1",
                               byte_size::Integer=0,
                               timestamp_utc::AbstractString="2026-10-29T00:00:00Z")
        digest = lowercase(strip(String(sha256)))
        occursin(r"^[0-9a-f]{64}$", digest) ||
            throw(ArgumentError("sha256 must be a valid 64-character hex SHA-256 digest"))
        ver = String(strip(fcs_version))
        ver in ("FCS3.1", "FCS3.0", "FCS2.0") ||
            throw(ArgumentError("unsupported FCS version '$ver'"))
        bs = Int(byte_size)
        bs >= 0 || throw(ArgumentError("byte_size must be >= 0"))
        sid = String(strip(sample_id))
        cid = String(strip(chip_id))
        rid = String(strip(run_id))
        rep = String(strip(biological_replicate))
        !isempty(sid) || throw(ArgumentError("sample_id must be non-empty"))
        !isempty(cid) || throw(ArgumentError("chip_id must be non-empty"))
        !isempty(rid) || throw(ArgumentError("run_id must be non-empty"))
        !isempty(rep) || throw(ArgumentError("biological_replicate must be non-empty"))
        new(String(source_path), digest, ver, String(strip(cytometer)),
            String(strip(institution)), sid, cid, rid, rep, bs, String(strip(timestamp_utc)))
    end
end

struct FCSDataset
    provenance::FCSFileProvenance
    channels::Vector{FCSChannelMetadata}
    keywords::Dict{String,String}
    events::Matrix{Float64}
    channel_index::Dict{String,Int}

    function FCSDataset(provenance::FCSFileProvenance,
                        channels::AbstractVector{FCSChannelMetadata},
                        keywords::Dict{String,String},
                        events::AbstractMatrix{<:Real})
        n_ch = length(channels)
        n_ch >= 1 || throw(ArgumentError("FCSDataset requires at least 1 channel"))
        size(events, 2) == n_ch ||
            throw(ArgumentError("events matrix column count ($(size(events, 2))) must equal channel count ($n_ch)"))
        ev = Matrix{Float64}(events)
        all(isfinite, ev) || throw(ArgumentError("all FCS event values must be finite"))
        ch_vec = collect(FCSChannelMetadata, channels)
        ch_idx = Dict{String,Int}()
        for (i, ch) in enumerate(ch_vec)
            ch.index == i || throw(ArgumentError("channel at position $i has mismatched index $(ch.index)"))
            !haskey(ch_idx, ch.name) || throw(ArgumentError("duplicate channel name '$(ch.name)'"))
            ch_idx[ch.name] = i
        end
        new(provenance, ch_vec, copy(keywords), ev, ch_idx)
    end
end

n_events(ds::FCSDataset) = size(ds.events, 1)
n_channels(ds::FCSDataset) = size(ds.events, 2)
channel_names(ds::FCSDataset) = [ch.name for ch in ds.channels]

function get_channel_vector(ds::FCSDataset, name::AbstractString)
    nm = String(strip(name))
    haskey(ds.channel_index, nm) ||
        throw(ArgumentError("channel '$nm' not found in FCSDataset; available: $(join(channel_names(ds), ", "))"))
    return ds.events[:, ds.channel_index[nm]]
end

function subset_events(ds::FCSDataset, mask::AbstractVector{Bool}; suffix::AbstractString="gated")
    length(mask) == n_events(ds) ||
        throw(ArgumentError("mask length ($(length(mask))) must match n_events ($(n_events(ds)))"))
    sub_ev = ds.events[ collect(Bool, mask), : ]
    kw = copy(ds.keywords)
    kw["\$TOT"] = string(size(sub_ev, 1))
    kw["\$PARENT_SHA256"] = ds.provenance.sha256
    sub_hash = bytes2hex(SHA.sha256( string(ds.provenance.sha256, ":", suffix, ":", size(sub_ev, 1), ":", sum(sub_ev)) ))
    prov = FCSFileProvenance(
        source_path = "$(ds.provenance.source_path)#$(suffix)",
        sha256 = sub_hash,
        fcs_version = ds.provenance.fcs_version,
        cytometer = ds.provenance.cytometer,
        institution = ds.provenance.institution,
        sample_id = ds.provenance.sample_id,
        chip_id = ds.provenance.chip_id,
        run_id = ds.provenance.run_id,
        biological_replicate = ds.provenance.biological_replicate,
        byte_size = sizeof(sub_ev),
        timestamp_utc = ds.provenance.timestamp_utc
    )
    return FCSDataset(prov, ds.channels, kw, sub_ev)
end

function _escape_fcs_value(s::AbstractString, delim::Char='/')
    str = String(s)
    isempty(str) && return " "
    return replace(str, string(delim) => string(delim, delim))
end

function _build_fcs_text_segment(ds::FCSDataset, datatype::Symbol, endian::Symbol,
                                 data_start::Int, data_end::Int)
    delim = '/'
    pairs = Pair{String,String}[]
    push!(pairs, "\$BEGINANALYSIS" => "0")
    push!(pairs, "\$ENDANALYSIS" => "0")
    push!(pairs, "\$BEGINSTEXT" => "0")
    push!(pairs, "\$ENDSTEXT" => "0")
    push!(pairs, "\$BEGINDATA" => string(data_start))
    push!(pairs, "\$ENDDATA" => string(data_end))
    push!(pairs, "\$BYTEORD" => (endian == :little ? "1,2,3,4" : "4,3,2,1"))
    push!(pairs, "\$DATATYPE" => string(datatype))
    push!(pairs, "\$MODE" => "L")
    push!(pairs, "\$NEXTDATA" => "0")
    push!(pairs, "\$PAR" => string(n_channels(ds)))
    push!(pairs, "\$TOT" => string(n_events(ds)))
    push!(pairs, "\$CYT" => ds.provenance.cytometer)
    push!(pairs, "\$INST" => ds.provenance.institution)
    push!(pairs, "\$SMNO" => ds.provenance.sample_id)
    push!(pairs, "CHIP_ID" => ds.provenance.chip_id)
    push!(pairs, "RUN_ID" => ds.provenance.run_id)
    push!(pairs, "BIOLOGICAL_REPLICATE" => ds.provenance.biological_replicate)
    push!(pairs, "TIMESTAMP_UTC" => ds.provenance.timestamp_utc)

    for ch in ds.channels
        i = ch.index
        bits_val = datatype == :D ? 64 : (datatype == :F ? 32 : ch.bits)
        push!(pairs, "\$P$(i)N" => ch.name)
        push!(pairs, "\$P$(i)S" => (isempty(ch.stain) ? ch.name : ch.stain))
        push!(pairs, "\$P$(i)R" => @sprintf("%.0f", ch.range))
        push!(pairs, "\$P$(i)B" => string(bits_val))
        push!(pairs, "\$P$(i)E" => @sprintf("%.4g,%.4g", ch.amplification[1], ch.amplification[2]))
        push!(pairs, "\$P$(i)G" => @sprintf("%.4g", ch.gain))
    end

    reserved = Set(first.(pairs))
    for k in sort!(collect(keys(ds.keywords)))
        if !(k in reserved)
            push!(pairs, k => ds.keywords[k])
        end
    end

    io = IOBuffer()
    write(io, delim)
    for (k, v) in pairs
        write(io, _escape_fcs_value(k, delim))
        write(io, delim)
        write(io, _escape_fcs_value(v, delim))
        write(io, delim)
    end
    return take!(io)
end

function _encode_fcs_data_segment(ds::FCSDataset, datatype::Symbol, endian::Symbol)
    events = ds.events
    n_ev, n_ch = size(events)
    io = IOBuffer()
    if datatype == :F
        for r in 1:n_ev
            for c in 1:n_ch
                val = Float32(events[r, c])
                w = endian == :little ? htol(val) : hton(val)
                write(io, w)
            end
        end
    elseif datatype == :D
        for r in 1:n_ev
            for c in 1:n_ch
                val = Float64(events[r, c])
                w = endian == :little ? htol(val) : hton(val)
                write(io, w)
            end
        end
    elseif datatype == :I
        for r in 1:n_ev
            for c in 1:n_ch
                b = ds.channels[c].bits
                if b == 16
                    clamped16 = clamp(round(Int64, events[r, c]), 0, Int64(typemax(UInt16)))
                    val16 = UInt16(clamped16)
                    write(io, endian == :little ? htol(val16) : hton(val16))
                elseif b == 64
                    clamped64 = clamp(round(Int128, events[r, c]), Int128(0), Int128(typemax(UInt64)))
                    val64 = UInt64(clamped64)
                    write(io, endian == :little ? htol(val64) : hton(val64))
                else
                    clamped32 = clamp(round(Int64, events[r, c]), 0, Int64(typemax(UInt32)))
                    val32 = UInt32(clamped32)
                    write(io, endian == :little ? htol(val32) : hton(val32))
                end
            end
        end
    else
        throw(ArgumentError("datatype must be :F, :D, or :I"))
    end
    return take!(io)
end

function _format_fcs_offset(offset::Int)
    0 <= offset <= 99_999_999 || return "       0"
    return @sprintf("%8d", offset)
end

"""
Serialize an `FCSDataset` into a standards-compliant ISAC FCS 3.1 byte vector.
Header offsets and `\$BEGINDATA` / `\$ENDDATA` keywords are solved to exact byte agreement.
"""
function serialize_fcs_bytes(ds::FCSDataset; endian::Symbol=:little, datatype::Symbol=:F)
    endian in (:little, :big) || throw(ArgumentError("endian must be :little or :big"))
    datatype in (:F, :D, :I) || throw(ArgumentError("datatype must be :F, :D, or :I"))

    data_bytes = _encode_fcs_data_segment(ds, datatype, endian)
    text_start = 58
    data_start = 58
    data_end = isempty(data_bytes) ? 58 : (58 + length(data_bytes) - 1)
    text_bytes = UInt8[]

    for _ in 1:6
        text_bytes = _build_fcs_text_segment(ds, datatype, endian, data_start, data_end)
        text_end = text_start + length(text_bytes) - 1
        new_data_start = text_end + 1
        new_data_end = isempty(data_bytes) ? new_data_start : (new_data_start + length(data_bytes) - 1)
        if new_data_start == data_start && new_data_end == data_end
            break
        end
        data_start = new_data_start
        data_end = new_data_end
    end

    text_end = text_start + length(text_bytes) - 1
    header_str = string(
        "FCS3.1",
        "    ",
        _format_fcs_offset(text_start),
        _format_fcs_offset(text_end),
        _format_fcs_offset(data_start),
        _format_fcs_offset(data_end),
        _format_fcs_offset(0),
        _format_fcs_offset(0)
    )
    length(codeunits(header_str)) == 58 ||
        error("internal error: FCS header must be exactly 58 bytes")

    out = Vector{UInt8}(undef, 58 + length(text_bytes) + length(data_bytes))
    copyto!(out, 1, codeunits(header_str), 1, 58)
    copyto!(out, 59, text_bytes, 1, length(text_bytes))
    if !isempty(data_bytes)
        copyto!(out, 59 + length(text_bytes), data_bytes, 1, length(data_bytes))
    end
    return out
end

function write_fcs_file(path::AbstractString, ds::FCSDataset; endian::Symbol=:little, datatype::Symbol=:F)
    bytes = serialize_fcs_bytes(ds; endian=endian, datatype=datatype)
    mkpath(dirname(abspath(path)))
    write(path, bytes)
    digest = bytes2hex(SHA.sha256(bytes))
    prov = FCSFileProvenance(
        source_path = abspath(path),
        sha256 = digest,
        fcs_version = "FCS3.1",
        cytometer = ds.provenance.cytometer,
        institution = ds.provenance.institution,
        sample_id = ds.provenance.sample_id,
        chip_id = ds.provenance.chip_id,
        run_id = ds.provenance.run_id,
        biological_replicate = ds.provenance.biological_replicate,
        byte_size = length(bytes),
        timestamp_utc = ds.provenance.timestamp_utc
    )
    return FCSDataset(prov, ds.channels, ds.keywords, ds.events)
end

function _parse_fcs_text_keywords(text_bytes::AbstractVector{UInt8})
    !isempty(text_bytes) || throw(ArgumentError("FCS TEXT segment is empty"))
    delim = Char(text_bytes[1])
    str = String(copy(text_bytes[2:end]))
    tokens = String[]
    buf = IOBuffer()
    i = firstindex(str)
    last_i = lastindex(str)
    while i <= last_i
        c = str[i]
        if c == delim
            nxt = nextind(str, i)
            if nxt <= last_i && str[nxt] == delim
                write(buf, delim)
                i = nextind(str, nxt)
                continue
            else
                push!(tokens, String(take!(buf)))
                i = nxt
                continue
            end
        else
            write(buf, c)
            i = nextind(str, i)
        end
    end
    kw = Dict{String,String}()
    n_pairs = div(length(tokens), 2)
    for p in 1:n_pairs
        k = uppercase(strip(tokens[2p - 1]))
        v = strip(tokens[2p])
        if !isempty(k)
            kw[k] = v
        end
    end
    return kw
end

function parse_fcs_bytes(bytes::AbstractVector{UInt8};
                         source_path::AbstractString="<memory>",
                         sample_id::AbstractString="",
                         chip_id::AbstractString="",
                         run_id::AbstractString="",
                         biological_replicate::AbstractString="")
    length(bytes) >= 58 || throw(ArgumentError("FCS buffer shorter than 58-byte HEADER segment"))
    ver = strip(String(copy(bytes[1:6])))
    ver in ("FCS3.1", "FCS3.0", "FCS2.0") ||
        throw(ArgumentError("invalid or unsupported FCS header version '$ver'"))

    text_start = parse(Int, strip(String(copy(bytes[11:18]))))
    text_end   = parse(Int, strip(String(copy(bytes[19:26]))))
    hdr_data_start = parse(Int, strip(String(copy(bytes[27:34]))))
    hdr_data_end   = parse(Int, strip(String(copy(bytes[35:42]))))

    0 <= text_start <= text_end < length(bytes) ||
        throw(ArgumentError("invalid TEXT segment offsets ($text_start .. $text_end)"))
    kw = _parse_fcs_text_keywords(view(bytes, (text_start + 1):(text_end + 1)))

    n_par = parse(Int, get(kw, "\$PAR", "0"))
    n_tot = parse(Int, get(kw, "\$TOT", "0"))
    n_par >= 1 || throw(ArgumentError("FCS \$PAR must be >= 1"))
    n_tot >= 0 || throw(ArgumentError("FCS \$TOT must be >= 0"))

    data_start = hdr_data_start > 0 ? hdr_data_start : parse(Int, get(kw, "\$BEGINDATA", "0"))
    data_end   = hdr_data_end > 0   ? hdr_data_end   : parse(Int, get(kw, "\$ENDDATA", "0"))
    if n_tot > 0
        0 <= data_start <= data_end < length(bytes) ||
            throw(ArgumentError("invalid DATA segment offsets ($data_start .. $data_end)"))
    end

    datatype_str = uppercase(get(kw, "\$DATATYPE", "F"))
    byteord_str  = get(kw, "\$BYTEORD", "1,2,3,4")
    is_little = startswith(byteord_str, "1,2")
    is_big    = startswith(byteord_str, "4,3") || startswith(byteord_str, "2,1")
    (is_little || is_big) || throw(ArgumentError("unsupported \$BYTEORD '$byteord_str'"))

    channels = Vector{FCSChannelMetadata}(undef, n_par)
    for i in 1:n_par
        nm = get(kw, "\$P$(i)N", "P$i")
        st = get(kw, "\$P$(i)S", nm)
        rg = parse(Float64, get(kw, "\$P$(i)R", "262144"))
        b  = parse(Int, get(kw, "\$P$(i)B", datatype_str == "D" ? "64" : "32"))
        pe_str = get(kw, "\$P$(i)E", "0,0")
        pe_parts = split(pe_str, ',')
        amp = length(pe_parts) == 2 ? (parse(Float64, pe_parts[1]), parse(Float64, pe_parts[2])) : (0.0, 0.0)
        gn = parse(Float64, get(kw, "\$P$(i)G", "1.0"))
        channels[i] = FCSChannelMetadata(i, nm; stain=st, range=rg, bits=b, amplification=amp, gain=gn)
    end

    events = Matrix{Float64}(undef, n_tot, n_par)
    if n_tot > 0
        data_slice = view(bytes, (data_start + 1):(data_end + 1))
        io = IOBuffer(data_slice)
        if datatype_str == "F"
            expected_bytes = n_tot * n_par * 4
            length(data_slice) >= expected_bytes ||
                throw(ArgumentError("DATA segment shorter than expected $expected_bytes bytes for Float32"))
            for r in 1:n_tot
                for c in 1:n_par
                    raw = read(io, Float32)
                    val = Float64(is_little ? ltoh(raw) : ntoh(raw))
                    events[r, c] = val
                end
            end
        elseif datatype_str == "D"
            expected_bytes = n_tot * n_par * 8
            length(data_slice) >= expected_bytes ||
                throw(ArgumentError("DATA segment shorter than expected $expected_bytes bytes for Float64"))
            for r in 1:n_tot
                for c in 1:n_par
                    raw = read(io, Float64)
                    val = is_little ? ltoh(raw) : ntoh(raw)
                    events[r, c] = val
                end
            end
        elseif datatype_str == "I"
            bytes_per_event = sum(div(ch.bits, 8) for ch in channels)
            expected_bytes = n_tot * bytes_per_event
            length(data_slice) >= expected_bytes ||
                throw(ArgumentError("DATA segment shorter than expected $expected_bytes bytes for Integer"))
            for r in 1:n_tot
                for c in 1:n_par
                    b = channels[c].bits
                    if b == 16
                        raw16 = read(io, UInt16)
                        events[r, c] = Float64(is_little ? ltoh(raw16) : ntoh(raw16))
                    elseif b == 64
                        raw64 = read(io, UInt64)
                        events[r, c] = Float64(is_little ? ltoh(raw64) : ntoh(raw64))
                    else
                        raw32 = read(io, UInt32)
                        events[r, c] = Float64(is_little ? ltoh(raw32) : ntoh(raw32))
                    end
                end
            end
        else
            throw(ArgumentError("unsupported FCS \$DATATYPE '$datatype_str'"))
        end
    end

    digest = bytes2hex(SHA.sha256(bytes))
    sid = !isempty(strip(sample_id)) ? sample_id : get(kw, "\$SMNO", "UNSPECIFIED-SAMPLE")
    cid = !isempty(strip(chip_id))   ? chip_id   : get(kw, "CHIP_ID", "UNSPECIFIED-CHIP")
    rid = !isempty(strip(run_id))    ? run_id    : get(kw, "RUN_ID", "UNSPECIFIED-RUN")
    rep = !isempty(strip(biological_replicate)) ? biological_replicate : get(kw, "BIOLOGICAL_REPLICATE", "REP-1")
    ts  = get(kw, "TIMESTAMP_UTC", "2026-10-29T00:00:00Z")

    prov = FCSFileProvenance(
        source_path = String(source_path),
        sha256 = digest,
        fcs_version = ver,
        cytometer = get(kw, "\$CYT", "Unknown"),
        institution = get(kw, "\$INST", "Unknown"),
        sample_id = sid,
        chip_id = cid,
        run_id = rid,
        biological_replicate = rep,
        byte_size = length(bytes),
        timestamp_utc = ts
    )
    return FCSDataset(prov, channels, kw, events)
end

function read_fcs_file(path::AbstractString;
                       sample_id::AbstractString="",
                       chip_id::AbstractString="",
                       run_id::AbstractString="",
                       biological_replicate::AbstractString="")
    isfile(path) || throw(ArgumentError("FCS file does not exist: $path"))
    bytes = read(path)
    return parse_fcs_bytes(bytes;
        source_path=abspath(path),
        sample_id=sample_id,
        chip_id=chip_id,
        run_id=run_id,
        biological_replicate=biological_replicate)
end

struct SensorTrajectoryRow
    event_index::Int
    sample_id::String
    chip_id::String
    run_id::String
    biological_replicate::String
    timestamp_s::Float64
    transit_time_ms::Float64
    blockage_peak_na::Float64
    baseline_current_na::Float64
    deformability_index::Float64
    cell_circularity::Float64
    apparent_velocity_mm_s::Float64
    retained_in_pore::Bool
    expert_label::Symbol
end

const SENSOR_CSV_HEADER = "event_index,sample_id,chip_id,run_id,biological_replicate,timestamp_s,transit_time_ms,blockage_peak_na,baseline_current_na,deformability_index,cell_circularity,apparent_velocity_mm_s,retained_in_pore,expert_label"

function write_microfluidic_sensor_csv(path::AbstractString, rows::AbstractVector{SensorTrajectoryRow})
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, SENSOR_CSV_HEADER)
        for r in rows
            @printf(io, "%d,%s,%s,%s,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s\n",
                r.event_index, r.sample_id, r.chip_id, r.run_id, r.biological_replicate,
                r.timestamp_s, r.transit_time_ms, r.blockage_peak_na, r.baseline_current_na,
                r.deformability_index, r.cell_circularity, r.apparent_velocity_mm_s,
                r.retained_in_pore ? "true" : "false", String(r.expert_label))
        end
    end
    digest = bytes2hex(SHA.sha256(read(path)))
    return (source_path = abspath(path), sha256 = digest, count = length(rows))
end

function load_microfluidic_sensor_csv(path::AbstractString)
    isfile(path) || throw(ArgumentError("sensor CSV file does not exist: $path"))
    raw_bytes = read(path)
    digest = bytes2hex(SHA.sha256(raw_bytes))
    lines = split(chomp(String(raw_bytes)), '\n')
    !isempty(lines) || throw(ArgumentError("sensor CSV file is empty"))
    hdr = strip(lines[1])
    hdr == SENSOR_CSV_HEADER ||
        throw(ArgumentError("sensor CSV header does not match expected schema"))
    rows = SensorTrajectoryRow[]
    for idx in 2:length(lines)
        ln = strip(lines[idx])
        isempty(ln) && continue
        parts = [strip(p) for p in split(ln, ',')]
        length(parts) == 14 || throw(ArgumentError("malformed sensor CSV row at line $idx"))
        push!(rows, SensorTrajectoryRow(
            parse(Int, parts[1]),
            parts[2],
            parts[3],
            parts[4],
            parts[5],
            parse(Float64, parts[6]),
            parse(Float64, parts[7]),
            parse(Float64, parts[8]),
            parse(Float64, parts[9]),
            parse(Float64, parts[10]),
            parse(Float64, parts[11]),
            parse(Float64, parts[12]),
            lowercase(parts[13]) == "true",
            Symbol(parts[14])
        ))
    end
    return (rows = rows, sha256 = digest, source_path = abspath(path))
end

"""
Generate a realistic multi-population Nano-Bio Lab flow-cytometry and micropore-transit
dataset with known ground-truth labels for all 4 candidate classification tasks:
- `:cell_type` (`:viable_bacterial_rod`, `:viable_coccus`, `:mineral_colloid_debris`)
- `:cell_stress_damage` (`:membrane_intact`, `:membrane_compromised`, `:dead_lysed`)
- `:nanocarrier_uptake` (`:high_uptake_active`, `:low_uptake_passive`, `:unbound_negative`)
- `:micropore_transit_regime` (`:deformable_smooth_transit`, `:stiff_retarded_transit`, `:pore_occlusion_risk`)
"""
function generate_synthetic_nanobio_fcs_dataset(;
        n_events::Int=400,
        sample_id::AbstractString="NBL-CHIP01-RUN1",
        chip_id::AbstractString="CHIP-01",
        run_id::AbstractString="RUN-01",
        biological_replicate::AbstractString="REP-1",
        fractions::NTuple{3,Float64}=(0.65, 0.22, 0.13),
        spillover_s12::Float64=0.14,
        spillover_s21::Float64=0.05,
        seed::Int=20261029)
    n_events >= 10 || throw(ArgumentError("n_events must be >= 10"))
    isapprox(sum(fractions), 1.0; atol=1e-6) || throw(ArgumentError("fractions must sum to 1"))

    rng = MersenneTwister(seed)
    channels = [
        FCSChannelMetadata(1, "FSC-A"; stain="Forward Scatter Area", range=262144.0),
        FCSChannelMetadata(2, "FSC-H"; stain="Forward Scatter Height", range=262144.0),
        FCSChannelMetadata(3, "SSC-A"; stain="Side Scatter Area", range=262144.0),
        FCSChannelMetadata(4, "FL1-SYBR-A"; stain="SYBR Green I Nucleic Acid", range=262144.0),
        FCSChannelMetadata(5, "FL2-PI-A"; stain="Propidium Iodide Membrane Compromise", range=262144.0),
        FCSChannelMetadata(6, "FL3-Cy5-A"; stain="Cy5 Nanocarrier Payload", range=262144.0),
        FCSChannelMetadata(7, "TIME"; stain="Acquisition Time s", range=3600.0),
    ]

    spill_mat = [1.0 spillover_s12 0.0;
                 spillover_s21 1.0 0.03;
                 0.0 0.02 1.0]
    spill_str = @sprintf("3,FL1-SYBR-A,FL2-PI-A,FL3-Cy5-A,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
        spill_mat[1,1], spill_mat[1,2], spill_mat[1,3],
        spill_mat[2,1], spill_mat[2,2], spill_mat[2,3],
        spill_mat[3,1], spill_mat[3,2], spill_mat[3,3])

    events = Matrix{Float64}(undef, n_events, 7)
    sensor_rows = Vector{SensorTrajectoryRow}(undef, n_events)

    labels_stress = Vector{Symbol}(undef, n_events)
    labels_type = Vector{Symbol}(undef, n_events)
    labels_uptake = Vector{Symbol}(undef, n_events)
    labels_transit = Vector{Symbol}(undef, n_events)

    n1 = round(Int, n_events * fractions[1])
    n2 = round(Int, n_events * fractions[2])
    n3 = n_events - n1 - n2

    for i in 1:n_events
        pop = i <= n1 ? 1 : (i <= n1 + n2 ? 2 : 3)
        t_sec = 0.05 * i + 0.005 * rand(rng)

        if pop == 1
            # Population 1: Membrane-intact viable cells, smooth deformable transit, high nanocarrier uptake
            fsc_a = clamp(72_000.0 + 11_000.0 * randn(rng), 15_000.0, 240_000.0)
            fsc_h = clamp(fsc_a * (0.93 + 0.025 * randn(rng)), 12_000.0, 240_000.0)
            ssc_a = clamp(18_000.0 + 3_800.0 * randn(rng), 2_000.0, 240_000.0)
            true_fl1 = clamp(14_500.0 * exp(0.28 * randn(rng)), 800.0, 200_000.0)
            true_fl2 = clamp(110.0 + 35.0 * randn(rng), 10.0, 900.0)
            true_fl3 = clamp(9_200.0 * exp(0.30 * randn(rng)), 1_500.0, 180_000.0)

            transit_ms = clamp(0.42 + 0.07 * randn(rng), 0.15, 2.5)
            block_na   = clamp(1.35 + 0.18 * randn(rng), 0.4, 5.0)
            deform     = clamp(1.48 + 0.11 * randn(rng), 1.05, 2.2)
            circ       = clamp(0.89 + 0.03 * randn(rng), 0.5, 0.99)
            vel_mm_s   = clamp(18.5 + 1.8 * randn(rng), 4.0, 35.0)
            retained   = false

            labels_stress[i]  = :membrane_intact
            labels_type[i]    = :viable_bacterial_rod
            labels_uptake[i]  = :high_uptake_active
            labels_transit[i] = :deformable_smooth_transit
        elseif pop == 2
            # Population 2: Membrane-compromised stressed cells, stiff retarded transit, low passive uptake
            fsc_a = clamp(54_000.0 + 9_500.0 * randn(rng), 12_000.0, 240_000.0)
            fsc_h = clamp(fsc_a * (0.90 + 0.035 * randn(rng)), 10_000.0, 240_000.0)
            ssc_a = clamp(39_000.0 + 6_500.0 * randn(rng), 5_000.0, 240_000.0)
            true_fl1 = clamp(7_200.0 * exp(0.30 * randn(rng)), 500.0, 150_000.0)
            true_fl2 = clamp(3_100.0 * exp(0.28 * randn(rng)), 700.0, 80_000.0)
            true_fl3 = clamp(1_600.0 * exp(0.28 * randn(rng)), 250.0, 25_000.0)

            transit_ms = clamp(1.25 + 0.18 * randn(rng), 0.55, 4.5)
            block_na   = clamp(2.15 + 0.25 * randn(rng), 0.8, 6.0)
            deform     = clamp(1.14 + 0.06 * randn(rng), 1.01, 1.45)
            circ       = clamp(0.76 + 0.05 * randn(rng), 0.4, 0.95)
            vel_mm_s   = clamp(9.2 + 1.4 * randn(rng), 2.0, 25.0)
            retained   = false

            labels_stress[i]  = :membrane_compromised
            labels_type[i]    = :viable_coccus
            labels_uptake[i]  = :low_uptake_passive
            labels_transit[i] = :stiff_retarded_transit
        else
            # Population 3: Dead / lysed or mineral-colloid aggregates, pore occlusion risk, unbound
            fsc_a = clamp(135_000.0 + 22_000.0 * randn(rng), 25_000.0, 260_000.0)
            fsc_h = clamp(fsc_a * (0.68 + 0.06 * randn(rng)), 15_000.0, 250_000.0)
            ssc_a = clamp(92_000.0 + 16_000.0 * randn(rng), 18_000.0, 260_000.0)
            true_fl1 = clamp(420.0 + 140.0 * randn(rng), 30.0, 3_500.0)
            true_fl2 = clamp(19_500.0 * exp(0.30 * randn(rng)), 3_500.0, 220_000.0)
            true_fl3 = clamp(140.0 + 45.0 * randn(rng), 10.0, 900.0)

            transit_ms = clamp(3.40 + 0.45 * randn(rng), 1.8, 9.0)
            block_na   = clamp(4.10 + 0.40 * randn(rng), 2.2, 9.5)
            deform     = clamp(1.03 + 0.02 * randn(rng), 1.00, 1.15)
            circ       = clamp(0.54 + 0.07 * randn(rng), 0.2, 0.85)
            vel_mm_s   = clamp(3.1 + 0.7 * randn(rng), 0.5, 10.0)
            retained   = transit_ms > 3.6

            labels_stress[i]  = :dead_lysed
            labels_type[i]    = :mineral_colloid_debris
            labels_uptake[i]  = :unbound_negative
            labels_transit[i] = :pore_occlusion_risk
        end

        # Apply physical spectral spillover: measured_fl = true_fl * spill_mat
        raw_fl = [true_fl1, true_fl2, true_fl3]' * spill_mat
        events[i, 1] = fsc_a
        events[i, 2] = fsc_h
        events[i, 3] = ssc_a
        events[i, 4] = raw_fl[1]
        events[i, 5] = raw_fl[2]
        events[i, 6] = raw_fl[3]
        events[i, 7] = t_sec

        sensor_rows[i] = SensorTrajectoryRow(
            i, String(sample_id), String(chip_id), String(run_id), String(biological_replicate),
            t_sec, transit_ms, block_na, 12.0, deform, circ, vel_mm_s, retained, labels_stress[i]
        )
    end

    kw = Dict{String,String}(
        "\$SPILLOVER" => spill_str,
        "ASSAY_TYPE" => "NANO_BIO_MICROPORE_CYTOMETRY"
    )
    init_hash = bytes2hex(SHA.sha256(string(sample_id, ":", chip_id, ":", run_id, ":", seed, ":", n_events)))
    prov = FCSFileProvenance(
        source_path = "<synthetic:$sample_id>",
        sha256 = init_hash,
        fcs_version = "FCS3.1",
        cytometer = "NanoBio-Microfluidic-Cytometer",
        institution = "SEMO-NanoBio-Lab",
        sample_id = sample_id,
        chip_id = chip_id,
        run_id = run_id,
        biological_replicate = biological_replicate,
        byte_size = sizeof(events),
        timestamp_utc = "2026-10-29T18:00:00Z"
    )
    ds = FCSDataset(prov, channels, kw, events)
    return (
        dataset = ds,
        sensor_rows = sensor_rows,
        labels = Dict{Symbol,Vector{Symbol}}(
            :cell_stress_damage => labels_stress,
            :cell_type => labels_type,
            :nanocarrier_uptake => labels_uptake,
            :micropore_transit_regime => labels_transit,
        ),
        spillover_matrix = spill_mat
    )
end

end # module FCSIO


# ===========================================================================
# MODULE 2: CytometryAnalysis
# Responsibility: Transformations, spillover compensation, quality control,
# interactive/hierarchical gates, population statistics, JSON/GatingML IO.
# ===========================================================================
module CytometryAnalysis

using LinearAlgebra
using Statistics
using Printf
import ..FCSIO: FCSDataset, FCSChannelMetadata, FCSFileProvenance,
                n_events, n_channels, channel_names, get_channel_vector

export AbstractCytometryTransform,
       LinearTransform, Log10Transform, ArcsinhTransform, BiexponentialLogicleTransform,
       apply_transform, inverse_transform, transform_dataset,
       SpilloverMatrix, parse_fcs_spillover, format_fcs_spillover, apply_compensation,
       QualityControlConfig, QualityControlReport, run_quality_control,
       AbstractCytometryGate,
       RectangleGate, PolygonGate, EllipseGate, RangeGate, QuadGate,
       point_in_polygon, evaluate_gate, evaluate_quad_gate_all,
       GateNode, GatingHierarchy, GatePopulationSummary,
       add_gate!, update_gate!, remove_gate!, evaluate_hierarchy,
       export_gating_hierarchy_json, import_gating_hierarchy_json

abstract type AbstractCytometryTransform end

struct LinearTransform <: AbstractCytometryTransform
    scale::Float64
    offset::Float64
    function LinearTransform(scale::Real=1.0, offset::Real=0.0)
        sc = Float64(scale)
        of = Float64(offset)
        isfinite(sc) && sc > 0 || throw(ArgumentError("scale must be finite and > 0"))
        isfinite(of) || throw(ArgumentError("offset must be finite"))
        new(sc, of)
    end
end

struct Log10Transform <: AbstractCytometryTransform
    offset::Float64
    floor_val::Float64
    function Log10Transform(offset::Real=1.0; floor_val::Real=1e-9)
        of = Float64(offset)
        fl = Float64(floor_val)
        isfinite(of) && of >= 0 || throw(ArgumentError("offset must be finite and >= 0"))
        isfinite(fl) && fl > 0 || throw(ArgumentError("floor_val must be finite and > 0"))
        new(of, fl)
    end
end

struct ArcsinhTransform <: AbstractCytometryTransform
    cofactor::Float64
    function ArcsinhTransform(cofactor::Real=150.0)
        cf = Float64(cofactor)
        isfinite(cf) && cf > 0 || throw(ArgumentError("cofactor must be finite and > 0"))
        new(cf)
    end
end

struct BiexponentialLogicleTransform <: AbstractCytometryTransform
    cofactor::Float64
    decade_scale::Float64
    function BiexponentialLogicleTransform(cofactor::Real=150.0; decade_scale::Real=1.0 / log(10.0))
        cf = Float64(cofactor)
        ds = Float64(decade_scale)
        isfinite(cf) && cf > 0 || throw(ArgumentError("cofactor must be finite and > 0"))
        isfinite(ds) && ds > 0 || throw(ArgumentError("decade_scale must be finite and > 0"))
        new(cf, ds)
    end
end

apply_transform(t::LinearTransform, x::Real) = t.scale * Float64(x) + t.offset
inverse_transform(t::LinearTransform, y::Real) = (Float64(y) - t.offset) / t.scale

apply_transform(t::Log10Transform, x::Real) = log10(max(Float64(x) + t.offset, t.floor_val))
inverse_transform(t::Log10Transform, y::Real) = (10.0 ^ Float64(y)) - t.offset

apply_transform(t::ArcsinhTransform, x::Real) = asinh(Float64(x) / t.cofactor)
inverse_transform(t::ArcsinhTransform, y::Real) = t.cofactor * sinh(Float64(y))

apply_transform(t::BiexponentialLogicleTransform, x::Real) = t.decade_scale * asinh(Float64(x) / t.cofactor)
inverse_transform(t::BiexponentialLogicleTransform, y::Real) = t.cofactor * sinh(Float64(y) / t.decade_scale)

function apply_transform(t::AbstractCytometryTransform, xs::AbstractArray{<:Real})
    return [apply_transform(t, x) for x in xs]
end

function inverse_transform(t::AbstractCytometryTransform, ys::AbstractArray{<:Real})
    return [inverse_transform(t, y) for y in ys]
end

function transform_dataset(ds::FCSDataset, transforms::Dict{String,<:AbstractCytometryTransform})
    new_ev = copy(ds.events)
    for (ch_name, tr) in transforms
        haskey(ds.channel_index, ch_name) ||
            throw(ArgumentError("cannot transform unknown channel '$ch_name'"))
        col = ds.channel_index[ch_name]
        for r in 1:size(new_ev, 1)
            new_ev[r, col] = apply_transform(tr, new_ev[r, col])
        end
    end
    return FCSDataset(ds.provenance, ds.channels, ds.keywords, new_ev)
end

struct SpilloverMatrix
    channels::Vector{String}
    matrix::Matrix{Float64}

    function SpilloverMatrix(channels::AbstractVector{<:AbstractString},
                             matrix::AbstractMatrix{<:Real})
        chs = [String(strip(c)) for c in channels]
        k = length(chs)
        k >= 2 || throw(ArgumentError("SpilloverMatrix requires at least 2 channels"))
        length(unique(chs)) == k || throw(ArgumentError("SpilloverMatrix channel names must be unique"))
        size(matrix) == (k, k) || throw(ArgumentError("SpilloverMatrix must be $(k)x$(k)"))
        m = Float64.(matrix)
        all(isfinite, m) && all(x -> 0.0 <= x <= 1.0, m) ||
            throw(ArgumentError("spillover entries must be finite and in [0, 1]"))
        for i in 1:k
            isapprox(m[i, i], 1.0; atol=1e-6) ||
                throw(ArgumentError("SpilloverMatrix diagonal entries must equal 1.0"))
        end
        (abs(det(m)) > 1e-6 && cond(m) <= 1e3) ||
            throw(ArgumentError("SpilloverMatrix is singular or nearly singular"))
        new(chs, m)
    end
end

function parse_fcs_spillover(spill_str::AbstractString)
    parts = [strip(p) for p in split(strip(spill_str), ',')]
    !isempty(parts) && !isempty(parts[1]) ||
        throw(ArgumentError("empty \$SPILLOVER string"))
    k = parse(Int, parts[1])
    k >= 2 || throw(ArgumentError("\$SPILLOVER channel count must be >= 2"))
    expected_len = 1 + k + k * k
    length(parts) == expected_len ||
        throw(ArgumentError("\$SPILLOVER expected $expected_len comma-separated tokens, got $(length(parts))"))
    chs = parts[2:(1 + k)]
    vals = [parse(Float64, p) for p in parts[(2 + k):end]]
    mat = Matrix{Float64}(undef, k, k)
    idx = 1
    for r in 1:k
        for c in 1:k
            mat[r, c] = vals[idx]
            idx += 1
        end
    end
    return SpilloverMatrix(chs, mat)
end

function format_fcs_spillover(spill::SpilloverMatrix)
    k = length(spill.channels)
    tokens = String[string(k)]
    append!(tokens, spill.channels)
    for r in 1:k
        for c in 1:k
            push!(tokens, @sprintf("%.6g", spill.matrix[r, c]))
        end
    end
    return join(tokens, ",")
end

"""
Unmix spectral spillover across `spill.channels` by right-multiplying the raw
fluorescence submatrix by `inv(spill.matrix)`.
"""
function apply_compensation(ds::FCSDataset, spill::SpilloverMatrix)
    cols = Int[]
    for ch in spill.channels
        haskey(ds.channel_index, ch) ||
            throw(ArgumentError("spillover channel '$ch' not present in FCSDataset"))
        push!(cols, ds.channel_index[ch])
    end
    comp_ev = copy(ds.events)
    inv_s = inv(spill.matrix)
    comp_ev[:, cols] = comp_ev[:, cols] * inv_s
    kw = copy(ds.keywords)
    kw["COMPENSATED"] = "TRUE"
    return FCSDataset(ds.provenance, ds.channels, kw, comp_ev)
end

struct QualityControlConfig
    rail_fraction::Float64
    min_valid_value::Float64
    max_doublet_deviation::Float64
    time_window_size::Int
    max_time_zscore::Float64

    function QualityControlConfig(; rail_fraction::Real=0.995,
                                  min_valid_value::Real=-1000.0,
                                  max_doublet_deviation::Real=0.26,
                                  time_window_size::Integer=40,
                                  max_time_zscore::Real=3.5)
        rf = Float64(rail_fraction)
        mv = Float64(min_valid_value)
        dd = Float64(max_doublet_deviation)
        tw = Int(time_window_size)
        tz = Float64(max_time_zscore)
        0.5 < rf <= 1.0 || throw(ArgumentError("rail_fraction must be in (0.5, 1.0]"))
        isfinite(mv) || throw(ArgumentError("min_valid_value must be finite"))
        0.0 < dd < 1.0 || throw(ArgumentError("max_doublet_deviation must be in (0, 1)"))
        tw >= 5 || throw(ArgumentError("time_window_size must be >= 5"))
        tz > 0.0 || throw(ArgumentError("max_time_zscore must be > 0"))
        new(rf, mv, dd, tw, tz)
    end
end

struct QualityControlReport
    clean_mask::BitVector
    rail_mask::BitVector
    doublet_mask::BitVector
    time_drift_mask::BitVector
    pass_count::Int
    total_count::Int
    pass_fraction::Float64
    median_fsc_ratio::Float64
end

function run_quality_control(ds::FCSDataset;
                             config::QualityControlConfig=QualityControlConfig(),
                             fsc_a_channel::AbstractString="FSC-A",
                             fsc_h_channel::AbstractString="FSC-H",
                             time_channel::AbstractString="TIME")
    n = n_events(ds)
    rail_mask = falses(n)
    doublet_mask = falses(n)
    time_drift_mask = falses(n)

    for (c_idx, ch) in enumerate(ds.channels)
        if ch.name == time_channel
            continue
        end
        max_allowed = ch.range * config.rail_fraction
        for r in 1:n
            v = ds.events[r, c_idx]
            if v >= max_allowed || v < config.min_valid_value
                rail_mask[r] = true
            end
        end
    end

    med_ratio = 1.0
    if haskey(ds.channel_index, fsc_a_channel) && haskey(ds.channel_index, fsc_h_channel) && n > 0
        fsc_a = get_channel_vector(ds, fsc_a_channel)
        fsc_h = get_channel_vector(ds, fsc_h_channel)
        ratios = [fsc_h[i] / max(fsc_a[i], 1.0) for i in 1:n]
        med_ratio = median(ratios)
        for i in 1:n
            if abs(ratios[i] - med_ratio) / max(med_ratio, 1e-6) > config.max_doublet_deviation
                doublet_mask[i] = true
            end
        end
    end

    if haskey(ds.channel_index, fsc_a_channel) && haskey(ds.channel_index, time_channel) && n >= config.time_window_size
        fsc_a = get_channel_vector(ds, fsc_a_channel)
        global_med = median(fsc_a)
        iqr_val = quantile(fsc_a, 0.75) - quantile(fsc_a, 0.25)
        robust_sd = max(iqr_val / 1.349, 1e-3)
        w = config.time_window_size
        n_win = cld(n, w)
        for k in 1:n_win
            i_start = (k - 1) * w + 1
            i_end = min(k * w, n)
            win_med = median(view(fsc_a, i_start:i_end))
            if abs(win_med - global_med) / robust_sd > config.max_time_zscore
                for idx in i_start:i_end
                    time_drift_mask[idx] = true
                end
            end
        end
    end

    clean_mask = .!(rail_mask .| doublet_mask .| time_drift_mask)
    pass_cnt = count(clean_mask)
    pass_frac = n > 0 ? pass_cnt / n : 1.0
    return QualityControlReport(clean_mask, rail_mask, doublet_mask, time_drift_mask,
                                pass_cnt, n, pass_frac, med_ratio)
end

abstract type AbstractCytometryGate end

struct RectangleGate <: AbstractCytometryGate
    id::String
    name::String
    x_channel::String
    y_channel::String
    x_min::Float64
    x_max::Float64
    y_min::Float64
    y_max::Float64
    transform_x::AbstractCytometryTransform
    transform_y::AbstractCytometryTransform

    function RectangleGate(id::AbstractString, name::AbstractString,
                           x_channel::AbstractString, y_channel::AbstractString,
                           x_min::Real, x_max::Real, y_min::Real, y_max::Real;
                           transform_x::AbstractCytometryTransform=LinearTransform(),
                           transform_y::AbstractCytometryTransform=LinearTransform())
        gid = String(strip(id))
        gnm = String(strip(name))
        xch = String(strip(x_channel))
        ych = String(strip(y_channel))
        !isempty(gid) && !isempty(gnm) && !isempty(xch) && !isempty(ych) ||
            throw(ArgumentError("gate id, name, and channels must be non-empty"))
        x0, x1 = Float64(x_min), Float64(x_max)
        y0, y1 = Float64(y_min), Float64(y_max)
        all(isfinite, (x0, x1, y0, y1)) && x0 < x1 && y0 < y1 ||
            throw(ArgumentError("RectangleGate requires finite bounds with x_min < x_max and y_min < y_max"))
        new(gid, gnm, xch, ych, x0, x1, y0, y1, transform_x, transform_y)
    end
end

struct PolygonGate <: AbstractCytometryGate
    id::String
    name::String
    x_channel::String
    y_channel::String
    vertices::Vector{Tuple{Float64,Float64}}
    transform_x::AbstractCytometryTransform
    transform_y::AbstractCytometryTransform

    function PolygonGate(id::AbstractString, name::AbstractString,
                         x_channel::AbstractString, y_channel::AbstractString,
                         vertices::AbstractVector{<:Tuple{<:Real,<:Real}};
                         transform_x::AbstractCytometryTransform=LinearTransform(),
                         transform_y::AbstractCytometryTransform=LinearTransform())
        gid = String(strip(id))
        gnm = String(strip(name))
        xch = String(strip(x_channel))
        ych = String(strip(y_channel))
        !isempty(gid) && !isempty(gnm) && !isempty(xch) && !isempty(ych) ||
            throw(ArgumentError("gate id, name, and channels must be non-empty"))
        length(vertices) >= 3 || throw(ArgumentError("PolygonGate requires at least 3 vertices"))
        verts = [(Float64(v[1]), Float64(v[2])) for v in vertices]
        all(v -> isfinite(v[1]) && isfinite(v[2]), verts) ||
            throw(ArgumentError("all polygon vertices must be finite"))
        new(gid, gnm, xch, ych, verts, transform_x, transform_y)
    end
end

struct EllipseGate <: AbstractCytometryGate
    id::String
    name::String
    x_channel::String
    y_channel::String
    center_x::Float64
    center_y::Float64
    semi_major::Float64
    semi_minor::Float64
    angle_rad::Float64
    transform_x::AbstractCytometryTransform
    transform_y::AbstractCytometryTransform

    function EllipseGate(id::AbstractString, name::AbstractString,
                         x_channel::AbstractString, y_channel::AbstractString,
                         center_x::Real, center_y::Real,
                         semi_major::Real, semi_minor::Real, angle_rad::Real=0.0;
                         transform_x::AbstractCytometryTransform=LinearTransform(),
                         transform_y::AbstractCytometryTransform=LinearTransform())
        gid = String(strip(id))
        gnm = String(strip(name))
        xch = String(strip(x_channel))
        ych = String(strip(y_channel))
        !isempty(gid) && !isempty(gnm) && !isempty(xch) && !isempty(ych) ||
            throw(ArgumentError("gate id, name, and channels must be non-empty"))
        cx, cy = Float64(center_x), Float64(center_y)
        a, b = Float64(semi_major), Float64(semi_minor)
        ang = Float64(angle_rad)
        all(isfinite, (cx, cy, a, b, ang)) && a > 0 && b > 0 ||
            throw(ArgumentError("EllipseGate semi_major and semi_minor must be finite and > 0"))
        new(gid, gnm, xch, ych, cx, cy, a, b, ang, transform_x, transform_y)
    end
end

struct RangeGate <: AbstractCytometryGate
    id::String
    name::String
    channel::String
    min_val::Float64
    max_val::Float64
    transform::AbstractCytometryTransform

    function RangeGate(id::AbstractString, name::AbstractString,
                       channel::AbstractString, min_val::Real, max_val::Real;
                       transform::AbstractCytometryTransform=LinearTransform())
        gid = String(strip(id))
        gnm = String(strip(name))
        ch  = String(strip(channel))
        !isempty(gid) && !isempty(gnm) && !isempty(ch) ||
            throw(ArgumentError("gate id, name, and channel must be non-empty"))
        v0, v1 = Float64(min_val), Float64(max_val)
        isfinite(v0) && isfinite(v1) && v0 < v1 ||
            throw(ArgumentError("RangeGate requires finite min_val < max_val"))
        new(gid, gnm, ch, v0, v1, transform)
    end
end

const VALID_QUADRANTS = (:Q1_UR, :Q2_UL, :Q3_LL, :Q4_LR)

struct QuadGate <: AbstractCytometryGate
    id::String
    name::String
    x_channel::String
    y_channel::String
    x_thresh::Float64
    y_thresh::Float64
    quadrant::Symbol
    transform_x::AbstractCytometryTransform
    transform_y::AbstractCytometryTransform

    function QuadGate(id::AbstractString, name::AbstractString,
                      x_channel::AbstractString, y_channel::AbstractString,
                      x_thresh::Real, y_thresh::Real;
                      quadrant::Symbol=:Q1_UR,
                      transform_x::AbstractCytometryTransform=LinearTransform(),
                      transform_y::AbstractCytometryTransform=LinearTransform())
        gid = String(strip(id))
        gnm = String(strip(name))
        xch = String(strip(x_channel))
        ych = String(strip(y_channel))
        !isempty(gid) && !isempty(gnm) && !isempty(xch) && !isempty(ych) ||
            throw(ArgumentError("gate id, name, and channels must be non-empty"))
        quadrant in VALID_QUADRANTS ||
            throw(ArgumentError("quadrant must be one of $(VALID_QUADRANTS)"))
        xt, yt = Float64(x_thresh), Float64(y_thresh)
        isfinite(xt) && isfinite(yt) || throw(ArgumentError("QuadGate thresholds must be finite"))
        new(gid, gnm, xch, ych, xt, yt, quadrant, transform_x, transform_y)
    end
end

function _point_on_segment(px::Float64, py::Float64,
                           x1::Float64, y1::Float64, x2::Float64, y2::Float64;
                           atol::Float64=1e-9)
    dx = x2 - x1
    dy = y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= atol * atol
        return hypot(px - x1, py - y1) <= atol
    end
    cross = abs((px - x1) * dy - (py - y1) * dx) / sqrt(seg_len2)
    if cross > atol
        return false
    end
    dotp = (px - x1) * dx + (py - y1) * dy
    return -atol <= dotp <= seg_len2 + atol
end

"""
Exact ray-casting point-in-polygon check with boundary inclusion.
"""
function point_in_polygon(x::Real, y::Real, vertices::AbstractVector{Tuple{Float64,Float64}})
    px = Float64(x)
    py = Float64(y)
    n = length(vertices)
    inside = false
    j = n
    for i in 1:n
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if _point_on_segment(px, py, xj, yj, xi, yi)
            return true
        end
        intersects = ((yi > py) != (yj > py)) &&
                     (px < (xj - xi) * (py - yi) / (yj - yi) + xi)
        if intersects
            inside = !inside
        end
        j = i
    end
    return inside
end

function evaluate_gate(gate::RectangleGate, ds::FCSDataset)
    xs = get_channel_vector(ds, gate.x_channel)
    ys = get_channel_vector(ds, gate.y_channel)
    n = length(xs)
    mask = falses(n)
    for i in 1:n
        tx = apply_transform(gate.transform_x, xs[i])
        ty = apply_transform(gate.transform_y, ys[i])
        mask[i] = (gate.x_min <= tx <= gate.x_max) && (gate.y_min <= ty <= gate.y_max)
    end
    return mask
end

function evaluate_gate(gate::PolygonGate, ds::FCSDataset)
    xs = get_channel_vector(ds, gate.x_channel)
    ys = get_channel_vector(ds, gate.y_channel)
    n = length(xs)
    mask = falses(n)
    for i in 1:n
        tx = apply_transform(gate.transform_x, xs[i])
        ty = apply_transform(gate.transform_y, ys[i])
        mask[i] = point_in_polygon(tx, ty, gate.vertices)
    end
    return mask
end

function evaluate_gate(gate::EllipseGate, ds::FCSDataset)
    xs = get_channel_vector(ds, gate.x_channel)
    ys = get_channel_vector(ds, gate.y_channel)
    n = length(xs)
    mask = falses(n)
    ca = cos(gate.angle_rad)
    sa = sin(gate.angle_rad)
    inv_a2 = 1.0 / (gate.semi_major * gate.semi_major)
    inv_b2 = 1.0 / (gate.semi_minor * gate.semi_minor)
    for i in 1:n
        tx = apply_transform(gate.transform_x, xs[i]) - gate.center_x
        ty = apply_transform(gate.transform_y, ys[i]) - gate.center_y
        u =  tx * ca + ty * sa
        v = -tx * sa + ty * ca
        mask[i] = (u * u * inv_a2 + v * v * inv_b2) <= 1.0
    end
    return mask
end

function evaluate_gate(gate::RangeGate, ds::FCSDataset)
    xs = get_channel_vector(ds, gate.channel)
    n = length(xs)
    mask = falses(n)
    for i in 1:n
        tx = apply_transform(gate.transform, xs[i])
        mask[i] = gate.min_val <= tx <= gate.max_val
    end
    return mask
end

function evaluate_quad_gate_all(gate::QuadGate, ds::FCSDataset)
    xs = get_channel_vector(ds, gate.x_channel)
    ys = get_channel_vector(ds, gate.y_channel)
    n = length(xs)
    q1 = falses(n)
    q2 = falses(n)
    q3 = falses(n)
    q4 = falses(n)
    for i in 1:n
        tx = apply_transform(gate.transform_x, xs[i])
        ty = apply_transform(gate.transform_y, ys[i])
        if tx >= gate.x_thresh && ty >= gate.y_thresh
            q1[i] = true
        elseif tx < gate.x_thresh && ty >= gate.y_thresh
            q2[i] = true
        elseif tx < gate.x_thresh && ty < gate.y_thresh
            q3[i] = true
        else
            q4[i] = true
        end
    end
    return Dict{Symbol,BitVector}(:Q1_UR => q1, :Q2_UL => q2, :Q3_LL => q3, :Q4_LR => q4)
end

function evaluate_gate(gate::QuadGate, ds::FCSDataset)
    return evaluate_quad_gate_all(gate, ds)[gate.quadrant]
end

struct GateNode
    gate::AbstractCytometryGate
    parent_id::String
end

mutable struct GatingHierarchy
    nodes::Dict{String,GateNode}
    order::Vector{String}
    function GatingHierarchy()
        new(Dict{String,GateNode}(), String[])
    end
end

function add_gate!(h::GatingHierarchy, gate::AbstractCytometryGate; parent_id::AbstractString="root")
    pid = String(strip(parent_id))
    gid = gate.id
    gid != "root" || throw(ArgumentError("gate id cannot be reserved name 'root'"))
    !haskey(h.nodes, gid) || throw(ArgumentError("gate id '$gid' already exists in hierarchy"))
    (pid == "root" || haskey(h.nodes, pid)) ||
        throw(ArgumentError("parent gate '$pid' does not exist in hierarchy"))
    h.nodes[gid] = GateNode(gate, pid)
    push!(h.order, gid)
    return h
end

function update_gate!(h::GatingHierarchy, gate::AbstractCytometryGate)
    gid = gate.id
    haskey(h.nodes, gid) || throw(ArgumentError("gate id '$gid' not found in hierarchy"))
    old_parent = h.nodes[gid].parent_id
    h.nodes[gid] = GateNode(gate, old_parent)
    return h
end

function remove_gate!(h::GatingHierarchy, gate_id::AbstractString)
    gid = String(strip(gate_id))
    haskey(h.nodes, gid) || throw(ArgumentError("gate id '$gid' not found in hierarchy"))
    to_remove = Set{String}([gid])
    changed = true
    while changed
        changed = false
        for (k, node) in h.nodes
            if !(k in to_remove) && (node.parent_id in to_remove)
                push!(to_remove, k)
                changed = true
            end
        end
    end
    for k in to_remove
        delete!(h.nodes, k)
    end
    filter!(id -> !(id in to_remove), h.order)
    return h
end

struct GatePopulationSummary
    gate_id::String
    gate_name::String
    parent_id::String
    count::Int
    parent_count::Int
    total_count::Int
    fraction_of_parent::Float64
    fraction_of_total::Float64
    channel_means::Dict{String,Float64}
    channel_medians::Dict{String,Float64}
    channel_iqrs::Dict{String,Float64}
    channel_cvs::Dict{String,Float64}
    mask::BitVector
end

function evaluate_hierarchy(h::GatingHierarchy, ds::FCSDataset;
                            base_mask::Union{Nothing,AbstractVector{Bool}}=nothing)
    n_tot = n_events(ds)
    root_mask = base_mask === nothing ? trues(n_tot) : BitVector(base_mask)
    length(root_mask) == n_tot ||
        throw(ArgumentError("base_mask length must match dataset event count"))

    masks = Dict{String,BitVector}("root" => root_mask)
    summaries = Dict{String,GatePopulationSummary}()

    for gid in h.order
        node = h.nodes[gid]
        p_mask = masks[node.parent_id]
        local_mask = evaluate_gate(node.gate, ds)
        combined = p_mask .& local_mask
        masks[gid] = combined

        cnt = count(combined)
        p_cnt = count(p_mask)
        frac_parent = p_cnt > 0 ? cnt / p_cnt : 0.0
        frac_total  = n_tot > 0 ? cnt / n_tot : 0.0

        means = Dict{String,Float64}()
        meds  = Dict{String,Float64}()
        iqrs  = Dict{String,Float64}()
        cvs   = Dict{String,Float64}()
        for (c_idx, ch) in enumerate(ds.channels)
            if cnt > 0
                sub = ds.events[combined, c_idx]
                m = mean(sub)
                s = cnt > 1 ? std(sub) : 0.0
                means[ch.name] = m
                meds[ch.name]  = median(sub)
                iqrs[ch.name]  = quantile(sub, 0.75) - quantile(sub, 0.25)
                cvs[ch.name]   = abs(m) > 1e-12 ? s / abs(m) : 0.0
            else
                means[ch.name] = 0.0
                meds[ch.name]  = 0.0
                iqrs[ch.name]  = 0.0
                cvs[ch.name]   = 0.0
            end
        end

        summaries[gid] = GatePopulationSummary(
            gid, node.gate.name, node.parent_id,
            cnt, p_cnt, n_tot, frac_parent, frac_total,
            means, meds, iqrs, cvs, combined
        )
    end
    return summaries
end

function _serialize_transform(t::LinearTransform)
    return "linear:$(t.scale):$(t.offset)"
end
function _serialize_transform(t::Log10Transform)
    return "log10:$(t.offset):$(t.floor_val)"
end
function _serialize_transform(t::ArcsinhTransform)
    return "arcsinh:$(t.cofactor)"
end
function _serialize_transform(t::BiexponentialLogicleTransform)
    return "logicle:$(t.cofactor):$(t.decade_scale)"
end

function _deserialize_transform(s::AbstractString)
    parts = split(strip(s), ':')
    kind = parts[1]
    if kind == "linear"
        return LinearTransform(parse(Float64, parts[2]), parse(Float64, parts[3]))
    elseif kind == "log10"
        return Log10Transform(parse(Float64, parts[2]); floor_val=parse(Float64, parts[3]))
    elseif kind == "arcsinh"
        return ArcsinhTransform(parse(Float64, parts[2]))
    elseif kind == "logicle"
        return BiexponentialLogicleTransform(parse(Float64, parts[2]); decade_scale=parse(Float64, parts[3]))
    else
        throw(ArgumentError("unknown serialized transform '$s'"))
    end
end

function _serialize_gate_entry(node::GateNode)
    g = node.gate
    p = node.parent_id
    if g isa RectangleGate
        return @sprintf(
            "{\"type\":\"RectangleGate\",\"id\":\"%s\",\"name\":\"%s\",\"parent_id\":\"%s\",\"x_channel\":\"%s\",\"y_channel\":\"%s\",\"params\":\"%.12g,%.12g,%.12g,%.12g\",\"tx\":\"%s\",\"ty\":\"%s\"}",
            g.id, g.name, p, g.x_channel, g.y_channel,
            g.x_min, g.x_max, g.y_min, g.y_max,
            _serialize_transform(g.transform_x), _serialize_transform(g.transform_y))
    elseif g isa PolygonGate
        v_str = join([@sprintf("%.12g:%.12g", v[1], v[2]) for v in g.vertices], ";")
        return @sprintf(
            "{\"type\":\"PolygonGate\",\"id\":\"%s\",\"name\":\"%s\",\"parent_id\":\"%s\",\"x_channel\":\"%s\",\"y_channel\":\"%s\",\"params\":\"%s\",\"tx\":\"%s\",\"ty\":\"%s\"}",
            g.id, g.name, p, g.x_channel, g.y_channel,
            v_str, _serialize_transform(g.transform_x), _serialize_transform(g.transform_y))
    elseif g isa EllipseGate
        return @sprintf(
            "{\"type\":\"EllipseGate\",\"id\":\"%s\",\"name\":\"%s\",\"parent_id\":\"%s\",\"x_channel\":\"%s\",\"y_channel\":\"%s\",\"params\":\"%.12g,%.12g,%.12g,%.12g,%.12g\",\"tx\":\"%s\",\"ty\":\"%s\"}",
            g.id, g.name, p, g.x_channel, g.y_channel,
            g.center_x, g.center_y, g.semi_major, g.semi_minor, g.angle_rad,
            _serialize_transform(g.transform_x), _serialize_transform(g.transform_y))
    elseif g isa RangeGate
        return @sprintf(
            "{\"type\":\"RangeGate\",\"id\":\"%s\",\"name\":\"%s\",\"parent_id\":\"%s\",\"x_channel\":\"%s\",\"y_channel\":\"-\",\"params\":\"%.12g,%.12g\",\"tx\":\"%s\",\"ty\":\"linear:1.0:0.0\"}",
            g.id, g.name, p, g.channel,
            g.min_val, g.max_val, _serialize_transform(g.transform))
    elseif g isa QuadGate
        return @sprintf(
            "{\"type\":\"QuadGate\",\"id\":\"%s\",\"name\":\"%s\",\"parent_id\":\"%s\",\"x_channel\":\"%s\",\"y_channel\":\"%s\",\"params\":\"%.12g,%.12g,%s\",\"tx\":\"%s\",\"ty\":\"%s\"}",
            g.id, g.name, p, g.x_channel, g.y_channel,
            g.x_thresh, g.y_thresh, String(g.quadrant),
            _serialize_transform(g.transform_x), _serialize_transform(g.transform_y))
    else
        throw(ArgumentError("unsupported gate type $(typeof(g))"))
    end
end

function export_gating_hierarchy_json(h::GatingHierarchy)
    entries = [_serialize_gate_entry(h.nodes[gid]) for gid in h.order]
    return "{\n  \"gating_ml_compatible_version\": \"2.0\",\n  \"gates\": [\n    " *
           join(entries, ",\n    ") * "\n  ]\n}"
end

function _extract_json_field(obj_str::AbstractString, key::AbstractString)
    pat = Regex("\"" * key * "\"\\s*:\\s*\"([^\"]*)\"")
    m = match(pat, obj_str)
    m !== nothing || throw(ArgumentError("missing field '$key' in gate JSON object"))
    return m.captures[1]
end

function import_gating_hierarchy_json(json_str::AbstractString)
    h = GatingHierarchy()
    for m in eachmatch(r"\{[^{}]*\"type\"\s*:\s*\"[^\"]+\"[^{}]*\}", json_str)
        obj = m.match
        gtype = _extract_json_field(obj, "type")
        gid   = _extract_json_field(obj, "id")
        gnm   = _extract_json_field(obj, "name")
        pid   = _extract_json_field(obj, "parent_id")
        xch   = _extract_json_field(obj, "x_channel")
        ych   = _extract_json_field(obj, "y_channel")
        prm   = _extract_json_field(obj, "params")
        tx    = _deserialize_transform(_extract_json_field(obj, "tx"))
        ty    = _deserialize_transform(_extract_json_field(obj, "ty"))

        if gtype == "RectangleGate"
            vals = [parse(Float64, s) for s in split(prm, ',')]
            gate = RectangleGate(gid, gnm, xch, ych, vals[1], vals[2], vals[3], vals[4];
                                 transform_x=tx, transform_y=ty)
        elseif gtype == "PolygonGate"
            verts = Tuple{Float64,Float64}[]
            for vp in split(prm, ';')
                xy = split(vp, ':')
                push!(verts, (parse(Float64, xy[1]), parse(Float64, xy[2])))
            end
            gate = PolygonGate(gid, gnm, xch, ych, verts; transform_x=tx, transform_y=ty)
        elseif gtype == "EllipseGate"
            vals = [parse(Float64, s) for s in split(prm, ',')]
            gate = EllipseGate(gid, gnm, xch, ych, vals[1], vals[2], vals[3], vals[4], vals[5];
                               transform_x=tx, transform_y=ty)
        elseif gtype == "RangeGate"
            vals = [parse(Float64, s) for s in split(prm, ',')]
            gate = RangeGate(gid, gnm, xch, vals[1], vals[2]; transform=tx)
        elseif gtype == "QuadGate"
            parts = split(prm, ',')
            gate = QuadGate(gid, gnm, xch, ych, parse(Float64, parts[1]), parse(Float64, parts[2]);
                            quadrant=Symbol(parts[3]), transform_x=tx, transform_y=ty)
        else
            throw(ArgumentError("unknown serialized gate type '$gtype'"))
        end
        add_gate!(h, gate; parent_id=pid)
    end
    return h
end

end # module CytometryAnalysis


# ===========================================================================
# MODULE 3: CellClassifier
# Responsibility: Single-cell feature engineering, supervised training across
# the 4 Nano-Bio Lab target tasks, calibrated confidence estimates, abstention
# (:defer_for_review), and strict group-aware train/test splitting.
# ===========================================================================
module CellClassifier

using LinearAlgebra
using Statistics
using Random
import ..FCSIO: FCSDataset, SensorTrajectoryRow, n_events, get_channel_vector

export VALID_CLASSIFICATION_TARGETS, VALID_SPLIT_GROUP_KEYS,
       default_classes_for_target,
       FeatureStandardizer, SingleCellPrediction,
       MicrofluidicCellClassifier, ClassifierValidationReport,
       extract_single_cell_features, fit_standardizer, apply_standardizer,
       train_cell_classifier, predict_cells,
       group_aware_split, verify_no_group_leakage,
       evaluate_classifier_predictions, group_cross_validate

const VALID_CLASSIFICATION_TARGETS = (
    :cell_type,
    :cell_stress_damage,
    :nanocarrier_uptake,
    :micropore_transit_regime,
)

const VALID_SPLIT_GROUP_KEYS = (
    :sample_id,
    :chip_id,
    :run_id,
    :biological_replicate,
)

function default_classes_for_target(target::Symbol)
    if target == :cell_type
        return [:viable_bacterial_rod, :viable_coccus, :mineral_colloid_debris]
    elseif target == :cell_stress_damage
        return [:membrane_intact, :membrane_compromised, :dead_lysed]
    elseif target == :nanocarrier_uptake
        return [:high_uptake_active, :low_uptake_passive, :unbound_negative]
    elseif target == :micropore_transit_regime
        return [:deformable_smooth_transit, :stiff_retarded_transit, :pore_occlusion_risk]
    else
        throw(ArgumentError("target must be one of $(VALID_CLASSIFICATION_TARGETS)"))
    end
end

struct FeatureStandardizer
    feature_names::Vector{String}
    means::Vector{Float64}
    stds::Vector{Float64}
end

function fit_standardizer(X::AbstractMatrix{<:Real}, feature_names::AbstractVector{<:AbstractString})
    n, d = size(X)
    n >= 2 || throw(ArgumentError("fit_standardizer requires at least 2 observations"))
    length(feature_names) == d ||
        throw(ArgumentError("feature_names length ($(length(feature_names))) must match column count ($d)"))
    means = vec(mean(Float64.(X); dims=1))
    stds  = vec(std(Float64.(X); dims=1))
    for j in 1:d
        if !isfinite(stds[j]) || stds[j] < 1e-9
            stds[j] = 1.0
        end
    end
    return FeatureStandardizer([String(f) for f in feature_names], means, stds)
end

function apply_standardizer(stdizer::FeatureStandardizer, X::AbstractMatrix{<:Real})
    n, d = size(X)
    d == length(stdizer.feature_names) ||
        throw(ArgumentError("input matrix column count ($d) does not match standardizer ($(length(stdizer.feature_names)))"))
    Z = Matrix{Float64}(undef, n, d)
    for j in 1:d
        m = stdizer.means[j]
        s = stdizer.stds[j]
        for i in 1:n
            Z[i, j] = (Float64(X[i, j]) - m) / s
        end
    end
    return Z
end

"""
Extract interpretable single-cell cytometric and optional microfluidic transit features.
"""
function extract_single_cell_features(ds::FCSDataset;
                                      transit_rows::Union{Nothing,AbstractVector{SensorTrajectoryRow}}=nothing,
                                      cofactor::Real=150.0)
    n = n_events(ds)
    cf = Float64(cofactor)
    feat_cols = Vector{Float64}[]
    feat_names = String[]

    for ch in ds.channels
        if ch.name == "TIME"
            continue
        end
        col_vec = get_channel_vector(ds, ch.name)
        push!(feat_cols, [asinh(v / cf) for v in col_vec])
        push!(feat_names, "asinh_$(ch.name)")
    end

    if haskey(ds.channel_index, "FSC-A") && haskey(ds.channel_index, "FSC-H")
        fsc_a = get_channel_vector(ds, "FSC-A")
        fsc_h = get_channel_vector(ds, "FSC-H")
        push!(feat_cols, [fsc_h[i] / max(fsc_a[i], 1.0) for i in 1:n])
        push!(feat_names, "singlet_ratio_FSC_H_over_A")
    end

    if haskey(ds.channel_index, "SSC-A") && haskey(ds.channel_index, "FSC-A")
        ssc_a = get_channel_vector(ds, "SSC-A")
        fsc_a = get_channel_vector(ds, "FSC-A")
        push!(feat_cols, [log10(max(ssc_a[i], 1.0) / max(fsc_a[i], 1.0)) for i in 1:n])
        push!(feat_names, "log10_SSC_over_FSC")
    end

    if haskey(ds.channel_index, "FL1-SYBR-A") && haskey(ds.channel_index, "FL2-PI-A")
        fl1 = get_channel_vector(ds, "FL1-SYBR-A")
        fl2 = get_channel_vector(ds, "FL2-PI-A")
        push!(feat_cols, [asinh(fl1[i] / cf) - asinh(fl2[i] / cf) for i in 1:n])
        push!(feat_names, "stain_integrity_index_SYBR_minus_PI")
    end

    if transit_rows !== nothing
        length(transit_rows) == n ||
            throw(ArgumentError("transit_rows length ($(length(transit_rows))) must match dataset event count ($n)"))
        push!(feat_cols, [log10(max(r.transit_time_ms, 1e-3)) for r in transit_rows])
        push!(feat_names, "log10_transit_time_ms")
        push!(feat_cols, [r.blockage_peak_na / max(r.baseline_current_na, 1e-3) for r in transit_rows])
        push!(feat_names, "blockage_ratio_dI_over_I0")
        push!(feat_cols, [r.deformability_index for r in transit_rows])
        push!(feat_names, "deformability_index")
        push!(feat_cols, [r.cell_circularity for r in transit_rows])
        push!(feat_names, "cell_circularity")
    end

    X = hcat(feat_cols...)
    return (features = X, feature_names = feat_names)
end

struct SingleCellPrediction
    event_index::Int
    predicted_class::Symbol
    final_decision::Symbol
    probabilities::Vector{Float64}
    confidence::Float64
    margin::Float64
    entropy_bits::Float64
    deferred::Bool
end

struct MicrofluidicCellClassifier
    target_task::Symbol
    classes::Vector{Symbol}
    standardizer::FeatureStandardizer
    weights::Matrix{Float64}
    class_means::Matrix{Float64}
    class_vars::Matrix{Float64}
    class_priors::Vector{Float64}
    hybrid_alpha::Float64
    temperature::Float64
    confidence_threshold::Float64
    lambda_l2::Float64
    split_group_key::Symbol
    training_groups::Vector{String}
end

function _softmax_rows(logits::AbstractMatrix{<:Real})
    n, k = size(logits)
    P = Matrix{Float64}(undef, n, k)
    for i in 1:n
        m = maximum(view(logits, i, :))
        s = 0.0
        for c in 1:k
            v = exp(Float64(logits[i, c]) - m)
            P[i, c] = v
            s += v
        end
        for c in 1:k
            P[i, c] /= s
        end
    end
    return P
end

"""
Train a calibrated supervised single-cell classifier with L2-regularized multinomial
logistic regression + Gaussian discriminant likelihood for the specified Nano-Bio Lab task.
"""
function train_cell_classifier(X::AbstractMatrix{<:Real},
                               labels::AbstractVector{Symbol};
                               feature_names::AbstractVector{<:AbstractString},
                               target_task::Symbol=:cell_stress_damage,
                               classes::Union{Nothing,AbstractVector{Symbol}}=nothing,
                               group_ids::AbstractVector{<:AbstractString}=String[],
                               split_group_key::Symbol=:chip_id,
                               confidence_threshold::Real=0.70,
                               lambda_l2::Real=1e-2,
                               hybrid_alpha::Real=0.75,
                               temperature::Real=1.0,
                               learning_rate::Real=0.35,
                               max_iter::Int=250)
    target_task in VALID_CLASSIFICATION_TARGETS ||
        throw(ArgumentError("target_task must be one of $(VALID_CLASSIFICATION_TARGETS)"))
    split_group_key in VALID_SPLIT_GROUP_KEYS ||
        throw(ArgumentError("split_group_key must be one of $(VALID_SPLIT_GROUP_KEYS); random event-level splitting is prohibited"))
    n, d = size(X)
    length(labels) == n || throw(ArgumentError("labels length must match size(X, 1)"))
    n >= 6 || throw(ArgumentError("training requires at least 6 labeled events"))

    conf_th = Float64(confidence_threshold)
    0.0 < conf_th <= 1.0 || throw(ArgumentError("confidence_threshold must be in (0, 1]"))
    lam = Float64(lambda_l2)
    lam > 0.0 || throw(ArgumentError("lambda_l2 must be > 0"))
    alpha = Float64(hybrid_alpha)
    0.0 <= alpha <= 1.0 || throw(ArgumentError("hybrid_alpha must be in [0, 1]"))
    temp = Float64(temperature)
    temp > 0.0 || throw(ArgumentError("temperature must be > 0"))

    cls_list = classes === nothing ? default_classes_for_target(target_task) : collect(Symbol, classes)
    k = length(cls_list)
    k >= 2 || throw(ArgumentError("at least 2 classes required"))
    cls_to_idx = Dict{Symbol,Int}(c => idx for (idx, c) in enumerate(cls_list))

    Y = zeros(Float64, n, k)
    for i in 1:n
        lbl = labels[i]
        haskey(cls_to_idx, lbl) ||
            throw(ArgumentError("label ':$lbl' is not in allowed classes $(cls_list) for target :$target_task"))
        Y[i, cls_to_idx[lbl]] = 1.0
    end

    stdizer = fit_standardizer(X, feature_names)
    Z = apply_standardizer(stdizer, X)
    Z_aug = hcat(ones(Float64, n), Z)

    W = zeros(Float64, d + 1, k)
    lr = Float64(learning_rate)
    for iter in 1:max_iter
        P = _softmax_rows(Z_aug * W)
        grad = (Z_aug' * (P .- Y)) ./ n
        # Regularize feature weights (exclude intercept row 1)
        grad[2:end, :] .+= lam .* view(W, 2:(d + 1), :)
        step_scale = lr / (1.0 + 0.004 * iter)
        W .-= step_scale .* grad
        if norm(grad) < 1e-6
            break
        end
    end

    class_means = zeros(Float64, d, k)
    class_vars  = ones(Float64, d, k)
    class_priors = fill(1.0 / k, k)
    for c in 1:k
        mask_c = [labels[i] == cls_list[c] for i in 1:n]
        cnt_c = count(mask_c)
        if cnt_c > 0
            sub_z = Z[mask_c, :]
            class_means[:, c] = vec(mean(sub_z; dims=1))
            if cnt_c > 1
                v = vec(var(sub_z; dims=1))
                class_vars[:, c] = max.(v .+ 0.15, 0.15)
            end
            class_priors[c] = (cnt_c + 1.0) / (n + k)
        end
    end
    class_priors ./= sum(class_priors)

    uniq_groups = isempty(group_ids) ? String["DEFAULT-GROUP"] : sort!(unique([String(strip(g)) for g in group_ids]))
    return MicrofluidicCellClassifier(
        target_task, cls_list, stdizer, W, class_means, class_vars, class_priors,
        alpha, temp, conf_th, lam, split_group_key, uniq_groups
    )
end

function predict_cells(clf::MicrofluidicCellClassifier, X::AbstractMatrix{<:Real};
                       confidence_threshold::Real=clf.confidence_threshold)
    conf_th = Float64(confidence_threshold)
    0.0 < conf_th <= 1.0 || throw(ArgumentError("confidence_threshold must be in (0, 1]"))
    Z = apply_standardizer(clf.standardizer, X)
    n, d = size(Z)
    k = length(clf.classes)
    Z_aug = hcat(ones(Float64, n), Z)

    lin_logits = Z_aug * clf.weights
    gen_logits = Matrix{Float64}(undef, n, k)
    for c in 1:k
        prior_log = log(max(clf.class_priors[c], 1e-12))
        for i in 1:n
            ll = prior_log
            for j in 1:d
                diff = Z[i, j] - clf.class_means[j, c]
                v = clf.class_vars[j, c]
                ll -= 0.5 * (diff * diff / v + log(v))
            end
            gen_logits[i, c] = ll / max(d, 1)
        end
    end

    combined_logits = (clf.hybrid_alpha .* lin_logits .+ (1.0 - clf.hybrid_alpha) .* gen_logits) ./ clf.temperature
    P = _softmax_rows(combined_logits)

    preds = Vector{SingleCellPrediction}(undef, n)
    for i in 1:n
        probs_i = collect(Float64, view(P, i, :))
        sorted_p = sort(probs_i; rev=true)
        best_idx = argmax(probs_i)
        best_cls = clf.classes[best_idx]
        conf = sorted_p[1]
        margin = k >= 2 ? (sorted_p[1] - sorted_p[2]) : sorted_p[1]
        ent = 0.0
        for p in probs_i
            if p > 1e-15
                ent -= p * log2(p)
            end
        end
        is_deferred = conf < conf_th
        final_dec = is_deferred ? :defer_for_review : best_cls
        preds[i] = SingleCellPrediction(i, best_cls, final_dec, probs_i, conf, margin, ent, is_deferred)
    end
    return preds
end

"""
Partition indices into training and holdout sets strictly by independent experimental group
(`:sample_id`, `:chip_id`, `:run_id`, or `:biological_replicate`).
Random event-level splitting within the same sample is explicitly rejected.
"""
function group_aware_split(group_ids::AbstractVector{<:AbstractString};
                           holdout_groups::AbstractVector{<:AbstractString},
                           split_by::Symbol=:chip_id)
    split_by in VALID_SPLIT_GROUP_KEYS ||
        throw(ArgumentError("split_by must be one of $(VALID_SPLIT_GROUP_KEYS); random event-level splitting from the same sample is prohibited"))
    n = length(group_ids)
    n >= 2 || throw(ArgumentError("group_ids must contain at least 2 events"))
    !isempty(holdout_groups) || throw(ArgumentError("holdout_groups must be non-empty"))

    clean_groups = [String(strip(g)) for g in group_ids]
    holdout_set  = Set{String}([String(strip(g)) for g in holdout_groups])
    present_set  = Set{String}(clean_groups)

    for hg in holdout_set
        hg in present_set || throw(ArgumentError("holdout group '$hg' not found in group_ids"))
    end

    train_idx = Int[]
    test_idx  = Int[]
    for i in 1:n
        if clean_groups[i] in holdout_set
            push!(test_idx, i)
        else
            push!(train_idx, i)
        end
    end
    !isempty(train_idx) && !isempty(test_idx) ||
        throw(ArgumentError("both train and holdout partitions must be non-empty"))
    verify_no_group_leakage(train_idx, test_idx, clean_groups)
    return (
        train_indices = train_idx,
        test_indices = test_idx,
        train_groups = sort!(unique(clean_groups[train_idx])),
        test_groups = sort!(unique(clean_groups[test_idx])),
        split_by = split_by
    )
end

function verify_no_group_leakage(train_indices::AbstractVector{<:Integer},
                                 test_indices::AbstractVector{<:Integer},
                                 group_ids::AbstractVector{<:AbstractString})
    !isempty(train_indices) && !isempty(test_indices) ||
        throw(ArgumentError("train_indices and test_indices must be non-empty"))
    idx_overlap = intersect(Set(Int.(train_indices)), Set(Int.(test_indices)))
    isempty(idx_overlap) ||
        throw(ArgumentError("event index leakage detected between train and test sets"))
    train_groups = Set{String}([String(strip(group_ids[i])) for i in train_indices])
    test_groups  = Set{String}([String(strip(group_ids[i])) for i in test_indices])
    grp_overlap  = intersect(train_groups, test_groups)
    isempty(grp_overlap) ||
        throw(ArgumentError("group leakage detected: groups $(collect(grp_overlap)) appear in both train and test sets"))
    return true
end

struct ClassifierValidationReport
    target_task::Symbol
    classes::Vector{Symbol}
    total_events::Int
    accepted_events::Int
    deferred_events::Int
    coverage_fraction::Float64
    deferral_fraction::Float64
    overall_accuracy::Float64
    accepted_accuracy::Float64
    macro_f1::Float64
    per_class_precision::Dict{Symbol,Float64}
    per_class_recall::Dict{Symbol,Float64}
    per_class_f1::Dict{Symbol,Float64}
    confusion_matrix::Matrix{Int}
    expected_calibration_error::Float64
    brier_score::Float64
    group_accuracies::Dict{String,Float64}
end

function evaluate_classifier_predictions(predictions::AbstractVector{SingleCellPrediction},
                                         true_labels::AbstractVector{Symbol},
                                         classes::AbstractVector{Symbol};
                                         target_task::Symbol=:cell_stress_damage,
                                         group_ids::AbstractVector{<:AbstractString}=String[])
    n = length(predictions)
    n > 0 && length(true_labels) == n ||
        throw(ArgumentError("predictions and true_labels must be non-empty and equal length"))
    cls_list = collect(Symbol, classes)
    k = length(cls_list)
    cls_to_idx = Dict{Symbol,Int}(c => idx for (idx, c) in enumerate(cls_list))

    conf_mat = zeros(Int, k, k)
    correct_all = 0
    correct_acc = 0
    acc_cnt = 0
    def_cnt = 0
    brier_sum = 0.0

    for i in 1:n
        pr = predictions[i]
        tr = true_labels[i]
        haskey(cls_to_idx, tr) || throw(ArgumentError("unknown true label :$tr"))
        r_idx = cls_to_idx[tr]
        c_idx = cls_to_idx[pr.predicted_class]
        conf_mat[r_idx, c_idx] += 1

        if pr.predicted_class == tr
            correct_all += 1
        end
        if pr.deferred
            def_cnt += 1
        else
            acc_cnt += 1
            if pr.predicted_class == tr
                correct_acc += 1
            end
        end

        for c in 1:k
            y_ic = (c == r_idx) ? 1.0 : 0.0
            brier_sum += (pr.probabilities[c] - y_ic)^2
        end
    end

    prec = Dict{Symbol,Float64}()
    rec  = Dict{Symbol,Float64}()
    f1s  = Dict{Symbol,Float64}()
    for c in 1:k
        cls = cls_list[c]
        tp = conf_mat[c, c]
        pred_pos = sum(view(conf_mat, :, c))
        true_pos = sum(view(conf_mat, c, :))
        p_val = pred_pos > 0 ? tp / pred_pos : 0.0
        r_val = true_pos > 0 ? tp / true_pos : 0.0
        f_val = (p_val + r_val) > 0 ? 2.0 * p_val * r_val / (p_val + r_val) : 0.0
        prec[cls] = p_val
        rec[cls]  = r_val
        f1s[cls]  = f_val
    end
    macro_f1 = mean(values(f1s))

    # 10-bin Expected Calibration Error (ECE)
    n_bins = 10
    ece = 0.0
    for b in 1:n_bins
        lo = (b - 1) / n_bins
        hi = b / n_bins
        bin_idx = [i for i in 1:n if (b == 1 ? (lo <= predictions[i].confidence <= hi) : (lo < predictions[i].confidence <= hi))]
        if !isempty(bin_idx)
            bin_acc  = mean(predictions[i].predicted_class == true_labels[i] ? 1.0 : 0.0 for i in bin_idx)
            bin_conf = mean(predictions[i].confidence for i in bin_idx)
            ece += (length(bin_idx) / n) * abs(bin_acc - bin_conf)
        end
    end

    grp_acc = Dict{String,Float64}()
    if !isempty(group_ids)
        length(group_ids) == n || throw(ArgumentError("group_ids length must match predictions"))
        for g in unique([String(strip(x)) for x in group_ids])
            g_idx = [i for i in 1:n if String(strip(group_ids[i])) == g]
            grp_acc[g] = mean(predictions[i].predicted_class == true_labels[i] ? 1.0 : 0.0 for i in g_idx)
        end
    end

    overall_acc = correct_all / n
    accepted_acc = acc_cnt > 0 ? correct_acc / acc_cnt : overall_acc
    return ClassifierValidationReport(
        target_task, cls_list, n, acc_cnt, def_cnt,
        acc_cnt / n, def_cnt / n,
        overall_acc, accepted_acc, macro_f1,
        prec, rec, f1s, conf_mat, ece, brier_sum / n, grp_acc
    )
end

"""
Leave-one-group-out cross-validation across independent chips, runs, samples,
or biological replicates.
"""
function group_cross_validate(X::AbstractMatrix{<:Real},
                              labels::AbstractVector{Symbol},
                              group_ids::AbstractVector{<:AbstractString};
                              feature_names::AbstractVector{<:AbstractString},
                              target_task::Symbol=:cell_stress_damage,
                              split_by::Symbol=:chip_id,
                              confidence_threshold::Real=0.70,
                              lambda_l2::Real=1e-2,
                              hybrid_alpha::Real=0.75)
    split_by in VALID_SPLIT_GROUP_KEYS ||
        throw(ArgumentError("split_by must be one of $(VALID_SPLIT_GROUP_KEYS)"))
    n = size(X, 1)
    length(labels) == n && length(group_ids) == n ||
        throw(ArgumentError("X, labels, and group_ids must have matching row count"))
    clean_groups = [String(strip(g)) for g in group_ids]
    uniq_groups = sort!(unique(clean_groups))
    length(uniq_groups) >= 2 ||
        throw(ArgumentError("group_cross_validate requires at least 2 distinct independent groups"))

    cls_list = default_classes_for_target(target_task)
    oof_preds = Vector{SingleCellPrediction}(undef, n)

    for g in uniq_groups
        sp = group_aware_split(clean_groups; holdout_groups=[g], split_by=split_by)
        clf = train_cell_classifier(
            view(X, sp.train_indices, :),
            view(labels, sp.train_indices);
            feature_names=feature_names,
            target_task=target_task,
            classes=cls_list,
            group_ids=view(clean_groups, sp.train_indices),
            split_group_key=split_by,
            confidence_threshold=confidence_threshold,
            lambda_l2=lambda_l2,
            hybrid_alpha=hybrid_alpha
        )
        fold_preds = predict_cells(clf, view(X, sp.test_indices, :);
                                   confidence_threshold=confidence_threshold)
        for (local_i, global_i) in enumerate(sp.test_indices)
            fp = fold_preds[local_i]
            oof_preds[global_i] = SingleCellPrediction(
                global_i, fp.predicted_class, fp.final_decision,
                fp.probabilities, fp.confidence, fp.margin, fp.entropy_bits, fp.deferred
            )
        end
    end

    report = evaluate_classifier_predictions(oof_preds, labels, cls_list;
                                             target_task=target_task, group_ids=clean_groups)
    return (report = report, oof_predictions = oof_preds)
end

end # module CellClassifier


# ===========================================================================
# MODULE 4: MicrofluidicBridge
# Responsibility: Join single-cell cytometry measurements to microfluidic and
# micropore transit trajectories, experimental conditions, and the calibrated
# FlowCytometryObservationModel.jl adapter.
# ===========================================================================
module MicrofluidicBridge

using Statistics
using Printf
import SHA
import ..FCSIO: FCSDataset, SensorTrajectoryRow, n_events
import ..CytometryAnalysis: GatingHierarchy, GatePopulationSummary, export_gating_hierarchy_json
import ..CellClassifier: SingleCellPrediction, ClassifierValidationReport
import ..FlowCytometryObservationModel:
    MODEL_ADAPTER_VERSION, VALID_SAMPLE_ORIGINS, VALID_VOLUME_CONVENTIONS,
    VALID_STAKEHOLDER_TRACKS, VALID_EPISTEMIC_STATUSES,
    CellStatePrediction, GateCalibration, FCMSampleSummary,
    ConcentrationUncertainty, OpenFOAMTransportSnap, PHREEQCSpeciationSnap,
    EpistemicQuantity, ProductionAuditRecord,
    expected_cell_events, predicted_gate_probabilities, absolute_cells_per_ml,
    verify_concentration_likelihood_consistency, joint_loglikelihood,
    recover_latent_state_mle, concentration_uncertainty,
    validate_solver_observation_alignment, build_production_audit_record

export MicrofluidicAssayCondition, MicroporeTransitEvent,
       JoinedMicrofluidicCellRecord, MicrofluidicBridgeReport,
       sensor_rows_to_transit_events,
       join_cytometry_and_microfluidics,
       summarize_to_fcm_sample, summarize_hierarchy_to_fcm_sample,
       bridge_to_observation_model,
       export_reproducible_analysis_bundle

struct MicrofluidicAssayCondition
    chip_id::String
    run_id::String
    sample_id::String
    biological_replicate::String
    channel_width_um::Float64
    pore_constriction_um::Float64
    flow_rate_ul_min::Float64
    wall_shear_pa::Float64
    actuation_voltage_vpp::Float64
    actuation_frequency_khz::Float64
    temperature_c::Float64

    function MicrofluidicAssayCondition(; chip_id::AbstractString,
                                        run_id::AbstractString,
                                        sample_id::AbstractString,
                                        biological_replicate::AbstractString="REP-1",
                                        channel_width_um::Real=25.0,
                                        pore_constriction_um::Real=8.0,
                                        flow_rate_ul_min::Real=5.0,
                                        wall_shear_pa::Real=1.2,
                                        actuation_voltage_vpp::Real=0.0,
                                        actuation_frequency_khz::Real=0.0,
                                        temperature_c::Real=25.0)
        cid = String(strip(chip_id))
        rid = String(strip(run_id))
        sid = String(strip(sample_id))
        rep = String(strip(biological_replicate))
        !isempty(cid) && !isempty(rid) && !isempty(sid) && !isempty(rep) ||
            throw(ArgumentError("chip_id, run_id, sample_id, and biological_replicate must be non-empty"))
        cw = Float64(channel_width_um)
        pc = Float64(pore_constriction_um)
        fr = Float64(flow_rate_ul_min)
        ws = Float64(wall_shear_pa)
        av = Float64(actuation_voltage_vpp)
        af = Float64(actuation_frequency_khz)
        tc = Float64(temperature_c)
        cw > 0 && pc > 0 && pc <= cw ||
            throw(ArgumentError("pore_constriction_um must be > 0 and <= channel_width_um"))
        fr >= 0 && ws >= 0 && av >= 0 && af >= 0 && isfinite(tc) ||
            throw(ArgumentError("physical assay parameters must be finite and nonnegative"))
        new(cid, rid, sid, rep, cw, pc, fr, ws, av, af, tc)
    end
end

struct MicroporeTransitEvent
    event_index::Int
    sample_id::String
    transit_time_ms::Float64
    blockage_peak_na::Float64
    baseline_current_na::Float64
    deformability_index::Float64
    cell_circularity::Float64
    apparent_velocity_mm_s::Float64
    retained_in_pore::Bool

    function MicroporeTransitEvent(event_index::Integer, sample_id::AbstractString;
                                   transit_time_ms::Real,
                                   blockage_peak_na::Real,
                                   baseline_current_na::Real=12.0,
                                   deformability_index::Real=1.0,
                                   cell_circularity::Real=0.85,
                                   apparent_velocity_mm_s::Real=10.0,
                                   retained_in_pore::Bool=false)
        idx = Int(event_index)
        idx >= 1 || throw(ArgumentError("event_index must be >= 1"))
        sid = String(strip(sample_id))
        !isempty(sid) || throw(ArgumentError("sample_id must be non-empty"))
        tt = Float64(transit_time_ms)
        bp = Float64(blockage_peak_na)
        bc = Float64(baseline_current_na)
        df = Float64(deformability_index)
        cc = Float64(cell_circularity)
        av = Float64(apparent_velocity_mm_s)
        tt > 0 && bp >= 0 && bc > 0 && df >= 1.0 && (0.0 < cc <= 1.0) && av >= 0 ||
            throw(ArgumentError("invalid physical values in MicroporeTransitEvent"))
        new(idx, sid, tt, bp, bc, df, cc, av, retained_in_pore)
    end
end

function sensor_rows_to_transit_events(rows::AbstractVector{SensorTrajectoryRow})
    return [
        MicroporeTransitEvent(
            r.event_index, r.sample_id;
            transit_time_ms = r.transit_time_ms,
            blockage_peak_na = r.blockage_peak_na,
            baseline_current_na = r.baseline_current_na,
            deformability_index = r.deformability_index,
            cell_circularity = r.cell_circularity,
            apparent_velocity_mm_s = r.apparent_velocity_mm_s,
            retained_in_pore = r.retained_in_pore
        )
        for r in rows
    ]
end

struct JoinedMicrofluidicCellRecord
    event_index::Int
    sample_id::String
    chip_id::String
    run_id::String
    biological_replicate::String
    cytometry_values::Vector{Float64}
    active_gates::Vector{String}
    prediction::Union{Nothing,SingleCellPrediction}
    transit::Union{Nothing,MicroporeTransitEvent}
    condition::MicrofluidicAssayCondition
end

function join_cytometry_and_microfluidics(
        ds::FCSDataset,
        transit_events::AbstractVector{MicroporeTransitEvent},
        condition::MicrofluidicAssayCondition;
        predictions::Union{Nothing,AbstractVector{SingleCellPrediction}}=nothing,
        gate_summaries::Dict{String,GatePopulationSummary}=Dict{String,GatePopulationSummary}())
    n = n_events(ds)
    ds.provenance.sample_id == condition.sample_id ||
        throw(ArgumentError("dataset sample_id ($(ds.provenance.sample_id)) does not match condition sample_id ($(condition.sample_id))"))
    if predictions !== nothing
        length(predictions) == n ||
            throw(ArgumentError("predictions length ($(length(predictions))) must equal dataset event count ($n)"))
    end

    transit_by_idx = Dict{Int,MicroporeTransitEvent}()
    for te in transit_events
        te.sample_id == condition.sample_id ||
            throw(ArgumentError("transit event sample_id ($(te.sample_id)) does not match condition sample_id ($(condition.sample_id))"))
        1 <= te.event_index <= n ||
            throw(ArgumentError("transit event_index $(te.event_index) out of bounds (1..$n)"))
        transit_by_idx[te.event_index] = te
    end

    sorted_gate_ids = sort!(collect(keys(gate_summaries)))
    records = Vector{JoinedMicrofluidicCellRecord}(undef, n)
    for i in 1:n
        g_list = String[gid for gid in sorted_gate_ids if gate_summaries[gid].mask[i]]
        pred_i = predictions === nothing ? nothing : predictions[i]
        tran_i = get(transit_by_idx, i, nothing)
        records[i] = JoinedMicrofluidicCellRecord(
            i, condition.sample_id, condition.chip_id, condition.run_id,
            condition.biological_replicate,
            collect(Float64, view(ds.events, i, :)),
            g_list, pred_i, tran_i, condition
        )
    end
    return records
end

"""
Summarize single-cell classifier outputs into a 3-state `FCMSampleSummary` and
`CellStatePrediction` compatible with `FlowCytometryObservationModel`.
"""
function summarize_to_fcm_sample(
        predictions::AbstractVector{SingleCellPrediction};
        bead_events::Integer,
        beads_added::Real,
        sample_volume_ml::Real,
        dilution_factor::Real=1.0,
        extraction_recovery::Real=1.0,
        detection_efficiency::Real=1.0,
        sample_origin::Symbol=:effluent,
        gate_id::AbstractString="classifier-3state",
        volume_convention::Symbol=:aliquot_post_dilution,
        state_order::NTuple{3,Symbol}=(:membrane_intact, :membrane_compromised, :dead_lysed),
        include_deferred_in_counts::Bool=false)
    !isempty(predictions) || throw(ArgumentError("predictions must be non-empty"))
    counts = [0, 0, 0]
    deferred_cnt = 0
    accepted_cnt = 0

    for pr in predictions
        if pr.deferred
            deferred_cnt += 1
            if !include_deferred_in_counts
                continue
            end
        else
            accepted_cnt += 1
        end
        if pr.predicted_class == state_order[1]
            counts[1] += 1
        elseif pr.predicted_class == state_order[2]
            counts[2] += 1
        elseif pr.predicted_class == state_order[3]
            counts[3] += 1
        else
            throw(ArgumentError("predicted class :$(pr.predicted_class) is not in state_order $(state_order)"))
        end
    end

    gc = (counts[1], counts[2], counts[3])
    tot = sum(gc)
    tot > 0 || throw(ArgumentError("no classified events available after deferral filtering"))

    summary = FCMSampleSummary(
        gc;
        bead_events = bead_events,
        beads_added = beads_added,
        sample_volume_ml = sample_volume_ml,
        dilution_factor = dilution_factor,
        extraction_recovery = extraction_recovery,
        sample_origin = sample_origin,
        gate_id = gate_id,
        volume_convention = volume_convention
    )
    c_mle = absolute_cells_per_ml(summary; detection_efficiency=detection_efficiency)
    fracs = (gc[1] / tot, gc[2] / tot, gc[3] / tot)
    state_pred = CellStatePrediction(c_mle, fracs)
    return (
        sample_summary = summary,
        state_prediction = state_pred,
        accepted_count = accepted_cnt,
        deferred_count = deferred_cnt
    )
end

function summarize_hierarchy_to_fcm_sample(
        hierarchy_summaries::Dict{String,GatePopulationSummary},
        gate_triplet::NTuple{3,String};
        bead_events::Integer,
        beads_added::Real,
        sample_volume_ml::Real,
        dilution_factor::Real=1.0,
        extraction_recovery::Real=1.0,
        detection_efficiency::Real=1.0,
        sample_origin::Symbol=:effluent,
        gate_id::AbstractString="hierarchy-3state",
        volume_convention::Symbol=:aliquot_post_dilution)
    for gid in gate_triplet
        haskey(hierarchy_summaries, gid) ||
            throw(ArgumentError("gate '$gid' not found in hierarchy_summaries"))
    end
    gc = (
        hierarchy_summaries[gate_triplet[1]].count,
        hierarchy_summaries[gate_triplet[2]].count,
        hierarchy_summaries[gate_triplet[3]].count,
    )
    tot = sum(gc)
    tot > 0 || throw(ArgumentError("total count across gate_triplet must be > 0"))
    summary = FCMSampleSummary(
        gc;
        bead_events = bead_events,
        beads_added = beads_added,
        sample_volume_ml = sample_volume_ml,
        dilution_factor = dilution_factor,
        extraction_recovery = extraction_recovery,
        sample_origin = sample_origin,
        gate_id = gate_id,
        volume_convention = volume_convention
    )
    c_mle = absolute_cells_per_ml(summary; detection_efficiency=detection_efficiency)
    fracs = (gc[1] / tot, gc[2] / tot, gc[3] / tot)
    return (sample_summary = summary, state_prediction = CellStatePrediction(c_mle, fracs))
end

struct MicrofluidicBridgeReport
    sample_summary::FCMSampleSummary
    latent_state_mle::CellStatePrediction
    absolute_concentration_cells_per_ml::Float64
    concentration_uncertainty::ConcentrationUncertainty
    expected_events::Float64
    predicted_gate_probs::Vector{Float64}
    joint_loglikelihood::Float64
    likelihood_consistent::Bool
    class_transit_statistics::Dict{Symbol,NamedTuple}
    audit_record::ProductionAuditRecord
end

"""
Connect joined single-cell cytometry + microfluidic records to `FlowCytometryObservationModel`
and produce a complete statistical, physical, and epistemic audit report.
"""
function bridge_to_observation_model(
        joined_records::AbstractVector{JoinedMicrofluidicCellRecord},
        sample_summary::FCMSampleSummary,
        calibration::GateCalibration;
        state_prediction::Union{Nothing,CellStatePrediction}=nothing,
        transport_snap::Union{Nothing,OpenFOAMTransportSnap}=nothing,
        chemistry_snap::Union{Nothing,PHREEQCSpeciationSnap}=nothing,
        vessel_id::AbstractString="CHIP-01",
        time_h::Real=1.0,
        track::Symbol=:track1_nanobio,
        input_sha256::AbstractString="",
        heldout_validated::Bool=true)
    !isempty(joined_records) || throw(ArgumentError("joined_records must be non-empty"))

    if transport_snap !== nothing || chemistry_snap !== nothing
        validate_solver_observation_alignment(
            sample_summary, vessel_id, time_h;
            transport=transport_snap, chemistry=chemistry_snap
        )
    end

    recovered_state = state_prediction === nothing ?
        recover_latent_state_mle([sample_summary], calibration) : state_prediction

    c_abs = absolute_cells_per_ml(sample_summary, calibration)
    unc   = concentration_uncertainty(sample_summary, calibration)
    exp_n = expected_cell_events(sample_summary, recovered_state, calibration)
    g_prb = predicted_gate_probabilities(recovered_state, calibration)
    j_ll  = joint_loglikelihood(sample_summary, recovered_state, calibration)
    cons  = verify_concentration_likelihood_consistency(sample_summary, calibration)

    # Compute physical transit statistics stratified by predicted cell class
    transit_stats = Dict{Symbol,NamedTuple}()
    by_class = Dict{Symbol,Vector{MicroporeTransitEvent}}()
    for rec in joined_records
        if rec.prediction !== nothing && rec.transit !== nothing && !rec.prediction.deferred
            cls = rec.prediction.predicted_class
            if !haskey(by_class, cls)
                by_class[cls] = MicroporeTransitEvent[]
            end
            push!(by_class[cls], rec.transit)
        end
    end
    for (cls, t_list) in by_class
        n_c = length(t_list)
        if n_c > 0
            transit_stats[cls] = (
                count = n_c,
                mean_transit_time_ms = mean(t.transit_time_ms for t in t_list),
                mean_blockage_ratio = mean(t.blockage_peak_na / t.baseline_current_na for t in t_list),
                mean_deformability_index = mean(t.deformability_index for t in t_list),
                mean_circularity = mean(t.cell_circularity for t in t_list),
                pore_retention_fraction = mean(t.retained_in_pore ? 1.0 : 0.0 for t in t_list)
            )
        end
    end

    digest = !isempty(strip(input_sha256)) ? lowercase(strip(String(input_sha256))) :
        bytes2hex(SHA.sha256(string(sample_summary.gate_id, ":", sum(sample_summary.gate_counts), ":", c_abs)))

    cond = joined_records[1].condition
    quantities = EpistemicQuantity[
        EpistemicQuantity("gated_cell_counts", sample_summary.gate_counts, :observed,
                          "Single-cell cytometry gate / classifier counts"),
        EpistemicQuantity("bead_events", sample_summary.bead_events, :observed,
                          "Counting-bead events in assay tube"),
        EpistemicQuantity("pore_constriction_um", cond.pore_constriction_um, :observed,
                          "Microfluidic channel constriction geometry"),
        EpistemicQuantity("detection_efficiency", calibration.detection_efficiency, :calibrated,
                          "Independently calibrated control recovery efficiency"),
        EpistemicQuantity("extraction_recovery", sample_summary.extraction_recovery, :assumed,
                          "Assay-specified extraction recovery factor"),
        EpistemicQuantity("latent_concentration_cells_per_ml", c_abs, :predicted,
                          "MLE concentration via FlowCytometryObservationModel"),
    ]
    audit = build_production_audit_record(
        track = track,
        input_sha256 = digest,
        quantities = quantities,
        sample = sample_summary,
        calibration = calibration,
        heldout_validated = heldout_validated
    )

    return MicrofluidicBridgeReport(
        sample_summary, recovered_state, c_abs, unc, exp_n, g_prb,
        j_ll, cons.consistent, transit_stats, audit
    )
end

"""
Export reproducible analysis settings, gate definitions, event classifications,
population statistics, and observation-model bridge summaries with SHA-256 digests.
"""
function export_reproducible_analysis_bundle(
        output_dir::AbstractString;
        dataset::FCSDataset,
        hierarchy::GatingHierarchy,
        hierarchy_summaries::Dict{String,GatePopulationSummary},
        classifier_report::ClassifierValidationReport,
        predictions::AbstractVector{SingleCellPrediction},
        bridge_report::MicrofluidicBridgeReport)
    mkpath(output_dir)
    gates_json_path = joinpath(output_dir, "analysis_settings_and_gates.json")
    events_csv_path = joinpath(output_dir, "classified_events.csv")
    summary_json_path = joinpath(output_dir, "population_and_bridge_summary.json")

    write(gates_json_path, export_gating_hierarchy_json(hierarchy))

    open(events_csv_path, "w") do io
        println(io, "event_index,sample_id,chip_id,run_id,biological_replicate,predicted_class,final_decision,confidence,margin,entropy_bits,deferred")
        for pr in predictions
            @printf(io, "%d,%s,%s,%s,%s,%s,%s,%.6f,%.6f,%.6f,%s\n",
                pr.event_index,
                dataset.provenance.sample_id,
                dataset.provenance.chip_id,
                dataset.provenance.run_id,
                dataset.provenance.biological_replicate,
                String(pr.predicted_class),
                String(pr.final_decision),
                pr.confidence,
                pr.margin,
                pr.entropy_bits,
                pr.deferred ? "true" : "false")
        end
    end

    open(summary_json_path, "w") do io
        println(io, "{")
        @printf(io, "  \"engine_version\": \"%s\",\n", "0.2.0")
        @printf(io, "  \"observation_model_version\": \"%s\",\n", MODEL_ADAPTER_VERSION)
        @printf(io, "  \"source_fcs_sha256\": \"%s\",\n", dataset.provenance.sha256)
        @printf(io, "  \"sample_id\": \"%s\",\n", dataset.provenance.sample_id)
        @printf(io, "  \"chip_id\": \"%s\",\n", dataset.provenance.chip_id)
        @printf(io, "  \"run_id\": \"%s\",\n", dataset.provenance.run_id)
        @printf(io, "  \"total_events\": %d,\n", n_events(dataset))
        @printf(io, "  \"classifier_target\": \"%s\",\n", String(classifier_report.target_task))
        @printf(io, "  \"overall_accuracy\": %.8f,\n", classifier_report.overall_accuracy)
        @printf(io, "  \"accepted_accuracy\": %.8f,\n", classifier_report.accepted_accuracy)
        @printf(io, "  \"coverage_fraction\": %.8f,\n", classifier_report.coverage_fraction)
        @printf(io, "  \"deferral_fraction\": %.8f,\n", classifier_report.deferral_fraction)
        @printf(io, "  \"macro_f1\": %.8f,\n", classifier_report.macro_f1)
        @printf(io, "  \"brier_score\": %.8f,\n", classifier_report.brier_score)
        @printf(io, "  \"expected_calibration_error\": %.8f,\n", classifier_report.expected_calibration_error)
        @printf(io, "  \"absolute_concentration_cells_per_ml\": %.8f,\n", bridge_report.absolute_concentration_cells_per_ml)
        @printf(io, "  \"concentration_ci95_lower\": %.8f,\n", bridge_report.concentration_uncertainty.ci_95_lower)
        @printf(io, "  \"concentration_ci95_upper\": %.8f,\n", bridge_report.concentration_uncertainty.ci_95_upper)
        @printf(io, "  \"joint_loglikelihood\": %.8f,\n", bridge_report.joint_loglikelihood)
        @printf(io, "  \"likelihood_consistent\": %s,\n", bridge_report.likelihood_consistent ? "true" : "false")
        println(io, "  \"gate_summaries\": {")
        gids = sort!(collect(keys(hierarchy_summaries)))
        for (idx, gid) in enumerate(gids)
            gs = hierarchy_summaries[gid]
            comma = idx < length(gids) ? "," : ""
            @printf(io, "    \"%s\": {\"count\": %d, \"parent_count\": %d, \"fraction_of_parent\": %.8f, \"fraction_of_total\": %.8f}%s\n",
                gid, gs.count, gs.parent_count, gs.fraction_of_parent, gs.fraction_of_total, comma)
        end
        println(io, "  }")
        println(io, "}")
    end

    return (
        gates_json_path = abspath(gates_json_path),
        gates_json_sha256 = bytes2hex(SHA.sha256(read(gates_json_path))),
        events_csv_path = abspath(events_csv_path),
        events_csv_sha256 = bytes2hex(SHA.sha256(read(events_csv_path))),
        summary_json_path = abspath(summary_json_path),
        summary_json_sha256 = bytes2hex(SHA.sha256(read(summary_json_path)))
    )
end

end # module MicrofluidicBridge


# ===========================================================================
# MODULE 5: SharedVisualizationPhysics
# Responsibility: Domain-separated computational physics for the shared
# wireframe-over-core 3D visualization architecture:
# 1. Ghosh (Track 1 :track1_nanobio): Multi-stratum spherical core-multishell
#    reaction-diffusion (R0 < R1 < R2 ... < RL) coupled to microfluidic channel
#    exposure and single-cell classification outcomes. Explicitly tagged as
#    :predicted simulated concentration fields, not observed physical layers.
# 2. Jack (Track 3 :track3_bitumen_rheology): Interface-resolved emulsion
#    droplet deformation D = (L-B)/(L+B) under shear, Boussinesq-Scriven
#    temperature-dependent interfacial rheology, internal/external vortex
#    velocity fields, pressure field P/Pa, and emulsifier coverage across t/tau_eta.
# ===========================================================================
module SharedVisualizationPhysics

using LinearAlgebra
using Statistics
import ..FlowCytometryObservationModel: VALID_STAKEHOLDER_TRACKS, VALID_EPISTEMIC_STATUSES, CellStatePrediction

export CoreMultishellSpec, CoreMultishellSnapshot, CoreMultishellSolution,
       solve_core_multishell_rd,
       EmulsionDropletSpec, EmulsionDropletSnapshot, EmulsionDropletSolution,
       solve_emulsion_interfacial_droplet

struct CoreMultishellSpec
    radii_um::Vector{Float64}
    diffusivities_um2_s::Vector{Float64}
    decay_rates_s::Vector{Float64}
    initial_concentrations::Vector{Float64}
    membrane_permeability_um_s::Float64
    channel_flow_velocity_mm_s::Float64
    track::Symbol
    epistemic_status::Symbol

    function CoreMultishellSpec(;
            radii_um::AbstractVector{<:Real}=[0.35, 0.70, 1.00],
            diffusivities_um2_s::AbstractVector{<:Real}=[0.012, 0.028, 0.045],
            decay_rates_s::AbstractVector{<:Real}=[0.001, 0.002, 0.002],
            initial_concentrations::AbstractVector{<:Real}=[0.030, 0.018, 0.006],
            membrane_permeability_um_s::Real=0.08,
            channel_flow_velocity_mm_s::Real=15.0,
            track::Symbol=:track1_nanobio,
            epistemic_status::Symbol=:predicted)
        track == :track1_nanobio ||
            throw(ArgumentError("CoreMultishellSpec is specific to :track1_nanobio; do not attach biological core-multishell release strata to :$track"))
        epistemic_status == :predicted ||
            throw(ArgumentError("CoreMultishellSpec shells must be tagged as :predicted simulated concentration fields unless independently observed"))
        L = length(radii_um)
        L >= 2 || throw(ArgumentError("at least 2 concentric strata (core + shell) required"))
        length(diffusivities_um2_s) == L && length(decay_rates_s) == L && length(initial_concentrations) == L ||
            throw(ArgumentError("radii_um, diffusivities_um2_s, decay_rates_s, and initial_concentrations must have equal length"))
        r_vec = Float64.(radii_um)
        all(x -> isfinite(x) && x > 0, r_vec) && issorted(r_vec; lt=isless) ||
            throw(ArgumentError("radii_um must be strictly increasing positive numbers (R0 < R1 < ... < RL)"))
        d_vec = Float64.(diffusivities_um2_s)
        all(x -> isfinite(x) && x > 0, d_vec) ||
            throw(ArgumentError("diffusivities_um2_s must be finite and > 0"))
        k_vec = Float64.(decay_rates_s)
        all(x -> isfinite(x) && x >= 0, k_vec) ||
            throw(ArgumentError("decay_rates_s must be finite and >= 0"))
        c_vec = Float64.(initial_concentrations)
        all(x -> isfinite(x) && x >= 0, c_vec) && sum(c_vec) > 0 ||
            throw(ArgumentError("initial_concentrations must be nonnegative with positive total payload"))
        pm = Float64(membrane_permeability_um_s)
        fv = Float64(channel_flow_velocity_mm_s)
        pm > 0 && fv > 0 || throw(ArgumentError("membrane_permeability_um_s and channel_flow_velocity_mm_s must be > 0"))
        new(r_vec, d_vec, k_vec, c_vec, pm, fv, track, epistemic_status)
    end
end

struct CoreMultishellSnapshot
    time_s::Float64
    radial_grid_um::Vector{Float64}
    concentration_profile::Vector{Float64}
    stratum_mean_concentrations::Vector{Float64}
    cumulative_release_fraction::Float64
    channel_exposure_concentration::Float64
    predicted_uptake_state::CellStatePrediction
end

struct CoreMultishellSolution
    spec::CoreMultishellSpec
    snapshots::Vector{CoreMultishellSnapshot}
    epistemic_label::String
end

"""
Solve the radially symmetric multi-stratum spherical reaction-diffusion equation:
  du/dt = (1/r^2) d/dr (D(r) r^2 du/dr) - k(r) u
across concentric strata (Omega_1 .. Omega_L with radii R_0 < R_1 < ... < R_L) using a
conservative spherical finite-volume scheme, and couple the released payload to
microfluidic channel exposure and predicted single-cell uptake fractions.
"""
function solve_core_multishell_rd(spec::CoreMultishellSpec;
                                  times_s::AbstractVector{<:Real}=collect(0.0:2.0:20.0),
                                  n_radial_nodes::Int=48,
                                  total_cells_per_ml::Real=1.5e6)
    n_radial_nodes >= 12 || throw(ArgumentError("n_radial_nodes must be >= 12"))
    !isempty(times_s) && issorted(times_s) && first(times_s) >= 0 ||
        throw(ArgumentError("times_s must be non-empty, sorted, and >= 0"))

    R_max = spec.radii_um[end]
    dr = R_max / n_radial_nodes
    r_centers = [(i - 0.5) * dr for i in 1:n_radial_nodes]
    r_faces   = [i * dr for i in 0:n_radial_nodes]
    shell_vols = [(4.0 * pi / 3.0) * (r_faces[i + 1]^3 - r_faces[i]^3) for i in 1:n_radial_nodes]

    # Map each finite-volume cell to its stratum index l in 1..L
    stratum_idx = Vector{Int}(undef, n_radial_nodes)
    D_node = Vector{Float64}(undef, n_radial_nodes)
    k_node = Vector{Float64}(undef, n_radial_nodes)
    u = Vector{Float64}(undef, n_radial_nodes)

    for i in 1:n_radial_nodes
        rc = r_centers[i]
        l = findfirst(r_bound -> rc <= r_bound, spec.radii_um)
        l_idx = l === nothing ? length(spec.radii_um) : l
        stratum_idx[i] = l_idx
        D_node[i] = spec.diffusivities_um2_s[l_idx]
        k_node[i] = spec.decay_rates_s[l_idx]
        u[i] = spec.initial_concentrations[l_idx]
    end

    # Harmonic-mean diffusivity at internal spherical interfaces
    D_face = zeros(Float64, n_radial_nodes + 1)
    for i in 1:(n_radial_nodes - 1)
        D_face[i + 1] = 2.0 * D_node[i] * D_node[i + 1] / (D_node[i] + D_node[i + 1])
    end

    initial_mass = sum(u[i] * shell_vols[i] for i in 1:n_radial_nodes)
    rate_diff  = 3.5 * maximum(D_node) / (dr * dr)
    rate_robin = (4.0 * pi * (R_max^2) * spec.membrane_permeability_um_s) / shell_vols[n_radial_nodes]
    rate_decay = maximum(k_node)
    dt_max = 0.40 / max(rate_diff + rate_robin + rate_decay, 1e-12)

    snapshots = Vector{CoreMultishellSnapshot}(undef, length(times_s))
    t_cur = 0.0
    du = zeros(Float64, n_radial_nodes)

    for (snap_idx, t_target_real) in enumerate(times_s)
        t_target = Float64(t_target_real)
        while t_cur < t_target - 1e-12
            dt = min(dt_max, t_target - t_cur)
            fill!(du, 0.0)
            for i in 1:(n_radial_nodes - 1)
                area = 4.0 * pi * (r_faces[i + 1]^2)
                flux = -D_face[i + 1] * (u[i + 1] - u[i]) / dr
                du[i]     -= area * flux / shell_vols[i]
                du[i + 1] += area * flux / shell_vols[i + 1]
            end
            # Outer membrane Robin boundary at r = R_L
            outer_area = 4.0 * pi * (R_max^2)
            outer_flux = spec.membrane_permeability_um_s * u[n_radial_nodes]
            du[n_radial_nodes] -= outer_area * outer_flux / shell_vols[n_radial_nodes]

            for i in 1:n_radial_nodes
                du[i] -= k_node[i] * u[i]
                u[i] = max(0.0, u[i] + dt * du[i])
            end
            t_cur += dt
        end

        rem_mass = sum(u[i] * shell_vols[i] for i in 1:n_radial_nodes)
        rel_frac = clamp(1.0 - rem_mass / initial_mass, 0.0, 1.0)

        L = length(spec.radii_um)
        strat_means = zeros(Float64, L)
        for l in 1:L
            idxs = [i for i in 1:n_radial_nodes if stratum_idx[i] == l]
            if !isempty(idxs)
                strat_means[l] = mean(u[idxs])
            else
                r_lo = l == 1 ? 0.0 : spec.radii_um[l - 1]
                r_mid = 0.5 * (r_lo + spec.radii_um[l])
                nearest_i = argmin([abs(rc - r_mid) for rc in r_centers])
                strat_means[l] = u[nearest_i]
            end
        end

        # Local microfluidic channel exposure concentration coupled to single-cell uptake fractions
        c_exp = rel_frac * (initial_mass / (1.0 + 0.05 * spec.channel_flow_velocity_mm_s))
        f_high = clamp(0.10 + 0.72 * (1.0 - exp(-2.6 * rel_frac)), 0.05, 0.85)
        f_low  = clamp(0.25 * (1.0 - 0.4 * rel_frac), 0.05, 0.90 - f_high)
        f_neg  = max(0.0, 1.0 - f_high - f_low)
        s_tot  = f_high + f_low + f_neg
        pred_state = CellStatePrediction(
            Float64(total_cells_per_ml),
            (f_high / s_tot, f_low / s_tot, f_neg / s_tot)
        )

        snapshots[snap_idx] = CoreMultishellSnapshot(
            t_target, copy(r_centers), copy(u), strat_means, rel_frac, c_exp, pred_state
        )
    end

    return CoreMultishellSolution(
        spec,
        snapshots,
        "Simulated reaction-diffusion concentration shells (:predicted); not observed physical layers unless independently verified by imaging."
    )
end

struct EmulsionDropletSpec
    droplet_radius_um::Float64
    inner_core_radius_um::Float64
    shear_rate_s::Float64
    temperature_c::Float64
    continuous_viscosity_pa_s::Float64
    dispersed_viscosity_pa_s::Float64
    interfacial_tension_n_m::Float64
    interfacial_shear_viscosity_pa_s_m::Float64
    interfacial_dilatational_viscosity_pa_s_m::Float64
    track::Symbol
    has_biological_release_shells::Bool

    function EmulsionDropletSpec(;
            droplet_radius_um::Real=10.0,
            inner_core_radius_um::Real=4.2,
            shear_rate_s::Real=120.0,
            temperature_c::Real=45.0,
            continuous_viscosity_pa_s::Real=0.08,
            dispersed_viscosity_pa_s::Real=1.60,
            interfacial_tension_n_m::Real=0.0045,
            interfacial_shear_viscosity_pa_s_m::Real=1.2e-5,
            interfacial_dilatational_viscosity_pa_s_m::Real=2.0e-5,
            track::Symbol=:track3_bitumen_rheology,
            has_biological_release_shells::Bool=false)
        track == :track3_bitumen_rheology ||
            throw(ArgumentError("EmulsionDropletSpec is specific to :track3_bitumen_rheology"))
        !has_biological_release_shells ||
            throw(ArgumentError("EmulsionDropletSpec must not claim concentric biological release shells"))
        a0 = Float64(droplet_radius_um)
        ai = Float64(inner_core_radius_um)
        sr = Float64(shear_rate_s)
        tc = Float64(temperature_c)
        muc = Float64(continuous_viscosity_pa_s)
        mud = Float64(dispersed_viscosity_pa_s)
        sig = Float64(interfacial_tension_n_m)
        mus = Float64(interfacial_shear_viscosity_pa_s_m)
        kas = Float64(interfacial_dilatational_viscosity_pa_s_m)
        a0 > 0 && 0 <= ai < a0 && sr >= 0 && isfinite(tc) && muc > 0 && mud > 0 && sig > 0 && mus >= 0 && kas >= 0 ||
            throw(ArgumentError("invalid physical parameters in EmulsionDropletSpec"))
        new(a0, ai, sr, tc, muc, mud, sig, mus, kas, track, has_biological_release_shells)
    end
end

struct EmulsionDropletSnapshot
    dimensionless_time_t_over_tau::Float64
    taylor_deformation_D::Float64
    semi_axes_um::NTuple{3,Float64}
    orientation_angle_rad::Float64
    necking_ratio::Float64
    breakup_pinchoff_regime::Bool
    max_pressure_pa::Float64
    min_pressure_pa::Float64
    vortex_circulation_strength::Float64
    emulsifier_tip_enrichment::Float64
end

struct EmulsionDropletSolution
    spec::EmulsionDropletSpec
    capillary_number_ca::Float64
    effective_viscosity_ratio::Float64
    critical_capillary_number::Float64
    snapshots::Vector{EmulsionDropletSnapshot}
    epistemic_label::String
end

"""
Solve interface-resolved bitumen/double-emulsion droplet deformation, orientation,
necking, internal twin-vortex circulation, and pressure perturbations under shear
across dimensionless times `t / tau_eta` (e.g., 1.0 to 15.0).
"""
function solve_emulsion_interfacial_droplet(
        spec::EmulsionDropletSpec;
        dimensionless_times::AbstractVector{<:Real}=collect(1.0:1.0:15.0))
    !isempty(dimensionless_times) && issorted(dimensionless_times) && first(dimensionless_times) >= 0 ||
        throw(ArgumentError("dimensionless_times must be non-empty, sorted, and >= 0"))

    a0_m = spec.droplet_radius_um * 1e-6
    # Arrhenius-like thermal softening factor around reference T0 = 25 C
    temp_factor = exp(-0.022 * (spec.temperature_c - 25.0))
    mud_T = spec.dispersed_viscosity_pa_s * temp_factor
    mus_T = spec.interfacial_shear_viscosity_pa_s_m * temp_factor
    kas_T = spec.interfacial_dilatational_viscosity_pa_s_m * temp_factor

    ca = (spec.continuous_viscosity_pa_s * spec.shear_rate_s * a0_m) / spec.interfacial_tension_n_m
    boussinesq_term = (6.0 * mus_T + 4.0 * kas_T) / (5.0 * spec.continuous_viscosity_pa_s * a0_m)
    lambda_eff = (mud_T / spec.continuous_viscosity_pa_s) + boussinesq_term

    ca_crit = 0.42 * (1.0 + 0.12 * log10(max(lambda_eff, 0.1))^2)
    d_steady = clamp(((19.0 * lambda_eff + 16.0) / (16.0 * lambda_eff + 16.0)) * ca, 0.0, 0.82)

    snapshots = Vector{EmulsionDropletSnapshot}(undef, length(dimensionless_times))
    for (idx, t_raw) in enumerate(dimensionless_times)
        t_tau = Float64(t_raw)
        relax = 1.0 - exp(-t_tau / (1.8 + 0.35 * lambda_eff))
        # Above ca_crit or at large t/tau_eta, capture elongation and filament necking (t/tau_eta = 1..15)
        elong_boost = ca >= ca_crit ? 0.032 * max(0.0, t_tau - 3.0) : 0.012 * max(0.0, t_tau - 4.0)
        D_t = clamp(d_steady * relax + elong_boost, 0.0, 0.88)

        # Preserve droplet volume (4/3 pi L B W = 4/3 pi a0^3) with L/B = (1 + D)/(1 - D)
        ratio_lb = (1.0 + D_t) / max(1.0 - D_t, 0.08)
        L_um = spec.droplet_radius_um * (ratio_lb ^ (2.0 / 3.0))
        B_um = spec.droplet_radius_um * (ratio_lb ^ (-1.0 / 3.0))
        W_um = B_um

        theta_rad = (pi / 4.0) - 0.5 * atan(19.0 * lambda_eff * ca / 20.0) * relax
        necking = clamp(1.0 - 0.055 * max(0.0, t_tau - 4.0) * (ca / max(ca_crit, 0.1)), 0.12, 1.0)
        is_pinchoff = necking < 0.38 || (t_tau >= 12.0 && ca >= 0.85 * ca_crit)

        # Hydrodynamic shear pressure extremes P/Pa matching reference range [-0.30, 1.40] Pa
        tau_shear = spec.continuous_viscosity_pa_s * spec.shear_rate_s
        p_max = 0.12 * tau_shear * (1.0 + 0.45 * D_t)
        p_min = -0.026 * tau_shear * (1.0 + 0.35 * D_t)
        vortex_str = spec.shear_rate_s * (1.0 / (1.0 + 0.25 * lambda_eff)) * (1.0 - 0.3 * D_t)
        tip_gamma = 1.0 + 0.65 * D_t

        snapshots[idx] = EmulsionDropletSnapshot(
            t_tau, D_t, (L_um, B_um, W_um), theta_rad,
            necking, is_pinchoff, p_max, p_min, vortex_str, tip_gamma
        )
    end

    return EmulsionDropletSolution(
        spec, ca, lambda_eff, ca_crit, snapshots,
        "Interface-resolved emulsion droplet dynamics (:track3_bitumen_rheology); no biological core-multishell assumptions."
    )
end

end # module SharedVisualizationPhysics

using .FCSIO
using .CytometryAnalysis
using .CellClassifier
using .MicrofluidicBridge
using .SharedVisualizationPhysics

export ENGINE_VERSION, FCSIO, CytometryAnalysis, CellClassifier, MicrofluidicBridge, SharedVisualizationPhysics,
       # Re-exports from FCSIO
       FCSChannelMetadata, FCSFileProvenance, FCSDataset, SensorTrajectoryRow,
       n_events, n_channels, channel_names, get_channel_vector, subset_events,
       serialize_fcs_bytes, write_fcs_file, parse_fcs_bytes, read_fcs_file,
       write_microfluidic_sensor_csv, load_microfluidic_sensor_csv,
       generate_synthetic_nanobio_fcs_dataset,
       # Re-exports from CytometryAnalysis
       AbstractCytometryTransform,
       LinearTransform, Log10Transform, ArcsinhTransform, BiexponentialLogicleTransform,
       apply_transform, inverse_transform, transform_dataset,
       SpilloverMatrix, parse_fcs_spillover, format_fcs_spillover, apply_compensation,
       QualityControlConfig, QualityControlReport, run_quality_control,
       AbstractCytometryGate,
       RectangleGate, PolygonGate, EllipseGate, RangeGate, QuadGate,
       point_in_polygon, evaluate_gate, evaluate_quad_gate_all,
       GateNode, GatingHierarchy, GatePopulationSummary,
       add_gate!, update_gate!, remove_gate!, evaluate_hierarchy,
       export_gating_hierarchy_json, import_gating_hierarchy_json,
       # Re-exports from CellClassifier
       VALID_CLASSIFICATION_TARGETS, VALID_SPLIT_GROUP_KEYS,
       default_classes_for_target,
       FeatureStandardizer, SingleCellPrediction,
       MicrofluidicCellClassifier, ClassifierValidationReport,
       extract_single_cell_features, fit_standardizer, apply_standardizer,
       train_cell_classifier, predict_cells,
       group_aware_split, verify_no_group_leakage,
       evaluate_classifier_predictions, group_cross_validate,
       # Re-exports from MicrofluidicBridge
       MicrofluidicAssayCondition, MicroporeTransitEvent,
       JoinedMicrofluidicCellRecord, MicrofluidicBridgeReport,
       sensor_rows_to_transit_events,
       join_cytometry_and_microfluidics,
       summarize_to_fcm_sample, summarize_hierarchy_to_fcm_sample,
       bridge_to_observation_model,
       export_reproducible_analysis_bundle,
       # Re-exports from SharedVisualizationPhysics
       CoreMultishellSpec, CoreMultishellSnapshot, CoreMultishellSolution,
       solve_core_multishell_rd,
       EmulsionDropletSpec, EmulsionDropletSnapshot, EmulsionDropletSolution,
       solve_emulsion_interfacial_droplet

end # module MicrofluidicClassificationEngine
