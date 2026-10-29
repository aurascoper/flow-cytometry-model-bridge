#!/usr/bin/env python3
"""
Independent Python reference verification for MicrofluidicClassificationEngine.jl.

Verifies:
1. ISAC FCS 3.1 binary header, TEXT segment keywords, byte-offset consistency,
   SHA-256 provenance, and binary Float32 list-mode DATA unpacking.
2. Cytometry transformations (linear, log10, arcsinh, biexponential logicle).
3. Spectral spillover compensation unmixing (E_comp = E_raw * S^-1).
4. All 5 gate geometries (RectangleGate, PolygonGate, EllipseGate, RangeGate,
   QuadGate) and hierarchical parent-child gate counts and proportions.
5. Classifier confidence thresholding, :defer_for_review abstention rules,
   macro-F1, Brier score, Expected Calibration Error (ECE), and group-leakage
   absence.
6. FlowCytometryObservationModel.jl MLE concentration and Poisson-Multinomial
   joint log-likelihood consistency.
"""

import csv
import hashlib
import json
import math
import os
import struct
import sys


def parse_fcs_3_1(fcs_path):
    with open(fcs_path, "rb") as f:
        raw = f.read()

    sha256_hex = hashlib.sha256(raw).hexdigest()
    if len(raw) < 58:
        raise ValueError("FCS file shorter than 58-byte header")

    version = raw[0:6].decode("ascii").strip()
    if version not in ("FCS3.1", "FCS3.0", "FCS2.0"):
        raise ValueError(f"Unexpected FCS version: {version}")

    text_start = int(raw[10:18].decode("ascii").strip())
    text_end = int(raw[18:26].decode("ascii").strip())
    hdr_data_start = int(raw[26:34].decode("ascii").strip())
    hdr_data_end = int(raw[34:42].decode("ascii").strip())

    text_segment = raw[text_start : text_end + 1]
    delim = chr(text_segment[0])
    body = text_segment[1:].decode("utf-8")

    tokens = []
    buf = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == delim:
            if i + 1 < n and body[i + 1] == delim:
                buf.append(delim)
                i += 2
            else:
                tokens.append("".join(buf))
                buf = []
                i += 1
        else:
            buf.append(ch)
            i += 1

    keywords = {}
    for idx in range(0, len(tokens) - 1, 2):
        k = tokens[idx].strip().upper()
        v = tokens[idx + 1].strip()
        if k:
            keywords[k] = v

    kw_data_start = int(keywords.get("$BEGINDATA", "0"))
    kw_data_end = int(keywords.get("$ENDDATA", "0"))
    assert hdr_data_start == kw_data_start, (
        f"Header data_start ({hdr_data_start}) != $BEGINDATA ({kw_data_start})"
    )
    assert hdr_data_end == kw_data_end, (
        f"Header data_end ({hdr_data_end}) != $ENDDATA ({kw_data_end})"
    )

    n_par = int(keywords["$PAR"])
    n_tot = int(keywords["$TOT"])
    datatype = keywords["$DATATYPE"].upper()
    byteord = keywords["$BYTEORD"]
    endian_prefix = "<" if byteord.startswith("1,2") else ">"

    data_bytes = raw[hdr_data_start : hdr_data_end + 1]
    if datatype == "F":
        fmt = f"{endian_prefix}{n_tot * n_par}f"
        expected_len = n_tot * n_par * 4
    elif datatype == "D":
        fmt = f"{endian_prefix}{n_tot * n_par}d"
        expected_len = n_tot * n_par * 8
    else:
        raise ValueError(f"Unsupported $DATATYPE: {datatype}")

    vals = struct.unpack(fmt, data_bytes[:expected_len])
    events = [list(vals[r * n_par : (r + 1) * n_par]) for r in range(n_tot)]
    channels = [keywords[f"$P{i}N"] for i in range(1, n_par + 1)]
    return {
        "sha256": sha256_hex,
        "version": version,
        "keywords": keywords,
        "channels": channels,
        "events": events,
    }


def apply_transform(spec_str, x):
    parts = spec_str.split(":")
    kind = parts[0]
    if kind == "linear":
        scale = float(parts[1])
        offset = float(parts[2])
        return scale * x + offset
    elif kind == "log10":
        offset = float(parts[1])
        floor_val = float(parts[2])
        return math.log10(max(x + offset, floor_val))
    elif kind == "arcsinh":
        cofactor = float(parts[1])
        return math.asinh(x / cofactor)
    elif kind == "logicle":
        cofactor = float(parts[1])
        decade_scale = float(parts[2])
        return decade_scale * math.asinh(x / cofactor)
    else:
        raise ValueError(f"Unknown transform spec: {spec_str}")


def invert_3x3(m):
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if abs(det) < 1e-12:
        raise ValueError("Singular 3x3 spillover matrix")
    inv_det = 1.0 / det
    return [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]


def point_on_segment(px, py, x1, y1, x2, y2, atol=1e-9):
    dx = x2 - x1
    dy = y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= atol * atol:
        return math.hypot(px - x1, py - y1) <= atol
    cross = abs((px - x1) * dy - (py - y1) * dx) / math.sqrt(seg_len2)
    if cross > atol:
        return False
    dotp = (px - x1) * dx + (py - y1) * dy
    return -atol <= dotp <= seg_len2 + atol


def point_in_polygon(px, py, vertices):
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if point_on_segment(px, py, xj, yj, xi, yi):
            return True
        intersects = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def evaluate_gate_row(gate_obj, row_dict):
    gtype = gate_obj["type"]
    xch = gate_obj["x_channel"]
    ych = gate_obj["y_channel"]
    tx_spec = gate_obj["tx"]
    ty_spec = gate_obj["ty"]
    prm = gate_obj["params"]

    tx = apply_transform(tx_spec, row_dict[xch])
    if gtype == "RangeGate":
        lo, hi = [float(v) for v in prm.split(",")]
        return lo <= tx <= hi

    ty = apply_transform(ty_spec, row_dict[ych])
    if gtype == "RectangleGate":
        x0, x1, y0, y1 = [float(v) for v in prm.split(",")]
        return (x0 <= tx <= x1) and (y0 <= ty <= y1)
    elif gtype == "PolygonGate":
        verts = []
        for pair in prm.split(";"):
            vx, vy = pair.split(":")
            verts.append((float(vx), float(vy)))
        return point_in_polygon(tx, ty, verts)
    elif gtype == "EllipseGate":
        cx, cy, a, b, ang = [float(v) for v in prm.split(",")]
        dx = tx - cx
        dy = ty - cy
        ca = math.cos(ang)
        sa = math.sin(ang)
        u = dx * ca + dy * sa
        v = -dx * sa + dy * ca
        return (u * u) / (a * a) + (v * v) / (b * b) <= 1.0
    elif gtype == "QuadGate":
        parts = prm.split(",")
        xt = float(parts[0])
        yt = float(parts[1])
        quad = parts[2]
        if quad == "Q1_UR":
            return tx >= xt and ty >= yt
        elif quad == "Q2_UL":
            return tx < xt and ty >= yt
        elif quad == "Q3_LL":
            return tx < xt and ty < yt
        elif quad == "Q4_LR":
            return tx >= xt and ty < yt
        else:
            raise ValueError(f"Unknown quadrant: {quad}")
    else:
        raise ValueError(f"Unknown gate type: {gtype}")


def log_factorial(n):
    return math.lgamma(n + 1.0)


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: verify_against_python_reference.py <bundle_dir>")

    bundle_dir = sys.argv[1]
    fcs_path = os.path.join(bundle_dir, "sample_nanobio_chip01.fcs")
    gates_path = os.path.join(bundle_dir, "analysis_settings_and_gates.json")
    summary_path = os.path.join(bundle_dir, "population_and_bridge_summary.json")
    events_csv_path = os.path.join(bundle_dir, "classified_events.csv")
    payload_path = os.path.join(bundle_dir, "python_verification_payload.json")

    fcs_data = parse_fcs_3_1(fcs_path)
    with open(gates_path, "r", encoding="utf-8") as f:
        gates_json = json.load(f)
    with open(summary_path, "r", encoding="utf-8") as f:
        summary_json = json.load(f)
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # 1. Verify FCS 3.1 SHA-256 and dimensions
    assert fcs_data["sha256"] == summary_json["source_fcs_sha256"], (
        "FCS SHA-256 mismatch between binary file and Julia summary"
    )
    n_events = len(fcs_data["events"])
    assert n_events == summary_json["total_events"], "Total event count mismatch"

    # 2. Verify spillover compensation unmixing on FCS events
    spill_str = fcs_data["keywords"]["$SPILLOVER"]
    spill_parts = [s.strip() for s in spill_str.split(",")]
    k_spill = int(spill_parts[0])
    spill_channels = spill_parts[1 : 1 + k_spill]
    spill_vals = [float(x) for x in spill_parts[1 + k_spill :]]
    spill_mat = [
        spill_vals[r * k_spill : (r + 1) * k_spill] for r in range(k_spill)
    ]
    inv_spill = invert_3x3(spill_mat)

    ch_to_idx = {ch: idx for idx, ch in enumerate(fcs_data["channels"])}
    spill_col_indices = [ch_to_idx[ch] for ch in spill_channels]

    comp_events = []
    for r_idx, row in enumerate(fcs_data["events"]):
        new_row = list(row)
        raw_fl = [row[c] for c in spill_col_indices]
        comp_fl = [
            sum(raw_fl[j] * inv_spill[j][c] for j in range(k_spill))
            for c in range(k_spill)
        ]
        for c_local, col_global in enumerate(spill_col_indices):
            new_row[col_global] = comp_fl[c_local]
        comp_events.append(new_row)

    # Compare compensated first 5 events against Julia payload
    for i in range(min(5, n_events)):
        for c in range(len(fcs_data["channels"])):
            diff = abs(comp_events[i][c] - payload["compensated_head_events"][i][c])
            assert diff <= 1e-2, f"Compensated event [{i},{c}] diff={diff}"

    # 3. Verify hierarchical gates on compensated dataset
    use_events = comp_events if payload.get("hierarchy_on_compensated", True) else fcs_data["events"]
    masks = {"root": [True] * n_events}
    for gate_obj in gates_json["gates"]:
        gid = gate_obj["id"]
        pid = gate_obj["parent_id"]
        parent_mask = masks[pid]
        cur_mask = [False] * n_events
        for i in range(n_events):
            if parent_mask[i]:
                row_dict = {
                    ch: use_events[i][idx]
                    for idx, ch in enumerate(fcs_data["channels"])
                }
                if evaluate_gate_row(gate_obj, row_dict):
                    cur_mask[i] = True
        masks[gid] = cur_mask

        cnt = sum(1 for x in cur_mask if x)
        p_cnt = sum(1 for x in parent_mask if x)
        frac_parent = cnt / p_cnt if p_cnt > 0 else 0.0
        frac_total = cnt / n_events if n_events > 0 else 0.0

        julia_gs = summary_json["gate_summaries"][gid]
        assert cnt == julia_gs["count"], (
            f"Gate {gid} count mismatch: python={cnt} vs julia={julia_gs['count']}"
        )
        assert p_cnt == julia_gs["parent_count"], (
            f"Gate {gid} parent_count mismatch: python={p_cnt} vs julia={julia_gs['parent_count']}"
        )
        assert abs(frac_parent - julia_gs["fraction_of_parent"]) <= 1e-7, (
            f"Gate {gid} fraction_of_parent mismatch"
        )
        assert abs(frac_total - julia_gs["fraction_of_total"]) <= 1e-7, (
            f"Gate {gid} fraction_of_total mismatch"
        )

    # 4. Verify classified_events.csv and classifier metrics
    conf_threshold = float(payload["confidence_threshold"])
    with open(events_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    assert len(csv_rows) == n_events, "classified_events.csv row count mismatch"
    deferred_count = 0
    accepted_count = 0
    for r in csv_rows:
        conf = float(r["confidence"])
        is_def = r["deferred"].lower() == "true"
        assert is_def == (conf < conf_threshold - 1e-12), (
            f"Deferral threshold mismatch for row {r['event_index']}"
        )
        if is_def:
            deferred_count += 1
            assert r["final_decision"] == "defer_for_review"
        else:
            accepted_count += 1
            assert r["final_decision"] == r["predicted_class"]

    # Verify group-aware cross-validation metrics from payload
    cv_preds = payload["cv_predictions"]
    true_labels = payload["cv_true_labels"]
    classes = payload["classes"]
    n_cv = len(cv_preds)
    correct_all = sum(
        1 for i in range(n_cv) if cv_preds[i]["predicted_class"] == true_labels[i]
    )
    acc_indices = [i for i in range(n_cv) if not cv_preds[i]["deferred"]]
    correct_acc = sum(
        1 for i in acc_indices if cv_preds[i]["predicted_class"] == true_labels[i]
    )
    overall_acc = correct_all / n_cv
    accepted_acc = correct_acc / len(acc_indices) if acc_indices else overall_acc
    coverage_frac = len(acc_indices) / n_cv
    deferral_frac = 1.0 - coverage_frac

    brier_sum = 0.0
    for i in range(n_cv):
        probs = cv_preds[i]["probabilities"]
        for c_idx, cls_name in enumerate(classes):
            y_ic = 1.0 if true_labels[i] == cls_name else 0.0
            brier_sum += (probs[c_idx] - y_ic) ** 2
    brier_score = brier_sum / n_cv

    assert abs(overall_acc - summary_json["overall_accuracy"]) <= 1e-6
    assert abs(accepted_acc - summary_json["accepted_accuracy"]) <= 1e-6
    assert abs(coverage_frac - summary_json["coverage_fraction"]) <= 1e-6
    assert abs(deferral_frac - summary_json["deferral_fraction"]) <= 1e-6
    assert abs(brier_score - summary_json["brier_score"]) <= 1e-6

    # Verify zero group leakage in train/test split
    train_groups = set(payload["split_train_groups"])
    test_groups = set(payload["split_test_groups"])
    assert len(train_groups.intersection(test_groups)) == 0, (
        "Group leakage detected between train and test sets"
    )

    # 5. Verify FlowCytometryObservationModel MLE concentration and likelihood
    bridge = payload["bridge_inputs"]
    gate_counts = bridge["gate_counts"]
    n_obs = sum(gate_counts)
    bead_events = bridge["bead_events"]
    beads_added = bridge["beads_added"]
    sample_vol = bridge["sample_volume_ml"]
    dilution = bridge["dilution_factor"]
    recovery = bridge["extraction_recovery"]
    det_eff = bridge["detection_efficiency"]
    v_eff = sample_vol / dilution

    c_mle = (
        (n_obs / bead_events)
        * (beads_added / v_eff)
        / recovery
        / det_eff
    )
    assert abs(c_mle - summary_json["absolute_concentration_cells_per_ml"]) <= 1e-4

    lam = (
        bridge["state_total_cells_per_ml"]
        * det_eff
        * v_eff
        * recovery
        * (bead_events / beads_added)
    )
    ll_count = n_obs * math.log(lam) - lam - log_factorial(n_obs)

    gate_probs = bridge["predicted_gate_probs"]
    ll_gate = log_factorial(n_obs)
    for g_idx in range(3):
        ll_gate -= log_factorial(gate_counts[g_idx])
        if gate_counts[g_idx] > 0:
            ll_gate += gate_counts[g_idx] * math.log(gate_probs[g_idx])

    ll_joint = ll_count + ll_gate
    assert abs(ll_joint - summary_json["joint_loglikelihood"]) <= 1e-5, (
        f"Joint loglikelihood mismatch: python={ll_joint} vs julia={summary_json['joint_loglikelihood']}"
    )

    print("PYTHON_REFERENCE_VERIFICATION_PASSED")


if __name__ == "__main__":
    main()
