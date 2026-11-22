# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Structure, parameter-provenance, and numerical validation suite."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import numpy_helper


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def graph_signature(model_path: str | Path) -> dict[str, Any]:
    m = onnx.load(str(model_path))
    nodes = sorted(
        (n.name or f"{n.op_type}#{i}", n.op_type) for i, n in enumerate(m.graph.node)
    )
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    ios = sorted(t.name for t in list(m.graph.input) + list(m.graph.output))
    return {"ops": nodes, "initializers": inits, "io": ios}


def compare(src_path: str | Path, rec_path: str | Path) -> dict[str, Any]:
    s, r = graph_signature(src_path), graph_signature(rec_path)
    res: dict[str, Any] = {}
    s_ops = [op for _, op in s["ops"]]
    r_ops = [op for _, op in r["ops"]]
    res["op_coverage"] = {
        "source": s_ops,
        "recovered": r_ops,
        "missing": sorted(set(s_ops) - set(r_ops)),
        "extra": sorted(set(r_ops) - set(s_ops)),
        "match": s_ops == r_ops,
    }
    res["io_match"] = s["io"] == r["io"]
    wres: dict[str, str] = {}
    for name, arr in s["initializers"].items():
        ra = r["initializers"].get(name)
        if ra is None:
            wres[name] = "MISSING"
        elif ra.shape == arr.shape and np.array_equal(ra, arr):
            wres[name] = "byte-exact"
        elif ra.shape == arr.shape:
            wres[name] = f"numerical (maxdiff {float(np.abs(ra - arr).max()):.3e})"
        else:
            wres[name] = f"SHAPE-MISMATCH {arr.shape}->{ra.shape}"
    res["weights"] = wres
    q_src = [
        n.op_type
        for n in onnx.load(str(src_path)).graph.node
        if "Quant" in n.op_type or "Dequant" in n.op_type
    ]
    q_rec = [
        n.op_type
        for n in onnx.load(str(rec_path)).graph.node
        if "Quant" in n.op_type or "Dequant" in n.op_type
    ]
    res["quant_params"] = {"source": q_src, "recovered": q_rec, "match": q_src == q_rec}
    return res


def numerical(
    src_onnx: str | Path,
    rec_onnx: str | Path,
    x_path: str | Path,
    ref_path: str | Path,
    input_name: str,
    classes: Sequence[int] | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    x = np.load(x_path)
    _ = np.load(ref_path)
    outs = {}
    for tag, p in (("src", src_onnx), ("rec", rec_onnx)):
        s = ort.InferenceSession(
            str(p), providers=["CPUExecutionProvider"], sess_options=so
        )
        outs[tag] = s.run(None, {input_name: x})[0]
    abserr = float(np.abs(outs["src"] - outs["rec"]).max())
    denom = np.maximum(np.abs(outs["src"]), 1e-6)
    relerr = float((np.abs(outs["src"] - outs["rec"]) / denom).max())
    res: dict[str, Any] = {"max_abs_err": abserr, "max_rel_err": relerr}
    if classes is not None and outs["src"].ndim >= 2:
        cs = int(np.prod(classes))
        a = outs["src"].reshape(-1, cs).argmax(1)
        b = outs["rec"].reshape(-1, cs).argmax(1)
        res["class_stability"] = float((a == b).mean())
    return res


def classify(
    cov: dict[str, Any],
    nums: dict[str, Any],
    htp_executed: bool,
) -> str:
    """Classify reconstruction fidelity; CPU-only checks cannot reach ``exact_recovery``."""
    exact = (
        cov["op_coverage"]["match"]
        and cov["io_match"]
        and all(v == "byte-exact" for v in cov["weights"].values())
        and "MISSING" not in cov["weights"].values()
        and "SHAPE-MISMATCH" not in " ".join(cov["weights"].values())
    )
    weights_ok = exact and nums["max_abs_err"] < 1e-5
    if weights_ok and htp_executed:
        return "exact_recovery"
    if weights_ok:
        return "functional_equivalence (CPU-validated only; HTP execution required)"
    if cov["op_coverage"]["match"] and cov["io_match"] and not cov["op_coverage"]["missing"]:
        return "partial_reconstruction"
    if cov["io_match"] or cov["op_coverage"]["match"]:
        return "partial_reconstruction"
    return "not_recovered"


def run_suite(
    case_dir: str | Path,
    src_onnx: str | Path,
    rec_onnx: str | Path,
    x_path: str | Path,
    ref_path: str | Path,
    input_name: str,
    htp_executed: bool = False,
    classes: Sequence[int] | None = None,
) -> dict[str, Any]:
    _ = case_dir
    cov = compare(src_onnx, rec_onnx)
    nums = numerical(src_onnx, rec_onnx, x_path, ref_path, input_name, classes)
    verdict = classify(cov, nums, htp_executed)
    return {
        "source": str(src_onnx),
        "recovered": str(rec_onnx),
        "source_sha256": _sha256_file(src_onnx),
        "recovered_sha256": _sha256_file(rec_onnx),
        "coverage": cov,
        "numerics": nums,
        "htp_executed": htp_executed,
        "classification": verdict,
    }


__all__ = ["graph_signature", "compare", "numerical", "classify", "run_suite"]
