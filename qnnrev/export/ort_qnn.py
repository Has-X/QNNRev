# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""ONNX Runtime QNN Execution Provider compiler helper.

Compiles an ONNX model into a QNN context binary and IR DLC via
``onnxruntime-qnn``, checks numerical parity against ORT CPU execution,
and writes a deterministic ``compile-receipt.json``.

Usage:
    python -m qnnrev.export.ort_qnn <onnx> <outdir> [backend=cpu|htp] [soc_model] [vtcm_mb] [arch] [input_npz]
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_inputs(
    sess: Any, onnx_path: str, input_npz: str | None = None
) -> dict[str, np.ndarray]:
    if input_npz and os.path.exists(input_npz):
        d = np.load(input_npz)
        return {inp.name: np.asarray(d[inp.name]) for inp in sess.get_inputs()}
    xfile = onnx_path.replace(".onnx", "_x.npy")
    sess_ins = sess.get_inputs()
    if len(sess_ins) == 1 and os.path.exists(xfile):
        return {sess_ins[0].name: np.load(xfile)}
    rng = np.random.default_rng(0)
    inputs: dict[str, np.ndarray] = {}
    for inp in sess_ins:
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        t = inp.type
        if "int32" in t:
            arr = (
                np.zeros(shape, dtype=np.int32)
                if ("token_type" in inp.name or inp.name == "t_5")
                else np.ones(shape, dtype=np.int32)
            )
        elif "int64" in t:
            arr = (
                np.zeros(shape, dtype=np.int64)
                if ("token_type" in inp.name or inp.name == "t_5")
                else np.ones(shape, dtype=np.int64)
            )
        elif "uint8" in t:
            arr = rng.integers(0, 255, size=shape, dtype=np.uint8)
        elif "uint16" in t:
            arr = rng.integers(0, 65535, size=shape, dtype=np.uint16)
        elif "bool" in t:
            arr = np.ones(shape, dtype=np.bool_)
        else:
            arr = rng.normal(0.0, 0.25, size=shape).astype(np.float32)
        inputs[inp.name] = arr
    return inputs


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2:
        print(
            "usage: python -m qnnrev.export.ort_qnn <onnx> <outdir> "
            "[backend=cpu|htp] [soc_model] [vtcm_mb] [arch] [input_npz]",
            file=sys.stderr,
        )
        return 2

    if os.uname().machine == "aarch64":
        os.environ.setdefault("ORT_QNN_ENABLE_CPU_BACKEND", "1")

    import onnxruntime as ort
    import onnxruntime_qnn as qnn_ep

    onnx_path, outdir = args[0], args[1]
    backend = args[2] if len(args) > 2 and args[2] else "cpu"
    soc = args[3] if len(args) > 3 and args[3] else None
    vtcm = args[4] if len(args) > 4 and args[4] else None
    arch = args[5] if len(args) > 5 and args[5] else None
    input_npz = args[6] if len(args) > 6 and args[6] else os.environ.get("QNNREV_INPUT_NPZ")
    os.makedirs(outdir, exist_ok=True)

    ep_registration = "QNNExecutionProvider"
    ort.register_execution_provider_library(ep_registration, qnn_ep.get_library_path())
    devs = [d for d in ort.get_ep_devices() if d.ep_name == ep_registration]
    if not devs:
        raise RuntimeError(
            "QNN EP device not found (set ORT_QNN_ENABLE_CPU_BACKEND=1 on arm64 hosts)"
        )

    ep_options: dict[str, str] = {
        "backend_path": (
            qnn_ep.get_qnn_cpu_path() if backend == "cpu" else qnn_ep.get_qnn_htp_path()
        )
    }
    if soc:
        ep_options["soc_model"] = soc
    if vtcm:
        ep_options["vtcm_mb"] = vtcm
    if arch:
        ep_options["htp_arch"] = arch
    so = ort.SessionOptions()
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", os.path.join(outdir, "context.bin"))
    ep_options["dump_qnn_ir_dlc"] = "1"
    ep_options["qnn_ir_backend_path"] = os.path.join(
        os.path.dirname(qnn_ep.get_library_path()), "libQnnIr.so"
    )
    so.add_provider_for_devices(devs, ep_options)
    sess = ort.InferenceSession(onnx_path, sess_options=so)
    sess.disable_fallback()

    inputs = _resolve_inputs(sess, onnx_path, input_npz=input_npz)
    outs = sess.run(None, inputs)
    reffile = onnx_path.replace(".onnx", "_ref.npy")
    if len(outs) == 1 and os.path.exists(reffile):
        refs = [np.load(reffile)]
    else:
        cpu_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        refs = cpu_sess.run(None, inputs)

    errs = [float(np.abs(o - r).max()) for o, r in zip(outs, refs)]
    coses = []
    for o, r in zip(outs, refs):
        ov = np.asarray(o, dtype=np.float64).ravel()
        rv = np.asarray(r, dtype=np.float64).ravel()
        denom = float(np.linalg.norm(ov) * np.linalg.norm(rv))
        coses.append(float(np.dot(ov, rv) / denom) if denom > 0 else 1.0)
    err = max(errs)
    np.save(os.path.join(outdir, "qnn_output.npy"), outs[0])
    np.savez(
        os.path.join(outdir, "qnn_outputs.npz"),
        **{f"out_{i}": o for i, o in enumerate(outs)},
    )

    arts = []
    for root, _, files in os.walk(outdir):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f == "compile-receipt.json":
                continue
            arts.append(
                {
                    "path": os.path.relpath(p, outdir),
                    "bytes": os.path.getsize(p),
                    "sha256": _sha256_file(p),
                }
            )
    receipt = {
        "case": os.path.basename(onnx_path),
        "backend": backend,
        "soc": soc,
        "vtcm_mb": vtcm,
        "htp_arch": arch,
        "inputs": list(inputs.keys()),
        "max_abs_err": err,
        "min_cos": min(coses),
        "qnn_ep_version": qnn_ep.__version__,
        "ort_version": ort.__version__,
        "artifacts": arts,
    }
    Path(outdir, "compile-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
