# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Unified CLI for QNNRev (``python -m qnnrev <command> ...`` or ``qnnrev ...``)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .core import (
    detect_version,
    inspect as inspect_artifact,
    parse_hnnx_context,
    unpack_auto,
)
from .export import (
    build_compile_command,
    export_bert_onnx,
    export_multihead_bert_onnx,
    export_vit_onnx,
)


def _parse_version_tuple(version: str) -> Optional[Tuple[int, ...]]:
    m = re.match(r"^\s*v?(\d+(?:\.\d+)*)", version)
    if not m:
        return None
    return tuple(int(part) for part in m.group(1).split("."))


def _default_qairt_root(qairt_version: str) -> Path:
    env_root = os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("QNN_SDK_ROOT")
    if env_root:
        return Path(env_root)
    if _parse_version_tuple(qairt_version) is not None:
        return Path(f"/opt/qairt/{qairt_version.strip()}")
    return Path("/opt/qairt/current")


def _default_py_x64() -> Path:
    env_py = os.environ.get("QNNREV_PY64")
    if env_py:
        return Path(env_py)
    found = shutil.which("python3.10") or sys.executable
    return Path(found)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 0.0 else 0.0


def _qairt_native_env(qairt_root: Path, py_x64: Path) -> Tuple[Dict[str, str], Optional[List[str]]]:
    """Prepare environment and optional cross-arch loader prefix for QAIRT x86_64 tools."""
    env = os.environ.copy()
    qairt_lib = qairt_root / "lib/x86_64-linux-clang"
    env["PYTHONPATH"] = str(qairt_root / "lib/python")

    eb_env = os.environ.get("QNNREV_EMUBUNDLE_DIR")
    eb = Path(eb_env) if eb_env else None
    if eb is not None and eb.exists():
        py_lib = py_x64.parent.parent / "lib"
        xlib = ":".join(
            str(p)
            for p in (
                eb / "glsys",
                eb / "llvm18",
                Path("/usr/x86_64-linux-gnu/lib"),
                eb / "syslibs/usr/lib/x86_64-linux-gnu",
                eb / "syslibs",
                qairt_lib,
                py_lib,
            )
        )
        ld_linux = os.environ.get("QNNREV_LD_LINUX", "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2")
        return env, [ld_linux, "--library-path", xlib]

    if platform.machine() == "x86_64":
        existing_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{qairt_lib}:{existing_ld}" if existing_ld else str(qairt_lib)
        return env, None

    ld_linux = os.environ.get("QNNREV_LD_LINUX", "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2")
    return env, [ld_linux, "--library-path", f"/usr/x86_64-linux-gnu/lib:{qairt_lib}"]


def build_qairt_plan(
    onnx_path: Path,
    out_dir: Path,
    graph_name: str,
    input_dims: Sequence[str],
    input_list: Optional[Path],
    qairt_root: Optional[Path] = None,
    vtcm_mb: int = 2,
    soc_model: int = 61,
    dsp_arch: str = "v73",
    opt_level: int = 3,
    quantization_overrides: Optional[Path] = None,
    qairt_version: str = "auto",
    htp_extended_opts: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build converter -> DLCQuantizer (asymmetric params) -> context-binary commands."""
    if qairt_root is None:
        qairt_root = _default_qairt_root(qairt_version)
    py_x64 = _default_py_x64()

    # Keep dlc_fp32 filename stem identical to graph_name so htp_in.json graph_names
    # matches; otherwise qnn-context-binary-generator falls back to O=2, vtcm_mb=4.
    dlc_fp32 = out_dir / f"{graph_name}.dlc"
    dlc_w8a16 = out_dir / f"{graph_name}_w8a16.dlc"
    bin_stem = f"{graph_name}_htp_{vtcm_mb}mb"
    htp_in = out_dir / "htp_in.json"
    htp_out = out_dir / "htp_out.json"
    _, ld_prefix = _qairt_native_env(qairt_root, py_x64)
    prefix = list(ld_prefix) if ld_prefix else []

    conv_cmd = prefix + [
        str(py_x64),
        str(qairt_root / "bin/x86_64-linux-clang/qairt-converter"),
        "--input_network",
        str(onnx_path),
    ]
    for spec in input_dims:
        name, dims = spec.split("=", 1)
        conv_cmd.extend(["-d", name, dims])
    if quantization_overrides is not None:
        conv_cmd.extend(["--quantization_overrides", str(quantization_overrides)])
    conv_cmd.extend(["--output_path", str(dlc_fp32)])

    # Keep param_quantizer_schema='asymmetric' with per-channel / per-row quantization
    # and quantized bias enabled for HTP LayerNorm/FullyConnected compatibility.
    quant_py = (
        "import qti.aisw.converters.common.dlc_quantizer as dq; "
        f"q = dq.DLCQuantizer(input_dlc={str(dlc_fp32)!r}, output_dlc={str(dlc_w8a16)!r}, "
        f"input_list={str(input_list) if input_list else None!r}, "
        "act_bitwidth=16, weights_bitwidth=8, bias_bitwidth=32, "
        "use_per_channel_quantization=True, use_per_row_quantization=True, "
        "enable_per_row_quantized_bias=True, "
        "use_native_input_files=True, disable_legacy_quantizer=True, dump_encoding_json=True); "
        "q.quantize(); "
        f"q.save({graph_name!r})"
    )
    quant_cmd = prefix + [str(py_x64), "-c", quant_py]

    ctx_cmd = prefix + [
        str(qairt_root / "bin/x86_64-linux-clang/qnn-context-binary-generator"),
        "--dlc_path",
        str(dlc_w8a16),
        "--backend",
        str(qairt_root / "lib/x86_64-linux-clang/libQnnHtp.so"),
        "--binary_file",
        bin_stem,
        "--output_dir",
        str(out_dir),
        "--config_file",
        str(htp_out),
    ]
    graph_cfg: Dict[str, Any] = {
        "graph_names": [graph_name],
        "vtcm_mb": vtcm_mb,
        "O": opt_level,
    }
    parsed_ver = _parse_version_tuple(qairt_version)
    use_extended = (
        bool(htp_extended_opts)
        if htp_extended_opts is not None
        else (parsed_ver is not None and parsed_ver >= (2, 45))
    )
    if use_extended:
        graph_cfg.update(
            {
                "dlbc": 1,
                "dlbc_weights": 1,
                "vtcm_retention": 1,
                "cached_weights": 1,
                "weights_packing": True,
                "precision_compensation": True,
                "advanced_activation_fusion": True,
                "high_precision_sigmoid": True,
            }
        )
    return {
        "qairt_version": qairt_version,
        "htp_extended_opts": use_extended,
        "htp_in": {
            "graphs": [graph_cfg],
            "devices": [{"soc_model": soc_model, "dsp_arch": dsp_arch}],
        },
        "htp_out": {
            "backend_extensions": {
                "shared_library_path": str(
                    qairt_root / "lib/x86_64-linux-clang/libQnnHtpNetRunExtensions.so"
                ),
                "config_file_path": str(htp_in),
            }
        },
        "htp_in_path": str(htp_in),
        "htp_out_path": str(htp_out),
        "converter_cmd": conv_cmd,
        "quantizer_cmd": quant_cmd,
        "context_cmd": ctx_cmd,
        "output_bin": str(out_dir / f"{bin_stem}.bin"),
    }


def _cmd_inspect(args: argparse.Namespace) -> int:
    target = Path(args.path)
    paths = (
        sorted(
            p
            for p in target.rglob("*")
            if p.is_file() and p.suffix in (".bin", ".minn", ".zip", ".dlc")
        )
        if target.is_dir()
        else [target]
    )
    results = []
    for p in paths:
        entry = inspect_artifact(str(p), sdk_dir=args.sdk_dir)
        if entry.get("format") == "qnn-context-raw":
            with p.open("rb") as f:
                ver, voff = detect_version(f.read(4096))
            entry["qnn_version"] = ver
            entry["version_offset"] = voff
            try:
                hnnx = parse_hnnx_context(p)
                entry["hnnx"] = {
                    "container_version": hnnx.container_version,
                    "compiler_version": hnnx.compiler_version,
                    "compiler_options": hnnx.compiler_options,
                    "mef_counts": hnnx.mef_counts,
                    "num_extents": len(hnnx.extents),
                    "extent_alignment": hnnx.extent_alignment,
                    "total_extent_bytes": sum(e.byte_length for e in hnnx.extents),
                    "num_op_classes": len(hnnx.op_class_names),
                }
            except Exception as exc:
                entry["hnnx_error"] = str(exc)
        results.append(entry)
    payload = json.dumps({"qnnrev": "0.2", "results": results}, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


def _cmd_unpack(args: argparse.Namespace) -> int:
    weights = unpack_auto(args.ctx, args.deser_dir, model_hint=args.model)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **weights)
    print(
        json.dumps(
            {
                "model": args.model,
                "tensors": len(weights),
                "out": str(out_path),
                "sample_keys": sorted(weights.keys())[:10],
            },
            indent=2,
        )
    )
    return 0


def _cmd_decompile(args: argparse.Namespace) -> int:
    if args.arch == "bert":
        out = export_bert_onnx(args.weights, args.out)
    elif args.arch == "bert-multihead":
        out = export_multihead_bert_onnx(args.weights, args.out)
    elif args.arch == "vit":
        out = export_vit_onnx(args.weights, args.out)
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")
    print(json.dumps({"arch": args.arch, "onnx": str(out)}, indent=2))
    return 0


def _cmd_compile(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(args.onnx)
    compiler = getattr(args, "compiler", "auto")
    parsed_ver = _parse_version_tuple(args.qairt_version)
    use_ort_ep = compiler == "ort-ep" or (
        compiler == "auto" and parsed_ver is not None and parsed_ver < (2, 40)
    )
    if use_ort_ep:
        cmd = build_compile_command(
            onnx_path,
            out_dir,
            backend=args.backend,
            soc=args.soc_name,
            vtcm_mb=args.vtcm_mb,
            arch=args.dsp_arch,
            input_npz=args.input_npz,
        )
        print(
            json.dumps(
                {
                    "qairt_version": args.qairt_version,
                    "compiler": "ort-ep",
                    "cmd": cmd,
                    "dry_run": args.dry_run,
                },
                indent=2,
            )
        )
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        return 0

    graph_name = args.graph_name or onnx_path.stem
    qairt_root = (
        Path(args.qairt_root)
        if args.qairt_root
        else _default_qairt_root(args.qairt_version)
    )
    plan = build_qairt_plan(
        onnx_path=onnx_path,
        out_dir=out_dir,
        graph_name=graph_name,
        input_dims=args.input_dim,
        input_list=Path(args.input_list) if args.input_list else None,
        qairt_root=qairt_root,
        vtcm_mb=args.vtcm_mb,
        soc_model=args.soc_model,
        dsp_arch=args.dsp_arch,
        opt_level=args.opt_level,
        quantization_overrides=(
            Path(args.quantization_overrides) if args.quantization_overrides else None
        ),
        qairt_version=args.qairt_version,
        htp_extended_opts=getattr(args, "htp_extended_opts", None),
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    Path(plan["htp_in_path"]).write_text(
        json.dumps(plan["htp_in"], indent=2) + "\n", encoding="utf-8"
    )
    Path(plan["htp_out_path"]).write_text(
        json.dumps(plan["htp_out"], indent=2) + "\n", encoding="utf-8"
    )
    env, _ = _qairt_native_env(qairt_root, _default_py_x64())
    subprocess.run(plan["converter_cmd"], env=env, check=True)
    subprocess.run(plan["quantizer_cmd"], env=env, check=True)
    subprocess.run(plan["context_cmd"], env=env, check=True)
    print(json.dumps({"status": "OK", "output_bin": plan["output_bin"]}, indent=2))
    return 0


def _load_vectors(
    path: Path, dtype: str, scale: float, offset: int
) -> Dict[str, np.ndarray]:
    if path.is_file() and path.suffix == ".npz":
        d = np.load(path)
        return {k: np.asarray(d[k], dtype=np.float32).ravel() for k in d.files}
    out: Dict[str, np.ndarray] = {}
    for fp in sorted(path.glob("*.raw")):
        if dtype == "u16":
            arr = (np.fromfile(fp, dtype="<u2").astype(np.float32) + float(offset)) * float(scale)
        else:
            arr = np.fromfile(fp, dtype="<f4")
        out[fp.stem] = arr.ravel()
    return out


def _cmd_verify(args: argparse.Namespace) -> int:
    a = _load_vectors(Path(args.a), args.a_dtype, args.a_scale, args.a_offset)
    b = _load_vectors(Path(args.b), args.b_dtype, args.b_scale, args.b_offset)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise ValueError("No overlapping probe keys between --a and --b")
    mat_a = np.stack([a[k] for k in keys], axis=0)
    mat_b = np.stack([b[k] for k in keys], axis=0)
    raw_cos = [_cosine(mat_a[i], mat_b[i]) for i in range(len(keys))]
    da = mat_a - mat_a.mean(axis=0, keepdims=True)
    db = mat_b - mat_b.mean(axis=0, keepdims=True)
    dev_cos = [_cosine(da[i], db[i]) for i in range(len(keys))] if len(keys) > 1 else raw_cos
    max_abs_err = float(np.abs(mat_a - mat_b).max())
    status = (
        "PASS"
        if (min(raw_cos) >= args.min_raw_cos and min(dev_cos) >= args.min_dev_cos)
        else "FAIL"
    )
    verdict = ("htp-verified" if args.htp_executed else "cpu-verified") if status == "PASS" else "failed"
    report = {
        "status": status,
        "count": len(keys),
        "raw_cos_min": min(raw_cos),
        "raw_cos_mean": float(np.mean(raw_cos)),
        "dev_cos_min": min(dev_cos),
        "dev_cos_mean": float(np.mean(dev_cos)),
        "max_abs_err": max_abs_err,
        "classification": verdict,
    }
    print(json.dumps(report, indent=2))
    return 0 if status == "PASS" else 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="qnnrev",
        description="QNNRev reverse-engineering, unpacking, and recompilation CLI",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ins = sub.add_parser("inspect", help="Inspect context binary, zip, or directory")
    p_ins.add_argument("path")
    p_ins.add_argument("--sdk-dir", default=None, help="Directory with official qnn-context-binary-utility JSON")
    p_ins.add_argument("-o", "--out")
    p_ins.set_defaults(func=_cmd_inspect)

    p_unp = sub.add_parser("unpack", help="Taplessly unpack weights from QNN HTP context binary")
    p_unp.add_argument(
        "--model",
        default="auto",
        choices=["auto", "generic", "bert", "bert-multihead", "vit"],
    )
    p_unp.add_argument("--ctx", required=True)
    p_unp.add_argument("--deser-dir", required=True)
    p_unp.add_argument("--out", required=True)
    p_unp.set_defaults(func=_cmd_unpack)

    p_dec = sub.add_parser("decompile", help="Export unpacked weights to canonical ONNX")
    p_dec.add_argument("--arch", required=True, choices=["bert", "bert-multihead", "vit"])
    p_dec.add_argument("--weights", required=True)
    p_dec.add_argument("--out", required=True)
    p_dec.set_defaults(func=_cmd_decompile)

    p_cmp = sub.add_parser("compile", help="Compile ONNX to QNN HTP context via ORT-EP or standalone QAIRT SDK")
    p_cmp.add_argument("--onnx", required=True)
    p_cmp.add_argument("--out-dir", required=True)
    p_cmp.add_argument("--qairt-version", default="auto", help="Target QAIRT version (e.g. 2.39, 2.44+) or 'auto'")
    p_cmp.add_argument(
        "--compiler",
        default="auto",
        choices=["auto", "sdk", "ort-ep"],
        help="Compilation backend (auto uses ort-ep for <2.40 and standalone sdk otherwise)",
    )
    p_cmp.add_argument("--graph-name")
    p_cmp.add_argument("--input-dim", action="append", default=[], help="name=1,seq or name=1,224,224,3")
    p_cmp.add_argument("--input-list")
    p_cmp.add_argument("--input-npz", help="Optional NPZ calibration inputs for ORT-QNN EP compile")
    p_cmp.add_argument("--quantization-overrides")
    p_cmp.add_argument("--qairt-root", default=None)
    p_cmp.add_argument("--backend", default="htp", choices=["cpu", "htp"])
    p_cmp.add_argument("--soc-name", default="SM7435")
    p_cmp.add_argument("--soc-model", type=int, default=61)
    p_cmp.add_argument("--dsp-arch", default="v73")
    p_cmp.add_argument("--vtcm-mb", type=int, default=2)
    p_cmp.add_argument("--opt-level", type=int, default=3)
    p_cmp.add_argument(
        "--htp-extended-opts",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Emit extended HTP graph options (auto-enabled when --qairt-version >= 2.45)",
    )
    p_cmp.add_argument("--dry-run", action="store_true")
    p_cmp.set_defaults(func=_cmd_compile)

    p_ver = sub.add_parser("verify", help="Compare two sets of probe outputs (raw & deviational cosine)")
    p_ver.add_argument("--a", required=True)
    p_ver.add_argument("--b", required=True)
    p_ver.add_argument("--a-dtype", default="u16", choices=["u16", "f32"])
    p_ver.add_argument("--b-dtype", default="f32", choices=["u16", "f32"])
    p_ver.add_argument("--a-scale", type=float, default=1.0)
    p_ver.add_argument("--a-offset", type=int, default=0)
    p_ver.add_argument("--b-scale", type=float, default=1.0)
    p_ver.add_argument("--b-offset", type=int, default=0)
    p_ver.add_argument("--min-raw-cos", type=float, default=0.98)
    p_ver.add_argument("--min-dev-cos", type=float, default=0.98)
    p_ver.add_argument("--htp-executed", action="store_true")
    p_ver.set_defaults(func=_cmd_verify)

    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
