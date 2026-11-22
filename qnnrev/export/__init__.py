# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Reproducible ONNX-to-QNN compilation and canonical ONNX exporters."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from .bert_onnx import export_bert_onnx
from .multihead_bert_onnx import export_multihead_bert_onnx
from .vit_onnx import export_vit_onnx


def build_compile_command(
    onnx_path: str | Path,
    output_dir: str | Path,
    backend: str = "cpu",
    soc: str | None = None,
    vtcm_mb: int | None = None,
    arch: str | None = None,
    input_npz: str | Path | None = None,
) -> list[str]:
    """Build the command invoking ``qnnrev.export.ort_qnn``."""
    if backend not in ("cpu", "htp"):
        raise ValueError(f"unsupported backend: {backend}")
    cmd = [
        sys.executable,
        "-m",
        "qnnrev.export.ort_qnn",
        str(onnx_path),
        str(output_dir),
        backend,
        soc or "",
        "" if vtcm_mb is None else str(vtcm_mb),
        arch or "",
    ]
    if input_npz is not None:
        cmd.append(str(input_npz))
    return cmd


def compile_onnx(
    onnx_path: str | Path,
    output_dir: str | Path,
    backend: str = "cpu",
    soc: str | None = None,
    vtcm_mb: int | None = None,
    arch: str | None = None,
    input_npz: str | Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run ORT-QNN EP compilation and write ``compile-receipt.json``."""
    return subprocess.run(
        build_compile_command(
            onnx_path, output_dir, backend, soc, vtcm_mb, arch, input_npz=input_npz
        ),
        check=True,
    )


__all__ = [
    "build_compile_command",
    "compile_onnx",
    "export_bert_onnx",
    "export_multihead_bert_onnx",
    "export_vit_onnx",
]
