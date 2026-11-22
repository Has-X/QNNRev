# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Evidence-bound QNNRev intermediate representation (IR v0.1)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


IR_VERSION = "0.1"
FORMATS = ("qnn-context-binary", "dlc", "dspx", "htp-context")


def build_ir(
    name: str,
    *,
    source_format: str,
    qnn_version: str,
    nodes: list[dict[str, Any]],
    tensors: list[dict[str, Any]],
    io: dict[str, Any],
    provenance: dict[str, Any],
    unknowns: Sequence[Any],
    notes: list[str] | None = None,
) -> dict[str, Any]:
    if source_format not in FORMATS:
        raise ValueError(f"Unsupported source_format: {source_format}")
    return {
        "qnnrev_ir_version": IR_VERSION,
        "model": name,
        "source_format": source_format,
        "qnn_version": qnn_version,
        "provenance": provenance,
        "graph": {"nodes": nodes, "tensors": tensors, "io": io},
        "unknowns": list(unknowns),
        "notes": notes or [],
    }


def validate_ir(ir: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    if ir.get("qnnrev_ir_version") != IR_VERSION:
        errs.append("IR version mismatch")
    tnames = {t["name"] for t in ir["graph"]["tensors"]}
    for n in ir["graph"]["nodes"]:
        for k in ("inputs", "outputs"):
            for t in n.get(k, []):
                if t not in tnames:
                    errs.append(f"node {n.get('name')}: tensor '{t}' not declared")
        for f in ("name", "op"):
            if f not in n:
                errs.append(f"node missing {f}: {n}")
    for k in ("inputs", "outputs"):
        for t in ir["graph"]["io"].get(k, []):
            if t["name"] not in tnames:
                errs.append(f"io tensor '{t['name']}' not declared")
    for t in ir["graph"]["tensors"]:
        if t.get("role") == "constant" and not any(
            p.get("tensor") == t["name"] for p in ir["provenance"].get("constants", [])
        ):
            errs.append(f"constant '{t['name']}' has no provenance entry")
    return errs


def ir_from_utility_json(
    name: str,
    util_json_path: str | Path,
    unknowns: Sequence[Any],
    source_format: str = "qnn-context-binary",
) -> dict[str, Any]:
    d = json.loads(Path(util_json_path).read_text(encoding="utf-8"))
    info = d["info"]
    g = info["graphs"][0]["info"]

    def _tdump(t: dict[str, Any]) -> dict[str, Any]:
        ti = t["info"]
        role = ti["type"].replace("QNN_TENSOR_TYPE_", "").lower()
        role = (
            "io-input"
            if role == "app_write"
            else "io-output"
            if role == "app_read"
            else role
        )
        return {
            "name": ti["name"],
            "dtype": ti["dataType"],
            "dims": ti["dimensions"],
            "role": role,
            "quant": ti.get("quantizeParams", {}),
        }

    in_tensors = [_tdump(t) for t in g.get("graphInputs", [])]
    out_tensors = [_tdump(t) for t in g.get("graphOutputs", [])]
    tensors = in_tensors + out_tensors
    return build_ir(
        name,
        source_format=source_format,
        qnn_version=info.get("buildId", ""),
        nodes=[
            {
                "name": g["graphName"],
                "op": "GRAPH",
                "src_evidence": "official utility JSON",
                "inputs": [t["name"] for t in in_tensors],
                "outputs": [t["name"] for t in out_tensors],
            }
        ],
        tensors=tensors,
        io={"inputs": in_tensors, "outputs": out_tensors},
        provenance={
            "sources": ["qnn-context-binary-utility official metadata"],
            "constants": [],
        },
        unknowns=unknowns,
    )


__all__ = ["IR_VERSION", "FORMATS", "build_ir", "validate_ir", "ir_from_utility_json"]
