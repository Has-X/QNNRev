# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Parser and SSA/tap query helpers for ``libhtp_deser_tap.so`` artifacts.

Handles the JSONL and binary outputs emitted by ``native/htp_deser_tap.c``:
  - ``extents.jsonl``: deserialized ``0x71c43c9b`` table & DMA extract ranges
  - ``runlist.jsonl``: ordered op sequence with input/output tensor descriptors
  - ``taps_index.jsonl`` + ``taps.bin``: untiled NHWC/row-major tensor dumps
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np


QNN_DTYPE_MAP: dict[int, np.dtype] = {
    1: np.dtype("<u1"),   # QUint8
    2: np.dtype("<u2"),   # QUint16
    3: np.dtype("<i2"),   # QInt16
    4: np.dtype("<f4"),   # Float32
    5: np.dtype("<i4"),   # Int32
    6: np.dtype("<i4"),   # QInt32
    7: np.dtype("<i1"),   # QInt8
    8: np.dtype("<f2"),   # Float16
    9: np.dtype("<i8"),   # Int64
    10: np.dtype("<u2"),  # BFloat16 (raw bits)
}


@dataclass
class SSAGraph:
    ops: list[dict[str, Any]]
    producers: dict[str, tuple[int, int, dict[str, Any]]]
    consumers: dict[str, list[tuple[int, int]]]
    by_idx: dict[int, dict[str, Any]]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_runlist(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def load_taps_index(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    by_idx: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in iter_jsonl(path):
        by_idx[int(rec["idx"])].append(rec)
    return dict(by_idx)


def build_ssa_graph(ops: list[dict[str, Any]]) -> SSAGraph:
    producers: dict[str, tuple[int, int, dict[str, Any]]] = {}
    consumers: dict[str, list[tuple[int, int]]] = defaultdict(list)
    by_idx: dict[int, dict[str, Any]] = {}
    for op in ops:
        idx = int(op["idx"])
        by_idx[idx] = op
        for oi, t in enumerate(op.get("outputs", [])):
            if t and t.get("ptr"):
                producers[t["ptr"]] = (idx, oi, t)
        for ii, t in enumerate(op.get("inputs", [])):
            if t and t.get("ptr"):
                consumers[t["ptr"]].append((idx, ii))
    return SSAGraph(
        ops=ops,
        producers=producers,
        consumers=dict(consumers),
        by_idx=by_idx,
    )


def summarize_runlist(path: str | Path) -> dict[str, Any]:
    seg_counts: Counter[int] = Counter()
    op_counts: Counter[str] = Counter()
    const_count = 0
    total = 0
    ln_groups: dict[tuple[float, int], int] = {}
    input_ops: list[dict[str, Any]] = []
    output_ops: list[dict[str, Any]] = []

    for op in iter_jsonl(path):
        total += 1
        seg_counts[int(op["seg"])] += 1
        if op.get("is_const"):
            const_count += 1
        tn = op["true_name"]
        op_counts[tn] += 1
        if "InputSlice" in tn or "InputGather" in tn:
            input_ops.append(
                {"idx": op["idx"], "true_name": tn, "outputs": op.get("outputs", [])}
            )
        if "OutputSlice" in tn:
            output_ops.append(
                {"idx": op["idx"], "true_name": tn, "inputs": op.get("inputs", [])}
            )
        if "layernorm_q16_2d_normalize" in tn and op.get("outputs"):
            out = op["outputs"][0]
            if out:
                key = (round(float(out["scale"]), 12), int(out["zero_offset"]))
                ln_groups.setdefault(key, int(op["idx"]))

    return {
        "total_ops": total,
        "const_ops": const_count,
        "non_const_ops": total - const_count,
        "segments": dict(seg_counts),
        "distinct_op_classes": len(op_counts),
        "top_ops": op_counts.most_common(25),
        "distinct_layernorm_groups": len(ln_groups),
        "input_ops": input_ops,
        "output_ops": output_ops,
    }


def read_tap_array(
    tap_rec: dict[str, Any],
    taps_bin: str | Path | Any,
    *,
    dequant: bool = False,
) -> np.ndarray:
    """Read a tapped tensor slice from ``taps.bin`` as a NumPy array."""
    dt = QNN_DTYPE_MAP.get(int(tap_rec["dtype"]))
    if dt is None:
        raise ValueError(
            f"Unsupported tap dtype={tap_rec['dtype']} ({tap_rec.get('dtype_name')})"
        )

    off = int(tap_rec["bin_off"])
    length = int(tap_rec["bin_len"])
    if hasattr(taps_bin, "read"):
        taps_bin.seek(off)
        raw = taps_bin.read(length)
    else:
        with Path(taps_bin).open("rb") as f:
            f.seek(off)
            raw = f.read(length)

    arr = np.frombuffer(raw, dtype=dt)
    dims = [int(d) for d in tap_rec.get("dims", []) if int(d) > 0]
    if tap_rec.get("enc") in ("untiled_nhwc", "flat_rowmajor") and dims:
        numel = 1
        for d in dims:
            numel *= d
        if numel <= arr.size:
            arr = arr[:numel].reshape(dims)
    else:
        arr = arr.copy()

    if dequant:
        zp = float(tap_rec.get("zero_offset", 0))
        scale = float(tap_rec.get("scale", 1.0))
        return (arr.astype(np.float32) - zp) * scale
    return arr
