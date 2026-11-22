# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""QAIRT 2.39 wide-MatMul A-region analyzer (2304x2304 probe family).

Exposes the closed-form A-region permutation and the kernel-sequencer metadata
tail classifier. The 129-byte boundary remainder after the mapped A-region
prefix is the head of the QAIRT 2.39 kernel-sequencer metadata region (runs of
32 f32 spill-fill constants ``0x1c3340c0`` = ``-3.003119468688965`` interleaved
with 128-byte ``(delta << 16 | val)`` placement tables and a trailing pointer
region), not weight data.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from qnnrev.core.decode_general import UnsupportedVersion, detect_version

K = N = 2304
WEIGHT_BYTES = K * N
KIND_MARKER = (0x3000000B).to_bytes(4, "little")
PAYLOAD_AFTER_MARKER = 4 + 29_040
A_REGION_REPORTED_BYTES = 5_284_225
A_REGION_MAPPED_BYTES = 5_284_096
RBASE = np.asarray([2048, 1024, 1280, 1536, 1792, 0, 256, 512, 768], dtype=np.int32)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_context(context: str | Path | bytes | bytearray) -> tuple[bytes, str | None]:
    if isinstance(context, (str, Path)):
        path = Path(context)
        return path.read_bytes(), str(path)
    return bytes(context), None


def _payload(data: bytes) -> tuple[int, int, bytes]:
    starts = []
    pos = 0
    while True:
        pos = data.find(KIND_MARKER, pos)
        if pos < 0:
            break
        starts.append(pos)
        pos += 1
    if len(starts) != 1:
        raise ValueError(
            f"expected one 2.39 kind-0x3000000b record, found {len(starts)}"
        )
    record_offset = starts[0]
    payload_offset = record_offset + PAYLOAD_AFTER_MARKER
    if payload_offset + WEIGHT_BYTES != len(data):
        raise ValueError(
            f"unsupported 2.39 context extent: payload candidate {payload_offset:#x} "
            f"+ {WEIGHT_BYTES} bytes != file size {len(data)}"
        )
    return record_offset, payload_offset, data[payload_offset:]


@lru_cache(maxsize=1)
def a_region_weight_indices() -> np.ndarray:
    """Map the verified payload prefix to row-major ``[C, R]`` weight indices."""
    p = np.arange(A_REGION_MAPPED_BYTES, dtype=np.int32)
    tile = p // 8192
    intra = p % 8192
    cb = (88 + 32 * tile) % N
    block = ((tile + 3) // 72) % len(RBASE)
    rbase = RBASE[block].copy()
    u0 = intra < 256

    wrap = u0 & (tile >= 69) & (((tile - 69) % 72) == 0)
    rbase[wrap] = RBASE[(block[wrap] - 1) % len(RBASE)]

    c = np.empty_like(p)
    r = np.empty_like(p)
    c[u0] = cb[u0] + 4 * (intra[u0] // 128) + intra[u0] % 4
    r[u0] = rbase[u0] + 224 + (intra[u0] % 128) // 4

    s = intra[~u0] - 256
    c[~u0] = cb[~u0] + 8 + 8 * ((s % 1024) // 256) + 4 * ((s % 256) // 128) + s % 4
    r[~u0] = rbase[~u0] + 32 * (s // 1024) + (s % 128) // 4
    c %= N

    indices = (c * N + r).astype(np.int32, copy=False)
    if int(indices.min()) < 0 or int(indices.max()) >= WEIGHT_BYTES:
        raise RuntimeError("2.39 A-region map produced an out-of-bounds weight index")
    if np.unique(indices).size != A_REGION_MAPPED_BYTES:
        raise RuntimeError("2.39 A-region map is not bijective over its verified prefix")
    indices.flags.writeable = False
    return indices


def analyze_tail_structure(tail: bytes) -> dict[str, Any]:
    """Decode the QAIRT 2.39 kernel-sequencer metadata tail."""
    pat = bytes.fromhex("1c3340c0")
    positions = []
    pos = tail.find(pat)
    while pos >= 0:
        positions.append(pos)
        pos = tail.find(pat, pos + 1)
    positions_arr = np.asarray(positions, dtype=np.int64)
    if positions_arr.size == 0 or positions_arr[0] != 0:
        raise ValueError("metadata tail: first 1c3340c0 constant not at tail offset 0")
    if positions_arr.size % 32 != 0:
        raise ValueError(
            f"metadata tail: {positions_arr.size} consts is not a multiple of 32"
        )
    diffs = np.diff(positions_arr)
    if not np.isin(diffs, [4, 132]).all():
        raise ValueError(
            f"metadata tail: unexpected inter-const spacing {sorted(set(diffs.tolist()))}"
        )
    gap_offsets = np.where(diffs == 132)[0]
    blocks = [tail[positions_arr[g] + 4 : positions_arr[g + 1]] for g in gap_offsets]
    words = [np.frombuffer(b, dtype="<u4") for b in blocks]
    val = np.concatenate([w & 0xFFFF for w in words]) if words else np.array([], dtype=np.uint32)
    delta = np.concatenate([(w >> 16) & 0xFFFF for w in words]) if words else np.array([], dtype=np.uint32)
    delta_signed = np.where(delta > 32767, delta.astype(np.int64) - 65536, delta)
    if val.size > 0 and not (val % 128 == 0).all():
        raise ValueError("metadata tail: placement val not 128-byte aligned")
    return {
        "kind": "kernel-sequencer-metadata-v1",
        "const_count": int(positions_arr.size),
        "expected_const_count": K,
        "const_value_f32": -3.003119468688965,
        "const_u32": 0x1C3340C0,
        "runs": int(positions_arr.size // 32),
        "run_size_consts": 32,
        "placement_tables": len(blocks),
        "placement_words": int(val.size),
        "val_mod256_classes": {str(k): int((val % 256 == k).sum()) for k in (0, 128)},
        "delta_range": [int(delta_signed.min()), int(delta_signed.max())] if delta_signed.size else [0, 0],
        "trailing_bytes": int(len(tail) - positions_arr[-1] - 4),
    }


def analyze_context(context: str | Path | bytes | bytearray) -> dict[str, Any]:
    """Return A-region codes/indices and explicit metadata-tail summary."""
    data, source_path = _read_context(context)
    version, version_offset = detect_version(data)
    if version != "2.39.0":
        raise UnsupportedVersion(f"2.39 wide adapter refuses version {version!r}")
    record_offset, payload_offset, payload = _payload(data)
    indices = a_region_weight_indices()
    mapped_codes = np.frombuffer(payload, dtype=np.uint8, count=A_REGION_MAPPED_BYTES)
    unresolved = payload[A_REGION_MAPPED_BYTES:]
    return {
        "adapter": "qairt239-wide-matmul-a-region-v1",
        "source_path": source_path,
        "context_sha256": _sha256(data),
        "qnn_version": version,
        "version_offset": version_offset,
        "record_marker_offset": record_offset,
        "payload_offset": payload_offset,
        "payload_bytes": len(payload),
        "weight_shape": [K, N],
        "reported_a_region_bytes": A_REGION_REPORTED_BYTES,
        "mapped_a_region_bytes": A_REGION_MAPPED_BYTES,
        "reported_a_boundary_unmapped_bytes": A_REGION_REPORTED_BYTES - A_REGION_MAPPED_BYTES,
        "bytes_after_mapped_prefix": len(unresolved),
        "unresolved_tail_sha256": _sha256(unresolved),
        "tail_structure": analyze_tail_structure(unresolved),
        "mapped_weight_indices": indices,
        "mapped_codes": mapped_codes,
    }


def compare_reference(
    context: str | Path | bytes | bytearray,
    reference_weights: str | Path | bytes | bytearray | memoryview | np.ndarray,
) -> dict[str, Any]:
    """Compare mapped codes with a known raw byte matrix for validation."""
    result = analyze_context(context)
    if isinstance(reference_weights, (str, Path)):
        reference = np.fromfile(reference_weights, dtype=np.uint8)
    elif isinstance(reference_weights, (bytes, bytearray, memoryview)):
        reference = np.frombuffer(reference_weights, dtype=np.uint8)
    else:
        reference = np.asarray(reference_weights, dtype=np.uint8).reshape(-1)
    if reference.size != WEIGHT_BYTES:
        raise ValueError(f"reference has {reference.size} bytes; expected {WEIGHT_BYTES}")
    idx = result["mapped_weight_indices"]
    codes = result["mapped_codes"]
    exact = reference[idx] == codes
    return {
        "context_sha256": result["context_sha256"],
        "reference_sha256": _sha256(reference.tobytes()),
        "qnn_version": result["qnn_version"],
        "record_marker_offset": result["record_marker_offset"],
        "payload_offset": result["payload_offset"],
        "mapped_bytes": int(exact.size),
        "exact_bytes": int(exact.sum()),
        "exact_fraction": float(exact.mean()),
        "reported_a_region_bytes": A_REGION_REPORTED_BYTES,
        "a_region_coverage_fraction": A_REGION_MAPPED_BYTES / A_REGION_REPORTED_BYTES,
        "unmapped_boundary_bytes": A_REGION_REPORTED_BYTES - A_REGION_MAPPED_BYTES,
        "bytes_after_mapped_prefix": result["bytes_after_mapped_prefix"],
    }
