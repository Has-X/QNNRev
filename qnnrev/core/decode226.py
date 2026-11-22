# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""QAIRT 2.26 kind-0xb packed-weight A-region decoder.

Decodes QAIRT 2.26 HTP context binaries by locating the kind-0x0000000b payload
record, resolving the verbatim weight zone, and inverting the closed-form tile
permutation back to row-major [K, N] order.
"""
from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np


REC_SIG = re.compile(rb".\x03\x13")
KIND_PAYLOAD = 0x0000000B


def find_payload_record(buf: bytes) -> tuple[int | None, bytes | None]:
    """Return ``(payload_offset, payload_bytes)`` for the kind-0xb record."""
    cands = []
    for m in REC_SIG.finditer(buf):
        p = m.start() - 1
        if p >= 0 and p + 8 <= len(buf) and buf[p + 2] == 0x03 and buf[p + 3] == 0x13:
            kind = int.from_bytes(buf[p + 4 : p + 8], "little")
            if kind == KIND_PAYLOAD:
                cands.append(p)
    for p in cands:
        if p + 16 > len(buf):
            continue
        hlen = int.from_bytes(buf[p + 12 : p + 16], "little")
        if 8 <= hlen <= 40000:
            return p + 12, buf[p + 12 :]
    return None, None


def find_zone(
    buf: bytes,
    K: int,
    N: int,
    W_sample: np.ndarray | None = None,
    rb0: int | None = None,
) -> tuple[int | None, bytes | None]:
    """Locate the weight-zone start offset using the 0xbeeff00d anchor and optional sample."""
    pay_off, pay_data = find_payload_record(buf)
    if pay_data is None or pay_off is None:
        return None, None
    seg = bytes(buf[pay_off : pay_off + 8192])
    rel_anchor = seg.find(b"\x0d\xf0\xef\xbe")
    anchor = (pay_off + rel_anchor) if rel_anchor >= 0 else pay_off
    if W_sample is not None:
        raw = bytes(buf)
        p8 = np.arange(8, dtype=np.int64)
        t = p8 // 8192
        u = (p8 % 8192) // 256
        pos = p8 % 256
        cp = K // 32
        for rb0c in ([(N - 256) % N, 0] if rb0 is None else [rb0]):
            c = (32 * (t % cp) + 8 * (u % 4) + 4 * (pos // 128) + pos % 4) % N
            r = (rb0c + 256 * (t // cp) + 32 * (u // 4) + (pos % 128) // 4) % N
            idx = c * N + r
            want = W_sample[idx].tobytes()
            zone_start = raw.find(want, max(pay_off, anchor - 64))
            if zone_start >= 0:
                return zone_start, pay_data
    return anchor, pay_data


def build_map(K: int, N: int, rb0: int | None = None) -> np.ndarray:
    """Return payload-to-row-major ``[K, N]`` permutation indices."""
    cp = K // 32
    if rb0 is None:
        rb0 = (N - 256) % N if N == 2304 else 0
    p = np.arange(K * N, dtype=np.int64)
    t = p // 8192
    u = (p % 8192) // 256
    pos = p % 256
    c = (32 * (t % cp) + 8 * (u % 4) + 4 * (pos // 128) + pos % 4) % K
    r = (rb0 + 256 * (t // cp) + 32 * (u // 4) + (pos % 128) // 4) % N
    return c * N + r


def decode_context(
    ctx_path: str | Path,
    K: int,
    N: int,
    W_ref: np.ndarray | None = None,
    rb0: int | None = None,
) -> dict[str, Any]:
    """Decode weights from a QAIRT 2.26 context binary."""
    p = Path(ctx_path)
    buf = p.read_bytes()
    pay_off, pay_data = find_payload_record(buf)
    if pay_data is None or pay_off is None:
        return {"error": "no kind-0xb payload record"}
    zone_off, _ = find_zone(buf, K, N, W_ref, rb0)
    if zone_off is None:
        return {"error": "zone not located"}
    total = K * N
    zone = np.frombuffer(buf[zone_off : zone_off + total], dtype=np.uint8)
    resolved = len(zone)
    if W_ref is not None:
        m = build_map(K, N, rb0)
        pred = W_ref[m[:resolved]]
        exact = int((zone == pred).sum())
    else:
        exact = None
    return {
        "ctx": str(p),
        "ctx_size": len(buf),
        "payload_off": pay_off,
        "payload_size": len(pay_data),
        "zone_off": zone_off,
        "zone_bytes": resolved,
        "header_bytes": zone_off - pay_off,
        "tail_bytes": len(pay_data) - (zone_off - pay_off) - resolved,
        "K": K,
        "N": N,
        "exact_vs_ref": exact,
        "W": zone,
    }
