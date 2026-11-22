# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Version guards, 2.26 A-region layout envelope, and statistical region locator."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import numpy as np


SUPPORTED_VERSIONS = ("2.26",)
VERSION_RE = re.compile(rb"2\.(?P<v>[0-9]{2})\.[0-9]")


class UnsupportedVersion(RuntimeError):
    """Raised when a binary carries an absent or unsupported QNN/QAIRT version."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_version(buf: bytes) -> tuple[str | None, int | None]:
    """Return ``(version_string, offset)`` for the first ``2.XX.0`` hit, or ``(None, None)``."""
    m = VERSION_RE.search(buf)
    if m is None:
        return None, None
    return f"2.{m.group('v').decode()}.0", m.start()


def guard_version(buf: bytes, component: str) -> tuple[str, int]:
    """Validate that ``buf`` contains a supported QNN version string."""
    v, off = detect_version(buf)
    if v is None or off is None:
        raise UnsupportedVersion(
            f"{component}: no QNN version string found — refusing to guess"
        )
    if not v.startswith(SUPPORTED_VERSIONS):
        raise UnsupportedVersion(
            f"{component}: QNN {v} not supported (supported: {SUPPORTED_VERSIONS})"
        )
    return v, off


def build_map(K: int, N: int, rb0: int | None = None) -> np.ndarray:
    """Closed-form A-region layout map ``payload[p] = W_flat[map[p]]`` for ``[K, N]``."""
    if K % 32 != 0:
        raise ValueError(f"K={K} must be a multiple of 32")
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


def layout_valid(K: int, N: int) -> bool:
    """Check whether ``(K, N)`` lies inside the bijective 2.26 A-region envelope."""
    if K <= 0 or N <= 0 or K % 32 != 0 or (K * N) % 8192 != 0:
        return False
    cp = K // 32
    tiles = (K * N) // 8192
    r_max = 256 * ((tiles - 1) // cp) + 255
    return r_max < N


def bijection_test(K: int, N: int) -> bool:
    """Return True if ``(K, N)`` is within envelope and ``build_map`` is bijective."""
    if not layout_valid(K, N):
        return False
    m = build_map(K, N)
    return len(np.unique(m)) == K * N


def locate_weight_region(
    buf: bytes,
    start: int = 0x100000,
    win: int = 4096,
    step: int = 512,
    code_lo: int = 100,
    code_hi: int = 175,
    opaque_std: float = 30.0,
) -> tuple[int, int] | None:
    """Locate the contiguous narrow-distribution u8 weight region before opaque tails."""
    a = np.frombuffer(buf, dtype=np.uint8)
    n = len(a)
    i = start
    first = None
    while i < n - win:
        w = a[i : i + win].astype(np.int32)
        if w.min() >= code_lo and w.max() <= code_hi and w.std() < 12.0:
            first = i
            break
        i += step
    if first is None:
        return None
    end = n
    i = first + win
    bad_since = None
    while i < n - win:
        w = a[i : i + win].astype(np.int32)
        bad = w.std() > opaque_std or w.min() < code_lo - 40 or w.max() > 255
        if bad:
            if bad_since is None:
                bad_since = i
            elif i - bad_since > 8 * win:
                end = bad_since
                break
        else:
            bad_since = None
            end = i + win
        i += step
    return first, end


def plane_stats(
    buf: bytes | np.ndarray,
    off: int,
    K: int,
    N: int,
    layout: str = "verbatim",
) -> dict[str, Any]:
    """Compute distribution statistics for a ``[K, N]`` byte plane at ``off``."""
    zone = np.frombuffer(buf[off : off + K * N], dtype=np.uint8)
    if layout != "verbatim":
        zone = zone[build_map(K, N)]
    return {
        "off": off,
        "K": K,
        "N": N,
        "zp_est": float(zone.mean()),
        "std": float(zone.std()),
        "min": int(zone.min()),
        "max": int(zone.max()),
        "u8_zp136_law": bool(abs(float(zone.mean()) - 136.0) < 8.0),
    }
