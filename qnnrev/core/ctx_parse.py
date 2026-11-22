# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Bounded raw QNN context header and string scanner."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import struct
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_qnn_context(
    path: str | Path,
    string_limit: int = 1 << 20,
    max_strings: int = 24,
) -> dict[str, Any]:
    """Parse bounded header metadata and ASCII strings from a raw QNN context file."""
    p = Path(path)
    data = p.read_bytes()
    n = len(data)
    out: dict[str, Any] = {"path": str(p), "bytes": n, "sha256": _sha256(data)}
    if n >= 64:
        out["head_u32"] = list(struct.unpack_from("<16I", data))
    else:
        out["head_u32"] = list(struct.unpack_from(f"<{n // 4}I", data))
    out["looks_like_qnn_context"] = bool(
        n > 8 and data[:8] == b"\x00\x00\x00\x02\x00\x00\x00\x03"
    )
    scan = data[: min(n, string_limit)]
    strs = [
        (m.start(), m.group().decode("latin1"))
        for m in re.finditer(rb"[\x20-\x7e]{6,}", scan)
    ]
    out["strings"] = [{"offset": pos, "text": s[:80]} for pos, s in strs[:max_strings]]
    out["strings_truncated"] = len(strs) > max_strings
    out["graph_name_candidates"] = [
        s for _, s in strs if re.match(r"^[A-Za-z][A-Za-z0-9_]*_\d{8,14}$", s)
    ]
    sample = data[: min(n, 2 << 20)]
    out["zero_share_sample"] = round(sample.count(0) / len(sample), 3) if sample else 0.0
    return out
