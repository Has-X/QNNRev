# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""QAIRT 2.26 adapter for the validated kind-0xb packed-weight decoder."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from qnnrev.core.decode226 import decode_context as _decode_context
from qnnrev.core.decode_general import UnsupportedVersion, guard_version


def decode_context(
    context_path: str | Path,
    K: int,
    N: int,
    reference_weights: Any = None,
    rb0: int | None = None,
) -> dict[str, Any]:
    """Decode a supported QAIRT 2.26 context; unsupported versions fail loudly."""
    path = Path(context_path)
    version, version_offset = guard_version(
        path.read_bytes(), "qnnrev.adapters.qairt226"
    )
    if version != "2.26.0":
        raise UnsupportedVersion(f"2.26 adapter refuses QNN {version}")
    result = _decode_context(path, K, N, reference_weights, rb0)
    if "error" in result:
        raise ValueError(f"2.26 context decode failed: {result['error']}")
    result.update(
        {
            "adapter": "qairt226-kind0xb-v1",
            "qnn_version": version,
            "version_offset": version_offset,
        }
    )
    return result
