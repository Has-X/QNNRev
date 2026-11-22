# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Metadata-to-IR pipeline helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .ir import ir_from_utility_json, validate_ir


def utility_json_to_ir(
    name: str,
    metadata_path: str | Path,
    unknowns: Sequence[Any] = (),
    source_format: str = "qnn-context-binary",
) -> dict[str, Any]:
    """Build and validate evidence-bound IR from official QNN utility JSON."""
    ir = ir_from_utility_json(
        name,
        Path(metadata_path),
        list(unknowns),
        source_format=source_format,
    )
    errors = validate_ir(ir)
    if errors:
        raise ValueError("invalid QNNRev IR: " + "; ".join(errors))
    return ir
