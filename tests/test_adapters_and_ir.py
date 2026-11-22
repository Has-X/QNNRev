# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Tests for decode_general, QAIRT 2.26 / 2.39-wide adapters, IR, and validation."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import pytest

from qnnrev.adapters.qairt226 import decode_context as decode_226_context
from qnnrev.adapters.qairt239_wide import (
    KIND_MARKER as MARKER_239,
    PAYLOAD_AFTER_MARKER,
    WEIGHT_BYTES,
    a_region_weight_indices,
    analyze_context,
    analyze_tail_structure,
    compare_reference,
)
from qnnrev.core.decode226 import build_map as build_map_226
from qnnrev.core.decode_general import (
    UnsupportedVersion,
    bijection_test,
    detect_version,
    guard_version,
    layout_valid,
    locate_weight_region,
    plane_stats,
)
from qnnrev.ir import validate_ir
from qnnrev.pipeline import utility_json_to_ir
from qnnrev.validation import classify, compare, run_suite


def test_decode_general_version_bijection_and_region() -> None:
    buf_226 = b"\x00" * 16 + b"APaP\x01\x00\x00\x002.26.0.240828\x00"
    ver, off = detect_version(buf_226)
    assert ver == "2.26.0"
    assert off == 24
    assert guard_version(buf_226, "test") == ("2.26.0", 24)

    buf_239 = b"\x00" * 8 + b"APaP\x01\x00\x00\x002.39.0.250926\x00"
    assert detect_version(buf_239) == ("2.39.0", 16)
    with pytest.raises(UnsupportedVersion):
        guard_version(buf_239, "test")

    for K, N in ((32, 256), (64, 512), (256, 256)):
        assert layout_valid(K, N)
        assert bijection_test(K, N) is True

    assert not layout_valid(16, 256)

    rng = np.random.default_rng(5)
    weights = rng.integers(120, 152, size=8192, dtype=np.uint8)
    blob = b"\x00" * 1024 + weights.tobytes() + b"\x00" * 1024
    span = locate_weight_region(blob, start=0, win=2048, step=256)
    assert span is not None
    assert span[0] <= 1024 + 256

    stats = plane_stats(np.full(32 * 256, 136, dtype=np.uint8).tobytes(), 0, 32, 256)
    assert stats["min"] == 136 and stats["max"] == 136 and stats["u8_zp136_law"] is True


def test_qairt226_adapter_roundtrip_and_refusal(tmp_path: Path) -> None:
    K, N = 32, 256
    rng = np.random.default_rng(123)
    ref = rng.integers(110, 160, size=K * N, dtype=np.uint8)
    perm = build_map_226(K, N, rb0=0)
    encoded = ref[perm].tobytes()

    # Record signature: 2B tag + \x03\x13 + u32(0x0000000b) + u32(0) + u32(hlen=64)
    hdr = b"APaP\x00\x00\x00\x002.26.0\x00" + b"\x00" * 16
    rec = (
        b"\x01\x00\x03\x13"
        + struct.pack("<3I", 0x0000000B, 0, 64)
        + b"\x00" * 12
        + b"\x0d\xf0\xef\xbe"
        + b"\x00" * 44
        + encoded
    )
    ctx_path = tmp_path / "ctx_226.bin"
    ctx_path.write_bytes(hdr + rec)

    decoded = decode_226_context(ctx_path, K, N, reference_weights=ref, rb0=0)
    assert decoded["adapter"] == "qairt226-kind0xb-v1"
    assert decoded["qnn_version"] == "2.26.0"
    assert decoded["exact_vs_ref"] == K * N

    bad_ctx = tmp_path / "ctx_239_hdr.bin"
    bad_ctx.write_bytes(b"APaP\x00\x00\x00\x002.39.0\x00" + b"\x00" * 64)
    with pytest.raises(UnsupportedVersion):
        decode_226_context(bad_ctx, K, N)


def test_qairt239_wide_adapter_and_tail_structure() -> None:
    indices = a_region_weight_indices()
    assert indices.size == 5_284_096

    # Synthetic sequencer tail: 32 consts + 128B placement table + 32 consts + 16B trailer
    pat = bytes.fromhex("1c3340c0")
    place_words = np.array(
        [((i + 1) << 16) | ((i % 2) * 128) for i in range(32)], dtype="<u4"
    ).tobytes()
    tail = pat * 32 + place_words + pat * 32 + b"\xaa" * 16
    ts = analyze_tail_structure(tail)
    assert ts["const_count"] == 64
    assert ts["runs"] == 2
    assert ts["placement_tables"] == 1
    assert ts["trailing_bytes"] == 16

    # Full synthetic 2.39 wide context (5.3 MB)
    rng = np.random.default_rng(99)
    ref = rng.integers(0, 256, size=WEIGHT_BYTES, dtype=np.uint8)
    payload = bytearray(WEIGHT_BYTES)
    payload_u8 = np.frombuffer(payload, dtype=np.uint8)
    payload_u8[: indices.size] = ref[indices]

    rem_len = WEIGHT_BYTES - indices.size
    syn_tail = pat * 32 + place_words + pat * 32
    syn_tail = syn_tail + b"\x00" * (rem_len - len(syn_tail))
    payload[indices.size :] = syn_tail

    ctx = (
        b"APaP\x00\x00\x00\x002.39.0\x00"
        + MARKER_239
        + b"\x00" * (PAYLOAD_AFTER_MARKER - 4)
        + bytes(payload)
    )
    comp = compare_reference(ctx, ref)
    assert comp["exact_fraction"] == 1.0
    assert analyze_context(ctx)["qnn_version"] == "2.39.0"


def test_ir_pipeline_and_validation_suite(tmp_path: Path) -> None:
    meta = {
        "info": {
            "graphs": [
                {
                    "info": {
                        "graphName": "tiny_g",
                        "graphInputs": [
                            {
                                "info": {
                                    "id": 1,
                                    "name": "x",
                                    "type": "QNN_TENSOR_TYPE_APP_WRITE",
                                    "dimensions": [1, 4],
                                    "dataType": "QNN_DATATYPE_FLOAT_32",
                                    "quantizeParams": {"quantizationEncoding": "UNDEFINED"},
                                }
                            }
                        ],
                        "graphOutputs": [
                            {
                                "info": {
                                    "id": 2,
                                    "name": "y",
                                    "type": "QNN_TENSOR_TYPE_APP_READ",
                                    "dimensions": [1, 4],
                                    "dataType": "QNN_DATATYPE_FLOAT_32",
                                    "quantizeParams": {"quantizationEncoding": "UNDEFINED"},
                                }
                            }
                        ],
                        "nodes": [
                            {
                                "type": "MatMul",
                                "packageName": "qti.aisw",
                                "inputs": [],
                                "outputs": [],
                            }
                        ],
                    }
                }
            ]
        }
    }
    meta_path = tmp_path / "utility.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    ir = utility_json_to_ir("tiny", meta_path, unknowns=[{"region": "tail"}])
    assert validate_ir(ir) == []
    assert ir["graph"]["nodes"][0]["name"] == "tiny_g"

    w = np.eye(4, dtype=np.float32) * 2.0
    node = helper.make_node("MatMul", ["x", "W"], ["y"], name="mm")
    g = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        initializer=[numpy_helper.from_array(w, name="W")],
    )
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    src_onnx = tmp_path / "src.onnx"
    rec_onnx = tmp_path / "rec.onnx"
    onnx.save(m, str(src_onnx))
    onnx.save(m, str(rec_onnx))

    x_arr = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    x_npy = tmp_path / "x.npy"
    ref_npy = tmp_path / "ref.npy"
    np.save(x_npy, x_arr)
    np.save(ref_npy, x_arr * 2.0)

    suite_cpu = run_suite(
        tmp_path, src_onnx, rec_onnx, x_npy, ref_npy, "x", htp_executed=False
    )
    assert suite_cpu["classification"].startswith("functional_equivalence")

    suite_htp = run_suite(
        tmp_path, src_onnx, rec_onnx, x_npy, ref_npy, "x", htp_executed=True
    )
    assert suite_htp["classification"] == "exact_recovery"

    cov = compare(src_onnx, rec_onnx)
    assert classify(cov, {"max_abs_err": 0.1}, False) == "partial_reconstruction"
    assert (
        classify(
            {
                "op_coverage": {"match": False, "missing": ["X"]},
                "io_match": False,
                "weights": {},
            },
            {"max_abs_err": 1.0},
            False,
        )
        == "not_recovered"
    )
