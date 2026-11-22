# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Tests for v3 and v4+ HTP container parsing and runlist/tap SSA helpers."""
from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import pytest

from qnnrev.core.hnnx_deser import (
    CONST_EXTENT_TABLE_MAGIC,
    EXTENT_UNIT_BYTES,
    QNN_CTX_MAGIC,
    parse_const_extent_table,
    parse_hnnx_context,
)
from qnnrev.core.runlist import (
    build_ssa_graph,
    load_runlist,
    load_taps_index,
    read_tap_array,
    summarize_runlist,
)


def _tlv_record(tag: int, payload: bytes) -> bytes:
    pad_len = (-len(payload)) % 4
    padded = payload + b"\x00" * pad_len
    nwords = 1 + len(padded) // 4
    return struct.pack("<I", (tag << 16) | nwords) + padded


def _build_synthetic_htp_context(
    *,
    container_ver: int,
    extent_alignment: int,
    compiler_version: str = "QNN HTP v2.x",
) -> tuple[bytes, bytes, bytes]:
    """Build a synthetic v3 or v4+ QNN context binary with two 64B-unit extents."""
    ext0 = bytes(range(256))
    ext1 = bytes((255 - i) & 0xFF for i in range(128))

    # Layout inside deser:
    #   [8B deser_size] + [12B magic/tag0] + TLVs + pad to table_rel
    ver_tlv = _tlv_record(0x7256, compiler_version.encode("ascii") + b"\x00")
    opt_tlv = _tlv_record(0x704F, b"-O3 --vtcm=2\x00")
    mm_tlv = _tlv_record(0x6D4D, struct.pack("<2I", 1, 2))
    mef_payload = struct.pack("<6I", 1, 0, 0, 1, 2, 0) + b"Const\x00MatMul\x00"
    mef_tlv = _tlv_record(0xEF4D, mef_payload)

    # First place a placeholder Sz TLV to measure table offset relative to deser_base
    sz_dummy = _tlv_record(0x7A53, struct.pack("<2I", 0, 0))
    tlvs_len = len(sz_dummy) + len(ver_tlv) + len(opt_tlv) + len(mm_tlv) + len(mef_tlv)
    table_rel_in_deser = 12 + tlvs_len
    assert table_rel_in_deser % 4 == 0
    sz_tlv = _tlv_record(0x7A53, struct.pack("<2I", table_rel_in_deser // 4, 0))
    deser_body_head = (
        struct.pack("<3I", 0xCAFE0001, 0xCAFE0002, 1)
        + sz_tlv
        + ver_tlv
        + opt_tlv
        + mm_tlv
        + mef_tlv
    )

    # ConstExtentTable: 1 hdr_item, 1 group, 2 extents, max_records_4w=16 (reserve=1024->0x1000)
    hdr_items = 1
    max_records_4w = 16
    n_groups = 1
    n_extents = 2
    ext0_units_len = len(ext0) // EXTENT_UNIT_BYTES
    ext1_units_len = len(ext1) // EXTENT_UNIT_BYTES
    # Place ext1 at unit 16 (1024 B) so both 64 B and 1024 B alignments hold.
    ext1_off_units = 16

    tbl = bytearray()
    tbl += struct.pack(
        "<4I",
        CONST_EXTENT_TABLE_MAGIC,
        (hdr_items << 24) | max_records_4w,
        n_groups,
        n_extents,
    )
    tbl += struct.pack("<4I", 0, extent_alignment, ext1_off_units + ext1_units_len, 0)
    tbl += struct.pack("<4I", 0, 0, 0, ext0_units_len)
    tbl += struct.pack("<4I", 1, 0, ext1_off_units, ext1_units_len)

    reserve_bytes = max(0x1000, max_records_4w * 64)
    pool_size = (ext1_off_units + ext1_units_len) * EXTENT_UNIT_BYTES
    pool = bytearray(pool_size)
    pool[0 : len(ext0)] = ext0
    pool[ext1_off_units * EXTENT_UNIT_BYTES : ext1_off_units * EXTENT_UNIT_BYTES + len(ext1)] = ext1

    deser_payload = deser_body_head + tbl + bytes(reserve_bytes - len(tbl)) + bytes(pool)
    deser_block = struct.pack("<Q", len(deser_payload)) + deser_payload

    blob_offset = 0x40
    if container_ver >= 4:
        # Container header (0x80 bytes) with PICKLE/256 and deser_rel at ((pickle_pos - 4) & ~7) - 16
        inner_hdr = bytearray(0x80)
        struct.pack_into("<I", inner_hdr, 0, container_ver)
        inner_hdr[0x30:0x3A] = b"PICKLE/256"
        deser_rel = len(inner_hdr)
        struct.pack_into("<Q", inner_hdr, 0x18, deser_rel)
    else:
        inner_hdr = bytearray(0x40)
        struct.pack_into("<I", inner_hdr, 0, container_ver)
        deser_rel = len(inner_hdr)
        struct.pack_into("<Q", inner_hdr, 0x18, deser_rel)

    blob = bytes(inner_hdr) + deser_block
    outer = bytearray(blob_offset)
    outer[0:8] = QNN_CTX_MAGIC
    struct.pack_into("<I", outer, 0x18, blob_offset)
    struct.pack_into("<Q", outer, 0x20, len(blob))
    return bytes(outer) + blob, ext0, ext1


def test_parse_const_extent_table_validation() -> None:
    bad = b"\x00" * 32
    assert parse_const_extent_table(bad, 0) is None
    assert parse_const_extent_table(bad, -1) is None

    # Valid table with v4+ 1024-byte alignment keeps unit_bytes == 64
    tbl = struct.pack(
        "<4I4I4I",
        CONST_EXTENT_TABLE_MAGIC,
        (1 << 24) | 8,
        1,
        1,
        0,
        1024,
        4,
        0,
        0,
        0,
        0,
        4,
    )
    parsed = parse_const_extent_table(tbl, 0)
    assert parsed is not None
    assert parsed["unit_bytes"] == 64
    assert parsed["extent_alignment"] == 1024
    assert parsed["n_extents"] == 1
    assert parsed["raw_extents"] == [[0, 0, 0, 4]]


@pytest.mark.parametrize(
    ("container_ver", "alignment"),
    [(3, 64), (4, 1024), (5, 1024)],
)
def test_parse_hnnx_context_containers(
    tmp_path: Path, container_ver: int, alignment: int
) -> None:
    raw, ext0, ext1 = _build_synthetic_htp_context(
        container_ver=container_ver,
        extent_alignment=alignment,
    )
    p = tmp_path / f"ctx_v{container_ver}.bin"
    p.write_bytes(raw)

    summary = parse_hnnx_context(p)
    assert summary.container_version == container_ver
    assert summary.extent_unit_bytes == 64
    assert summary.extent_alignment == alignment
    assert summary.compiler_version == "QNN HTP v2.x"
    assert summary.compiler_options == "-O3 --vtcm=2"
    assert summary.op_class_names == ["Const", "MatMul"]
    assert len(summary.extents) == 2

    e0, e1 = summary.extents
    assert raw[e0.file_offset : e0.file_end] == ext0
    assert raw[e1.file_offset : e1.file_end] == ext1
    assert summary.to_dict()["container_version"] == container_ver


def test_runlist_ssa_summary_and_tap_read(tmp_path: Path) -> None:
    rl_path = tmp_path / "runlist.jsonl"
    ops = [
        {
            "idx": 0,
            "seg": 0,
            "true_name": "Const",
            "is_const": 1,
            "inputs": [],
            "outputs": [{"ptr": "0x1000", "dims": [4]}],
        },
        {
            "idx": 1,
            "seg": 1,
            "true_name": "InputGather",
            "is_const": 0,
            "inputs": [],
            "outputs": [{"ptr": "0x2000", "dims": [1, 4]}],
        },
        {
            "idx": 2,
            "seg": 1,
            "true_name": "layernorm_q16_2d_normalize",
            "is_const": 0,
            "inputs": [{"ptr": "0x2000"}, {"ptr": "0x1000"}],
            "outputs": [{"ptr": "0x3000", "scale": 0.25, "zero_offset": -128}],
        },
        {
            "idx": 3,
            "seg": 1,
            "true_name": "OutputSlice",
            "is_const": 0,
            "inputs": [{"ptr": "0x3000"}],
            "outputs": [],
        },
    ]
    rl_path.write_text("\n".join(json.dumps(o) for o in ops) + "\n", encoding="utf-8")

    loaded = load_runlist(rl_path)
    ssa = build_ssa_graph(loaded)
    assert ssa.producers["0x2000"][0] == 1
    assert ssa.consumers["0x3000"] == [(3, 0)]

    summary = summarize_runlist(rl_path)
    assert summary["total_ops"] == 4
    assert summary["const_ops"] == 1
    assert summary["non_const_ops"] == 3
    assert summary["distinct_layernorm_groups"] == 1
    assert len(summary["input_ops"]) == 1
    assert len(summary["output_ops"]) == 1

    # Write taps_index.jsonl and taps.bin
    arr_u16 = np.array([[100, 104, 108, 112]], dtype="<u2")
    taps_bin = tmp_path / "taps.bin"
    taps_bin.write_bytes(arr_u16.tobytes())
    taps_idx = tmp_path / "taps_index.jsonl"
    rec = {
        "seq": 0,
        "idx": 2,
        "dtype": 2,
        "dtype_name": "QUint16",
        "enc": "flat_rowmajor",
        "dims": [1, 4],
        "zero_offset": 100,
        "scale": 0.5,
        "bin_off": 0,
        "bin_len": arr_u16.nbytes,
    }
    taps_idx.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    by_idx = load_taps_index(taps_idx)
    raw = read_tap_array(by_idx[2][0], taps_bin, dequant=False)
    np.testing.assert_array_equal(raw, arr_u16)
    deq = read_tap_array(by_idx[2][0], taps_bin, dequant=True)
    np.testing.assert_allclose(deq, np.array([[0.0, 2.0, 4.0, 6.0]], dtype=np.float32))
