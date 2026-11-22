# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Static parser for QNN HTP context binaries (``hnnx::Deserializer`` & ``0x71c43c9b``).

Supports outer container v3 and container v4+ (``PICKLE/`` headers), parsing
``hnnx::Deserializer`` TLV headers and ``0x71c43c9b`` constant-extent tables.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import struct
from typing import Any


QNN_CTX_MAGIC = b"\x00\x00\x00\x02\x00\x00\x00\x03"
CONST_EXTENT_TABLE_MAGIC = 0x71C43C9B
EXTENT_UNIT_BYTES = 64


@dataclass
class ConstExtentRecord:
    extent_id: int
    group_id: int
    offset_units: int
    length_units: int
    byte_offset_in_pool: int
    byte_length: int
    file_offset: int
    file_end: int


@dataclass
class DeserHeaderSummary:
    file_path: str
    file_size: int
    container_version: int
    blob_offset: int
    blob_size: int
    deser_hdr_offset: int
    deser_base: int
    deser_size: int
    deser_magic_lo: int
    deser_magic_hi: int
    deser_tag0: int
    compiler_version: str
    compiler_options: str
    sz_words: list[int]
    mm_words: list[int]
    op_class_names: list[str]
    mef_counts: list[int]
    extent_table_file_offset: int | None
    extent_data_base_file_offset: int | None
    extent_unit_bytes: int
    extent_alignment: int
    extent_groups: list[list[int]]
    extents: list[ConstExtentRecord]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["extents"] = [asdict(e) for e in self.extents]
        return d


def parse_const_extent_table(buf: bytes, offset: int = 0) -> dict[str, Any] | None:
    """Parse a ``0x71c43c9b`` ConstExtentTable from ``buf`` at ``offset``.

    Across container v3 and v4+, extent offsets and lengths are measured in
    64-byte units (``EXTENT_UNIT_BYTES = 64``), while ``groups[0][1]`` records
    the extent start alignment in bytes (e.g., 64 B in v3, 1024 B in v4+).
    """
    if offset < 0 or offset + 16 > len(buf):
        return None
    w0, w1, w2, w3 = struct.unpack_from("<4I", buf, offset)
    if w0 != CONST_EXTENT_TABLE_MAGIC:
        return None
    hdr_items = w1 >> 24
    max_records_4w = w1 & 0xFFFFFF
    n_groups = w2 & 0xFFFFFF
    n_extents = w3 & 0xFFFFFF
    if hdr_items < 1 or max_records_4w == 0 or n_groups > n_extents:
        return None
    total_items = hdr_items + n_groups + n_extents
    if total_items > max_records_4w * 4:
        return None
    table_bytes = total_items * 16
    if offset + table_bytes > len(buf):
        return None

    groups: list[list[int]] = []
    g_off = offset + hdr_items * 16
    for g in range(n_groups):
        groups.append(list(struct.unpack_from("<4I", buf, g_off + g * 16)))

    extents: list[list[int]] = []
    e_off = g_off + n_groups * 16
    for e in range(n_extents):
        extents.append(list(struct.unpack_from("<4I", buf, e_off + e * 16)))

    extent_alignment = groups[0][1] if groups and groups[0][1] > 0 else EXTENT_UNIT_BYTES
    return {
        "magic": w0,
        "hdr1": w1,
        "hdr_items": hdr_items,
        "max_records_4w": max_records_4w,
        "n_groups": n_groups,
        "n_extents": n_extents,
        "table_bytes": table_bytes,
        "unit_bytes": EXTENT_UNIT_BYTES,
        "extent_alignment": extent_alignment,
        "groups": groups,
        "raw_extents": extents,
    }


def parse_hnnx_context(path: str | Path) -> DeserHeaderSummary:
    """Parse QNN context framing, ``hnnx::Deserializer`` header TLVs, and extents."""
    p = Path(path)
    fsize = p.stat().st_size
    with p.open("rb") as f:
        head = f.read(min(fsize, 0x10000))
        if len(head) < 0x30 or head[:8] != QNN_CTX_MAGIC:
            raise ValueError(f"{p} does not start with QNN context magic")

        blob_offset = struct.unpack_from("<I", head, 0x18)[0]
        blob_size = struct.unpack_from("<Q", head, 0x20)[0]
        if blob_offset + 0x20 > fsize:
            raise ValueError(f"{p}: invalid blob_offset=0x{blob_offset:x}")

        if blob_offset + 0x200 > len(head):
            f.seek(0)
            head = f.read(min(fsize, blob_offset + 0x200))

        container_ver = struct.unpack_from("<I", head, blob_offset)[0]
        if container_ver >= 4:
            pickle_pos = head.find(
                b"PICKLE/", blob_offset, min(len(head), blob_offset + 0x200)
            )
            if pickle_pos >= blob_offset + 20:
                rel_off = ((pickle_pos - 4) & ~7) - 16
            else:
                rel_off = blob_offset + 0x70
        else:
            rel_off = blob_offset + 0x18

        if rel_off + 8 > len(head):
            raise ValueError(f"{p}: invalid container rel_off=0x{rel_off:x}")
        deser_rel = struct.unpack_from("<Q", head, rel_off)[0]

        deser_hdr_offset = blob_offset + deser_rel
        if deser_hdr_offset + 24 > fsize:
            raise ValueError(f"{p}: invalid deser_hdr_offset=0x{deser_hdr_offset:x}")

        f.seek(deser_hdr_offset)
        dh = f.read(min(0x40000, fsize - deser_hdr_offset))
        deser_size = struct.unpack_from("<Q", dh, 0)[0]
        deser_base = deser_hdr_offset + 8
        magic_lo, magic_hi, tag0 = struct.unpack_from("<3I", dh, 8)

        off = 8 + 12
        while off + 4 <= min(len(dh), 0x800) and struct.unpack_from("<I", dh, off)[0] == 0:
            off += 4

        compiler_version = ""
        compiler_options = ""
        sz_words: list[int] = []
        mm_words: list[int] = []
        op_class_names: list[str] = []
        mef_counts: list[int] = []

        for _ in range(32):
            if off + 4 > len(dh):
                break
            w = struct.unpack_from("<I", dh, off)[0]
            nwords = w & 0xFFFF
            tag = (w >> 16) & 0xFFFF
            if tag == 0 or nwords == 0 or off + nwords * 4 > len(dh):
                break
            payload = dh[off + 4 : off + nwords * 4]
            if tag == 0x7A53:  # 'Sz'
                sz_words = list(
                    struct.unpack_from(f"<{min(len(payload) // 4, 16)}I", payload, 0)
                )
            elif tag == 0x7256:  # 'Vr'
                compiler_version = payload.split(b"\x00", 1)[0].decode(
                    "ascii", errors="replace"
                )
            elif tag == 0x704F:  # 'Op'
                compiler_options = payload.split(b"\x00", 1)[0].decode(
                    "ascii", errors="replace"
                )
            elif tag == 0x6D4D:  # 'Mm'
                mm_words = list(
                    struct.unpack_from(f"<{min(len(payload) // 4, 16)}I", payload, 0)
                )
            elif tag == 0xEF4D:  # 'M\xef' class index table
                if len(payload) >= 24:
                    mef_hdr = struct.unpack_from("<6I", payload, 0)
                    mef_counts = list(mef_hdr[3:6])
                    n_op_classes = mef_hdr[4]
                    pos = 24
                    for _k in range(n_op_classes):
                        nul = payload.find(b"\x00", pos)
                        if nul < 0:
                            break
                        op_class_names.append(
                            payload[pos:nul].decode("ascii", errors="replace")
                        )
                        pos = nul + 1
            off += nwords * 4

        extent_table_off: int | None = None
        extent_data_base: int | None = None
        unit_bytes = EXTENT_UNIT_BYTES
        extent_alignment = EXTENT_UNIT_BYTES
        groups: list[list[int]] = []
        extents: list[ConstExtentRecord] = []

        cand_offsets: list[int] = []
        if sz_words:
            cand_offsets.append(deser_base + sz_words[0] * 4)

        parsed_tbl = None
        for cand in cand_offsets:
            if 0 <= cand <= fsize - 64:
                f.seek(cand)
                tbl_buf = f.read(min(0x4000, fsize - cand))
                parsed_tbl = parse_const_extent_table(tbl_buf, 0)
                if parsed_tbl is not None:
                    extent_table_off = cand
                    break

        if parsed_tbl is None:
            magic_bytes = struct.pack("<I", CONST_EXTENT_TABLE_MAGIC)
            scan_limit = min(fsize, 64 << 20)
            f.seek(0)
            chunk = f.read(scan_limit)
            pos = 0
            while True:
                idx = chunk.find(magic_bytes, pos)
                if idx < 0:
                    break
                if idx % 4 == 0:
                    pt = parse_const_extent_table(chunk, idx)
                    if pt is not None:
                        parsed_tbl = pt
                        extent_table_off = idx
                        break
                pos = idx + 4

        if parsed_tbl is not None and extent_table_off is not None:
            unit_bytes = parsed_tbl["unit_bytes"]
            extent_alignment = parsed_tbl["extent_alignment"]
            groups = parsed_tbl["groups"]
            reserve_bytes = max(0x1000, parsed_tbl["max_records_4w"] * 64)
            extent_data_base = extent_table_off + reserve_bytes
            for ext_id, grp_id, off_u, len_u in parsed_tbl["raw_extents"]:
                b_off = off_u * unit_bytes
                b_len = len_u * unit_bytes
                foff = extent_data_base + b_off
                extents.append(
                    ConstExtentRecord(
                        extent_id=ext_id,
                        group_id=grp_id,
                        offset_units=off_u,
                        length_units=len_u,
                        byte_offset_in_pool=b_off,
                        byte_length=b_len,
                        file_offset=foff,
                        file_end=foff + b_len,
                    )
                )

    return DeserHeaderSummary(
        file_path=str(p),
        file_size=fsize,
        container_version=container_ver,
        blob_offset=blob_offset,
        blob_size=blob_size,
        deser_hdr_offset=deser_hdr_offset,
        deser_base=deser_base,
        deser_size=deser_size,
        deser_magic_lo=magic_lo,
        deser_magic_hi=magic_hi,
        deser_tag0=tag0,
        compiler_version=compiler_version,
        compiler_options=compiler_options,
        sz_words=sz_words,
        mm_words=mm_words,
        op_class_names=op_class_names,
        mef_counts=mef_counts,
        extent_table_file_offset=extent_table_off,
        extent_data_base_file_offset=extent_data_base,
        extent_unit_bytes=unit_bytes,
        extent_alignment=extent_alignment,
        extent_groups=groups,
        extents=extents,
    )
