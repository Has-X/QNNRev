# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Level-1 artifact format detector and metadata inspector."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any
import zipfile


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_format(path: str | Path) -> dict[str, Any]:
    """Detect container format (ZIP, MiNN NNiM wrapper, raw QNN context, or unknown)."""
    p = str(path)
    out: dict[str, Any] = {
        "path": p,
        "bytes": os.path.getsize(p),
        "sha256": _sha256_file(p),
    }
    with open(p, "rb") as f:
        head = f.read(8)
    if zipfile.is_zipfile(p):
        out["format"] = "zip"
    elif head[:4] == b"NNiM":
        out["format"] = "minn"
    elif head[:8] == b"\x00\x00\x00\x02\x00\x00\x00\x03":
        out["format"] = "qnn-context-raw"
    else:
        out["format"] = "unknown"
    return out


def minn_header(head: bytes, member_size: int) -> dict[str, Any]:
    """Parse a 30-byte ``NNiM`` wrapper header and validate payload boundaries."""
    if len(head) < 30 or head[:4] != b"NNiM":
        return {"magic": False}
    ver = struct.unpack_from("<I", head, 4)[0]
    fmt = struct.unpack_from("<I", head, 8)[0]
    qnn_minor = struct.unpack_from("<I", head, 12)[0]
    total_len = int.from_bytes(head[22:28], "little")
    return {
        "magic": True,
        "header_version": ver,
        "format": hex(fmt),
        "qnn_minor_version": qnn_minor,
        "total_content_len": total_len,
        "boundary_validated": total_len == member_size - 30,
        "layout": "wrapper=30B; content=[30, 30+total_len); tail=member_size-30-total_len",
    }


def xor36_strings(data: bytes, minlen: int = 6, limit: int = 24) -> list[dict[str, Any]]:
    """Extract ASCII runs under a single-byte ``0x36`` XOR mask."""
    dec = bytes(b ^ 0x36 for b in data)
    runs = [
        (m.start(), m.group())
        for m in re.finditer(rb"[\x20-\x7e]{%d,}" % minlen, dec)
    ]
    return [{"offset": pos, "text": r[:100].decode("latin1")} for pos, r in runs[:limit]]


def inspect_zip(path: str | Path) -> dict[str, Any]:
    """Inspect model bundle ZIP members (contexts, NNiM payloads, DLCs, runtime libs)."""
    r: dict[str, Any] = {"container": "zip", "context_members": []}
    keys = ("serialized", ".minn", ".dlc")
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.endswith("/") or not any(k in n for k in keys):
                continue
            info = z.getinfo(n)
            m: dict[str, Any] = {
                "name": n,
                "bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
            }
            if n.endswith(".minn"):
                with z.open(n) as f:
                    head = f.read(4096)
                if head[:4] == b"NNiM":
                    m["magic"] = "NNiM"
                    m["minn_header"] = minn_header(head, info.file_size)
                else:
                    m["magic"] = head[:4].hex()
            r["context_members"].append(m)
        r["qnn_runtime_dirs"] = sorted(
            {x.split("/")[1] for x in z.namelist() if "qnn_lib" in x and "/" in x}
        )
    return r


def sdk_json(path_json: str | Path) -> dict[str, Any] | None:
    """Summarize official ``qnn-context-binary-utility`` JSON metadata if available."""
    p = Path(path_json)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    info = d.get("info", {})
    out: dict[str, Any] = {
        "buildId": info.get("buildId"),
        "coreApiVersion": info.get("coreApiVersion"),
        "backendApiVersion": info.get("backendApiVersion"),
        "contextBlobVersion": info.get("contextBlobVersion"),
        "spillFillBufferSize": [
            h["info"].get("spillFillBufferSize")
            for h in info.get("hwInfoBlob", [])
            if isinstance(h, dict) and isinstance(h.get("info"), dict)
        ],
    }
    graphs = []
    for g in info.get("graphs", []):
        gi = g.get("info", {})
        graphs.append(
            {
                "name": gi.get("graphName"),
                "inputs": [
                    {
                        "name": t["info"]["name"],
                        "type": t["info"]["dataType"],
                        "dims": t["info"]["dimensions"],
                    }
                    for t in gi.get("graphInputs", [])
                ],
                "outputs": [
                    {
                        "name": t["info"]["name"],
                        "type": t["info"]["dataType"],
                        "dims": t["info"]["dimensions"],
                    }
                    for t in gi.get("graphOutputs", [])
                ],
            }
        )
    out["graphs"] = graphs
    return out


def inspect(path: str | Path, sdk_dir: str | Path | None = None) -> dict[str, Any]:
    """Inspect a raw QNN context binary, NNiM wrapper, or ZIP bundle."""
    p = str(path)
    d = detect_format(p)
    if d["format"] == "zip":
        d.update(inspect_zip(p))
    elif d["format"] == "qnn-context-raw":
        name = os.path.basename(p)
        base_dir = Path(sdk_dir) if sdk_dir is not None else Path("reports/sdk-json")
        sdk_data = None
        for cand in (
            base_dir / f"{name}.json",
            base_dir / f"{os.path.splitext(name)[0]}.json",
        ):
            sdk_data = sdk_json(cand)
            if sdk_data is not None:
                break
        d["sdk"] = sdk_data or {
            "_note": "run qnn-context-binary-utility for official metadata"
        }
        d["heur"] = {"_class": "heuristic-plain"}
        with open(p, "rb") as f:
            blob = f.read(1 << 20)
        runs = [
            (m.start(), m.group()) for m in re.finditer(rb"[\x20-\x7e]{6,}", blob)
        ]
        d["heur"]["strings_first1MB"] = [
            {"offset": pos, "text": r[:100].decode("latin1")}
            for pos, r in runs[:12]
        ]
    elif d["format"] == "minn":
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            head = f.read(4096)
            d["minn_header"] = minn_header(head, size)
            d["boundary"] = {
                "validated": d["minn_header"].get("boundary_validated", False),
                "candidate": None,
            }
            f.seek(min(30, size))
            d["heur"] = {
                "_class": "heuristic-xor36",
                "strings": xor36_strings(f.read(1 << 20), limit=16),
            }
    return d
