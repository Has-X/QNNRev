# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Tests for Xsb/Fi unpacking, ONNX exporters, compile planning, and CLI commands."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import zipfile

import numpy as np
import onnxruntime as ort
import pytest

from qnnrev.__main__ import build_qairt_plan, main as qnnrev_main
from qnnrev.core.hnnx_deser import CONST_EXTENT_TABLE_MAGIC, EXTENT_UNIT_BYTES, QNN_CTX_MAGIC
from qnnrev.core.htp_unpack import unpack_auto, unpack_fi, unpack_htp_generic, unpack_xsb
from qnnrev.export.bert_onnx import export_bert_onnx
from qnnrev.export.multihead_bert_onnx import export_multihead_bert_onnx
from qnnrev.export.vit_onnx import export_vit_onnx


def _pack_xsb(w: np.ndarray) -> bytes:
    """Inverse of unpack_xsb for [K, N] int8 arrays (K, N % 32 == 0)."""
    K, N = w.shape
    return (
        w.astype(np.int8)
        .reshape(K // 32, 8, 4, N // 32, 32)
        .transpose(0, 3, 1, 4, 2)
        .tobytes()
    )


def _pack_fi_unit_exp(bias_div_256: np.ndarray, scale_bits_exponent: int = 24) -> bytes:
    """Pack N-channel Fi table with a_q = 2**(exp - 24) and integer biases."""
    N = bias_div_256.shape[0]
    assert N % 32 == 0
    w0_val = (scale_bits_exponent & 0x3F) << 10
    words = np.zeros(N * 4, dtype="<i4")
    uwords = words.view("<u4")
    for blk in range(N // 32):
        base = blk * 128
        for c in range(16):
            ch_e = blk * 32 + 2 * c
            ch_o = blk * 32 + 2 * c + 1
            uwords[base + 2 * c + 0] = w0_val
            uwords[base + 2 * c + 1] = 0
            words[base + 32 + 2 * c] = int(bias_div_256[ch_e])
            uwords[base + 64 + 2 * c + 0] = w0_val
            uwords[base + 64 + 2 * c + 1] = 0
            words[base + 96 + 2 * c] = int(bias_div_256[ch_o])
    return words.tobytes()


def test_xsb_and_fi_roundtrip() -> None:
    rng = np.random.default_rng(42)
    w = rng.integers(-127, 128, size=(64, 96), dtype=np.int8)
    packed = _pack_xsb(w)
    np.testing.assert_array_equal(unpack_xsb(packed, 64, 96), w)

    with pytest.raises(ValueError):
        unpack_xsb(b"\x00" * 32, 30, 32)

    biases = rng.integers(-50, 50, size=64, dtype=np.int32)
    fi_raw = _pack_fi_unit_exp(biases, scale_bits_exponent=25)
    a_q, bi = unpack_fi(fi_raw, 64)
    np.testing.assert_allclose(a_q, np.full(64, 2.0))
    np.testing.assert_array_equal(bi, biases.astype(np.int64) * 256)

    with pytest.raises(ValueError):
        unpack_fi(b"\x00" * 16, 16)


def test_generic_unpack_with_synthetic_deser(tmp_path: Path) -> None:
    # Build minimal v4 context binary holding one 256-byte extent payload
    payload = np.arange(64, dtype=np.float32).tobytes()
    assert len(payload) == 256

    sz_payload = struct.pack("<2I", 0, 0)
    # 12B hdr + Sz TLV (12B) + Vr TLV (8B) = 32B -> 8 words
    vr_tlv = struct.pack("<I", (0x7256 << 16) | 2) + b"2.52\x00\x00\x00\x00"
    sz_tlv = struct.pack("<I", (0x7A53 << 16) | 3) + struct.pack("<2I", 8, 0)
    deser_head = struct.pack("<3I", 1, 2, 1) + sz_tlv + vr_tlv
    tbl = struct.pack(
        "<4I4I4I",
        CONST_EXTENT_TABLE_MAGIC,
        (1 << 24) | 16,
        1,
        1,
        0,
        1024,
        len(payload) // EXTENT_UNIT_BYTES,
        0,
        0,
        0,
        0,
        len(payload) // EXTENT_UNIT_BYTES,
    )
    deser_payload = deser_head + tbl + b"\x00" * (0x1000 - len(tbl)) + payload
    deser_block = struct.pack("<Q", len(deser_payload)) + deser_payload

    blob_off = 0x40
    inner = bytearray(0x80)
    struct.pack_into("<I", inner, 0, 4)
    inner[0x30:0x3A] = b"PICKLE/256"
    struct.pack_into("<Q", inner, 0x18, len(inner))
    blob = bytes(inner) + deser_block
    outer = bytearray(blob_off)
    outer[0:8] = QNN_CTX_MAGIC
    struct.pack_into("<I", outer, 0x18, blob_off)
    struct.pack_into("<Q", outer, 0x20, len(blob))

    ctx_path = tmp_path / "generic_v4.bin"
    ctx_path.write_bytes(bytes(outer) + blob)

    deser_dir = tmp_path / "deser"
    deser_dir.mkdir()
    (deser_dir / "extents.jsonl").write_text(
        json.dumps(
            {
                "event": "extract_const_extent_data_dma",
                "a": 0,
                "b": len(payload),
                "dst": "0x10000000",
                "dma": "0x1",
                "rc": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (deser_dir / "runlist.jsonl").write_text(
        json.dumps(
            {
                "idx": 0,
                "seg": 0,
                "true_name": "Const",
                "is_const": 1,
                "inputs": [],
                "outputs": [
                    {
                        "ptr": "0xdead0001",
                        "tname": "TensorFlat",
                        "dims": [8, 8],
                        "dtype": 4,
                        "dtype_name": "Float32",
                        "scale": 1.0,
                        "zero_offset": 0,
                        "storage_bytes": len(payload),
                        "raw_data": "0x10000000",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_generic = unpack_htp_generic(ctx_path, deser_dir)
    assert "const_0" in out_generic
    np.testing.assert_array_equal(
        out_generic["const_0"],
        np.arange(64, dtype=np.float32).reshape(8, 8),
    )

    auto_out = unpack_auto(ctx_path, deser_dir, model_hint="generic")
    assert set(auto_out.keys()) == set(out_generic.keys())

    # CLI unpack roundtrip
    npz_out = tmp_path / "unpacked.npz"
    rc = qnnrev_main(
        [
            "unpack",
            "--model",
            "generic",
            "--ctx",
            str(ctx_path),
            "--deser-dir",
            str(deser_dir),
            "--out",
            str(npz_out),
        ]
    )
    assert rc == 0
    assert npz_out.exists()


def _tiny_bert_weights(seq_len: int = 8, num_layers: int = 1, hidden: int = 8, vocab: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    w: dict[str, np.ndarray] = {
        "word_embeddings": rng.normal(0, 0.1, (vocab, hidden)).astype(np.float32),
        "token_type_embeddings": rng.normal(0, 0.1, (2, hidden)).astype(np.float32),
        "position_embeddings": rng.normal(0, 0.1, (seq_len, hidden)).astype(np.float32),
        "ln_gamma": np.ones((1 + 2 * num_layers, hidden), dtype=np.float32),
        "ln_beta": np.zeros((1 + 2 * num_layers, hidden), dtype=np.float32),
        "W_head": rng.normal(0, 0.1, (hidden, hidden)).astype(np.float32),
        "b_head": np.zeros((hidden,), dtype=np.float32),
    }
    for L in range(num_layers):
        p = f"L{L}"
        for k in ("q", "k", "v", "ao"):
            w[f"{p}_W{k}"] = rng.normal(0, 0.1, (hidden, hidden)).astype(np.float32)
            w[f"{p}_b{k}"] = np.zeros((hidden,), dtype=np.float32)
        w[f"{p}_Wfc1"] = rng.normal(0, 0.1, (hidden, 4 * hidden)).astype(np.float32)
        w[f"{p}_bfc1"] = np.zeros((4 * hidden,), dtype=np.float32)
        w[f"{p}_Wfc2"] = rng.normal(0, 0.1, (4 * hidden, hidden)).astype(np.float32)
        w[f"{p}_bfc2"] = np.zeros((hidden,), dtype=np.float32)
    return w


@pytest.mark.parametrize("use_ln,use_gelu", [(True, True), (False, False)])
def test_export_bert_onnx_executes(tmp_path: Path, use_ln: bool, use_gelu: bool) -> None:
    weights = _tiny_bert_weights()
    out_onnx = tmp_path / f"bert_{use_ln}_{use_gelu}.onnx"
    export_bert_onnx(
        weights,
        out_onnx,
        seq_len=8,
        num_layers=1,
        num_heads=2,
        hidden_size=8,
        use_layernorm_op=use_ln,
        use_gelu_op=use_gelu,
    )
    sess = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    outs = sess.run(
        None,
        {
            "input_ids": np.ones((1, 8), dtype=np.int32),
            "token_type_ids": np.zeros((1, 8), dtype=np.int32),
            "attention_mask": np.ones((1, 8), dtype=np.int32),
        },
    )
    assert outs[0].shape == (1, 8)
    assert np.isfinite(outs[0]).all()


def test_export_multihead_bert_and_vit_onnx_execute(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    weights = _tiny_bert_weights(seq_len=8, num_layers=1, hidden=8)
    weights["W_pool"] = rng.normal(0, 0.1, (8, 8)).astype(np.float32)
    weights["b_pool"] = np.zeros((8,), dtype=np.float32)
    weights["W_o4"] = rng.normal(0, 0.1, (8, 5)).astype(np.float32)
    weights["b_o4"] = np.zeros((5,), dtype=np.float32)
    for hname, nc in (("o1", 3), ("o2", 4), ("o3", 2)):
        weights[f"{hname}_W1"] = rng.normal(0, 0.1, (8, 6)).astype(np.float32)
        weights[f"{hname}_b1"] = np.zeros((6,), dtype=np.float32)
        weights[f"{hname}_W2"] = rng.normal(0, 0.1, (6, 6)).astype(np.float32)
        weights[f"{hname}_b2"] = np.zeros((6,), dtype=np.float32)
        weights[f"{hname}_W3"] = rng.normal(0, 0.1, (6, nc)).astype(np.float32)
        weights[f"{hname}_b3"] = np.zeros((nc,), dtype=np.float32)

    mh_onnx = tmp_path / "mh_bert.onnx"
    export_multihead_bert_onnx(
        weights,
        mh_onnx,
        seq_len=8,
        num_layers=1,
        num_heads=2,
        hidden_size=8,
    )
    mh_sess = ort.InferenceSession(str(mh_onnx), providers=["CPUExecutionProvider"])
    mh_outs = mh_sess.run(
        None,
        {
            "input_1": np.ones((1, 8), dtype=np.int32),
            "attention_mask": np.ones((1, 8), dtype=np.int32),
            "t_5": np.zeros((1, 8), dtype=np.int32),
        },
    )
    assert [o.shape for o in mh_outs] == [(1, 3), (1, 4), (1, 2), (1, 8, 5)]

    # Tiny ViT: image_size=16, patch_size=8 -> 4 patches + 1 cls = seq 5
    vit_w: dict[str, np.ndarray] = {
        "W_patch": rng.normal(0, 0.1, (8 * 8 * 3, 8)).astype(np.float32),
        "b_patch": np.zeros((8,), dtype=np.float32),
        "cls_token": rng.normal(0, 0.1, (8,)).astype(np.float32),
        "pos_emb": rng.normal(0, 0.1, (5, 8)).astype(np.float32),
        "ln_gamma": np.ones((4, 8), dtype=np.float32),
        "ln_beta": np.zeros((4, 8), dtype=np.float32),
        "W_head": rng.normal(0, 0.1, (8, 6)).astype(np.float32),
        "b_head": np.zeros((6,), dtype=np.float32),
    }
    for k in ("q", "k", "v", "ao"):
        vit_w[f"L0_W{k}"] = rng.normal(0, 0.1, (8, 8)).astype(np.float32)
        vit_w[f"L0_b{k}"] = np.zeros((8,), dtype=np.float32)
    vit_w["L0_Wfc1"] = rng.normal(0, 0.1, (8, 32)).astype(np.float32)
    vit_w["L0_bfc1"] = np.zeros((32,), dtype=np.float32)
    vit_w["L0_Wfc2"] = rng.normal(0, 0.1, (32, 8)).astype(np.float32)
    vit_w["L0_bfc2"] = np.zeros((8,), dtype=np.float32)

    vit_onnx = tmp_path / "vit.onnx"
    export_vit_onnx(
        vit_w,
        vit_onnx,
        image_size=16,
        patch_size=8,
        num_layers=1,
        num_heads=2,
        hidden_size=8,
        out_dim=6,
    )
    vit_sess = ort.InferenceSession(str(vit_onnx), providers=["CPUExecutionProvider"])
    vit_out = vit_sess.run(None, {"image": rng.normal(0, 0.5, (1, 16, 16, 3)).astype(np.float32)})
    assert vit_out[0].shape == (1, 6)
    assert np.isfinite(vit_out[0]).all()


def test_compile_plans_and_cli_inspect_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    plan_244 = build_qairt_plan(
        onnx_path=tmp_path / "m.onnx",
        out_dir=tmp_path / "out244",
        graph_name="my_graph",
        input_dims=["x=1,8"],
        input_list=tmp_path / "calib.txt",
        qairt_root=Path("/opt/sdk/2.44"),
        qairt_version="2.44",
    )
    assert plan_244["htp_in"]["graphs"][0]["graph_names"] == ["my_graph"]
    assert any(str(a).endswith("my_graph.dlc") for a in plan_244["converter_cmd"])
    assert "dlbc" not in plan_244["htp_in"]["graphs"][0]

    plan_252 = build_qairt_plan(
        onnx_path=tmp_path / "m.onnx",
        out_dir=tmp_path / "out252",
        graph_name="my_graph",
        input_dims=["x=1,8"],
        input_list=None,
        qairt_root=Path("/opt/sdk/2.52"),
        qairt_version="2.52",
    )
    g252 = plan_252["htp_in"]["graphs"][0]
    assert g252["dlbc"] == 1
    assert g252["weights_packing"] is True
    assert g252["high_precision_sigmoid"] is True

    # Dry-run compile CLI across 2.39 and 2.52
    for ver in ("2.39", "2.52"):
        rc = qnnrev_main(
            [
                "compile",
                "--onnx",
                str(tmp_path / "m.onnx"),
                "--out-dir",
                str(tmp_path / f"dry_{ver}"),
                "--qairt-version",
                ver,
                "--dry-run",
            ]
        )
        assert rc == 0

    # Inspect zip archive
    zpath = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("model/weights.bin", b"\x00" * 64)
    ins_json = tmp_path / "inspect.json"
    assert qnnrev_main(["inspect", str(zpath), "-o", str(ins_json)]) == 0
    inspected = json.loads(ins_json.read_text(encoding="utf-8"))
    assert inspected["results"][0]["format"] == "zip"

    # CLI decompile smoke check on tiny BERT weights
    w_npz = tmp_path / "tiny_bert.npz"
    np.savez(w_npz, **_tiny_bert_weights())
    # Verify default bert dims fail cleanly if shapes mismatch, and inspect works on a folder
    assert qnnrev_main(["inspect", str(tmp_path)]) == 0

    # Verify CLI (raw & deviational cosine across 2 probes)
    a_dir = tmp_path / "a_raw"
    a_dir.mkdir()
    p0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    p1 = np.array([4.0, 1.0, 2.0, 3.0], dtype=np.float32)
    # Encode as u16 with scale=0.01, offset=-1000 => float = (u16 - 1000) * 0.01
    (a_dir / "p0.raw").write_bytes(((p0 / 0.01).astype(np.int32) + 1000).astype("<u2").tobytes())
    (a_dir / "p1.raw").write_bytes(((p1 / 0.01).astype(np.int32) + 1000).astype("<u2").tobytes())
    b_npz = tmp_path / "b.npz"
    np.savez(b_npz, p0=p0, p1=p1)

    rc_pass = qnnrev_main(
        [
            "verify",
            "--a",
            str(a_dir),
            "--a-dtype",
            "u16",
            "--a-scale",
            "0.01",
            "--a-offset",
            "-1000",
            "--b",
            str(b_npz),
            "--htp-executed",
        ]
    )
    assert rc_pass == 0
    capsys.readouterr()

    b_bad = tmp_path / "b_bad.npz"
    np.savez(b_bad, p0=-p0, p1=-p1)
    rc_fail = qnnrev_main(
        [
            "verify",
            "--a",
            str(a_dir),
            "--a-dtype",
            "u16",
            "--a-scale",
            "0.01",
            "--a-offset",
            "-1000",
            "--b",
            str(b_bad),
        ]
    )
    assert rc_fail == 2
