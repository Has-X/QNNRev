# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Tapless HTP weight, scale, and LayerNorm unpackers over ``hnnx::Deserializer`` runlists."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .hnnx_deser import parse_hnnx_context
from .runlist import SSAGraph, build_ssa_graph, load_runlist


@dataclass
class HTPUnpackContext:
    ctx_path: Path
    ctx_bytes: bytes
    extents_map: List[Tuple[int, int, int]]  # (lo_va, hi_va, file_offset)
    ops: List[Dict[str, Any]]
    ssa: SSAGraph
    deser_extents: List[Any]
    f32_head: np.ndarray

    @classmethod
    def load(cls, ctx_path: Path | str, deser_dir: Path | str) -> "HTPUnpackContext":
        cpath = Path(ctx_path)
        ddir = Path(deser_dir)
        ctx_bytes = cpath.read_bytes()
        summary = parse_hnnx_context(cpath)
        ops = load_runlist(ddir / "runlist.jsonl")
        ssa = build_ssa_graph(ops)
        dma_ev = [
            json.loads(line)
            for line in (ddir / "extents.jsonl").read_text(encoding="utf-8").splitlines()
            if "extract_const_extent_data_dma" in line
        ]
        ext_map = [
            (int(ev["dst"], 16), int(ev["dst"], 16) + int(ev["b"]), ex.file_offset)
            for ev, ex in zip(dma_ev, summary.extents)
        ]
        head_end = summary.extents[0].file_offset if summary.extents else len(ctx_bytes)
        f32_head = np.frombuffer(ctx_bytes[: (head_end // 4) * 4], dtype="<f4")
        return cls(
            ctx_path=cpath,
            ctx_bytes=ctx_bytes,
            extents_map=ext_map,
            ops=ops,
            ssa=ssa,
            deser_extents=summary.extents,
            f32_head=f32_head,
        )

    def ptr_to_file_off(self, raw_ptr_str: Optional[str]) -> Optional[int]:
        if not raw_ptr_str or raw_ptr_str == "(nil)":
            return None
        p = int(raw_ptr_str, 16)
        for lo, hi, foff in self.extents_map:
            if lo <= p < hi:
                return foff + (p - lo)
        return None

    def trace_const_info(
        self, ptr: Optional[str]
    ) -> Tuple[Optional[int], Optional[int], Optional[Dict[str, Any]]]:
        cur = ptr
        seen = set()
        while cur and cur not in seen:
            seen.add(cur)
            prod = self.ssa.producers.get(cur)
            if not prod:
                break
            op = self.ssa.by_idx[prod[0]]
            if op["true_name"] == "Const":
                o = op["outputs"][0]
                return op["idx"], self.ptr_to_file_off(o.get("raw_data")), o
            if not op["inputs"]:
                break
            cur = op["inputs"][0].get("ptr")
        return None, None, None


def unpack_xsb(raw: bytes, K: int, N: int) -> np.ndarray:
    """Unpack HTP Xsb weight tile layout ``(K/32, N/32, 8, 32, 4)`` int8 to ``[K, N]``."""
    if K % 32 != 0 or N % 32 != 0:
        raise ValueError(f"Xsb requires K, N multiples of 32; got ({K}, {N})")
    return (
        np.frombuffer(raw, dtype=np.int8)
        .reshape(K // 32, N // 32, 8, 32, 4)
        .transpose(0, 2, 4, 1, 3)
        .reshape(K, N)
    )


def unpack_fi(raw: bytes, N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack HTP Fi interleaved (``a_q`` scale + ``int32`` bias) per-32 channel block."""
    if N % 32 != 0:
        raise ValueError(f"Fi requires N multiple of 32; got {N}")
    b_i32 = np.frombuffer(raw, dtype="<i4")
    b_u32 = np.frombuffer(raw, dtype="<u4")
    sc_raw = np.zeros((N, 2), dtype=np.uint32)
    bi_raw = np.zeros(N, dtype=np.int64)
    for blk in range(N // 32):
        base = blk * 128
        sc_raw[blk * 32 + 0 : blk * 32 + 32 : 2] = b_u32[base + 0 : base + 32].reshape(16, 2)
        bi_raw[blk * 32 + 0 : blk * 32 + 32 : 2] = b_i32[base + 32 : base + 64 : 2]
        sc_raw[blk * 32 + 1 : blk * 32 + 32 : 2] = b_u32[base + 64 : base + 96].reshape(16, 2)
        bi_raw[blk * 32 + 1 : blk * 32 + 32 : 2] = b_i32[base + 96 : base + 128 : 2]
    w0, w1 = sc_raw[:, 0].astype(np.int64), sc_raw[:, 1].astype(np.int64)
    exp_f16 = (w0 >> 10) & 0x3F
    mant_hi = w1 & 0x3FF
    mant_lo = w0 & 0x3FF
    a_q = (1.0 + (mant_hi + mant_lo / 1024.0) / 1024.0) * (2.0 ** (exp_f16 - 24))
    return a_q, bi_raw * 256


def decode_conv_s1(
    ctx: HTPUnpackContext,
    op: Dict[str, Any],
    out_channels: Optional[int] = None,
) -> Dict[str, Any]:
    """Decode one const ``ConvLayer_s1`` op into quantized and FP32 linear parameters."""
    K = int(op["inputs"][1]["dims"][2])
    N_decl = int(op["inputs"][1]["dims"][3])
    N_eff = N_decl if out_channels is None else int(out_channels)
    N_pad = ((N_decl + 31) // 32) * 32
    in_sc = float(op["inputs"][0]["scale"])
    in_zp = int(op["inputs"][0]["zero_offset"])
    out_sc = float(op["outputs"][0]["scale"])
    out_zp = int(op["outputs"][0]["zero_offset"])
    _, w_off, _ = ctx.trace_const_info(op["inputs"][1]["ptr"])
    _, b_off, b_desc = ctx.trace_const_info(op["inputs"][2]["ptr"])
    if w_off is None or b_off is None or b_desc is None:
        raise ValueError(f"Op {op['idx']} missing const extent offset")
    W_i8 = unpack_xsb(ctx.ctx_bytes[w_off : w_off + K * N_pad], K, N_pad)[:, :N_eff]
    a_q, eff_b = unpack_fi(ctx.ctx_bytes[b_off : b_off + b_desc["storage_bytes"]], N_pad)
    a_q = a_q[:N_eff]
    eff_b = eff_b[:N_eff]
    w_scale = (a_q * out_sc) / in_sc
    W_f32 = (W_i8.astype(np.float64) * w_scale[None, :]).astype(np.float32)
    b_f32 = (
        (eff_b.astype(np.float64) + in_zp * W_i8.astype(np.int64).sum(axis=0))
        * (a_q * out_sc)
    ).astype(np.float32)
    return {
        "K": K,
        "N": N_eff,
        "W_i8": W_i8,
        "a_q": a_q,
        "eff_b": eff_b,
        "W_f32": W_f32,
        "b_f32": b_f32,
        "in_sc": in_sc,
        "in_zp": in_zp,
        "out_sc": out_sc,
        "out_zp": out_zp,
        "w_off": w_off,
        "b_off": b_off,
    }


def concat_conv_group(
    ctx: HTPUnpackContext,
    ops: List[Dict[str, Any]],
    out_channels: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode and concatenate split ``ConvLayer_s1`` ops along output channels."""
    unique_by_woff: Dict[int, Dict[str, Any]] = {}
    for op in ops:
        _, w_off, _ = ctx.trace_const_info(op["inputs"][1]["ptr"])
        if w_off is None:
            raise ValueError(f"Op {op['idx']} missing w_off")
        if w_off not in unique_by_woff:
            unique_by_woff[w_off] = op
    parts = [
        decode_conv_s1(ctx, unique_by_woff[k], out_channels=out_channels)
        for k in sorted(unique_by_woff)
    ]
    return (
        np.concatenate([p["W_f32"] for p in parts], axis=1).astype(np.float32),
        np.concatenate([p["b_f32"] for p in parts], axis=0).astype(np.float32),
    )


def extract_ln_gscale(ctx: HTPUnpackContext, out_scale: float) -> float:
    """Recover the LayerNorm Q15 gamma float32 scale taplessly from the context header."""
    head_end = ctx.deser_extents[0].file_offset if ctx.deser_extents else len(ctx.ctx_bytes)
    pat = struct.pack("<f", float(out_scale))
    matches: List[float] = []
    pos = 0
    while True:
        pos = ctx.ctx_bytes.find(pat, pos, head_end)
        if pos < 0:
            break
        if ctx.ctx_bytes[pos + 12 : pos + 16] == pat and pos + 28 <= head_end:
            matches.append(float(struct.unpack("<f", ctx.ctx_bytes[pos + 24 : pos + 28])[0]))
        pos += 1
    if len(matches) != 1:
        raise ValueError(
            f"Expected unique gscale struct for LN out_scale={out_scale}; found {len(matches)}"
        )
    return matches[0]


def extract_ln_params_from_norms(
    ctx: HTPUnpackContext, norm_ops: List[Dict[str, Any]]
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Taplessly extract ``(gamma_f32, beta_f32, out_sc, out_zp)`` from one LayerNorm."""
    out0 = norm_ops[0]["outputs"][0]
    out_sc, out_zp = float(out0["scale"]), int(out0["zero_offset"])
    by_off2: Dict[int, Tuple[int, int]] = {}
    for op in norm_ops:
        _, off2, _ = ctx.trace_const_info(op["inputs"][2]["ptr"])
        _, off3, _ = ctx.trace_const_info(op["inputs"][3]["ptr"])
        if off2 is None or off3 is None:
            raise ValueError(f"LN op {op['idx']} missing Fh/Fi const offset")
        by_off2.setdefault(off2, (off2, off3))
    v2_parts, v3_parts = [], []
    for off2 in sorted(by_off2):
        o2, o3 = by_off2[off2]
        v2_parts.append(np.frombuffer(ctx.ctx_bytes[o2 : o2 + 1024], dtype="<i2")[0::2])
        v3_parts.append(np.frombuffer(ctx.ctx_bytes[o3 : o3 + 1024], dtype="<i4"))
    v2 = np.concatenate(v2_parts).astype(np.float64)
    v3 = np.concatenate(v3_parts).astype(np.float64)
    gscale = extract_ln_gscale(ctx, out_sc)
    gamma = (v2 * gscale).astype(np.float32)
    beta = ((v3 - out_zp) * out_sc).astype(np.float32)
    return gamma, beta, out_sc, out_zp


def extract_ln_vectors_only(ctx: HTPUnpackContext, ln_ops: List[Dict[str, Any]]):
    """Extract raw Q15 integer gamma vector and output-grid shifted beta."""
    out0 = ln_ops[0]["outputs"][0]
    out_sc, out_zp = float(out0["scale"]), int(out0["zero_offset"])
    in2_vecs, in3_vecs = [], []
    for gop in ln_ops[0::2]:
        n2 = ctx.ssa.by_idx[ctx.ssa.producers[gop["inputs"][2]["ptr"]][0]]["inputs"][0]["ptr"]
        n3 = ctx.ssa.by_idx[ctx.ssa.producers[gop["inputs"][3]["ptr"]][0]]["inputs"][0]["ptr"]
        bo = ctx.ptr_to_file_off(ctx.ssa.by_idx[ctx.ssa.producers[n2][0]]["outputs"][0]["raw_data"])
        go = ctx.ptr_to_file_off(ctx.ssa.by_idx[ctx.ssa.producers[n3][0]]["outputs"][0]["raw_data"])
        in2_vecs.append(np.frombuffer(ctx.ctx_bytes[bo : bo + 1024], dtype="<i2")[0::2])
        in3_vecs.append(np.frombuffer(ctx.ctx_bytes[go : go + 1024], dtype="<i4"))
    v2 = np.concatenate(in2_vecs).astype(np.float64)
    v3 = np.concatenate(in3_vecs).astype(np.float64)
    beta = ((v3 - out_zp) * out_sc).astype(np.float32)
    return v2, beta, out_sc, out_zp


def _collect_layernorms(ctx: HTPUnpackContext) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    ln_groups: Dict[Tuple[float, int], List[Dict[str, Any]]] = {}
    for op in ctx.ops:
        if "layernorm_q16_2d_normalize" not in op["true_name"]:
            continue
        out = op["outputs"][0]
        key = (round(float(out["scale"]), 12), int(out["zero_offset"]))
        ln_groups.setdefault(key, []).append(op)
    gammas, betas, scales = [], [], []
    for gops in ln_groups.values():
        g, b, sc, _ = extract_ln_params_from_norms(ctx, gops)
        gammas.append(g)
        betas.append(b)
        scales.append(sc)
    return gammas, betas, scales


def unpack_bert_profile(ctx_path: Path | str, deser_dir: Path | str) -> Dict[str, np.ndarray]:
    """Taplessly unpack a Post-LN BERT encoder context (embeddings, LNs, layers, linear head)."""
    ctx = HTPUnpackContext.load(ctx_path, deser_dir)
    weights: Dict[str, np.ndarray] = {}

    word_cands: List[Tuple[int, np.ndarray]] = []
    tt_cands: List[Tuple[int, np.ndarray]] = []
    pos_chunks: List[Tuple[int, np.ndarray]] = []

    for op in ctx.ops:
        if op["true_name"] != "Const":
            continue
        out = op["outputs"][0]
        dims = tuple(int(d) for d in out["dims"])
        off = ctx.ptr_to_file_off(out.get("raw_data"))
        if off is None:
            continue
        sc, zp = float(out["scale"]), int(out["zero_offset"])
        dname = out.get("dtype_name", "")
        if len(dims) == 4 and dims[0] == 1 and dims[1] == 1 and dname == "QUint8":
            rows, cols = dims[2], dims[3]
            arr = np.frombuffer(ctx.ctx_bytes[off : off + rows * cols], dtype=np.uint8).reshape(rows, cols)
            deq = ((arr.astype(np.float32) - zp) * sc).astype(np.float32)
            if rows >= 1000:
                word_cands.append((rows, deq))
            elif 2 <= rows <= 8:
                tt_cands.append((rows, deq))
        elif dims == (1, 8, 16, 256) and dname == "QUint16":
            arr = np.frombuffer(ctx.ctx_bytes[off : off + 8 * 16 * 256 * 2], dtype="<u2").reshape(8, 16, 256)
            pos_chunks.append((off, ((arr.astype(np.float32) - zp) * sc).astype(np.float32)))

    if len(word_cands) != 1 or len(tt_cands) != 1:
        raise ValueError(
            f"Expected unique word/token_type embeddings in BERT profile; got {len(word_cands)}, {len(tt_cands)}"
        )
    weights["word_embeddings"] = word_cands[0][1]
    weights["token_type_embeddings"] = tt_cands[0][1]
    hidden_size = int(weights["word_embeddings"].shape[1])

    if not pos_chunks or len(pos_chunks) % 4 != 0:
        raise ValueError(f"Expected crouton position_embeddings chunks (multiple of 4), found {len(pos_chunks)}")
    pos_chunks.sort(key=lambda x: x[0])
    n_col_groups = len(pos_chunks) // 4
    pos_cols = [
        np.stack([pos_chunks[c * 4 + k][1] for k in range(4)], axis=1).reshape(-1, 256)
        for c in range(n_col_groups)
    ]
    weights["position_embeddings"] = np.concatenate(pos_cols, axis=1).astype(np.float32)

    gammas, betas, _ = _collect_layernorms(ctx)
    if len(gammas) < 3 or (len(gammas) - 1) % 2 != 0:
        raise ValueError(f"Expected 2*L+1 LayerNorms in BERT profile, found {len(gammas)}")
    num_layers = (len(gammas) - 1) // 2
    weights["ln_gamma"] = np.stack(gammas, axis=0)
    weights["ln_beta"] = np.stack(betas, axis=0)

    convs = [op for op in ctx.ops if op["true_name"].startswith("ConvLayer_s1")]
    by_in: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for op in convs:
        _, w_off, _ = ctx.trace_const_info(op["inputs"][1]["ptr"])
        if w_off is not None:
            by_in[op["inputs"][0]["ptr"]].append(op)
    groups = list(by_in.values())

    kq_groups = [
        g for g in groups
        if len(g) == 15 and {tuple(o["inputs"][1]["dims"]) for o in g} == {(1, 1, hidden_size, 64), (1, 1, hidden_size, 256)}
    ]
    v_groups = [
        g for g in groups
        if len(g) == hidden_size // 64 and all(o["inputs"][1]["dims"] == [1, 1, hidden_size, 64] for o in g)
    ]
    ao_groups = [
        g for g in groups
        if len(g) == hidden_size // 256
        and g[0]["inputs"][0]["dims"][2] > 1
        and g[0]["inputs"][1]["dims"] == [1, 1, hidden_size, 256]
    ]
    fc1_groups = [
        g for g in groups
        if len(g) == (4 * hidden_size) // 64 and g[0]["inputs"][1]["dims"] == [1, 1, hidden_size, 64]
    ]
    fc2_groups = [
        g for g in groups
        if len(g) == hidden_size // 64 and g[0]["inputs"][1]["dims"] == [1, 1, 4 * hidden_size, 64]
    ]
    head_groups = [
        g for g in groups
        if len(g) == hidden_size // 256
        and g[0]["inputs"][0]["dims"] == [1, 1, 1, hidden_size]
        and g[0]["inputs"][1]["dims"] == [1, 1, hidden_size, 256]
    ]

    if not (
        len(kq_groups) == len(v_groups) == len(ao_groups) == len(fc1_groups) == len(fc2_groups) == num_layers
        and len(head_groups) == 1
    ):
        raise ValueError(
            f"Unexpected BERT conv group counts for L={num_layers}: "
            f"kq={len(kq_groups)} v={len(v_groups)} ao={len(ao_groups)} "
            f"fc1={len(fc1_groups)} fc2={len(fc2_groups)} head={len(head_groups)}"
        )

    for b in range(num_layers):
        p = f"L{b}_"
        k_ops = [o for o in kq_groups[b] if o["inputs"][1]["dims"] == [1, 1, hidden_size, 256]]
        q_ops = [o for o in kq_groups[b] if o["inputs"][1]["dims"] == [1, 1, hidden_size, 64]]
        weights[p + "Wq"], weights[p + "bq"] = concat_conv_group(ctx, q_ops)
        weights[p + "Wk"], weights[p + "bk"] = concat_conv_group(ctx, k_ops)
        weights[p + "Wv"], weights[p + "bv"] = concat_conv_group(ctx, v_groups[b])
        weights[p + "Wao"], weights[p + "bao"] = concat_conv_group(ctx, ao_groups[b])
        weights[p + "Wfc1"], weights[p + "bfc1"] = concat_conv_group(ctx, fc1_groups[b])
        weights[p + "Wfc2"], weights[p + "bfc2"] = concat_conv_group(ctx, fc2_groups[b])

    weights["W_head"], weights["b_head"] = concat_conv_group(ctx, head_groups[0])
    return weights


def unpack_multihead_bert_profile(
    ctx_path: Path | str,
    deser_dir: Path | str,
    *,
    slot_classes: int = 27,
) -> Dict[str, np.ndarray]:
    """Taplessly unpack a multi-head Post-LN BERT + pooler + 3 MLP heads + slot head context."""
    ctx = HTPUnpackContext.load(ctx_path, deser_dir)
    weights: Dict[str, np.ndarray] = {}

    word_cands: List[np.ndarray] = []
    tt_cands: List[np.ndarray] = []
    pos_chunks: List[Tuple[int, np.ndarray]] = []
    for op in ctx.ops:
        if op["true_name"] != "Const":
            continue
        out = op["outputs"][0]
        dims = tuple(int(d) for d in out["dims"])
        off = ctx.ptr_to_file_off(out.get("raw_data"))
        if off is None:
            continue
        sc, zp = float(out["scale"]), int(out["zero_offset"])
        dname = out.get("dtype_name", "")
        if len(dims) == 4 and dims[0] == 1 and dims[1] == 1 and dname == "QUint8":
            rows, cols = dims[2], dims[3]
            arr = np.frombuffer(ctx.ctx_bytes[off : off + rows * cols], dtype=np.uint8).reshape(rows, cols)
            deq = ((arr.astype(np.float32) - zp) * sc).astype(np.float32)
            if rows >= 1000:
                word_cands.append(deq)
            elif 2 <= rows <= 8:
                tt_cands.append(deq)
        elif len(dims) == 4 and dims[0] == 1 and dims[1] == 1 and dims[3] == 256 and dname == "QUint16":
            rows = dims[2]
            arr = np.frombuffer(ctx.ctx_bytes[off : off + rows * 256 * 2], dtype="<u2").reshape(rows, 256)
            pos_chunks.append((off, ((arr.astype(np.float32) - zp) * sc).astype(np.float32)))

    if len(word_cands) != 1 or len(tt_cands) != 1 or len(pos_chunks) != 2:
        raise ValueError("Unexpected embedding structure in multi-head BERT profile")
    weights["word_embeddings"] = word_cands[0]
    weights["token_type_embeddings"] = tt_cands[0]
    weights["position_embeddings"] = np.concatenate(
        [c for _, c in sorted(pos_chunks, key=lambda x: x[0])], axis=1
    )
    hidden_size = int(weights["word_embeddings"].shape[1])
    seq_len = int(weights["position_embeddings"].shape[0])

    gammas, betas, _ = _collect_layernorms(ctx)
    if len(gammas) < 3 or (len(gammas) - 1) % 2 != 0:
        raise ValueError(f"Expected 2*L+1 LayerNorms in multi-head BERT profile, found {len(gammas)}")
    num_layers = (len(gammas) - 1) // 2
    weights["ln_gamma"] = np.stack(gammas, axis=0)
    weights["ln_beta"] = np.stack(betas, axis=0)

    convs = [op for op in ctx.ops if op["true_name"].startswith("ConvLayer_s1")]
    by_in: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for op in convs:
        by_in[op["inputs"][0]["ptr"]].append(op)
    groups = list(by_in.values())

    qkv_groups = [
        g for g in groups
        if len(g) == 18 and g[0]["inputs"][0]["dims"] == [1, 1, seq_len, hidden_size]
    ]
    fc1_groups = [
        g for g in groups
        if len(g) == 48 and g[0]["inputs"][1]["dims"] == [1, 1, hidden_size, 64]
    ]
    fc2_groups = [
        g for g in groups
        if len(g) == 8 and g[0]["inputs"][1]["dims"] == [1, 1, 6 * hidden_size, 64]
    ]
    ao_by_woff_set: Dict[Tuple[int, ...], List[Dict[str, Any]]] = {}
    for g in groups:
        if (
            g[0]["inputs"][1]["dims"] == [1, 1, hidden_size, 256]
            and 1 < g[0]["inputs"][0]["dims"][2] <= seq_len
        ):
            wset = tuple(sorted({ctx.trace_const_info(o["inputs"][1]["ptr"])[1] for o in g}))
            if len(wset) == 2 and wset not in ao_by_woff_set:
                ao_by_woff_set[wset] = g
    ao_groups = list(ao_by_woff_set.values())
    if not (
        len(qkv_groups) == len(ao_groups) == len(fc1_groups) == len(fc2_groups) == num_layers
    ):
        raise ValueError(
            f"Unexpected multi-head BERT block counts for L={num_layers}: "
            f"qkv={len(qkv_groups)} ao={len(ao_groups)} fc1={len(fc1_groups)} fc2={len(fc2_groups)}"
        )

    for b in range(num_layers):
        p = f"L{b}_"
        qkv = qkv_groups[b]
        k_ops = [o for o in qkv if o["inputs"][1]["dims"][3] == 256]
        q_ops, v_ops = [], []
        for o in qkv:
            if o["inputs"][1]["dims"][3] == 64:
                cons = [
                    ctx.ssa.by_idx[c]["true_name"]
                    for c, _ in ctx.ssa.consumers.get(o["outputs"][0]["ptr"], [])
                ]
                if any("space_rearrange" in cn for cn in cons):
                    q_ops.append(o)
                else:
                    v_ops.append(o)
        weights[p + "Wq"], weights[p + "bq"] = concat_conv_group(ctx, q_ops)
        weights[p + "Wk"], weights[p + "bk"] = concat_conv_group(ctx, k_ops)
        weights[p + "Wv"], weights[p + "bv"] = concat_conv_group(ctx, v_ops)
        weights[p + "Wao"], weights[p + "bao"] = concat_conv_group(ctx, ao_groups[b])
        weights[p + "Wfc1"], weights[p + "bfc1"] = concat_conv_group(ctx, fc1_groups[b])
        weights[p + "Wfc2"], weights[p + "bfc2"] = concat_conv_group(ctx, fc2_groups[b])

    pool_ops = [
        o for o in convs
        if o["inputs"][0]["dims"] == [1, 1, 1, hidden_size]
        and o["inputs"][1]["dims"] == [1, 1, hidden_size, 256]
        and any(
            "tanh_16" in ctx.ssa.by_idx[c]["true_name"]
            for c, _ in ctx.ssa.consumers.get(o["outputs"][0]["ptr"], [])
        )
    ]
    weights["W_pool"], weights["b_pool"] = concat_conv_group(ctx, pool_ops)

    post_pool_convs = [o for o in convs if o["idx"] > max(po["idx"] for po in pool_ops)]
    w1_sorted = sorted(
        [o for o in post_pool_convs if o["inputs"][1]["dims"] == [1, 1, hidden_size, 256]],
        key=lambda o: ctx.trace_const_info(o["inputs"][1]["ptr"])[1],
    )
    w2_sorted = sorted(
        [o for o in post_pool_convs if o["inputs"][1]["dims"] == [1, 1, 6 * hidden_size, 256]],
        key=lambda o: ctx.trace_const_info(o["inputs"][1]["ptr"])[1],
    )
    if len(w1_sorted) != 36 or len(w2_sorted) != 6:
        raise ValueError(
            f"Expected 36 W1 and 6 W2 head ops; got {len(w1_sorted)}, {len(w2_sorted)}"
        )

    out_slices = [op for op in ctx.ops if "OutputSlice" in op["true_name"]]
    for out_op in out_slices:
        head_idx = int(out_op["inputs"][3]["dims"][3])
        last_conv = ctx.ssa.by_idx[ctx.ssa.producers[out_op["inputs"][0]["ptr"]][0]]
        if head_idx == 3:
            weights["W_o4"], weights["b_o4"] = concat_conv_group(
                ctx, [last_conv], out_channels=slot_classes
            )
            continue
        hname = f"o{head_idx + 1}"
        n_cls = int(last_conv["inputs"][1]["dims"][3])
        weights[f"{hname}_W1"], weights[f"{hname}_b1"] = concat_conv_group(
            ctx, w1_sorted[head_idx * 12 : (head_idx + 1) * 12]
        )
        weights[f"{hname}_W2"], weights[f"{hname}_b2"] = concat_conv_group(
            ctx, w2_sorted[head_idx * 2 : (head_idx + 1) * 2]
        )
        weights[f"{hname}_W3"], weights[f"{hname}_b3"] = concat_conv_group(
            ctx, [last_conv], out_channels=n_cls
        )

    return weights


def unpack_vit_profile(ctx_path: Path | str, deser_dir: Path | str) -> Dict[str, np.ndarray]:
    """Taplessly unpack patch/pos/cls embeddings, LayerNorms, ViT blocks, and head."""
    ctx = HTPUnpackContext.load(ctx_path, deser_dir)
    weights: Dict[str, np.ndarray] = {}

    cls_consts: List[Tuple[int, np.ndarray]] = []
    pos_consts: List[Tuple[int, int, np.ndarray]] = []
    for op in ctx.ops:
        if op["true_name"] != "Const":
            continue
        out = op["outputs"][0]
        off = ctx.ptr_to_file_off(out.get("raw_data"))
        if off is None:
            continue
        dims = tuple(out["dims"])
        sc, zp = float(out["scale"]), int(out["zero_offset"])
        if dims == (1, 1, 1, 128) and int(out["storage_bytes"]) == 256:
            arr = (np.frombuffer(ctx.ctx_bytes[off : off + 256], dtype="<u2").astype(np.float32) - zp) * sc
            cls_consts.append((off, arr.astype(np.float32)))
        elif dims in ((1, 1, 68, 256), (1, 1, 53, 256)):
            n_rows = dims[2]
            arr = (
                np.frombuffer(ctx.ctx_bytes[off : off + n_rows * 256 * 2], dtype="<u2")
                .reshape(n_rows, 256)
                .astype(np.float32)
                - zp
            ) * sc
            pos_consts.append((off, n_rows, arr.astype(np.float32)))

    if len(cls_consts) != 8 or len(pos_consts) != 16:
        raise ValueError(
            f"Expected 8 cls_token and 16 pos_emb chunks in ViT profile; got {len(cls_consts)}, {len(pos_consts)}"
        )
    cls_consts.sort(key=lambda x: x[0])
    pos_consts.sort(key=lambda x: x[0])
    weights["cls_token"] = np.concatenate([x[1] for x in cls_consts], axis=0).astype(np.float32)
    weights["pos_emb"] = np.concatenate(
        [np.concatenate([pos_consts[cg * 4 + k][2] for k in range(4)], axis=0) for cg in range(4)],
        axis=1,
    ).astype(np.float32)
    pos_min_off = pos_consts[0][0]
    pos_max_off = pos_consts[-1][0]

    gammas, betas, ln_scales = _collect_layernorms(ctx)
    if len(gammas) < 4 or (len(gammas) - 2) % 2 != 0:
        raise ValueError(f"Expected 2*L+2 LayerNorms in ViT profile, found {len(gammas)}")
    num_layers = (len(gammas) - 2) // 2
    weights["ln_gamma"] = np.stack(gammas, axis=0)
    weights["ln_beta"] = np.stack(betas, axis=0)

    convs = [op for op in ctx.ops if op["true_name"].startswith("ConvLayer_s1")]
    by_woff: Dict[int, Dict[str, Any]] = {}
    for op in convs:
        _, w_off, _ = ctx.trace_const_info(op["inputs"][1]["ptr"])
        if w_off is not None and w_off not in by_woff:
            by_woff[w_off] = op
    sorted_convs = [by_woff[w] for w in sorted(by_woff)]

    patch_ops = [op for op in sorted_convs if op["inputs"][1]["dims"] == [1, 1, 608, 256]]
    if len(patch_ops) != 4:
        raise ValueError(f"Expected 4 patch_emb ops in ViT profile, found {len(patch_ops)}")
    W_patch_raw, b_patch = concat_conv_group(ctx, patch_ops)
    weights["W_patch"] = W_patch_raw[:588, :].astype(np.float32)
    weights["b_patch"] = b_patch.astype(np.float32)

    qkv_ops = [
        op
        for w, op in sorted(by_woff.items())
        if op["inputs"][1]["dims"] == [1, 1, 1024, 64] and w < pos_min_off
    ]
    ao_ops = [
        op
        for w, op in sorted(by_woff.items())
        if op["inputs"][1]["dims"] == [1, 1, 1024, 64] and w > pos_max_off
    ]
    fc2_ops = [op for op in sorted_convs if op["inputs"][1]["dims"] == [1, 1, 4096, 256]]

    out_slices = [op for op in ctx.ops if "OutputSlice" in op["true_name"]]
    if len(out_slices) != 1:
        raise ValueError(f"Expected 1 OutputSlice in ViT profile, found {len(out_slices)}")
    out_sc = float(out_slices[0]["inputs"][0]["scale"])
    ln_post_sc = ln_scales[-1]
    k1024_256 = [
        (w, op) for w, op in sorted(by_woff.items()) if op["inputs"][1]["dims"] == [1, 1, 1024, 256]
    ]
    head_ops = [
        op
        for _, op in k1024_256
        if abs(float(op["inputs"][0]["scale"]) - ln_post_sc) < 1e-10
        and abs(float(op["outputs"][0]["scale"]) - out_sc) < 1e-10
    ]
    head_idx_set = {op["idx"] for op in head_ops}
    fc1_ops = [op for _, op in k1024_256 if op["idx"] not in head_idx_set]

    if not (
        len(qkv_ops) == num_layers * 48
        and len(ao_ops) == num_layers * 16
        and len(fc1_ops) == num_layers * 16
        and len(fc2_ops) == num_layers * 4
        and len(head_ops) == 3
    ):
        raise ValueError(
            f"Unexpected ViT op counts for L={num_layers}: qkv={len(qkv_ops)} ao={len(ao_ops)} "
            f"fc1={len(fc1_ops)} fc2={len(fc2_ops)} head={len(head_ops)}"
        )

    for b in range(num_layers):
        p = f"L{b}_"
        weights[p + "Wq"], weights[p + "bq"] = concat_conv_group(ctx, qkv_ops[b * 48 : b * 48 + 16])
        weights[p + "Wk"], weights[p + "bk"] = concat_conv_group(ctx, qkv_ops[b * 48 + 16 : b * 48 + 32])
        weights[p + "Wv"], weights[p + "bv"] = concat_conv_group(ctx, qkv_ops[b * 48 + 32 : b * 48 + 48])
        weights[p + "Wao"], weights[p + "bao"] = concat_conv_group(ctx, ao_ops[b * 16 : (b + 1) * 16])
        weights[p + "Wfc1"], weights[p + "bfc1"] = concat_conv_group(ctx, fc1_ops[b * 16 : (b + 1) * 16])
        weights[p + "Wfc2"], weights[p + "bfc2"] = concat_conv_group(ctx, fc2_ops[b * 4 : (b + 1) * 4])

    weights["W_head"], weights["b_head"] = concat_conv_group(ctx, head_ops)
    return weights


def unpack_htp_generic(ctx_path: Path | str, deser_dir: Path | str) -> Dict[str, np.ndarray]:
    """Generic tapless unpacker for arbitrary QNN HTP context binaries.

    Extracts:
      - Const tensors backed by constant extents (``const_{idx}``)
      - LayerNorm gamma/beta groups whose header ``gscale`` resolves uniquely
      - Const ``ConvLayer_s1`` groups grouped by shared input activation pointer
    """
    ctx = HTPUnpackContext.load(ctx_path, deser_dir)
    weights: Dict[str, np.ndarray] = {}

    for op in ctx.ops:
        if op["true_name"] != "Const":
            continue
        out = op["outputs"][0]
        off = ctx.ptr_to_file_off(out.get("raw_data"))
        if off is None:
            continue
        dims = tuple(int(d) for d in out["dims"])
        sbytes = int(out.get("storage_bytes", 0))
        dname = out.get("dtype_name", "")
        if sbytes <= 0 or off + sbytes > len(ctx.ctx_bytes):
            continue
        raw = ctx.ctx_bytes[off : off + sbytes]
        sc, zp = float(out["scale"]), int(out["zero_offset"])
        n_elem = int(np.prod(dims)) if dims else 0
        if dname == "QUint8" and sbytes == n_elem:
            arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - zp) * sc
            weights[f"const_{op['idx']}"] = arr.reshape(dims).astype(np.float32)
        elif dname == "QUint16" and sbytes == n_elem * 2:
            arr = (np.frombuffer(raw, dtype="<u2").astype(np.float32) - zp) * sc
            weights[f"const_{op['idx']}"] = arr.reshape(dims).astype(np.float32)
        elif dname == "Float32" and sbytes == n_elem * 4:
            weights[f"const_{op['idx']}"] = np.frombuffer(raw, dtype="<f4").reshape(dims).copy()

    ln_groups: Dict[Tuple[float, int], List[Dict[str, Any]]] = {}
    for op in ctx.ops:
        if "layernorm_q16_2d_normalize" in op["true_name"]:
            out = op["outputs"][0]
            key = (round(float(out["scale"]), 12), int(out["zero_offset"]))
            ln_groups.setdefault(key, []).append(op)
    for li, gops in enumerate(ln_groups.values()):
        try:
            g, b, _, _ = extract_ln_params_from_norms(ctx, gops)
            weights[f"ln{li}_gamma"] = g
            weights[f"ln{li}_beta"] = b
        except Exception:
            pass

    convs = [op for op in ctx.ops if op["true_name"].startswith("ConvLayer_s1")]
    by_in: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for op in convs:
        _, w_off, _ = ctx.trace_const_info(op["inputs"][1]["ptr"])
        _, b_off, _ = ctx.trace_const_info(op["inputs"][2]["ptr"])
        if w_off is not None and b_off is not None:
            by_in[op["inputs"][0]["ptr"]].append(op)
    for gi, gops in enumerate(by_in.values()):
        try:
            W, b = concat_conv_group(ctx, gops)
            weights[f"conv{gi}_W"] = W
            weights[f"conv{gi}_b"] = b
        except Exception:
            pass

    return weights


def unpack_auto(
    ctx_path: Path | str,
    deser_dir: Path | str,
    model_hint: str = "auto",
) -> Dict[str, np.ndarray]:
    """Dispatch tapless unpacker by explicit profile hint or structural fingerprint."""
    hint = (model_hint or "auto").lower()
    if hint == "bert":
        return unpack_bert_profile(ctx_path, deser_dir)
    if hint == "bert-multihead":
        return unpack_multihead_bert_profile(ctx_path, deser_dir)
    if hint == "vit":
        return unpack_vit_profile(ctx_path, deser_dir)
    if hint == "generic":
        return unpack_htp_generic(ctx_path, deser_dir)

    ops = load_runlist(Path(deser_dir) / "runlist.jsonl")
    const_dims = {
        tuple(op["outputs"][0]["dims"])
        for op in ops
        if op["true_name"] == "Const" and op.get("outputs")
    }
    if (1, 8, 16, 256) in const_dims:
        try:
            return unpack_bert_profile(ctx_path, deser_dir)
        except Exception:
            return unpack_htp_generic(ctx_path, deser_dir)
    if any("OutputSlice" in op["true_name"] for op in ops):
        out_slices = [op for op in ops if "OutputSlice" in op["true_name"]]
        if len(out_slices) >= 4:
            try:
                return unpack_multihead_bert_profile(ctx_path, deser_dir)
            except Exception:
                return unpack_htp_generic(ctx_path, deser_dir)
    if (1, 1, 68, 256) in const_dims:
        try:
            return unpack_vit_profile(ctx_path, deser_dir)
        except Exception:
            return unpack_htp_generic(ctx_path, deser_dir)
    return unpack_htp_generic(ctx_path, deser_dir)
