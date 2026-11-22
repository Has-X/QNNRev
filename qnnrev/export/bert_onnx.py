# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Canonical ONNX exporter for reconstructed Post-LN BERT encoders."""
from __future__ import annotations

from pathlib import Path
from typing import Dict
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _add_init(inits, name: str, arr: np.ndarray, dtype=np.float32) -> str:
    a = np.ascontiguousarray(arr.astype(dtype))
    inits.append(numpy_helper.from_array(a, name=name))
    return name


def export_bert_onnx(
    weights: Path | str | Dict[str, np.ndarray],
    out_onnx: Path | str,
    seq_len: int = 512,
    num_layers: int = 12,
    num_heads: int = 12,
    hidden_size: int = 768,
    opset: int = 20,
    use_layernorm_op: bool = True,
    use_gelu_op: bool = True,
    attn_mask_value: float = -200.0,
    graph_name: str = "bert_encoder",
) -> Path:
    """Export a canonical Post-LN BERT encoder with masked mean pooling and linear head."""
    w = np.load(weights) if isinstance(weights, (str, Path)) else weights
    head_dim = hidden_size // num_heads
    inits = []
    nodes = []

    _add_init(inits, "c_one", np.array([1.0], dtype=np.float32))
    _add_init(inits, "c_neg1e4", np.array([float(attn_mask_value)], dtype=np.float32))
    _add_init(inits, "c_inv_sqrt_dk", np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32))
    if not use_layernorm_op:
        _add_init(inits, "c_eps", np.array([1e-12], dtype=np.float32))
        _add_init(inits, "c_two", np.array([2.0], dtype=np.float32))
        if opset >= 18:
            _add_init(inits, "c_axes_neg1", np.array([-1], dtype=np.int64), dtype=np.int64)
    if not use_gelu_op:
        _add_init(inits, "c_half", np.array([0.5], dtype=np.float32))
        _add_init(inits, "c_sqrt2", np.array([np.sqrt(2.0)], dtype=np.float32))
    _add_init(inits, "shape_b1_seq_hid", np.array([1, seq_len, hidden_size], dtype=np.int64), dtype=np.int64)
    _add_init(inits, "shape_1_1_1_seq", np.array([1, 1, 1, seq_len], dtype=np.int64), dtype=np.int64)
    _add_init(inits, "shape_1_seq_1", np.array([1, seq_len, 1], dtype=np.int64), dtype=np.int64)
    _add_init(inits, "shape_b_seq_nh_dh", np.array([1, seq_len, num_heads, head_dim], dtype=np.int64), dtype=np.int64)

    def add_ln(x_name: str, ln_idx: int, prefix: str) -> str:
        g_name = _add_init(inits, f"{prefix}_gamma", w["ln_gamma"][ln_idx])
        b_name = _add_init(inits, f"{prefix}_beta", w["ln_beta"][ln_idx])
        out_name = f"{prefix}_out"
        if use_layernorm_op:
            nodes.append(
                helper.make_node(
                    "LayerNormalization",
                    [x_name, g_name, b_name],
                    [out_name],
                    name=prefix,
                    axis=-1,
                    epsilon=1e-12,
                )
            )
            return out_name
        mu, xc, sq, var = f"{prefix}_mu", f"{prefix}_xc", f"{prefix}_sq", f"{prefix}_var"
        veps, std, norm, scaled = f"{prefix}_veps", f"{prefix}_std", f"{prefix}_norm", f"{prefix}_scaled"
        if opset >= 18:
            nodes.append(helper.make_node("ReduceMean", [x_name, "c_axes_neg1"], [mu], name=f"{prefix}_mean", keepdims=1))
        else:
            nodes.append(helper.make_node("ReduceMean", [x_name], [mu], name=f"{prefix}_mean", axes=[-1], keepdims=1))
        nodes.append(helper.make_node("Sub", [x_name, mu], [xc], name=f"{prefix}_sub"))
        nodes.append(helper.make_node("Pow", [xc, "c_two"], [sq], name=f"{prefix}_pow"))
        if opset >= 18:
            nodes.append(helper.make_node("ReduceMean", [sq, "c_axes_neg1"], [var], name=f"{prefix}_var_mean", keepdims=1))
        else:
            nodes.append(helper.make_node("ReduceMean", [sq], [var], name=f"{prefix}_var_mean", axes=[-1], keepdims=1))
        nodes.append(helper.make_node("Add", [var, "c_eps"], [veps], name=f"{prefix}_add_eps"))
        nodes.append(helper.make_node("Sqrt", [veps], [std], name=f"{prefix}_sqrt"))
        nodes.append(helper.make_node("Div", [xc, std], [norm], name=f"{prefix}_div"))
        nodes.append(helper.make_node("Mul", [norm, g_name], [scaled], name=f"{prefix}_mul_g"))
        nodes.append(helper.make_node("Add", [scaled, b_name], [out_name], name=f"{prefix}_add_b"))
        return out_name

    _add_init(inits, "word_embeddings", w["word_embeddings"])
    _add_init(inits, "token_type_embeddings", w["token_type_embeddings"])
    _add_init(inits, "position_embeddings", w["position_embeddings"].reshape(1, seq_len, hidden_size))

    nodes.append(helper.make_node("Gather", ["word_embeddings", "input_ids"], ["we_out"], name="gather_word", axis=0))
    nodes.append(helper.make_node("Gather", ["token_type_embeddings", "token_type_ids"], ["tte_out"], name="gather_tt", axis=0))
    nodes.append(helper.make_node("Add", ["we_out", "tte_out"], ["emb_wt"], name="add_word_tt"))
    nodes.append(helper.make_node("Add", ["emb_wt", "position_embeddings"], ["emb_sum"], name="add_pos"))
    x = add_ln("emb_sum", 0, "ln0")

    nodes.append(helper.make_node("Cast", ["attention_mask"], ["am_f32"], name="cast_am", to=TensorProto.FLOAT))
    nodes.append(helper.make_node("Reshape", ["am_f32", "shape_1_1_1_seq"], ["am_4d"], name="reshape_am_4d"))
    nodes.append(helper.make_node("Sub", ["c_one", "am_4d"], ["am_inv"], name="sub_am"))
    nodes.append(helper.make_node("Mul", ["am_inv", "c_neg1e4"], ["attn_bias"], name="mul_am_bias"))

    for L in range(num_layers):
        p = f"L{L}"
        for kind in ("q", "k", "v"):
            W = _add_init(inits, f"{p}_W{kind}", w[f"{p}_W{kind}"])
            b = _add_init(inits, f"{p}_b{kind}", w[f"{p}_b{kind}"])
            mm, bi, rs, tp = f"{p}_{kind}_mm", f"{p}_{kind}_lin", f"{p}_{kind}_4d", f"{p}_{kind}"
            nodes.append(helper.make_node("MatMul", [x, W], [mm], name=f"{p}_mm_{kind}"))
            nodes.append(helper.make_node("Add", [mm, b], [bi], name=f"{p}_add_{kind}"))
            nodes.append(helper.make_node("Reshape", [bi, "shape_b_seq_nh_dh"], [rs], name=f"{p}_rs_{kind}"))
            perm = [0, 2, 3, 1] if kind == "k" else [0, 2, 1, 3]
            nodes.append(helper.make_node("Transpose", [rs], [tp], name=f"{p}_tp_{kind}", perm=perm))

        nodes.append(helper.make_node("MatMul", [f"{p}_q", f"{p}_k"], [f"{p}_qkt"], name=f"{p}_qkt"))
        nodes.append(helper.make_node("Mul", [f"{p}_qkt", "c_inv_sqrt_dk"], [f"{p}_scores_scaled"], name=f"{p}_scale"))
        nodes.append(helper.make_node("Add", [f"{p}_scores_scaled", "attn_bias"], [f"{p}_scores"], name=f"{p}_mask"))
        nodes.append(helper.make_node("Softmax", [f"{p}_scores"], [f"{p}_probs"], name=f"{p}_softmax", axis=-1))
        nodes.append(helper.make_node("MatMul", [f"{p}_probs", f"{p}_v"], [f"{p}_ctx_4d"], name=f"{p}_attv"))
        nodes.append(helper.make_node("Transpose", [f"{p}_ctx_4d"], [f"{p}_ctx_tp"], name=f"{p}_ctx_t", perm=[0, 2, 1, 3]))
        nodes.append(helper.make_node("Reshape", [f"{p}_ctx_tp", "shape_b1_seq_hid"], [f"{p}_ctx"], name=f"{p}_ctx_r"))

        Wao = _add_init(inits, f"{p}_Wao", w[f"{p}_Wao"])
        bao = _add_init(inits, f"{p}_bao", w[f"{p}_bao"])
        nodes.append(helper.make_node("MatMul", [f"{p}_ctx", Wao], [f"{p}_ao_mm"], name=f"{p}_mm_ao"))
        nodes.append(helper.make_node("Add", [f"{p}_ao_mm", bao], [f"{p}_ao"], name=f"{p}_add_ao"))
        nodes.append(helper.make_node("Add", [x, f"{p}_ao"], [f"{p}_res1"], name=f"{p}_res1"))
        x_mid = add_ln(f"{p}_res1", 2 * L + 1, f"{p}_ln1")

        Wfc1 = _add_init(inits, f"{p}_Wfc1", w[f"{p}_Wfc1"])
        bfc1 = _add_init(inits, f"{p}_bfc1", w[f"{p}_bfc1"])
        nodes.append(helper.make_node("MatMul", [x_mid, Wfc1], [f"{p}_fc1_mm"], name=f"{p}_mm_fc1"))
        nodes.append(helper.make_node("Add", [f"{p}_fc1_mm", bfc1], [f"{p}_fc1"], name=f"{p}_add_fc1"))
        if use_gelu_op:
            nodes.append(helper.make_node("Gelu", [f"{p}_fc1"], [f"{p}_gelu"], name=f"{p}_gelu_op", approximate="none"))
        else:
            nodes.append(helper.make_node("Div", [f"{p}_fc1", "c_sqrt2"], [f"{p}_gelu_s"], name=f"{p}_gelu_s"))
            nodes.append(helper.make_node("Erf", [f"{p}_gelu_s"], [f"{p}_gelu_e"], name=f"{p}_gelu_erf"))
            nodes.append(helper.make_node("Add", [f"{p}_gelu_e", "c_one"], [f"{p}_gelu_p"], name=f"{p}_gelu_p"))
            nodes.append(helper.make_node("Mul", [f"{p}_fc1", "c_half"], [f"{p}_gelu_h"], name=f"{p}_gelu_h"))
            nodes.append(helper.make_node("Mul", [f"{p}_gelu_h", f"{p}_gelu_p"], [f"{p}_gelu"], name=f"{p}_gelu"))

        Wfc2 = _add_init(inits, f"{p}_Wfc2", w[f"{p}_Wfc2"])
        bfc2 = _add_init(inits, f"{p}_bfc2", w[f"{p}_bfc2"])
        nodes.append(helper.make_node("MatMul", [f"{p}_gelu", Wfc2], [f"{p}_fc2_mm"], name=f"{p}_mm_fc2"))
        nodes.append(helper.make_node("Add", [f"{p}_fc2_mm", bfc2], [f"{p}_fc2"], name=f"{p}_add_fc2"))
        nodes.append(helper.make_node("Add", [x_mid, f"{p}_fc2"], [f"{p}_res2"], name=f"{p}_res2"))
        x = add_ln(f"{p}_res2", 2 * L + 2, f"{p}_ln2")

    _add_init(inits, "c_axes_1", np.array([1], dtype=np.int64), dtype=np.int64)
    nodes.append(helper.make_node("Reshape", ["am_f32", "shape_1_seq_1"], ["am_3d"], name="reshape_am_3d"))
    nodes.append(helper.make_node("Mul", [x, "am_3d"], ["masked_seq"], name="mul_mask_seq"))
    nodes.append(helper.make_node("ReduceSum", ["masked_seq", "c_axes_1"], ["seq_sum"], name="sum_seq", keepdims=0))
    nodes.append(helper.make_node("ReduceSum", ["am_f32", "c_axes_1"], ["am_sum"], name="sum_am", keepdims=1))
    nodes.append(helper.make_node("Max", ["am_sum", "c_one"], ["am_denom"], name="max_denom"))
    nodes.append(helper.make_node("Div", ["seq_sum", "am_denom"], ["pooled"], name="div_pool"))

    W_head = _add_init(inits, "W_head", w["W_head"])
    b_head = _add_init(inits, "b_head", w["b_head"])
    nodes.append(helper.make_node("MatMul", ["pooled", W_head], ["head_mm"], name="mm_head"))
    nodes.append(helper.make_node("Add", ["head_mm", b_head], ["last_hidden_state"], name="add_head"))

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT32, [1, seq_len]),
        helper.make_tensor_value_info("token_type_ids", TensorProto.INT32, [1, seq_len]),
        helper.make_tensor_value_info("attention_mask", TensorProto.INT32, [1, seq_len]),
    ]
    outputs = [
        helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, [1, hidden_size]),
    ]

    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9 if opset >= 19 else 8
    onnx.checker.check_model(model)

    out_path = Path(out_onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path
