# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Canonical ONNX exporter for reconstructed Pre-LN Vision Transformer (ViT) encoders."""
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


def export_vit_onnx(
    weights: Path | str | Dict[str, np.ndarray],
    out_onnx: Path | str,
    image_size: int = 224,
    patch_size: int = 14,
    num_layers: int = 24,
    num_heads: int = 16,
    hidden_size: int = 1024,
    out_dim: int = 768,
    opset: int = 20,
    ln_eps: float = 1e-5,
    graph_name: str = "vit_encoder",
) -> Path:
    """Export canonical Pre-LN ViT with NHWC ``[1, H, W, 3]`` input and QuickGELU MLP."""
    w = np.load(weights) if isinstance(weights, (str, Path)) else weights
    grid = image_size // patch_size
    num_patches = grid * grid
    seq_len = num_patches + 1
    head_dim = hidden_size // num_heads

    inits = []
    nodes = []

    _add_init(inits, "c_inv_sqrt_dk", np.array([1.0 / np.sqrt(head_dim)], dtype=np.float32))
    _add_init(inits, "c_quickgelu", np.array([1.702], dtype=np.float32))
    _add_init(inits, "shape_1_hid_np", np.array([1, hidden_size, num_patches], dtype=np.int64), dtype=np.int64)
    _add_init(inits, "shape_b_seq_nh_dh", np.array([1, seq_len, num_heads, head_dim], dtype=np.int64), dtype=np.int64)
    _add_init(inits, "shape_b1_seq_hid", np.array([1, seq_len, hidden_size], dtype=np.int64), dtype=np.int64)
    _add_init(inits, "shape_1_hid", np.array([1, hidden_size], dtype=np.int64), dtype=np.int64)
    _add_init(inits, "idx_cls", np.array([0], dtype=np.int64), dtype=np.int64)

    def add_ln(x_name: str, ln_idx: int, prefix: str) -> str:
        g_name = _add_init(inits, f"{prefix}_gamma", w["ln_gamma"][ln_idx])
        b_name = _add_init(inits, f"{prefix}_beta", w["ln_beta"][ln_idx])
        out_name = f"{prefix}_out"
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                [x_name, g_name, b_name],
                [out_name],
                name=prefix,
                axis=-1,
                epsilon=float(ln_eps),
            )
        )
        return out_name

    w_patch_conv = (
        w["W_patch"]
        .reshape(patch_size, patch_size, 3, hidden_size)
        .transpose(3, 2, 0, 1)
    )
    _add_init(inits, "W_patch_conv", w_patch_conv)
    _add_init(inits, "b_patch", w["b_patch"])
    _add_init(inits, "cls_token", w["cls_token"].reshape(1, 1, hidden_size))
    _add_init(inits, "pos_emb", w["pos_emb"].reshape(1, seq_len, hidden_size))

    nodes.append(helper.make_node("Transpose", ["image"], ["img_nchw"], name="img_to_nchw", perm=[0, 3, 1, 2]))
    nodes.append(
        helper.make_node(
            "Conv",
            ["img_nchw", "W_patch_conv", "b_patch"],
            ["patch_conv"],
            name="conv_patch",
            kernel_shape=[patch_size, patch_size],
            strides=[patch_size, patch_size],
        )
    )
    nodes.append(helper.make_node("Reshape", ["patch_conv", "shape_1_hid_np"], ["patch_flat"], name="rs_patch"))
    nodes.append(helper.make_node("Transpose", ["patch_flat"], ["patches"], name="tp_patch", perm=[0, 2, 1]))
    nodes.append(helper.make_node("Concat", ["cls_token", "patches"], ["tok_seq"], name="concat_cls", axis=1))
    nodes.append(helper.make_node("Add", ["tok_seq", "pos_emb"], ["emb_out"], name="add_pos"))
    x = add_ln("emb_out", 0, "ln_pre")

    for b in range(num_layers):
        p = f"L{b}"
        xn1 = add_ln(x, 1 + 2 * b, f"{p}_ln1")
        for kind in ("q", "k", "v"):
            W = _add_init(inits, f"{p}_W{kind}", w[f"{p}_W{kind}"])
            bias = _add_init(inits, f"{p}_b{kind}", w[f"{p}_b{kind}"])
            mm, bi, rs, tp = f"{p}_{kind}_mm", f"{p}_{kind}_lin", f"{p}_{kind}_4d", f"{p}_{kind}"
            nodes.append(helper.make_node("MatMul", [xn1, W], [mm], name=f"{p}_mm_{kind}"))
            nodes.append(helper.make_node("Add", [mm, bias], [bi], name=f"{p}_add_{kind}"))
            nodes.append(helper.make_node("Reshape", [bi, "shape_b_seq_nh_dh"], [rs], name=f"{p}_rs_{kind}"))
            perm = [0, 2, 3, 1] if kind == "k" else [0, 2, 1, 3]
            nodes.append(helper.make_node("Transpose", [rs], [tp], name=f"{p}_tp_{kind}", perm=perm))

        nodes.append(helper.make_node("MatMul", [f"{p}_q", f"{p}_k"], [f"{p}_qkt"], name=f"{p}_qkt"))
        nodes.append(helper.make_node("Mul", [f"{p}_qkt", "c_inv_sqrt_dk"], [f"{p}_scores"], name=f"{p}_scale"))
        nodes.append(helper.make_node("Softmax", [f"{p}_scores"], [f"{p}_probs"], name=f"{p}_softmax", axis=-1))
        nodes.append(helper.make_node("MatMul", [f"{p}_probs", f"{p}_v"], [f"{p}_ctx_4d"], name=f"{p}_attv"))
        nodes.append(helper.make_node("Transpose", [f"{p}_ctx_4d"], [f"{p}_ctx_tp"], name=f"{p}_ctx_t", perm=[0, 2, 1, 3]))
        nodes.append(helper.make_node("Reshape", [f"{p}_ctx_tp", "shape_b1_seq_hid"], [f"{p}_ctx"], name=f"{p}_ctx_r"))

        Wao = _add_init(inits, f"{p}_Wao", w[f"{p}_Wao"])
        bao = _add_init(inits, f"{p}_bao", w[f"{p}_bao"])
        nodes.append(helper.make_node("MatMul", [f"{p}_ctx", Wao], [f"{p}_ao_mm"], name=f"{p}_mm_ao"))
        nodes.append(helper.make_node("Add", [f"{p}_ao_mm", bao], [f"{p}_ao"], name=f"{p}_add_ao"))
        nodes.append(helper.make_node("Add", [x, f"{p}_ao"], [f"{p}_res1"], name=f"{p}_res1"))

        xn2 = add_ln(f"{p}_res1", 2 + 2 * b, f"{p}_ln2")
        Wfc1 = _add_init(inits, f"{p}_Wfc1", w[f"{p}_Wfc1"])
        bfc1 = _add_init(inits, f"{p}_bfc1", w[f"{p}_bfc1"])
        nodes.append(helper.make_node("MatMul", [xn2, Wfc1], [f"{p}_fc1_mm"], name=f"{p}_mm_fc1"))
        nodes.append(helper.make_node("Add", [f"{p}_fc1_mm", bfc1], [f"{p}_fc1"], name=f"{p}_add_fc1"))
        nodes.append(helper.make_node("Mul", [f"{p}_fc1", "c_quickgelu"], [f"{p}_qg_arg"], name=f"{p}_qg_mul1"))
        nodes.append(helper.make_node("Sigmoid", [f"{p}_qg_arg"], [f"{p}_qg_sig"], name=f"{p}_qg_sig"))
        nodes.append(helper.make_node("Mul", [f"{p}_fc1", f"{p}_qg_sig"], [f"{p}_gelu"], name=f"{p}_qg_mul2"))

        Wfc2 = _add_init(inits, f"{p}_Wfc2", w[f"{p}_Wfc2"])
        bfc2 = _add_init(inits, f"{p}_bfc2", w[f"{p}_bfc2"])
        nodes.append(helper.make_node("MatMul", [f"{p}_gelu", Wfc2], [f"{p}_fc2_mm"], name=f"{p}_mm_fc2"))
        nodes.append(helper.make_node("Add", [f"{p}_fc2_mm", bfc2], [f"{p}_fc2"], name=f"{p}_add_fc2"))
        nodes.append(helper.make_node("Add", [f"{p}_res1", f"{p}_fc2"], [f"{p}_res2"], name=f"{p}_res2"))
        x = f"{p}_res2"

    nodes.append(helper.make_node("Gather", [x, "idx_cls"], ["cls_3d"], name="gather_cls", axis=1))
    nodes.append(helper.make_node("Reshape", ["cls_3d", "shape_1_hid"], ["cls_2d"], name="reshape_cls"))
    cls_post = add_ln("cls_2d", 1 + 2 * num_layers, "ln_post")
    _add_init(inits, "W_head", w["W_head"])
    _add_init(inits, "b_head", w["b_head"])
    nodes.append(helper.make_node("MatMul", [cls_post, "W_head"], ["head_mm"], name="mm_head"))
    nodes.append(helper.make_node("Add", ["head_mm", "b_head"], ["feature"], name="add_head"))

    inputs = [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, image_size, image_size, 3])]
    outputs = [helper.make_tensor_value_info("feature", TensorProto.FLOAT, [1, out_dim])]

    graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 9 if opset >= 19 else 8
    onnx.checker.check_model(model)

    out_path = Path(out_onnx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    return out_path
