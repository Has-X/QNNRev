# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Chromatic and contributors

"""Version guards, layout utilities, container parsers, and HTP unpackers."""
from .ctx_parse import parse_qnn_context
from .decode_general import (
    UnsupportedVersion,
    build_map as build_map_226,
    detect_version,
    guard_version,
    layout_valid,
    locate_weight_region,
    plane_stats,
)
from .hnnx_deser import (
    ConstExtentRecord,
    DeserHeaderSummary,
    parse_const_extent_table,
    parse_hnnx_context,
)
from .htp_unpack import (
    HTPUnpackContext,
    concat_conv_group,
    decode_conv_s1,
    extract_ln_gscale,
    extract_ln_params_from_norms,
    unpack_auto,
    unpack_bert_profile,
    unpack_fi,
    unpack_htp_generic,
    unpack_multihead_bert_profile,
    unpack_vit_profile,
    unpack_xsb,
)
from .inspect import detect_format, inspect
from .runlist import (
    SSAGraph,
    build_ssa_graph,
    iter_jsonl,
    load_runlist,
    load_taps_index,
    read_tap_array,
    summarize_runlist,
)

__all__ = [
    "UnsupportedVersion",
    "detect_version",
    "guard_version",
    "layout_valid",
    "build_map_226",
    "locate_weight_region",
    "plane_stats",
    "detect_format",
    "inspect",
    "parse_qnn_context",
    "ConstExtentRecord",
    "DeserHeaderSummary",
    "parse_const_extent_table",
    "parse_hnnx_context",
    "SSAGraph",
    "build_ssa_graph",
    "iter_jsonl",
    "load_runlist",
    "load_taps_index",
    "read_tap_array",
    "summarize_runlist",
    "HTPUnpackContext",
    "concat_conv_group",
    "decode_conv_s1",
    "extract_ln_gscale",
    "extract_ln_params_from_norms",
    "unpack_auto",
    "unpack_bert_profile",
    "unpack_fi",
    "unpack_htp_generic",
    "unpack_multihead_bert_profile",
    "unpack_vit_profile",
    "unpack_xsb",
]
