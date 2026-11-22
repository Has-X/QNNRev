# QNNRev Architecture

## 1. Context Binary Framing (v3 and v4)

QNN HTP serialized context binaries start with the 8-byte magic header
`\x00\x00\x00\x02\x00\x00\x00\x03`.

- `0x18` (`uint32`): `blob_offset` — file offset of the inner HTP container.
- `0x20` (`uint64`): `blob_size` — byte length of the inner container.

At `blob_offset`, the first `uint32` identifies `container_version`:

- **Container v3 (QAIRT 2.26–2.44)**: `deser_rel` (`uint64`) sits at `blob_offset + 0x18`.
- **Container v4 (QAIRT 2.52 `PICKLE/256`)**: A `PICKLE/` header precedes `deser_rel`; the parser locates `PICKLE/` within `[blob_offset, blob_offset + 0x200)` and reads `deser_rel` from `((pickle_pos - 4) & ~7) - 16` (falling back to `blob_offset + 0x70`).

In both versions:

```
deser_hdr_offset = blob_offset + deser_rel
deser_size       = u64[deser_hdr_offset]
deser_base       = deser_hdr_offset + 8
```

## 2. `hnnx::Deserializer` Header TLVs

Following the 12-byte magic/tag header at `deser_base` (and optional zero padding words), 32-bit TLV headers encode `(tag << 16) | nwords` (each record spans `nwords * 4` bytes):

| Tag | ASCII | Contents |
| :--- | :--- | :--- |
| `0x7A53` | `Sz` | Size words; `deser_base + sz_words[0] * 4` points directly to the `0x71c43c9b` extent table |
| `0x7256` | `Vr` | Compiler / HTP serializer version string (null-terminated) |
| `0x704F` | `Op` | Compiler options string |
| `0x6D4D` | `Mm` | Memory layout words |
| `0xEF4D` | `M\xef` | Op class index table (`mef_counts` + `n_op_classes` null-terminated class strings) |

## 3. `0x71c43c9b` ConstExtentTable Geometry

`parse_const_extent_table` decodes the table located via `sz_words[0]` (or aligned 4-byte scan):

- Word 0: `0x71c43c9b` magic
- Word 1: `hdr_items = w1 >> 24`, `max_records_4w = w1 & 0xffffff`
- Word 2: `n_groups = w2 & 0xffffff`
- Word 3: `n_extents = w3 & 0xffffff`

Each group and extent entry is 4 `uint32` words (16 bytes). Extent records are `[extent_id, group_id, offset_units, length_units]`.

- **Pool base**: `extent_data_base = extent_table_file_offset + max(0x1000, max_records_4w * 64)`.
- **Unit size vs. alignment**: In both v3 and v4, `offset_units`, `length_units`, and `groups[0][2]` are measured in **64-byte units** (`EXTENT_UNIT_BYTES = 64`). In v4 containers, `groups[0][1]` changes from `64` to `1024`, recording the **extent start alignment in bytes** rather than the unit size:
  ```
  byte_offset_in_pool = offset_units * 64
  byte_length         = length_units * 64
  file_offset         = extent_data_base + byte_offset_in_pool
  ```

## 4. Mapping Runlist Ops to Context File Offsets

When `libhtp_deser_tap.so` intercepts context loading under `libQnnHtp.so`:

1. `extract_const_extent_data_dma` logs `(dst, b)` into `extents.jsonl` in the same order as `ConstExtentTable.extents`.
2. Pairing each DMA `(dst, dst + b)` virtual address range with `extent.file_offset` yields a deterministic virtual-address-to-file-offset map.
3. `Graph::do_execute` logs every op and tensor descriptor (`raw_data`, `dims`, `dtype`, `zero_offset`, `scale`) to `runlist.jsonl`.
4. `HTPUnpackContext.trace_const_info` walks SSA producer edges back to `Const` ops and translates `raw_data` into exact byte slices in the `.bin` file.

## 5. `Xsb` Weights and `Fi` Bias/Scale Layout

- **`Xsb` packed `int8` weights** (`unpack_xsb`): Stored as `(K / 32, N / 32, 8, 32, 4)` and transposed `(0, 2, 4, 1, 3)` back to row-major `[K, N]`.
- **`Fi` interleaved scales + biases** (`unpack_fi`): Processed in 32-output-channel blocks (128 `int32`/`uint32` words per block: even channels then odd channels).
  - Each channel stores a 64-bit `(w0, w1)` custom `a_q` exponent/mantissa encoding alongside an `int32` bias term (scaled by `256` relative to the `int32` field):
    ```
    exp_f16 = (w0 >> 10) & 0x3F
    mant    = (w1 & 0x3FF) + (w0 & 0x3FF) / 1024.0
    a_q     = (1.0 + mant / 1024.0) * 2**(exp_f16 - 24)
    bias    = bi_raw * 256
    ```

## 6. Evidence-Bound IR & Version Adapters

- `qnnrev.ir` constructs an evidence-bound IR dictionary from official `qnn-context-binary-utility` JSON (`ir_from_utility_json`, `validate_ir`), tracking unverified unknowns explicitly.
- `qnnrev.core.decode226` / `qnnrev.adapters.qairt226` handle QAIRT 2.26 `kind=0xb` layouts with explicit version guards (`0x50615041` / `"2.26."`).
- `qnnrev.adapters.qairt239_wide` handles QAIRT 2.39 wide context structures and `0x1c3340c0` sequencer tail metadata (`analyze_tail_structure`).
