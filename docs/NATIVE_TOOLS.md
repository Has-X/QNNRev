# Native Tools

`native/` provides two small C tools to support runtime inspection and target execution when you supply a Qualcomm QAIRT SDK.

## 1. `libhtp_deser_tap.so` (`htp_deser_tap.c`)

An `LD_PRELOAD` shared library targeting `libQnnHtp.so` (`x86_64` QAIRT host emulator runtime). It intercepts `dlopen("...libQnnHtp.so", ...)` (handling `RTLD_LOCAL`) and resolves internal `hnnx::Deserializer` and `Graph` symbols directly from the loaded handle:

- `hnnx::Deserializer::extract_const_extent_table`
- `hnnx::Deserializer::auxdata_read_const_extent_descriptor`
- `Graph::handle_auxdata_deser`
- `hnnx::Deserializer::extract_const_extent_data` (DMA and memory variants) -> `extents.jsonl`
- `Graph::compile_exec_list` & `Graph::do_execute` -> `runlist.jsonl`
- `continue_execution_fast` / `continue_execution_bkgrnd_thread` -> untiles Crouton NHWC/flat activations to `taps_index.jsonl` + `taps.bin`

### Build

```bash
make -C native libhtp_deser_tap.so
```

### Runtime Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HTP_TAP_DIR` | unset (disabled) | Output directory created automatically for `extents.jsonl`, `runlist.jsonl`, `taps_index.jsonl`, `taps.bin` |
| `HTP_TAP_MODE` | full | Set to `deser` or `none` to skip `taps_index.jsonl` / `taps.bin` |
| `HTP_TAP_SKIP_EXEC` | `0` | Set to `1` to dump `extents.jsonl` and `runlist.jsonl` in `do_execute` and return immediately without running ops |
| `HTP_TAP_DUMP_CONSTS` | `0` | Set to `1` to dump `consts_index.jsonl` and `consts.bin` |
| `HTP_TAP_SKIP_CONST` | `1` | Set to `0` to include constant ops in activation taps |
| `HTP_TAP_SKIP_REPEAT` | `0` | Set to `1` to skip executing subsequent `do_execute` invocations |
| `HTP_TAP_MAX_TAP_BYTES` | `0` (unlimited) | Cap on `taps.bin` output size in bytes |

For tapless weight unpacking (`qnnrev unpack`), set `HTP_TAP_MODE=deser HTP_TAP_SKIP_EXEC=1` and run a single `contextCreateFromBinary` + `graphExecute` dummy call to emit `extents.jsonl` and `runlist.jsonl` quickly.

## 2. `vload_batch` (`vload_batch.c`)

A standalone QAIRT HTP `contextCreateFromBinary` / `graphExecute` runner supporting single-probe and `--batch` execution (loading the context once for `N` probes). It configures `QNN_HTP_DEVICE_CONFIG_OPTION_SOC`, `QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3` burst performance voting, and `QNN_HTP_CONTEXT_CONFIG_OPTION_PARALLEL_WEIGHT_LOADING`.

### Build

Requires a local QAIRT SDK (`include/QNN`):

```bash
make -C native vload_batch QNN_SDK_ROOT=/path/to/qairt/sdk CC=aarch64-linux-android34-clang
```

### Usage

```bash
# Single run
./native/vload_batch /path/to/qnn/lib ctx.bin graph_name 61 \
  in0:1,128:u16:1:0:in0.raw -- out0:1,128:u16:1:0:out0.raw

# Batch run (manifest lines: in0,in1|out0)
./native/vload_batch /path/to/qnn/lib ctx.bin graph_name 61 \
  --batch manifest.txt in0,in1 out0
```

Set `QNNREV_LOG=/path/to/run.log` (or `VLOAD_LOG`) to redirect output and timing lines (`ITEM_OK idx=... ns=...`).
