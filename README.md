<p align="center">
  <img src="assets/qnnrev-banner.png" alt="QNNRev — QNN / QAIRT Hexagon HTP Context Binary Toolkit" width="100%">
</p>

<h1 align="center">QNNRev — Qualcomm QNN / QAIRT HTP Binary Analysis &amp; Reconstruction</h1>

<p align="center">
  <a href="https://github.com/Has-X/QNNRev/actions/workflows/ci.yml"><img src="https://github.com/Has-X/QNNRev/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.12-0078D4" alt="Python 3.10 and 3.12">
  <a href="https://github.com/Has-X/QNNRev/actions/workflows/security.yml"><img src="https://img.shields.io/badge/security-CodeQL%20Advanced-2EA44F" alt="CodeQL Advanced"></a>
  <img src="https://img.shields.io/badge/Made%20in-the%20EU%20%F0%9F%87%AA%F0%9F%87%BA-003399" alt="Made in the EU 🇪🇺">
  <img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-000000" alt="AGPL 3.0 or later">
</p>

QNNRev is a Python and C toolkit for inspecting Qualcomm QNN / QAIRT Hexagon HTP context binaries, intercepting `hnnx::Deserializer` runlists, unpacking tiled weights and quantization parameters (`Xsb`, `Fi`, flat constants), reconstructing canonical W8A16 QDQ ONNX graphs, and driving SDK recompilation across QAIRT versions.

Developed and published by [Chromatic](https://chromatic.hu). Send product feedback to [feedback@chromatic.hu](mailto:feedback@chromatic.hu).

> [!NOTE]
> QNNRev does not redistribute Qualcomm QAIRT SDK binaries, Hexagon toolchains, or third-party model weights. Features that invoke `libQnnHtp.so`, `qairt-converter`, `DLCQuantizer`, or `qnn-context-binary-generator` expect user-supplied SDK installations under their respective terms.

## Capabilities

- **Static container & extent parsing**: Reads QNN context binary headers (`container_version` 3 and `container_version` 4+ `PICKLE/` containers), `hnnx::Deserializer` TLVs (`Sz`, `Vr`, `Op`, `Mm`, `M\xef`), and `0x71c43c9b` constant-extent tables in 64-byte units (`docs/ARCHITECTURE.md`).
- **Native execution & deserializer hook**: `native/htp_deser_tap.c` (`LD_PRELOAD`) captures `extract_const_extent_table` / `extract_const_extent_data` mappings, `Graph::do_execute` runlists (`runlist.jsonl`), and optionally untiles HTP Crouton intermediate activations.
- **Tapless weight & constant unpacking**: Maps `Const` op pointers through recorded DMA extents directly back into `.bin` file offsets (`Xsb` int8 tiles, `Fi` interleaved float scale + int32 bias pairs, and flat tensors) across `generic`, `bert`, `bert-multihead`, and `vit` profiles.
- **ONNX reconstruction**: Exports unpacked quantized weights and LayerNorm / embedding parameters to canonical float+QDQ ONNX models runnable under `onnxruntime` CPU EP or QAIRT tools.
- **SDK compilation & parity verification**: Builds plans or executes ORT-QNN EP (`onnxruntime-qnn`, QAIRT < 2.40) and standalone QAIRT SDK (`qairt-converter` → `DLCQuantizer` → `qnn-context-binary-generator`, QAIRT 2.40+) workflows, plus raw and deviational cosine verification (`qnnrev verify`).
- **Evidence-bound IR & adapters**: Includes static adapters for QAIRT 2.26 (`0xb`) and QAIRT 2.39 wide containers (`0x1c3340c0` sequencer metadata), an evidence-bound JSON IR validator, and numerical classification helpers (`qnnrev.validation`).

## Install

Python 3.10+ and GCC (for the optional `LD_PRELOAD` interceptor) are required.

```bash
git clone https://github.com/Has-X/QNNRev.git
cd QNNRev
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make -C native libhtp_deser_tap.so
```

## CLI Usage

```bash
# Inspect a context binary, .zip/.minn package, or directory
qnnrev inspect model.bin

# Unpack constants/weights from a context binary + native/htp_deser_tap.c capture dir
qnnrev unpack --model auto --ctx model.bin --deser-dir deser_out/ --out weights.npz

# Export unpacked weights to ONNX (bert | bert-multihead | vit)
qnnrev decompile --arch bert --weights weights.npz --out model.onnx

# Emit a standalone QAIRT SDK compilation plan (dry-run) or execute compilation
qnnrev compile --onnx model.onnx --out-dir build_htp/ --htp-extended-opts \
  --input-dim t_0=1,128 --input-dim t_1=1,128 --input-list calib.txt --dry-run

# Verify probe outputs (raw + deviational cosine)
qnnrev verify --a htp_out/ --a-dtype u16 --a-scale 0.001 --a-offset -32768 \
  --b ref_out/ --b-dtype f32 --htp-executed
```

Environment variables used by `qnnrev compile`:

| Variable | Purpose |
| :--- | :--- |
| `QAIRT_SDK_ROOT` | Default QAIRT SDK root (e.g., `/opt/qairt/current`) |
| `QNN_SDK_ROOT` | Fallback QNN / QAIRT SDK root when `QAIRT_SDK_ROOT` is unset |
| `QNNREV_PY64` | `x86_64` Python interpreter for `qairt-converter` / `DLCQuantizer` |
| `QNNREV_EMUBUNDLE_DIR` | Optional sysroot/LLVM directory when running `x86_64` QAIRT tools under `ld-linux-x86-64.so.2` on `aarch64` hosts |
| `QNNREV_LD_LINUX` | Override dynamic linker path for cross-arch `x86_64` SDK execution |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — Context binary containers (v3 & v4+), `hnnx::Deserializer` TLVs, `0x71c43c9b` extent geometry, `Xsb`/`Fi` packing, and IR design.
- [docs/COMPILATION.md](docs/COMPILATION.md) — ORT-EP vs standalone QAIRT SDK compilation rules, extended HTP graph options, and asymmetric quantization requirements.
- [docs/NATIVE_TOOLS.md](docs/NATIVE_TOOLS.md) — Building and using `libhtp_deser_tap.so` (`LD_PRELOAD`) and `vload_batch`.

## Known Boundaries

- **ONNX architectural exporters** (`bert`, `bert-multihead`, `vit`) reconstruct parameterized transformer encoder families; arbitrary unknown graphs should use `--model generic` (`unpack_htp_generic`) or the SSA graph in `qnnrev.core.runlist` to inspect tensor bindings directly.
- **`vload_batch`** requires Qualcomm QAIRT SDK headers (`include/QNN`) to compile and a target Hexagon runtime / `libQnnHtp.so` to execute.
- **Taps vs. tapless unpacking**: `unpack` recovers constants purely from context bytes + `extents.jsonl` + `runlist.jsonl`; activation quantization scales come from op metadata in `runlist.jsonl`, whereas dynamic runtime values require running under `libhtp_deser_tap.so` or `vload_batch`.

## Development & Tests

```bash
python -m compileall -q qnnrev tests
pytest -v
make -C native libhtp_deser_tap.so
```

## License

QNNRev is licensed under the GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`). See [LICENSE](LICENSE) and [NOTICE](NOTICE).
