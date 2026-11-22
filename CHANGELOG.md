# Changelog

All notable changes to QNNRev are documented in this file.

## [0.2.0] - 2026-11-22

### Added
- Support for container v4+ (`PICKLE/`) HTP context headers, `0x71c43c9b` extent
  tables with 1024-byte start alignment, and `O=3` extended HTP compilation
  options (`--htp-extended-opts`, `--compiler`, version-agnostic `--qairt-version`).
- Dynamic-layer HTP unpackers (`unpack_bert_profile`, `unpack_multihead_bert_profile`,
  `unpack_vit_profile`, `unpack_generic`, `unpack_auto`) and matching W8A16 ONNX
  QDQ exporters.
- Native `LD_PRELOAD` `hnnx::Deserializer` + `Graph::do_execute` interceptor
  (`native/htp_deser_tap.c`) and QAIRT HTP single/batch runner
  (`native/vload_batch.c`).
- Unified CLI (`qnnrev inspect|unpack|decompile|compile|verify`), package layout,
  CI workflow, and synthetic test suite.

### Fixed
- Kept `ConstExtentTable` `offset_units` and `length_units` fixed at 64-byte
  units on v4+ containers instead of conflating the 1024-byte extent start
  alignment field with unit size.

## [0.1.0] - 2026-08-22

### Added
- Initial container inspection (`ctx_parse`, `decode226`, `decode_general`,
  `inspect`), QAIRT 2.26 / 2.39-wide adapters, evidence-bound IR schema, and
  validation suite.
