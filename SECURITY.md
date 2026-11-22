# Security Policy

## Supported Versions

Security fixes are handled on the `main` branch.

## Reporting a Vulnerability

Please report security issues privately instead of opening a public issue.

Use GitHub private vulnerability reporting when available for this repository.
If that is not available, email [feedback@chromatic.hu](mailto:feedback@chromatic.hu)
with `QNNRev security report` in the subject and include:

- affected version or commit,
- component involved (container parser, unpacker, ONNX exporter, or native C hook),
- steps to reproduce with minimal synthetic input,
- expected and actual behavior.

## Scope

Useful reports include:

- out-of-bounds reads, integer overflows, or memory corruption in `native/` C code,
- malformed context binary headers or extent tables triggering unbounded allocation,
- path traversal when inspecting or extracting `.zip` / container archives,
- command-injection or unsafe subprocess handling in CLI wrappers.

Do not attach proprietary model weights or confidential vendor SDK artifacts
unless explicitly requested and cleared for sharing.
