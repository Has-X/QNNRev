# Contributing to QNNRev

Contributions that improve container parsing, tile unpacking, ONNX graph
reconstruction, or native inspection tooling are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make -C native libhtp_deser_tap.so
```

Run checks before opening a pull request:

```bash
python -m compileall -q qnnrev tests
pytest -v
```

## Rules for contributions

- Keep all new files licensed under `AGPL-3.0-or-later` with SPDX headers.
- Do not commit proprietary SDK binaries, vendor shared libraries, model
  weights, or captured device dumps.
- Base format decoders and unpackers on reproducible structural checks and add
  synthetic unit tests that run without external binaries.
- Keep pull requests focused and concise.

## License

By contributing to QNNRev, you agree that your contributions will be licensed
under the GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`).
See [LICENSE](LICENSE).
