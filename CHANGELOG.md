# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- ONNX interpreter support for Conv2d, fused BatchNormalization, MaxPool2d,
  AveragePool2d, and GlobalAveragePool.
- GitHub Actions CI: `cargo build`, `cargo fmt` / `cargo clippy` (non-blocking),
  and Python tests via `maturin develop`.
- Branch protection on `main`: pull requests required, `cargo build` and
  `python tests (maturin)` checks required to merge.
- ONNX interpreter with Gemm, Linear, Relu, Sigmoid, Tanh, Gelu, Softmax,
  LogSoftmax, Add, Sub, Mul, Div, Reshape, Flatten, and Transpose ops.
- Zero-copy numpy ↔ Burn tensor bridge (`roundtrip`).
- Initial maturin/pyo3 project scaffold, `burn-flex` + `onnx-ir` dependencies.

[Unreleased]: https://github.com/asjad2401/burn-python/commits/main
