# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance
- Static weights that onnx-ir passes as anonymous (unnamed) constants — including all
  conv weights — are now pre-loaded once instead of converted on every forward pass.
- BatchNormalization directly after a Conv2d is folded into the conv's weight and bias
  at load time and removed from the graph. ResNet-18 (Model Zoo v1) at batch 8: 279 → 103 ms.

### Changed
- Errors now surface as normal Python exceptions instead of `PanicException`:
  - `TypeError` for non-array / non-float32 inputs (with a `.astype(np.float32)` hint).
  - `ValueError` for wrong input count, rank, or a static dim that doesn't match the model.
  - `RuntimeError` naming the failing node and op (e.g. `node 'gemm1' (Gemm): ...`) for op
    and backend failures, including panics raised inside Burn.
- Rust panic output on stderr is suppressed (the message is in the exception);
  set `RUST_BACKTRACE=1` to get it back.

### Fixed
- BatchNormalization on non-4D inputs (e.g. `[N, C]` after a Linear) no longer
  broadcasts to a 4D output; the pre-fused fast path is only used for NCHW.
- Fortran-ordered inputs (e.g. `x.T`) were read in memory order, silently scrambling data.
  Non-C-contiguous arrays are now copied in logical order.
- Conv2d / MaxPool2d / AveragePool2d with asymmetric pads no longer average the two
  sides; uneven pads are applied to the input explicitly (exact for `count_include_pad=0`).
- `auto_pad=SAME_UPPER` / `SAME_LOWER` / `VALID` are now honoured instead of treated as no padding.
- Pooling now respects `ceil_mode`.
- Reshape with both `0` and `-1` in the target shape no longer panics (divide by zero),
  and the opset < 5 `shape` attribute form is supported.
- Unsupported ops now fail at `load_onnx` with a list of the missing ops, instead of
  being skipped with a warning and failing (or silently misbehaving) at inference time.

### Added
- ReduceMean, ReduceSum, ReduceMax, ReduceMin (needed by the torch dynamo exporter's
  global average pooling).
- `tests/test_resnet.py`: ResNet-18 end to end vs ONNX Runtime on Model Zoo and
  torchvision (legacy + dynamo) exports, with optional `--bench`; run in CI.
- `tests/test_ops.py`: per-op edge-case tests against ONNX Runtime, run in CI.
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
