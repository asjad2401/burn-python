# burn-python

[![CI](https://github.com/asjad2401/burn-python/actions/workflows/ci.yml/badge.svg)](https://github.com/asjad2401/burn-python/actions/workflows/ci.yml)

Python inference frontend for the [Burn](https://github.com/tracel-ai/burn) deep learning framework.

Load an ONNX model and run inference from Python — numpy in, numpy out. No Rust required.

```python
import burn_python as burn
import numpy as np

model = burn.load_onnx("model.onnx")
x = np.random.randn(1, 3, 224, 224).astype(np.float32)
output = model([x])[0]
```

## Backends

```python
burn.available_backends()                          # ['flex', 'wgpu']
model = burn.load_onnx("model.onnx", backend="wgpu")  # run on the GPU
```

| backend | runs on | notes |
|---------|---------|-------|
| `"flex"` (default) | CPU | pure-Rust [burn-flex](https://github.com/tracel-ai/burn) backend |
| `"wgpu"` | GPU | Metal on macOS, Vulkan on Linux / Windows; first call per input shape compiles and autotunes kernels |

Both use Burn's dispatch backend, so the same interpreter runs on either. If no usable
GPU is found, `load_onnx(..., backend="wgpu")` raises a `RuntimeError` naming the problem.

## Status

Early development. The numpy ↔ Burn tensor bridge is done, and the ONNX interpreter
currently supports:

- **Linear algebra**: Gemm, Linear
- **Activations**: Relu, Sigmoid, Tanh, Gelu, Softmax, LogSoftmax
- **Elementwise**: Add, Sub, Mul, Div
- **Shape ops**: Reshape, Flatten, Transpose
- **Conv/pooling**: Conv2d, BatchNormalization (folded into the preceding Conv when possible), MaxPool2d, AveragePool2d, GlobalAveragePool
- **Reductions**: ReduceMean, ReduceSum, ReduceMax, ReduceMin

### Verified models

ResNet-18 runs end to end and matches ONNX Runtime (max logit diff < 1e-5, same top-1)
for the ONNX Model Zoo exports (`resnet18-v1-7`, `resnet18-v2-7`) and torchvision's
`resnet18` exported with both `torch.onnx` exporters (legacy and dynamo).

Latency vs ONNX Runtime on CPU (torchvision export, median of 15 runs after warm-up).
Numbers vary a lot by platform:

| platform | backend | batch 1 | batch 8 |
|----------|---------|--------:|--------:|
| Apple M-series (10 cores) | burn-python `wgpu` (Metal GPU) | **6.1 ms** | **35.6 ms** |
| Apple M-series (10 cores) | burn-python `flex` (CPU) | 22.4 ms | 135.7 ms |
| Apple M-series (10 cores) | ONNX Runtime (CPU) | 14.9 ms | 123.0 ms |
| Linux x86 (GitHub Actions runner) | burn-python `flex` (CPU) | 58.0 ms | 335.7 ms |
| Linux x86 (GitHub Actions runner) | ONNX Runtime (CPU) | 18.7 ms | 145.9 ms |

Run `python tests/test_resnet.py --bench` to measure on your machine.

More ops are being added incrementally.

## Building

```bash
pip install maturin
maturin develop --release
```

## Testing

```bash
python tests/make_test_model.py   # generates tests/mlp.onnx
python tests/test_bridge.py       # numpy <-> Burn tensor bridge
python tests/compare_ort.py       # correctness + perf vs ONNX Runtime
python tests/test_ops.py          # op edge cases (padding, pooling, reshape) vs ORT
python tests/test_resnet.py       # ResNet-18 end to end vs ORT (add --bench for timings)

BURN_BACKEND=wgpu python tests/test_ops.py     # same tests on the GPU backend
```

## License

MIT OR Apache-2.0
