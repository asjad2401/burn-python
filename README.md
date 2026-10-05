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

CPU latency vs ONNX Runtime (torchvision export, burn `flex` backend, median of 10 runs).
Performance varies a lot by platform:

| platform | batch | burn-python | ONNX Runtime | burn / ORT |
|----------|------:|------------:|-------------:|-----------:|
| Apple M-series (10 cores) | 1 | 22.6 ms  | 14.7 ms  | 1.5× slower |
| Apple M-series (10 cores) | 8 | 105.0 ms | 118.5 ms | 0.9× (faster) |
| Linux x86 (GitHub Actions runner) | 1 | 58.0 ms  | 18.7 ms  | 3.1× slower |
| Linux x86 (GitHub Actions runner) | 8 | 335.7 ms | 145.9 ms | 2.3× slower |

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
```

## License

MIT OR Apache-2.0
