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
- **Conv/pooling**: Conv2d, BatchNormalization (fused), MaxPool2d, AveragePool2d, GlobalAveragePool

Enough to run simple MLPs and small CNNs; more ops are being added incrementally.

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
```

## License

MIT OR Apache-2.0
