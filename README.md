# burn-python

Python inference frontend for the [Burn](https://github.com/tracel-ai/burn) deep learning framework.

Load any ONNX model and run inference from Python — numpy in, numpy out. No Rust required.

```python
import burn_python as burn
import numpy as np

model = burn.load_onnx("resnet50.onnx")
output = model(np.random.randn(1, 3, 224, 224).astype(np.float32))
```

## Status

Early development. Stage 1 (numpy ↔ Burn tensor bridge) in progress.

## Building

```bash
pip install maturin
maturin develop
```

## License

MIT OR Apache-2.0
