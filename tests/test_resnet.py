"""
End-to-end ResNet-18 vs ONNX Runtime, on real exported models:
  - ONNX Model Zoo resnet18-v1-7 / v2-7 (MXNet exports, downloaded + cached)
  - torchvision resnet18 via torch.onnx legacy + dynamo exporters (if torch is installed)

Checks logits match ORT and top-1 agrees, across batch sizes.
Run: python tests/test_resnet.py [--bench]
"""

import os
import sys
import time
import urllib.request
import numpy as np
import onnxruntime as ort
import burn_python as burn

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
ZOO_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/{}.onnx"
BATCHES = [1, 4]
BENCH = "--bench" in sys.argv

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

failures = 0

def check(label, cond):
    global failures
    print(f"  {'[' + PASS + ']' if cond else '[' + FAIL + ']'} {label}")
    if not cond:
        failures += 1
    return cond

def zoo_model(name):
    path = os.path.join(MODEL_DIR, f"{name}.onnx")
    if not os.path.exists(path):
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(ZOO_URL.format(name), path + ".part")
        os.rename(path + ".part", path)
    return path

def torch_models():
    try:
        import torch
        import torchvision
    except ImportError:
        print("torch/torchvision not installed — skipping PyTorch exports")
        return []

    model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()
    # batch 2, not 1: torch.export specializes size-1 dims, baking batch=1 into Reshape
    x = torch.randn(2, 3, 224, 224)
    paths = []

    legacy = os.path.join(MODEL_DIR, "resnet18-torch-legacy.onnx")
    if not os.path.exists(legacy):
        torch.onnx.export(model, x, legacy, input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
                          opset_version=17, dynamo=False)
    paths.append(legacy)

    dynamo = os.path.join(MODEL_DIR, "resnet18-torch-dynamo.onnx")
    if not os.path.exists(dynamo):
        try:
            torch.onnx.export(model, (x,), dynamo, input_names=["input"], output_names=["logits"],
                              dynamic_shapes={"x": {0: torch.export.Dim("N")}}, dynamo=True)
        except Exception as e:  # dynamo exporter needs onnxscript
            print(f"dynamo export unavailable ({type(e).__name__}) — skipping")
            return paths
    paths.append(dynamo)
    return paths

def median_ms(fn, runs=10):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000

os.makedirs(MODEL_DIR, exist_ok=True)
models = [zoo_model("resnet18-v1-7"), zoo_model("resnet18-v2-7")] + torch_models()

for path in models:
    print(f"\n=== {os.path.basename(path)} ===")
    model = burn.load_onnx(path)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.RandomState(0)
    for batch in BATCHES:
        x = rng.randn(batch, 3, 224, 224).astype(np.float32)
        got = model([x])[0]
        want = sess.run(None, {input_name: x})[0]
        diff = np.abs(got - want).max()
        check(f"batch={batch}  max_abs_diff={diff:.2e}",
              got.shape == want.shape and np.allclose(got, want, atol=1e-4, rtol=1e-4))
        check(f"batch={batch}  top-1 matches ORT", np.array_equal(got.argmax(1), want.argmax(1)))

    if BENCH:
        for batch in (1, 8):
            x = rng.randn(batch, 3, 224, 224).astype(np.float32)
            for _ in range(3):  # warm up
                model([x])
                sess.run(None, {input_name: x})
            burn_ms = median_ms(lambda: model([x]))
            ort_ms = median_ms(lambda: sess.run(None, {input_name: x}))
            print(f"  bench batch={batch:<2}  burn {burn_ms:6.1f} ms  |  ORT {ort_ms:6.1f} ms  |  ratio {burn_ms / ort_ms:.2f}x")

print()

if failures:
    print(f"{failures} check(s) failed")
    sys.exit(1)
