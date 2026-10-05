"""
Op edge-case tests vs ONNX Runtime — padding modes, pooling, reshape, load errors.
Each case builds a single-node ONNX model on the fly.
Run with: python tests/test_ops.py
"""

import os
import sys
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
import burn_python as burn

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

failures = 0
tmpdir = tempfile.mkdtemp()

def check(label, cond):
    global failures
    print(f"  {'[' + PASS + ']' if cond else '[' + FAIL + ']'} {label}")
    if not cond:
        failures += 1
    return cond

def save_model(name, node, x_shape, initializers=(), opset=17):
    graph = helper.make_graph(
        [node],
        name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10  # older onnxruntime builds reject newer IR versions
    path = os.path.join(tmpdir, f"{name}.onnx")
    onnx.save(model, path)
    return path

def compare(label, node, x_shape, initializers=(), opset=17):
    path = save_model(label.replace(" ", "_"), node, x_shape, initializers, opset)
    x = np.random.randn(*x_shape).astype(np.float32)
    try:
        got = burn.load_onnx(path)([x])[0]
    except BaseException as e:  # pyo3 panics aren't Exception subclasses
        check(f"{label}  ({type(e).__name__}: {e})", False)
        return
    want = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
    if got.shape != want.shape:
        check(f"{label}  shape {got.shape} != ORT {want.shape}", False)
        return
    diff = np.abs(got - want).max()
    check(f"{label}  max_abs_diff={diff:.2e}", np.allclose(got, want, atol=1e-4))

np.random.seed(0)
X = (1, 2, 7, 6)

# ── conv padding ─────────────────────────────────────────────────────────────

print("\n=== conv2d padding ===")

W = numpy_helper.from_array(np.random.randn(3, 2, 3, 3).astype(np.float32), "W")
B = numpy_helper.from_array(np.random.randn(3).astype(np.float32), "B")

def conv(**attrs):
    return helper.make_node("Conv", ["x", "W", "B"], ["y"], kernel_shape=[3, 3], **attrs)

compare("conv symmetric", conv(pads=[1, 1, 1, 1]), X, [W, B])
compare("conv asymmetric", conv(pads=[0, 1, 2, 0]), X, [W, B])
compare("conv SAME_UPPER s2", conv(auto_pad="SAME_UPPER", strides=[2, 2]), X, [W, B])
compare("conv SAME_LOWER s2", conv(auto_pad="SAME_LOWER", strides=[2, 2]), X, [W, B])
compare("conv VALID", conv(auto_pad="VALID"), X, [W, B])

# ── pooling ──────────────────────────────────────────────────────────────────

print("\n=== pooling ===")

def pool(op, **attrs):
    return helper.make_node(op, ["x"], ["y"], kernel_shape=[3, 3], **attrs)

compare("maxpool asymmetric", pool("MaxPool", pads=[0, 1, 2, 0]), X)
compare("maxpool SAME_UPPER s2", pool("MaxPool", auto_pad="SAME_UPPER", strides=[2, 2]), X)
compare("maxpool ceil_mode", pool("MaxPool", strides=[2, 2], ceil_mode=1), X)
compare("avgpool asymmetric incl pad",
        pool("AveragePool", pads=[0, 1, 2, 0], count_include_pad=1), X)
compare("avgpool asymmetric excl pad",
        pool("AveragePool", pads=[0, 1, 2, 0], count_include_pad=0), X)
compare("avgpool SAME_LOWER s2",
        pool("AveragePool", auto_pad="SAME_LOWER", strides=[2, 2]), X)

# ── reshape ──────────────────────────────────────────────────────────────────

print("\n=== reshape ===")

def reshape(shape):
    s = numpy_helper.from_array(np.array(shape, dtype=np.int64), "s")
    return helper.make_node("Reshape", ["x", "s"], ["y"]), [s]

node, init = reshape([0, -1])
compare("reshape [0, -1]", node, (2, 3, 4), init)
node, init = reshape([0, 0, -1])
compare("reshape [0, 0, -1]", node, (2, 3, 4), init)
node, init = reshape([-1, 4])
compare("reshape [-1, 4]", node, (2, 3, 4), init)

# ── load errors ──────────────────────────────────────────────────────────────

print("\n=== load errors ===")

path = save_model("unsupported", helper.make_node("Exp", ["x"], ["y"]), (2, 3))
try:
    burn.load_onnx(path)
    check("unsupported op raises at load", False)
except RuntimeError as e:
    check(f"unsupported op raises at load  ({e})", "Exp" in str(e))

print()

if failures:
    print(f"{failures} check(s) failed")
    sys.exit(1)
