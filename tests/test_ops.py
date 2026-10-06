"""
Op edge-case tests vs ONNX Runtime — padding modes, pooling, reshape, load errors.
Each case builds a single-node ONNX model on the fly.
Run with: python tests/test_ops.py
"""

import os
import re
import sys
import tempfile
import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper, TensorProto
import burn_python as burn

BACKEND = os.environ.get("BURN_BACKEND", "flex")  # e.g. BURN_BACKEND=wgpu

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
    path = os.path.join(tmpdir, re.sub(r"[^\w-]+", "_", name) + ".onnx")
    onnx.save(model, path)
    return path

def compare(label, node, x_shape, initializers=(), opset=17):
    path = save_model(label.replace(" ", "_"), node, x_shape, initializers, opset)
    x = np.random.randn(*x_shape).astype(np.float32)
    try:
        got = burn.load_onnx(path, backend=BACKEND)([x])[0]
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

# ── batchnorm (folded into conv where possible) ───────────────────────────────

print("\n=== batchnorm ===")

def graph_model(name, nodes, x_shape, initializers, outputs=("y",)):
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x_shape))],
        [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outputs],
        initializer=list(initializers),
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    path = os.path.join(tmpdir, f"{name}.onnx")
    onnx.save(model, path)
    return path

def compare_graph(label, nodes, x_shape, initializers, outputs=("y",), n_nodes=None):
    path = graph_model(label.replace(" ", "_"), nodes, x_shape, initializers, outputs)
    x = np.random.randn(*x_shape).astype(np.float32)
    try:
        model = burn.load_onnx(path, backend=BACKEND)
        if n_nodes is not None:  # e.g. BN folded away into the conv
            check(f"{label}  runs {n_nodes} node(s)", f"nodes={n_nodes}," in repr(model))
        got = model([x])
    except BaseException as e:
        check(f"{label}  ({type(e).__name__}: {e})", False)
        return
    want = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"x": x})
    ok = all(g.shape == w.shape and np.allclose(g, w, atol=1e-4) for g, w in zip(got, want))
    diff = max(np.abs(g - w).max() for g, w in zip(got, want)) if ok else float("nan")
    check(f"{label}  max_abs_diff={diff:.2e}", ok)

def bn_params(c, prefix="bn"):
    return [
        numpy_helper.from_array(np.random.rand(c).astype(np.float32) + 0.5, f"{prefix}_g"),
        numpy_helper.from_array(np.random.randn(c).astype(np.float32), f"{prefix}_b"),
        numpy_helper.from_array(np.random.randn(c).astype(np.float32), f"{prefix}_m"),
        numpy_helper.from_array(np.random.rand(c).astype(np.float32) + 0.5, f"{prefix}_v"),
    ]

def bn(x, y, prefix="bn"):
    return helper.make_node("BatchNormalization",
                            [x, f"{prefix}_g", f"{prefix}_b", f"{prefix}_m", f"{prefix}_v"], [y])

compare_graph("conv+bn (folded)",
              [helper.make_node("Conv", ["x", "W", "B"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
               bn("c", "y")], X, [W, B] + bn_params(3), n_nodes=1)
compare_graph("conv(no bias)+bn (folded)",
              [helper.make_node("Conv", ["x", "W"], ["c"], kernel_shape=[3, 3]),
               bn("c", "y")], X, [W] + bn_params(3), n_nodes=1)
compare_graph("conv+bn, conv output reused (not folded)",
              [helper.make_node("Conv", ["x", "W", "B"], ["c"], kernel_shape=[3, 3]),
               bn("c", "n"), helper.make_node("Add", ["c", "n"], ["y"])], X, [W, B] + bn_params(3),
              n_nodes=3)
compare_graph("conv+bn, conv output is graph output (not folded)",
              [helper.make_node("Conv", ["x", "W", "B"], ["c"], kernel_shape=[3, 3]),
               bn("c", "y")], X, [W, B] + bn_params(3), outputs=("y", "c"), n_nodes=2)
compare_graph("bn after relu (runtime path)",
              [helper.make_node("Relu", ["x"], ["r"]), bn("r", "y", "bn2")], X, bn_params(2, "bn2"))
compare_graph("bn on rank-2 input",
              [bn("x", "y", "bn3")], (5, 4), bn_params(4, "bn3"))

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

# ── broadcasting across ranks (numpy-style, as ONNX allows) ──────────────────

print("\n=== broadcasting ===")

for op, x_shape, c_shape in [("Add", (4, 3), (3,)), ("Mul", (2, 3, 5, 4), (3, 1, 1)),
                             ("Sub", (2, 3, 5, 4), (4,)), ("Div", (3,), (2, 3))]:
    c = numpy_helper.from_array(np.random.rand(*c_shape).astype(np.float32) + 0.5, "c")
    compare(f"{op} {list(x_shape)} with {list(c_shape)}",
            helper.make_node(op, ["x", "c"], ["y"]), x_shape, [c])

gemm_c = numpy_helper.from_array(np.random.randn(2).astype(np.float32), "C")
compare("Gemm alpha/beta, rank-1 C",
        helper.make_node("Gemm", ["x", "W2", "C"], ["y"], alpha=0.5, beta=2.0),
        (3, 4), [numpy_helper.from_array(np.random.randn(4, 2).astype(np.float32), "W2"), gemm_c])

# ── reductions ───────────────────────────────────────────────────────────────

print("\n=== reductions ===")

def reduce_node(op, axes, keepdims):
    a = numpy_helper.from_array(np.array(axes, dtype=np.int64), "axes")
    return helper.make_node(op, ["x", "axes"], ["y"], keepdims=keepdims), [a]

for op in ["ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin"]:
    node, init = reduce_node(op, [-1, -2], 1)
    compare(f"{op} axes=[-1,-2] keepdims=1", node, X, init, opset=18)
    node, init = reduce_node(op, [1], 0)
    compare(f"{op} axes=[1] keepdims=0", node, X, init, opset=18)
compare("ReduceMean all axes keepdims=1",
        helper.make_node("ReduceMean", ["x"], ["y"], keepdims=1), X, opset=18)

# ── load errors ──────────────────────────────────────────────────────────────

print("\n=== load errors ===")

path = save_model("unsupported", helper.make_node("Exp", ["x"], ["y"]), (2, 3))
try:
    burn.load_onnx(path, backend=BACKEND)
    check("unsupported op raises at load", False)
except RuntimeError as e:
    check(f"unsupported op raises at load  ({e})", "Exp" in str(e))

def raises(label, fn, exc, *needles):
    try:
        fn()
        check(f"{label}  (no exception)", False)
    except exc as e:
        check(f"{label}  ({type(e).__name__}: {e})", all(n in str(e) for n in needles))
    except BaseException as e:  # e.g. pyo3 PanicException
        check(f"{label}  (wrong exception {type(e).__name__}: {e})", False)

raises("unknown backend", lambda: burn.load_onnx(path, backend="tpu"), ValueError, "tpu", "flex")

# ── inference errors ─────────────────────────────────────────────────────────

print("\n=== inference errors ===")

relu = burn.load_onnx(save_model("relu", helper.make_node("Relu", ["x"], ["y"]), ("N", 3)),
                      backend=BACKEND)
raises("float64 input", lambda: relu([np.zeros((2, 3))]), TypeError, "float32", "float64")
raises("non-array input", lambda: relu([[1.0, 2.0, 3.0]]), TypeError, "numpy array")
raises("wrong rank", lambda: relu([np.zeros((2, 3, 1), np.float32)]), ValueError, "[?, 3]")
raises("wrong static dim", lambda: relu([np.zeros((2, 4), np.float32)]), ValueError, "[?, 3]")
raises("wrong input count", lambda: relu([]), ValueError, "expected 1 input")

# shape mismatch caught inside the backend: Gemm with an input whose inner dim
# doesn't match W, declared as fully dynamic so the input check lets it through
gemm_w = numpy_helper.from_array(np.ones((4, 2), np.float32), "W")
gemm = burn.load_onnx(save_model(
    "gemm_dyn", helper.make_node("Gemm", ["x", "W"], ["y"]), ("N", "K"), [gemm_w]),
    backend=BACKEND)
raises("backend panic -> RuntimeError with node context",
       lambda: gemm([np.zeros((2, 3), np.float32)]), RuntimeError, "(Gemm)")

# ── non-contiguous inputs ────────────────────────────────────────────────────

print("\n=== non-contiguous inputs ===")

x = np.random.randn(3, 4).astype(np.float32)
for label, view in [("transposed (F-order)", x.T), ("strided slice", x[:, ::2].T),
                    ("negative stride", np.ascontiguousarray(x.T)[::-1])]:
    check(label, np.array_equal(relu([view])[0], np.maximum(view, 0)))

print()

if failures:
    print(f"{failures} check(s) failed")
    sys.exit(1)
