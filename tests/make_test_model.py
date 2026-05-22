"""
Generate a tiny MLP ONNX model for testing the interpreter.
Architecture: Linear(4->8) -> Relu -> Linear(8->2) -> Softmax
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

np.random.seed(42)

W1 = np.random.randn(8, 4).astype(np.float32)
b1 = np.random.randn(8).astype(np.float32)
W2 = np.random.randn(2, 8).astype(np.float32)
b2 = np.random.randn(2).astype(np.float32)

# nodes
gemm1 = helper.make_node("Gemm", ["x", "W1", "b1"], ["h1"], transB=1)
relu  = helper.make_node("Relu", ["h1"], ["h2"])
gemm2 = helper.make_node("Gemm", ["h2", "W2", "b2"], ["h3"], transB=1)
softmax = helper.make_node("Softmax", ["h3"], ["out"], axis=1)

graph = helper.make_graph(
    [gemm1, relu, gemm2, softmax],
    "mlp",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 4])],
    [helper.make_tensor_value_info("out", TensorProto.FLOAT, [None, 2])],
    initializer=[
        numpy_helper.from_array(W1, "W1"),
        numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(W2, "W2"),
        numpy_helper.from_array(b2, "b2"),
    ],
)

model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(model, "tests/mlp.onnx")
print("saved tests/mlp.onnx")

# also compute reference output with numpy for comparison
def ref_forward(x):
    h1 = x @ W1.T + b1
    h2 = np.maximum(h1, 0)
    h3 = h2 @ W2.T + b2
    e = np.exp(h3 - h3.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

x = np.random.randn(3, 4).astype(np.float32)
ref = ref_forward(x)
np.save("tests/mlp_ref_input.npy", x)
np.save("tests/mlp_ref_output.npy", ref)
print("reference input shape:", x.shape)
print("reference output:", ref)
