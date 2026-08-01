import burn_python as burn
import numpy as np

# Load an ONNX model
model = burn.load_onnx("tests/mlp.onnx")

# Generate sample input matching the model expected input shape (batch_size=1, features=4)
x = np.array([[0.5, -0.2, 0.1, 0.9]], dtype=np.float32)

# Run inference
output = model([x])[0]

print("Model Output:", output)
