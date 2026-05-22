"""
Compare burn-python vs ONNX Runtime:
  - numerical correctness across batch sizes
  - inference throughput (burn debug build vs ORT, so don't read too much into perf yet)

Run: python tests/compare_ort.py
"""

import time
import numpy as np
import onnxruntime as ort
import burn_python as burn

MODEL = "tests/mlp.onnx"
RUNS  = 500

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(label, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return cond

# ── load both ──────────────────────────────────────────────────────────────────
burn_model = burn.load_onnx(MODEL)
ort_sess   = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
ort_input  = ort_sess.get_inputs()[0].name

print(f"\nmodel: {MODEL}")
print(f"burn: {burn_model}")
print(f"ort:  {ort_sess.get_inputs()[0]} -> {ort_sess.get_outputs()[0]}")

# ── correctness ────────────────────────────────────────────────────────────────
print("\n=== correctness (burn vs ORT) ===")

np.random.seed(0)
for batch in [1, 4, 16, 64]:
    x = np.random.randn(batch, 4).astype(np.float32)

    burn_out = burn_model([x])[0]
    ort_out  = ort_sess.run(None, {ort_input: x})[0]

    max_diff = np.abs(burn_out - ort_out).max()
    ok = np.allclose(burn_out, ort_out, atol=1e-5)
    check(f"batch={batch:<3}  max_abs_diff={max_diff:.2e}", ok)

# ── throughput ─────────────────────────────────────────────────────────────────
print(f"\n=== throughput ({RUNS} runs each, debug build) ===")
print(f"  (run 'maturin develop --release' for optimized build)\n")

for batch in [1, 16, 64, 256]:
    x = np.random.randn(batch, 4).astype(np.float32)

    # warm up
    for _ in range(20):
        burn_model([x])
        ort_sess.run(None, {ort_input: x})

    t0 = time.perf_counter()
    for _ in range(RUNS):
        burn_model([x])
    burn_ms = (time.perf_counter() - t0) / RUNS * 1000

    t0 = time.perf_counter()
    for _ in range(RUNS):
        ort_sess.run(None, {ort_input: x})
    ort_ms = (time.perf_counter() - t0) / RUNS * 1000

    ratio = burn_ms / ort_ms
    print(f"  batch={batch:<4}  burn {burn_ms:.3f} ms  |  ORT {ort_ms:.3f} ms  |  ratio {ratio:.1f}x")

print()
