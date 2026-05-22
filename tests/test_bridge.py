"""
Tensor bridge tests — correctness + perf vs raw numpy.
Run with: python tests/test_bridge.py
"""

import time
import numpy as np
import burn_python as burn

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(label, cond):
    print(f"  {'[' + PASS + ']' if cond else '[' + FAIL + ']'} {label}")
    return cond

# ── correctness ──────────────────────────────────────────────────────────────

print("\n=== correctness ===")

# basic shapes
for shape in [(1,), (128,), (4, 4), (2, 3, 4), (1, 3, 224, 224), (2, 4, 8, 8, 2)]:
    x = np.random.randn(*shape).astype(np.float32)
    y = burn.roundtrip(x)
    ok = x.shape == y.shape and np.array_equal(x, y) and y.dtype == np.float32
    check(f"shape {shape}", ok)

# contiguous after transpose (non-contiguous input should raise, contiguous should pass)
x = np.random.randn(4, 4).astype(np.float32)
x_contig = np.ascontiguousarray(x.T)
y = burn.roundtrip(x_contig)
check("contiguous transposed array", np.array_equal(x_contig, y))

# values: zeros, ones, negatives, large
for name, arr in [
    ("zeros",     np.zeros((32, 32), dtype=np.float32)),
    ("ones",      np.ones((32, 32), dtype=np.float32)),
    ("negatives", np.full((16,), -1.23456, dtype=np.float32)),
    ("large vals",np.full((8,), 1e38, dtype=np.float32)),
    ("small vals",np.full((8,), 1e-38, dtype=np.float32)),
    ("nan",       np.array([float('nan')], dtype=np.float32)),
    ("inf",       np.array([float('inf'), float('-inf')], dtype=np.float32)),
]:
    y = burn.roundtrip(arr)
    # nan != nan, so handle separately
    ok = np.array_equal(arr, y, equal_nan=True) and arr.shape == y.shape
    check(name, ok)

# single element
x = np.array([42.0], dtype=np.float32)
y = burn.roundtrip(x)
check("single element", y[0] == 42.0)

# non-contiguous input is auto-contiguified (F-order transpose)
x_nc = np.random.randn(4, 4).astype(np.float32).T
y_nc = burn.roundtrip(x_nc)
check("non-contiguous auto-fixed shape", x_nc.shape == y_nc.shape)
check("non-contiguous auto-fixed values", np.array_equal(np.ascontiguousarray(x_nc), y_nc))

# ── performance ──────────────────────────────────────────────────────────────

print("\n=== performance (roundtrip vs numpy copy) ===")

RUNS = 200

for n_elems, label in [(1_000, "1K"), (100_000, "100K"), (1_000_000, "1M"), (10_000_000, "10M")]:
    x = np.random.randn(n_elems).astype(np.float32)

    # warm up
    for _ in range(5):
        burn.roundtrip(x)
        x.copy()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        burn.roundtrip(x)
    burn_ms = (time.perf_counter() - t0) / RUNS * 1000

    t0 = time.perf_counter()
    for _ in range(RUNS):
        x.copy()
    np_ms = (time.perf_counter() - t0) / RUNS * 1000

    ratio = burn_ms / np_ms
    print(f"  {label:>6} f32  |  burn {burn_ms:.3f} ms  |  np.copy {np_ms:.3f} ms  |  ratio {ratio:.1f}x")

print()
