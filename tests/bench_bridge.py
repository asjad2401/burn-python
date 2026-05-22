"""
Bridge throughput benchmark — shows GB/s to reveal whether overhead is
PyO3 call cost or actual memory bandwidth.

Real compute benchmarks (matmul, conv2d, etc.) come in Stage 2 once ops are wired up.
"""

import time
import numpy as np
import burn_python as burn

RUNS = 300

print(f"\n{'size':>10} {'elements':>12} {'burn GB/s':>12} {'np.copy GB/s':>14} {'overhead':>10}")
print("-" * 64)

for n_elems in [1_000, 10_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]:
    x = np.random.randn(n_elems).astype(np.float32)
    bytes_total = x.nbytes

    # warm up
    for _ in range(10):
        burn.roundtrip(x)
        x.copy()

    t0 = time.perf_counter()
    for _ in range(RUNS):
        burn.roundtrip(x)
    burn_s = (time.perf_counter() - t0) / RUNS

    t0 = time.perf_counter()
    for _ in range(RUNS):
        x.copy()
    np_s = (time.perf_counter() - t0) / RUNS

    # burn does 2 copies (in + out), np does 1 — normalize for fair comparison
    burn_gbs = (bytes_total * 2) / burn_s / 1e9
    np_gbs   = (bytes_total * 1) / np_s   / 1e9

    label = f"{bytes_total / 1024:.0f} KB" if bytes_total < 1024**2 else f"{bytes_total / 1024**2:.0f} MB"
    overhead_us = (burn_s - np_s * 2) * 1e6  # fixed overhead above 2x copy

    print(f"{label:>10} {n_elems:>12,} {burn_gbs:>11.1f}  {np_gbs:>13.1f}  {overhead_us:>+8.1f} µs")

print()
print("note: burn roundtrip = 2 memcopies (in + out); np.copy = 1.")
print("      GB/s columns are normalized per-copy for comparability.")
print("      overhead col = burn_time - 2×np_copy_time (PyO3 fixed cost).")
