import numpy as np
from ._burn_python import __version__
from . import _burn_python as _ext

def roundtrip(arr: np.ndarray) -> np.ndarray:
    """numpy -> Burn -> numpy (dev/test utility)."""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return _ext.roundtrip(arr)

__all__ = ["__version__", "roundtrip"]
