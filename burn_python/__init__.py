import numpy as np
from ._burn_python import __version__, OnnxModel
from . import _burn_python as _ext

def roundtrip(arr: np.ndarray) -> np.ndarray:
    """numpy -> Burn -> numpy (dev/test utility)."""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return _ext.roundtrip(arr)

def load_onnx(path: str, backend: str = "flex") -> OnnxModel:
    """Load an ONNX model for inference."""
    if backend != "flex":
        raise ValueError(f"unsupported backend '{backend}' — only 'flex' is available right now")
    return _ext.load_onnx(path)

__all__ = ["__version__", "load_onnx", "roundtrip", "OnnxModel"]
