import numpy as np
from ._burn_python import __version__, OnnxModel, available_backends
from . import _burn_python as _ext

def roundtrip(arr: np.ndarray) -> np.ndarray:
    """numpy -> Burn -> numpy (dev/test utility)."""
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return _ext.roundtrip(arr)

def load_onnx(path: str, backend: str = "flex") -> OnnxModel:
    """Load an ONNX model for inference.

    backend: "flex" (CPU, default) or "wgpu" (GPU via Metal / Vulkan / DX12).
    See available_backends() for what this build supports.
    """
    return _ext.load_onnx(str(path), backend)

__all__ = ["__version__", "load_onnx", "available_backends", "roundtrip", "OnnxModel"]
