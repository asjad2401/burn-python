// numpy <-> Burn tensor bridge
//
// in:  numpy f32 -> TensorData (one copy) -> backend tensor (upload, for GPU backends)
// out: backend tensor -> TensorData (sync read) -> numpy f32 (one copy)
//
// All ops run on burn-dispatch's `Dispatch` backend, which forwards each op to the
// backend selected by the tensor's device (Flex on CPU, wgpu on GPU, ...).

use burn_backend::{TensorData, backend::ops::FloatTensorOps};
use burn_dispatch::{Dispatch, DispatchDevice, DispatchTensor};
use burn_flex::FlexDevice;
use cubecl_common::reader::read_sync;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

pub type B = Dispatch;
pub type FloatPrim = DispatchTensor;
pub type Device = DispatchDevice;

/// Backend names accepted by `load_onnx(..., backend=...)`.
pub const BACKENDS: &[&str] = &["flex", "wgpu"];

/// Map a user-facing backend name to a device.
pub fn device_for(backend: &str) -> Result<Device, String> {
    match backend {
        "flex" | "cpu" => Ok(cpu_device()),
        "wgpu" | "gpu" => Ok(gpu_device()),
        other => Err(format!(
            "unknown or unavailable backend '{other}' (available: {})",
            BACKENDS.join(", ")
        )),
    }
}

/// Default GPU, through the wgpu flavor this platform is built with (see Cargo.toml).
fn gpu_device() -> Device {
    let device = burn_wgpu::WgpuDevice::DefaultDevice;
    #[cfg(target_os = "macos")]
    return DispatchDevice::Metal(device);
    #[cfg(not(target_os = "macos"))]
    return DispatchDevice::Vulkan(device);
}

/// Device of the default CPU backend.
pub fn cpu_device() -> Device {
    DispatchDevice::Flex(FlexDevice)
}

// numpy f32 ndarray -> backend tensor (one copy at the Python boundary)
pub fn numpy_to_tensor(arr: &PyReadonlyArrayDyn<'_, f32>, device: &Device) -> FloatPrim {
    let shape: Vec<usize> = arr.shape().to_vec();
    // as_slice() also accepts Fortran order (e.g. x.T), which would be read in memory
    // order; anything not C-contiguous is gathered in logical order instead
    let values: Vec<f32> = match arr.as_slice() {
        Ok(slice) if arr.is_c_contiguous() => slice.to_vec(),
        _ => arr.as_array().iter().copied().collect(),
    };
    let data = TensorData::new(values, shape);

    B::float_from_data(data, device)
}

// backend tensor -> numpy f32 ndarray (sync read + one copy out)
pub fn tensor_to_numpy<'py>(
    py: Python<'py>,
    prim: FloatPrim,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    // blocks until the device has finished (immediate for CPU backends)
    let data: TensorData = read_sync(B::float_into_data(prim))
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read tensor: {e:?}")))?;
    // a backend may compute in another float type; the numpy side is always f32
    let data = data.convert::<f32>();

    let shape: Vec<usize> = data.shape.iter().copied().collect();
    let floats: &[f32] = bytemuck::cast_slice(data.as_bytes());

    PyArray1::from_slice(py, floats).reshape(shape)
}
