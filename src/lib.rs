use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

mod interpreter;
mod panic;
mod tensor;

use interpreter::OnnxModel;

/// Round-trip a numpy f32 array through Burn (dev/test utility).
#[pyfunction]
fn roundtrip<'py>(
    py: Python<'py>,
    arr: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let prim = tensor::numpy_to_tensor(&arr, &tensor::cpu_device());
    tensor::tensor_to_numpy(py, prim)
}

/// Load an ONNX model from a file path, to run on the given backend.
#[pyfunction]
#[pyo3(signature = (path, backend = "flex"))]
fn load_onnx(path: &str, backend: &str) -> PyResult<OnnxModel> {
    tensor::device_for(backend).map_err(pyo3::exceptions::PyValueError::new_err)?;
    interpreter::load_onnx(path, backend).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

/// Backends compiled into this build.
#[pyfunction]
fn available_backends() -> Vec<&'static str> {
    tensor::BACKENDS.to_vec()
}

#[pymodule]
fn _burn_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    panic::install_hook();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(load_onnx, m)?)?;
    m.add_function(wrap_pyfunction!(available_backends, m)?)?;
    m.add_class::<OnnxModel>()?;
    Ok(())
}
