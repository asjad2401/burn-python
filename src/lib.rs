use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

mod interpreter;
mod tensor;

use interpreter::OnnxModel;

/// Round-trip a numpy f32 array through Burn (dev/test utility).
#[pyfunction]
fn roundtrip<'py>(
    py: Python<'py>,
    arr: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let prim = tensor::numpy_to_flex(&arr);
    tensor::flex_to_numpy(py, prim)
}

/// Load an ONNX model from a file path.
#[pyfunction]
fn load_onnx(path: &str) -> PyResult<OnnxModel> {
    interpreter::load_onnx(path).map_err(pyo3::exceptions::PyRuntimeError::new_err)
}

#[pymodule]
fn _burn_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Panics inside ops are caught and re-raised as Python exceptions that carry the
    // message, so the default hook's stderr dump is just noise. RUST_BACKTRACE keeps it.
    if std::env::var_os("RUST_BACKTRACE").is_none() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(roundtrip, m)?)?;
    m.add_function(wrap_pyfunction!(load_onnx, m)?)?;
    m.add_class::<OnnxModel>()?;
    Ok(())
}
