use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

mod tensor;

/// Round-trip a numpy f32 array through Burn (numpy -> FlexTensor -> numpy).
/// Useful for testing the bridge; will be replaced by actual ops in later stages.
#[pyfunction]
fn roundtrip<'py>(
    py: Python<'py>,
    arr: PyReadonlyArrayDyn<'py, f32>,
) -> Bound<'py, PyArrayDyn<f32>> {
    let prim = tensor::numpy_to_flex(&arr);
    tensor::flex_to_numpy(py, prim)
}

#[pymodule]
fn _burn_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(roundtrip, m)?)?;
    Ok(())
}
