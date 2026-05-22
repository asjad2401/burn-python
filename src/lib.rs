use pyo3::prelude::*;

mod tensor;

#[pymodule]
fn _burn_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
