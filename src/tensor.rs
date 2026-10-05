// numpy <-> Burn tensor bridge
//
// in:  numpy f32 -> TensorData (one copy) -> FlexTensor
// out: FlexTensor -> TensorData (sync CPU read) -> numpy f32 (one copy)

use burn_backend::{TensorData, backend::ops::FloatTensorOps};
use burn_flex::{Flex, FlexDevice, FlexTensor};
use cubecl_common::reader::read_sync;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

pub type B = Flex;
pub type FloatPrim = FlexTensor;

pub fn default_device() -> FlexDevice {
    FlexDevice
}

// numpy f32 ndarray -> FlexTensor (one copy at the Python boundary)
pub fn numpy_to_flex(arr: &PyReadonlyArrayDyn<'_, f32>) -> FloatPrim {
    let shape: Vec<usize> = arr.shape().to_vec();
    // as_slice() also accepts Fortran order (e.g. x.T), which would be read in memory
    // order; anything not C-contiguous is gathered in logical order instead
    let values: Vec<f32> = match arr.as_slice() {
        Ok(slice) if arr.is_c_contiguous() => slice.to_vec(),
        _ => arr.as_array().iter().copied().collect(),
    };
    let data = TensorData::new(values, shape);

    B::float_from_data(data, &default_device())
}

// FlexTensor -> numpy f32 ndarray (sync read + one copy out)
pub fn flex_to_numpy<'py>(
    py: Python<'py>,
    prim: FloatPrim,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    // for CPU backends the future resolves immediately
    let data: TensorData = read_sync(B::float_into_data(prim))
        .map_err(|e| PyRuntimeError::new_err(format!("failed to read tensor: {e:?}")))?;

    let shape: Vec<usize> = data.shape.iter().copied().collect();
    let floats: &[f32] = bytemuck::cast_slice(data.as_bytes());

    PyArray1::from_slice(py, floats).reshape(shape)
}
