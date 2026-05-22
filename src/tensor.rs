// numpy <-> Burn tensor bridge
//
// in:  numpy f32 -> TensorData (one copy) -> FlexTensor
// out: FlexTensor -> TensorData (sync CPU read) -> numpy f32 (one copy)

use burn_backend::{backend::ops::FloatTensorOps, TensorData, DType};
use burn_flex::{Flex, FlexDevice, FlexTensor};
use cubecl_common::reader::read_sync;
use numpy::{PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;

pub type B = Flex;
pub type FloatPrim = FlexTensor;

pub fn default_device() -> FlexDevice {
    FlexDevice
}

// numpy f32 ndarray -> FlexTensor (one copy at the Python boundary)
pub fn numpy_to_flex(arr: &PyReadonlyArrayDyn<'_, f32>) -> FloatPrim {
    let shape: Vec<usize> = arr.shape().to_vec();
    let slice = arr
        .as_slice()
        .expect("array must be contiguous — call np.ascontiguousarray() first");

    let bytes: Vec<u8> = bytemuck::cast_slice(slice).to_vec();
    let data = TensorData::from_bytes_vec(bytes, shape, DType::F32);

    B::float_from_data(data, &default_device())
}

// FlexTensor -> numpy f32 ndarray (sync read + one copy out)
pub fn flex_to_numpy<'py>(py: Python<'py>, prim: FloatPrim) -> Bound<'py, PyArrayDyn<f32>> {
    // for CPU backends the future resolves immediately
    let data: TensorData = read_sync(B::float_into_data(prim))
        .expect("float_into_data error");

    let shape: Vec<usize> = data.shape.iter().copied().collect();
    let floats: &[f32] = bytemuck::cast_slice(data.as_bytes());

    PyArray1::from_slice(py, floats)
        .reshape(shape)
        .expect("reshape failed")
}
