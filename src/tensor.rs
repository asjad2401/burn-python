// numpy <-> Burn tensor bridge
// uses zero-copy path from burn issue #4411:
//   NumpyOwner -> bytes::Bytes::from_owner -> TensorData::from_bytes -> FlexTensor
