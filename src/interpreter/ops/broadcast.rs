// ONNX uses numpy-style broadcasting, where operands may differ in rank
// ([N, C] + [C], [N, C, H, W] * [C, 1, 1]). Burn broadcasts size-1 dims but needs
// equal ranks, so the lower-rank operand gets leading 1-dims first.

use burn_backend::{TensorMetadata, backend::ops::FloatTensorOps};

use crate::tensor::{B, FloatPrim};

fn unsqueeze_to(t: FloatPrim, rank: usize) -> FloatPrim {
    let shape: Vec<usize> = t.shape().iter().copied().collect();
    if shape.len() >= rank {
        return t;
    }
    let padded: Vec<usize> = std::iter::repeat_n(1, rank - shape.len())
        .chain(shape)
        .collect();
    B::float_reshape(t, padded.into())
}

/// Bring two operands to the same rank for an elementwise op.
pub fn align(a: FloatPrim, b: FloatPrim) -> (FloatPrim, FloatPrim) {
    let rank = a.shape().num_dims().max(b.shape().num_dims());
    (unsqueeze_to(a, rank), unsqueeze_to(b, rank))
}
