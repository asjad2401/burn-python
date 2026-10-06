// Spatial padding shared by conv and pooling ops.
//
// Burn's conv/pool kernels only take symmetric padding ([h, w] applied to both sides).
// ONNX allows asymmetric pads and auto_pad=SAME_*, so we resolve the real per-side
// pads here and, when they're uneven, pad the input explicitly instead.

use burn_backend::{Slice, TensorData, TensorMetadata, backend::ops::FloatTensorOps};
use onnx_ir::node::padding::{AutoPad, PaddingConfig2d};

use crate::tensor::{B, FloatPrim};

/// Per-side pads for an NCHW tensor: [top, left, bottom, right].
pub type Pads2d = [usize; 4];

/// Resolve the actual pads for a 2D window op, honouring auto_pad.
pub fn resolve_pads(
    x: &FloatPrim,
    padding: &PaddingConfig2d,
    auto_pad: &AutoPad,
    kernel: [usize; 2],
    stride: [usize; 2],
    dilation: [usize; 2],
) -> Pads2d {
    match auto_pad {
        AutoPad::Valid => [0; 4],
        AutoPad::SameUpper | AutoPad::SameLower => {
            let shape = x.shape();
            let mut begin = [0; 2];
            let mut end = [0; 2];
            for i in 0..2 {
                // output = ceil(input / stride); pad just enough to produce it
                let input = shape[2 + i];
                let out = input.div_ceil(stride[i]);
                let span = (kernel[i] - 1) * dilation[i] + 1;
                let total = ((out - 1) * stride[i] + span).saturating_sub(input);
                // SAME_UPPER puts the extra pixel at the end, SAME_LOWER at the start
                let small = total / 2;
                (begin[i], end[i]) = match auto_pad {
                    AutoPad::SameUpper => (small, total - small),
                    _ => (total - small, small),
                };
            }
            [begin[0], begin[1], end[0], end[1]]
        }
        AutoPad::NotSet => match padding {
            PaddingConfig2d::Valid => [0; 4],
            PaddingConfig2d::Explicit(top, left, bottom, right) => [*top, *left, *bottom, *right],
        },
    }
}

/// Some([h, w]) if the pads can be passed straight to a Burn kernel.
pub fn symmetric(pads: Pads2d) -> Option<[usize; 2]> {
    let [top, left, bottom, right] = pads;
    (top == bottom && left == right).then_some([top, left])
}

/// Pad an NCHW tensor's spatial dims with a constant value.
pub fn pad2d(x: FloatPrim, pads: Pads2d, value: f32) -> FloatPrim {
    let [top, left, bottom, right] = pads;
    let shape = x.shape();
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

    let padded = B::float_from_data(
        TensorData::full(vec![n, c, h + top + bottom, w + left + right], value),
        &B::float_device(&x),
    );
    let range =
        |start: usize, len: usize| Slice::new(start as isize, Some((start + len) as isize), 1);
    let slices = [range(0, n), range(0, c), range(top, h), range(left, w)];
    B::float_slice_assign(padded, &slices, x)
}
