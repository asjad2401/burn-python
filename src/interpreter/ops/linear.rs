use burn_backend::backend::ops::FloatTensorOps;
use onnx_ir::{linear::LinearNode, gemm::GemmNode};
use crate::tensor::{B, default_device};
use super::super::context::ExecutionContext;

pub fn linear(node: &LinearNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("linear: missing input");
    let w = ctx.get(&node.inputs[1].name).expect("linear: missing weight");

    // Gemm layout: w is [out, in], needs transpose before matmul
    // MatMul layout: w is [in, out], use as-is
    let w = if node.config.transpose_weight {
        B::float_swap_dims(w, 0, 1)
    } else {
        w
    };

    let y = B::float_matmul(x, w);

    // bias is optional (3rd input)
    let y = match node.inputs.get(2) {
        Some(bias_arg) if !bias_arg.name.is_empty() => {
            if let Some(b) = ctx.get(&bias_arg.name) {
                B::float_add(y, b)
            } else {
                y
            }
        }
        Some(bias_arg) => {
            // inline static (name="")
            if let Some(data) = bias_arg.value() {
                let b = B::float_from_data(data, &default_device());
                B::float_add(y, b)
            } else {
                y
            }
        }
        None => y,
    };

    ctx.insert(node.outputs[0].name.clone(), y);
}

// General matrix multiply: Y = alpha * A' * B' + beta * C
// In practice alpha=1, beta=1 for neural net layers.
pub fn gemm(node: &GemmNode, ctx: &mut ExecutionContext) {
    let a = ctx.get(&node.inputs[0].name).expect("gemm: missing A");
    let b = ctx.get(&node.inputs[1].name).expect("gemm: missing B");

    let a = if node.config.trans_a != 0 { B::float_swap_dims(a, 0, 1) } else { a };
    let b = if node.config.trans_b != 0 { B::float_swap_dims(b, 0, 1) } else { b };

    let mut y = B::float_matmul(a, b);

    if node.config.alpha != 1.0 {
        let alpha = B::float_from_data(
            burn_backend::TensorData::from([node.config.alpha]),
            &default_device(),
        );
        y = B::float_mul(y, alpha);
    }

    if let Some(c_arg) = node.inputs.get(2) {
        let c = if !c_arg.name.is_empty() {
            ctx.get(&c_arg.name)
        } else {
            c_arg.value().map(|d| B::float_from_data(d, &default_device()))
        };
        if let Some(c) = c {
            let c = if node.config.beta != 1.0 {
                let beta = B::float_from_data(
                    burn_backend::TensorData::from([node.config.beta]),
                    &default_device(),
                );
                B::float_mul(c, beta)
            } else {
                c
            };
            y = B::float_add(y, c);
        }
    }

    ctx.insert(node.outputs[0].name.clone(), y);
}
