use super::super::context::{ExecutionContext, OpResult};
use super::broadcast::align;
use crate::tensor::B;
use burn_backend::backend::ops::FloatTensorOps;
use onnx_ir::{gemm::GemmNode, linear::LinearNode};

pub fn linear(node: &LinearNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let w = ctx.require(&node.inputs[1])?;

    // Gemm layout: w is [out, in], needs transpose before matmul
    // MatMul layout: w is [in, out], use as-is
    let w = if node.config.transpose_weight {
        B::float_swap_dims(w, 0, 1)
    } else {
        w
    };

    let y = B::float_matmul(x, w);

    // bias is optional (3rd input)
    let y = match node.inputs.get(2).and_then(|a| ctx.resolve(a)) {
        Some(b) => {
            let (y, b) = align(y, b);
            B::float_add(y, b)
        }
        None => y,
    };

    ctx.insert(node.outputs[0].name.clone(), y);
    Ok(())
}

// General matrix multiply: Y = alpha * A' * B' + beta * C
// In practice alpha=1, beta=1 for neural net layers.
pub fn gemm(node: &GemmNode, ctx: &mut ExecutionContext) -> OpResult {
    let a = ctx.require(&node.inputs[0])?;
    let b = ctx.require(&node.inputs[1])?;

    let a = if node.config.trans_a != 0 {
        B::float_swap_dims(a, 0, 1)
    } else {
        a
    };
    let b = if node.config.trans_b != 0 {
        B::float_swap_dims(b, 0, 1)
    } else {
        b
    };

    let mut y = B::float_matmul(a, b);

    if node.config.alpha != 1.0 {
        y = B::float_mul_scalar(y, node.config.alpha.into());
    }

    if let Some(c) = node.inputs.get(2).and_then(|a| ctx.resolve(a)) {
        let c = if node.config.beta != 1.0 {
            B::float_mul_scalar(c, node.config.beta.into())
        } else {
            c
        };
        let (y_aligned, c) = align(y, c);
        y = B::float_add(y_aligned, c);
    }

    ctx.insert(node.outputs[0].name.clone(), y);
    Ok(())
}
