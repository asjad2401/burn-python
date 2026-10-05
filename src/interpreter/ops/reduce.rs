use super::super::context::{ExecutionContext, OpResult};
use crate::tensor::{B, FloatPrim};
use burn_backend::{TensorMetadata, backend::ops::FloatTensorOps};
use onnx_ir::{
    ir::Argument,
    reduce::{ReduceConfig, ReduceMaxNode, ReduceMeanNode, ReduceMinNode, ReduceSumNode},
};

// onnx-ir hands us sorted, non-negative dims; empty means "reduce everything".
// Burn's *_dim ops keep the reduced dim as size 1, matching keepdims=1.
fn reduce(
    inputs: &[Argument],
    outputs: &[Argument],
    cfg: &ReduceConfig,
    ctx: &mut ExecutionContext,
    op: fn(FloatPrim, usize) -> FloatPrim,
) -> OpResult {
    let x = ctx.require(&inputs[0])?;
    let rank = x.shape().num_dims();
    let dims: Vec<usize> = if cfg.dims.is_empty() {
        (0..rank).collect()
    } else {
        cfg.dims.clone()
    };

    let mut y = dims.iter().fold(x, |t, &d| op(t, d));
    if !cfg.keepdims {
        let shape: Vec<usize> = y
            .shape()
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &s)| s)
            .collect();
        y = B::float_reshape(y, shape.into());
    }

    ctx.insert(outputs[0].name.clone(), y);
    Ok(())
}

pub fn mean(node: &ReduceMeanNode, ctx: &mut ExecutionContext) -> OpResult {
    reduce(
        &node.inputs,
        &node.outputs,
        &node.config,
        ctx,
        B::float_mean_dim,
    )
}

pub fn sum(node: &ReduceSumNode, ctx: &mut ExecutionContext) -> OpResult {
    reduce(
        &node.inputs,
        &node.outputs,
        &node.config,
        ctx,
        B::float_sum_dim,
    )
}

pub fn max(node: &ReduceMaxNode, ctx: &mut ExecutionContext) -> OpResult {
    reduce(
        &node.inputs,
        &node.outputs,
        &node.config,
        ctx,
        B::float_max_dim,
    )
}

pub fn min(node: &ReduceMinNode, ctx: &mut ExecutionContext) -> OpResult {
    reduce(
        &node.inputs,
        &node.outputs,
        &node.config,
        ctx,
        B::float_min_dim,
    )
}
