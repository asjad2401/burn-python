use super::super::context::{ExecutionContext, OpResult};
use crate::tensor::B;
use burn_backend::{TensorMetadata, backend::ops::FloatTensorOps};
use onnx_ir::{
    flatten::FlattenNode,
    reshape::{ReshapeInput, ReshapeNode},
    transpose::TransposeNode,
};

pub fn reshape(node: &ReshapeNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let orig_shape: Vec<usize> = x.shape().iter().copied().collect();

    // shape comes from the attribute (opset < 5) or a constant input (opset 5+)
    let raw: Vec<i64> = match &node.config.shape {
        ReshapeInput::Static(shape) => shape.clone(),
        ReshapeInput::Runtime(r) => node.inputs[r.input_index]
            .value()
            .ok_or("shape must be a constant (dynamic shapes are not supported yet)")?
            .to_vec()
            .map_err(|e| format!("shape must be int64: {e:?}"))?,
    };

    // 0 copies the input dim (allowzero=0); -1 is inferred from what's left
    let mut shape: Vec<usize> = raw
        .iter()
        .enumerate()
        .map(|(i, &s)| match s {
            0 => orig_shape[i],
            -1 => 1,
            s => s as usize,
        })
        .collect();
    if let Some(i) = raw.iter().position(|&s| s == -1) {
        let total: usize = orig_shape.iter().product();
        let known: usize = shape.iter().product();
        shape[i] = total / known;
    }

    ctx.insert(
        node.outputs[0].name.clone(),
        B::float_reshape(x, shape.into()),
    );
    Ok(())
}

pub fn flatten(node: &FlattenNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let shape: Vec<usize> = x.shape().iter().copied().collect();
    let axis = node.config.axis;
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = shape[axis..].iter().product();
    ctx.insert(
        node.outputs[0].name.clone(),
        B::float_reshape(x, vec![outer, inner].into()),
    );
    Ok(())
}

pub fn transpose(node: &TransposeNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let rank = x.shape().num_dims();

    let perm: Vec<usize> = if node.config.perm.is_empty() {
        (0..rank).rev().collect()
    } else {
        node.config.perm.iter().map(|&i| i as usize).collect()
    };

    ctx.insert(node.outputs[0].name.clone(), B::float_permute(x, &perm));
    Ok(())
}
