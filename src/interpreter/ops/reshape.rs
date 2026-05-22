use burn_backend::{backend::ops::FloatTensorOps, TensorMetadata};
use onnx_ir::{flatten::FlattenNode, reshape::ReshapeNode, transpose::TransposeNode};
use crate::tensor::B;
use super::super::context::ExecutionContext;

pub fn reshape(node: &ReshapeNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("reshape: missing input");
    let orig_shape: Vec<usize> = x.shape().iter().copied().collect();
    let total: usize = orig_shape.iter().product();

    let shape_data = node.inputs[1].value().expect("reshape: shape must be static");
    let raw: Vec<i64> = shape_data.to_vec().expect("reshape: shape must be i64");

    let shape: Vec<usize> = raw.iter().enumerate().map(|(i, &s)| {
        if s == -1 {
            let known: usize = raw.iter().enumerate()
                .filter(|(j, v)| *j != i && **v != -1)
                .map(|(_, &v)| v as usize)
                .product();
            total / known
        } else if s == 0 {
            orig_shape[i]
        } else {
            s as usize
        }
    }).collect();

    ctx.insert(node.outputs[0].name.clone(), B::float_reshape(x, shape.into()));
}

pub fn flatten(node: &FlattenNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("flatten: missing input");
    let shape: Vec<usize> = x.shape().iter().copied().collect();
    let axis = node.config.axis as usize;
    let outer: usize = shape[..axis].iter().product::<usize>().max(1);
    let inner: usize = shape[axis..].iter().product();
    ctx.insert(node.outputs[0].name.clone(), B::float_reshape(x, vec![outer, inner].into()));
}

pub fn transpose(node: &TransposeNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("transpose: missing input");
    let rank = x.shape().num_dims();

    let perm: Vec<usize> = if node.config.perm.is_empty() {
        (0..rank).rev().collect()
    } else {
        node.config.perm.iter().map(|&i| i as usize).collect()
    };

    ctx.insert(node.outputs[0].name.clone(), B::float_permute(x, &perm));
}
