use super::super::context::{ExecutionContext, OpResult};
use crate::tensor::B;
use burn_backend::backend::ops::FloatTensorOps;
use onnx_ir::arithmetic::{AddNode, DivNode, MulNode, SubNode};

pub fn add(node: &AddNode, ctx: &mut ExecutionContext) -> OpResult {
    let a = ctx.require(&node.inputs[0])?;
    let b = ctx.require(&node.inputs[1])?;
    ctx.insert(node.outputs[0].name.clone(), B::float_add(a, b));
    Ok(())
}

pub fn sub(node: &SubNode, ctx: &mut ExecutionContext) -> OpResult {
    let a = ctx.require(&node.inputs[0])?;
    let b = ctx.require(&node.inputs[1])?;
    ctx.insert(node.outputs[0].name.clone(), B::float_sub(a, b));
    Ok(())
}

pub fn mul(node: &MulNode, ctx: &mut ExecutionContext) -> OpResult {
    let a = ctx.require(&node.inputs[0])?;
    let b = ctx.require(&node.inputs[1])?;
    ctx.insert(node.outputs[0].name.clone(), B::float_mul(a, b));
    Ok(())
}

pub fn div(node: &DivNode, ctx: &mut ExecutionContext) -> OpResult {
    let a = ctx.require(&node.inputs[0])?;
    let b = ctx.require(&node.inputs[1])?;
    ctx.insert(node.outputs[0].name.clone(), B::float_div(a, b));
    Ok(())
}
