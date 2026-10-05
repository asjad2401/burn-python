use super::super::context::{ExecutionContext, OpResult};
use crate::tensor::B;
use burn_backend::backend::ops::{ActivationOps, FloatTensorOps};
use onnx_ir::{
    gelu::GeluNode, log_softmax::LogSoftmaxNode, relu::ReluNode, sigmoid::SigmoidNode,
    softmax::SoftmaxNode, tanh::TanhNode,
};

pub fn relu(node: &ReluNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    ctx.insert(node.outputs[0].name.clone(), B::relu(x));
    Ok(())
}

pub fn sigmoid(node: &SigmoidNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    ctx.insert(node.outputs[0].name.clone(), B::sigmoid(x));
    Ok(())
}

pub fn tanh(node: &TanhNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    ctx.insert(node.outputs[0].name.clone(), B::float_tanh(x));
    Ok(())
}

pub fn gelu(node: &GeluNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    ctx.insert(node.outputs[0].name.clone(), B::gelu(x));
    Ok(())
}

pub fn softmax(node: &SoftmaxNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let dim = node.config.axis;
    ctx.insert(node.outputs[0].name.clone(), B::softmax(x, dim));
    Ok(())
}

pub fn log_softmax(node: &LogSoftmaxNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let dim = node.config.axis;
    ctx.insert(node.outputs[0].name.clone(), B::log_softmax(x, dim));
    Ok(())
}
