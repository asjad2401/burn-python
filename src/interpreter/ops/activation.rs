use burn_backend::backend::ops::{ActivationOps, FloatTensorOps};
use onnx_ir::{
    gelu::GeluNode, log_softmax::LogSoftmaxNode, relu::ReluNode, sigmoid::SigmoidNode,
    softmax::SoftmaxNode, tanh::TanhNode,
};
use crate::tensor::B;
use super::super::context::ExecutionContext;

pub fn relu(node: &ReluNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("relu: missing input");
    ctx.insert(node.outputs[0].name.clone(), B::relu(x));
}

pub fn sigmoid(node: &SigmoidNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("sigmoid: missing input");
    ctx.insert(node.outputs[0].name.clone(), B::sigmoid(x));
}

pub fn tanh(node: &TanhNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("tanh: missing input");
    ctx.insert(node.outputs[0].name.clone(), B::float_tanh(x));
}

pub fn gelu(node: &GeluNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("gelu: missing input");
    ctx.insert(node.outputs[0].name.clone(), B::gelu(x));
}

pub fn softmax(node: &SoftmaxNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("softmax: missing input");
    let dim = node.config.axis as usize;
    ctx.insert(node.outputs[0].name.clone(), B::softmax(x, dim));
}

pub fn log_softmax(node: &LogSoftmaxNode, ctx: &mut ExecutionContext) {
    let x = ctx.get(&node.inputs[0].name).expect("log_softmax: missing input");
    let dim = node.config.axis as usize;
    ctx.insert(node.outputs[0].name.clone(), B::log_softmax(x, dim));
}
