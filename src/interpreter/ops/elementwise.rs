use burn_backend::backend::ops::FloatTensorOps;
use onnx_ir::arithmetic::{AddNode, SubNode, MulNode, DivNode};
use crate::tensor::B;
use super::super::context::ExecutionContext;

pub fn add(node: &AddNode, ctx: &mut ExecutionContext) {
    let a = ctx.get(&node.inputs[0].name).expect("add: missing lhs");
    let b = ctx.get(&node.inputs[1].name).expect("add: missing rhs");
    ctx.insert(node.outputs[0].name.clone(), B::float_add(a, b));
}

pub fn sub(node: &SubNode, ctx: &mut ExecutionContext) {
    let a = ctx.get(&node.inputs[0].name).expect("sub: missing lhs");
    let b = ctx.get(&node.inputs[1].name).expect("sub: missing rhs");
    ctx.insert(node.outputs[0].name.clone(), B::float_sub(a, b));
}

pub fn mul(node: &MulNode, ctx: &mut ExecutionContext) {
    let a = ctx.get(&node.inputs[0].name).expect("mul: missing lhs");
    let b = ctx.get(&node.inputs[1].name).expect("mul: missing rhs");
    ctx.insert(node.outputs[0].name.clone(), B::float_mul(a, b));
}

pub fn div(node: &DivNode, ctx: &mut ExecutionContext) {
    let a = ctx.get(&node.inputs[0].name).expect("div: missing lhs");
    let b = ctx.get(&node.inputs[1].name).expect("div: missing rhs");
    ctx.insert(node.outputs[0].name.clone(), B::float_div(a, b));
}
