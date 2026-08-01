use burn_backend::backend::ops::ModuleOps;
use onnx_ir::{
    max_pool2d::MaxPool2dNode,
    avg_pool2d::AveragePool2dNode,
    global_avg_pool::GlobalAveragePoolNode,
    node::padding::PaddingConfig2d,
};
use crate::tensor::B;
use super::super::context::ExecutionContext;

fn symmetric_padding(cfg: &PaddingConfig2d) -> [usize; 2] {
    match cfg {
        PaddingConfig2d::Valid => [0, 0],
        PaddingConfig2d::Explicit(top, left, bottom, right) => {
            [(top + bottom) / 2, (left + right) / 2]
        }
    }
}

pub fn max_pool2d(node: &MaxPool2dNode, ctx: &mut ExecutionContext) {
    let x = ctx.resolve(&node.inputs[0]).expect("max_pool2d: missing input");
    let pad = symmetric_padding(&node.config.padding);
    let y = B::max_pool2d(
        x,
        node.config.kernel_size,
        node.config.strides,
        pad,
        node.config.dilation,
        false,
    );
    ctx.insert(node.outputs[0].name.clone(), y);
}

pub fn avg_pool2d(node: &AveragePool2dNode, ctx: &mut ExecutionContext) {
    let x = ctx.resolve(&node.inputs[0]).expect("avg_pool2d: missing input");
    let pad = symmetric_padding(&node.config.padding);
    let y = B::avg_pool2d(
        x,
        node.config.kernel_size,
        node.config.strides,
        pad,
        node.config.count_include_pad,
        false,
    );
    ctx.insert(node.outputs[0].name.clone(), y);
}

pub fn global_avg_pool(node: &GlobalAveragePoolNode, ctx: &mut ExecutionContext) {
    let x = ctx.resolve(&node.inputs[0]).expect("global_avg_pool: missing input");
    let y = B::adaptive_avg_pool2d(x, [1, 1]);
    ctx.insert(node.outputs[0].name.clone(), y);
}
