use super::super::context::{ExecutionContext, OpResult};
use super::padding::{pad2d, resolve_pads, symmetric};
use crate::tensor::{B, default_device};
use burn_backend::{
    TensorData, TensorMetadata,
    backend::ops::{FloatTensorOps, ModuleOps},
};
use onnx_ir::{
    avg_pool2d::AveragePool2dNode, global_avg_pool::GlobalAveragePoolNode,
    max_pool2d::MaxPool2dNode,
};

pub fn max_pool2d(node: &MaxPool2dNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let cfg = &node.config;
    let pads = resolve_pads(
        &x,
        &cfg.padding,
        &cfg.auto_pad,
        cfg.kernel_size,
        cfg.strides,
        cfg.dilation,
    );
    // uneven pads: -inf padding never wins the max, so pre-padding is exact
    let (x, pad) = match symmetric(pads) {
        Some(pad) => (x, pad),
        None => (pad2d(x, pads, f32::NEG_INFINITY), [0, 0]),
    };
    let y = B::max_pool2d(
        x,
        cfg.kernel_size,
        cfg.strides,
        pad,
        cfg.dilation,
        cfg.ceil_mode,
    );
    ctx.insert(node.outputs[0].name.clone(), y);
    Ok(())
}

pub fn avg_pool2d(node: &AveragePool2dNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let cfg = &node.config;
    let pads = resolve_pads(
        &x,
        &cfg.padding,
        &cfg.auto_pad,
        cfg.kernel_size,
        cfg.strides,
        cfg.dilation,
    );
    let pool = |t, pad, include_pad| {
        B::avg_pool2d(
            t,
            cfg.kernel_size,
            cfg.strides,
            pad,
            include_pad,
            cfg.ceil_mode,
        )
    };

    let y = match symmetric(pads) {
        Some(pad) => pool(x, pad, cfg.count_include_pad),
        None if cfg.count_include_pad => pool(pad2d(x, pads, 0.0), [0, 0], true),
        None => {
            // uneven pads that must not count towards the average:
            // sum(window) / count(real pixels in window), via a padded ones-mask
            let shape = x.shape();
            let ones = B::float_from_data(
                TensorData::full(vec![1, 1, shape[2], shape[3]], 1.0f32),
                &default_device(),
            );
            let sum = pool(pad2d(x, pads, 0.0), [0, 0], true);
            let count = pool(pad2d(ones, pads, 0.0), [0, 0], true);
            B::float_div(sum, count)
        }
    };
    ctx.insert(node.outputs[0].name.clone(), y);
    Ok(())
}

pub fn global_avg_pool(node: &GlobalAveragePoolNode, ctx: &mut ExecutionContext) -> OpResult {
    let x = ctx.require(&node.inputs[0])?;
    let y = B::adaptive_avg_pool2d(x, [1, 1]);
    ctx.insert(node.outputs[0].name.clone(), y);
    Ok(())
}
