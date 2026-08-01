use burn_backend::{
    backend::ops::{FloatTensorOps, ModuleOps},
    backend::ops::ConvOptions,
    TensorData, TensorMetadata,
};
use onnx_ir::{
    conv2d::Conv2dNode,
    batch_norm::{BatchNormalizationNode, BatchNormConfig},
    node::padding::PaddingConfig2d,
};
use crate::tensor::{B, default_device};
use super::super::context::ExecutionContext;

fn symmetric_padding(cfg: &PaddingConfig2d) -> [usize; 2] {
    match cfg {
        PaddingConfig2d::Valid => [0, 0],
        PaddingConfig2d::Explicit(top, left, bottom, right) => {
            [(top + bottom) / 2, (left + right) / 2]
        }
    }
}

pub fn conv2d(node: &Conv2dNode, ctx: &mut ExecutionContext) {
    let x    = ctx.resolve(&node.inputs[0]).expect("conv2d: missing input");
    let w    = ctx.resolve(&node.inputs[1]).expect("conv2d: missing weight");
    let bias = node.inputs.get(2).and_then(|a| ctx.resolve(a));

    let pad = symmetric_padding(&node.config.padding);
    let options = ConvOptions::new(
        node.config.stride,
        pad,
        node.config.dilation,
        node.config.groups,
    );

    ctx.insert(node.outputs[0].name.clone(), B::conv2d(x, w, bias, options));
}

// BatchNorm dispatch — uses pre-fused scale/offset computed at load time.
// Falls back to full computation if pre-fusion wasn't possible.
pub fn batch_norm_fused(node: &BatchNormalizationNode, ctx: &mut ExecutionContext) {
    let x = ctx.resolve(&node.inputs[0]).expect("batch_norm: missing input");

    let scale_key  = format!("{}::scale",  node.name);
    let offset_key = format!("{}::offset", node.name);

    if let (Some(scale), Some(offset)) = (ctx.get(&scale_key), ctx.get(&offset_key)) {
        // fast path: 2 ops, no reshape (scale/offset are already [1,C,1,1])
        let y = B::float_mul(x, scale);
        let y = B::float_add(y, offset);
        ctx.insert(node.outputs[0].name.clone(), y);
    } else {
        // fallback: full manual BN (slow, but correct)
        let gamma = ctx.resolve(&node.inputs[1]).expect("batch_norm: missing gamma");
        let beta  = ctx.resolve(&node.inputs[2]).expect("batch_norm: missing beta");
        let mean  = ctx.resolve(&node.inputs[3]).expect("batch_norm: missing mean");
        let var   = ctx.resolve(&node.inputs[4]).expect("batch_norm: missing var");

        let eps = match &node.config {
            BatchNormConfig::Static(c)  => c.epsilon as f32,
            BatchNormConfig::Runtime(c) => c.epsilon as f32,
        };
        let rank = x.shape().num_dims();
        let c = gamma.shape().iter().next().copied().unwrap_or(1);
        let bshape: Vec<usize> = std::iter::once(1)
            .chain(std::iter::once(c))
            .chain(std::iter::repeat(1).take(rank - 2))
            .collect();
        let bcast = |t| B::float_reshape(t, bshape.clone().into());
        let eps_t = B::float_from_data(TensorData::from([eps]), &default_device());
        let denom = B::float_sqrt(B::float_add(bcast(var), eps_t));
        let y = B::float_div(B::float_sub(x, bcast(mean)), denom);
        let y = B::float_mul(y, bcast(gamma));
        let y = B::float_add(y, bcast(beta));
        ctx.insert(node.outputs[0].name.clone(), y);
    }
}
