// ONNX runtime interpreter:
// parse -> pre-load weights (with BN fusion) -> dispatch nodes on each forward pass

mod context;
pub mod ops;

use std::collections::HashMap;

use burn_backend::{backend::ops::FloatTensorOps, DType, TensorData, TensorMetadata};
use onnx_ir::{
    ir::{Argument, ValueSource},
    batch_norm::{BatchNormConfig, BatchNormalizationNode},
    Node,
    OnnxGraphBuilder,
};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

use crate::tensor::{default_device, flex_to_numpy, numpy_to_flex, FloatPrim, B};
use context::ExecutionContext;

fn load_weight(arg: &Argument) -> Option<(String, FloatPrim)> {
    if arg.name.is_empty() {
        return None;
    }
    match arg.value_source {
        ValueSource::Static(_) | ValueSource::Constant => {
            let data = arg.value()?;
            if !matches!(data.dtype, DType::F32 | DType::F16 | DType::BF16) {
                return None;
            }
            Some((arg.name.clone(), B::float_from_data(data, &default_device())))
        }
        _ => None,
    }
}

/// Pre-compute BN scale/bias as [1,C,1,1] tensors so runtime BN is just 2 elementwise ops.
/// key: "{node_name}::scale" and "{node_name}::bias"
/// inputs: [x, gamma, beta, running_mean, running_var]
fn precompute_bn(node: &BatchNormalizationNode, weights: &HashMap<String, FloatPrim>)
    -> Option<(String, FloatPrim, FloatPrim)>
{
    let eps = match &node.config {
        BatchNormConfig::Static(c)  => c.epsilon as f32,
        BatchNormConfig::Runtime(c) => c.epsilon as f32,
    };

    // all params must be pre-loaded (static/constant)
    let get = |arg: &Argument| -> Option<FloatPrim> {
        if !arg.name.is_empty() {
            weights.get(&arg.name).cloned()
        } else {
            arg.value().map(|d| B::float_from_data(d, &default_device()))
        }
    };

    let gamma = get(&node.inputs[1])?;
    let beta  = get(&node.inputs[2])?;
    let mean  = get(&node.inputs[3])?;
    let var   = get(&node.inputs[4])?;

    let c = gamma.shape().iter().next().copied().unwrap_or(1);

    // scale = gamma / sqrt(var + eps),  shaped [1, C, 1, 1]
    let eps_t = B::float_from_data(TensorData::from([eps]), &default_device());
    let scale_flat = B::float_div(
        gamma,
        B::float_sqrt(B::float_add(var, eps_t)),
    );
    // offset = beta - mean * scale,  shaped [1, C, 1, 1]
    let offset_flat = B::float_sub(beta, B::float_mul(mean, scale_flat.clone()));

    // pre-expand to [1, C, 1, 1] so no reshape at inference time
    let shape_4d: Vec<usize> = vec![1, c, 1, 1];
    let scale  = B::float_reshape(scale_flat,  shape_4d.clone().into());
    let offset = B::float_reshape(offset_flat, shape_4d.into());

    Some((node.name.clone(), scale, offset))
}

#[pyclass]
pub struct OnnxModel {
    nodes: Vec<Node>,
    weights: HashMap<String, FloatPrim>,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

#[pymethods]
impl OnnxModel {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<PyReadonlyArrayDyn<'py, f32>>,
    ) -> PyResult<Vec<Bound<'py, PyArrayDyn<f32>>>> {
        if inputs.len() != self.input_names.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} input(s), got {}",
                self.input_names.len(),
                inputs.len()
            )));
        }

        let mut ctx = ExecutionContext::new(&self.weights);
        for (name, arr) in self.input_names.iter().zip(inputs) {
            ctx.insert(name.clone(), numpy_to_flex(&arr));
        }

        for node in &self.nodes {
            dispatch(node, &mut ctx);
        }

        self.output_names.iter()
            .map(|name| {
                let t = ctx.get(name)
                    .unwrap_or_else(|| panic!("output '{}' not found", name));
                Ok(flex_to_numpy(py, t))
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "OnnxModel(inputs={:?}, outputs={:?}, nodes={})",
            self.input_names, self.output_names, self.nodes.len()
        )
    }
}

pub fn load_onnx(path: &str) -> Result<OnnxModel, String> {
    let graph = OnnxGraphBuilder::new()
        .parse_file(path)
        .map_err(|e| e.to_string())?;

    let mut weights = HashMap::new();

    // pass 1: load all named static weights
    for node in &graph.nodes {
        for arg in node.inputs() {
            if let Some((name, tensor)) = load_weight(arg) {
                weights.insert(name, tensor);
            }
        }
    }

    // pass 2: pre-compute BN scale/bias so dispatch is just 2 ops
    for node in &graph.nodes {
        if let Node::BatchNormalization(bn) = node {
            if let Some((name, scale, offset)) = precompute_bn(bn, &weights) {
                weights.insert(format!("{name}::scale"),  scale);
                weights.insert(format!("{name}::offset"), offset);
            }
        }
    }

    Ok(OnnxModel {
        input_names: graph.inputs.iter().map(|a| a.name.clone()).collect(),
        output_names: graph.outputs.iter().map(|a| a.name.clone()).collect(),
        nodes: graph.nodes,
        weights,
    })
}

fn dispatch(node: &Node, ctx: &mut ExecutionContext) {
    match node {
        Node::Relu(n)           => ops::activation::relu(n, ctx),
        Node::Sigmoid(n)        => ops::activation::sigmoid(n, ctx),
        Node::Tanh(n)           => ops::activation::tanh(n, ctx),
        Node::Gelu(n)           => ops::activation::gelu(n, ctx),
        Node::Softmax(n)        => ops::activation::softmax(n, ctx),
        Node::LogSoftmax(n)     => ops::activation::log_softmax(n, ctx),
        Node::Linear(n)         => ops::linear::linear(n, ctx),
        Node::Gemm(n)           => ops::linear::gemm(n, ctx),
        Node::Add(n)            => ops::elementwise::add(n, ctx),
        Node::Sub(n)            => ops::elementwise::sub(n, ctx),
        Node::Mul(n)            => ops::elementwise::mul(n, ctx),
        Node::Div(n)            => ops::elementwise::div(n, ctx),
        Node::Reshape(n)        => ops::reshape::reshape(n, ctx),
        Node::Flatten(n)        => ops::reshape::flatten(n, ctx),
        Node::Transpose(n)      => ops::reshape::transpose(n, ctx),
        Node::Conv2d(n)         => ops::conv::conv2d(n, ctx),
        Node::BatchNormalization(n) => ops::conv::batch_norm_fused(n, ctx),
        Node::MaxPool2d(n)      => ops::pool::max_pool2d(n, ctx),
        Node::AveragePool2d(n)  => ops::pool::avg_pool2d(n, ctx),
        Node::GlobalAveragePool(n) => ops::pool::global_avg_pool(n, ctx),
        Node::Constant(_)       => {}
        other => {
            eprintln!("warn: unimplemented op '{}' — skipping", other.name());
        }
    }
}
