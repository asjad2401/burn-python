// ONNX runtime interpreter:
// parse -> pre-load weights (with BN fusion) -> dispatch nodes on each forward pass

mod context;
pub mod ops;

use std::collections::HashMap;

use burn_backend::{DType, TensorData, TensorMetadata, backend::ops::FloatTensorOps};
use numpy::{
    PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArray, PyUntypedArrayMethods,
};
use onnx_ir::{
    Node, OnnxGraphBuilder,
    batch_norm::{BatchNormConfig, BatchNormalizationNode},
    ir::{ArgType, Argument, ValueSource},
};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};

use crate::tensor::{B, Device, FloatPrim, device_for, numpy_to_tensor, tensor_to_numpy};
use context::{ExecutionContext, OpResult, weight_key};

fn load_weight(arg: &Argument, device: &Device) -> Option<(String, FloatPrim)> {
    match arg.value_source {
        ValueSource::Static(_) | ValueSource::Constant => {
            let key = weight_key(arg)?;
            let data = arg.value()?;
            if !matches!(data.dtype, DType::F32 | DType::F16 | DType::BF16) {
                return None;
            }
            Some((key, B::float_from_data(data, device)))
        }
        _ => None,
    }
}

/// Pre-compute BN scale/bias as [1,C,1,1] tensors so runtime BN is just 2 elementwise ops.
/// key: "{node_name}::scale" and "{node_name}::bias"
/// inputs: [x, gamma, beta, running_mean, running_var]
fn precompute_bn(
    node: &BatchNormalizationNode,
    weights: &HashMap<String, FloatPrim>,
) -> Option<(String, FloatPrim, FloatPrim)> {
    let eps = match &node.config {
        BatchNormConfig::Static(c) => c.epsilon as f32,
        BatchNormConfig::Runtime(c) => c.epsilon as f32,
    };

    // all params must be pre-loaded (static/constant)
    let get = |arg: &Argument| weights.get(&weight_key(arg)?).cloned();

    let gamma = get(&node.inputs[1])?;
    let beta = get(&node.inputs[2])?;
    let mean = get(&node.inputs[3])?;
    let var = get(&node.inputs[4])?;

    let c = gamma.shape().iter().next().copied().unwrap_or(1);

    // scale = gamma / sqrt(var + eps),  shaped [1, C, 1, 1]
    let scale_flat = B::float_div(gamma, B::float_sqrt(B::float_add_scalar(var, eps.into())));
    // offset = beta - mean * scale,  shaped [1, C, 1, 1]
    let offset_flat = B::float_sub(beta, B::float_mul(mean, scale_flat.clone()));

    // pre-expand to [1, C, 1, 1] so no reshape at inference time
    let shape_4d: Vec<usize> = vec![1, c, 1, 1];
    let scale = B::float_reshape(scale_flat, shape_4d.clone().into());
    let offset = B::float_reshape(offset_flat, shape_4d.into());

    Some((node.name.clone(), scale, offset))
}

/// Fold each BatchNorm that directly follows a Conv2d into the conv's weight and bias:
///   W' = W * scale[c_out],  b' = b * scale + offset
/// then drop the BN node and have the conv write the BN's output directly.
/// Only done when the conv output feeds nothing but that BN, and the BN's scale/offset
/// were pre-computed (`precompute_bn`). Unfolded BNs keep the runtime mul + add path.
fn fold_conv_bn(
    mut nodes: Vec<Node>,
    weights: &mut HashMap<String, FloatPrim>,
    graph_outputs: &[String],
) -> Vec<Node> {
    let mut consumers: HashMap<String, usize> = HashMap::new();
    for name in nodes
        .iter()
        .flat_map(|n| n.inputs().iter().map(|a| &a.name))
        .chain(graph_outputs)
    {
        *consumers.entry(name.clone()).or_default() += 1;
    }
    let conv_by_output: HashMap<String, usize> = nodes
        .iter()
        .enumerate()
        .filter(|(_, n)| matches!(n, Node::Conv2d(_)))
        .map(|(i, n)| (n.outputs()[0].name.clone(), i))
        .collect();

    let mut folded = Vec::new(); // (conv index, bn index)
    for (j, node) in nodes.iter().enumerate() {
        let Node::BatchNormalization(bn) = node else {
            continue;
        };
        let x = &bn.inputs[0].name;
        let (Some(&i), Some(1)) = (conv_by_output.get(x), consumers.get(x)) else {
            continue;
        };
        let Node::Conv2d(conv) = &nodes[i] else {
            continue;
        };
        let (Some(scale), Some(offset), Some(w)) = (
            weights.get(&format!("{}::scale", bn.name)).cloned(),
            weights.get(&format!("{}::offset", bn.name)).cloned(),
            weight_key(&conv.inputs[1]).and_then(|k| weights.get(&k).cloned()),
        ) else {
            continue;
        };
        let c_out = w.shape()[0];
        let bias = match conv.inputs.get(2) {
            Some(arg) => match weight_key(arg).and_then(|k| weights.get(&k).cloned()) {
                Some(b) => b,
                None => continue, // dynamic bias: leave unfolded
            },
            None => B::float_from_data(TensorData::zeros::<f32, _>([c_out]), &B::float_device(&w)),
        };

        let w = B::float_mul(
            w,
            B::float_reshape(scale.clone(), vec![c_out, 1, 1, 1].into()),
        );
        let b = B::float_add(
            B::float_mul(bias, B::float_reshape(scale, vec![c_out].into())),
            B::float_reshape(offset, vec![c_out].into()),
        );
        weights.insert(format!("{}::folded_weight", conv.name), w);
        weights.insert(format!("{}::folded_bias", conv.name), b);
        folded.push((i, j));
    }

    for &(i, j) in &folded {
        let bn_out = nodes[j].outputs()[0].name.clone();
        let Node::Conv2d(conv) = &mut nodes[i] else {
            unreachable!()
        };
        // point the conv at the folded params (static args are looked up by name first)
        let mut w_arg = conv.inputs[1].clone();
        w_arg.name = format!("{}::folded_weight", conv.name);
        let mut b_arg = w_arg.clone();
        b_arg.name = format!("{}::folded_bias", conv.name);
        conv.inputs.truncate(1);
        conv.inputs.extend([w_arg, b_arg]);
        conv.outputs[0].name = bn_out;
    }
    let dropped: std::collections::HashSet<usize> = folded.iter().map(|&(_, j)| j).collect();
    nodes
        .into_iter()
        .enumerate()
        .filter(|(j, _)| !dropped.contains(j))
        .map(|(_, n)| n)
        .collect()
}

/// A graph input, with whatever shape info the model declares (None = symbolic dim).
struct InputSpec {
    name: String,
    rank: Option<usize>,
    shape: Option<Vec<Option<usize>>>,
}

impl InputSpec {
    fn from_arg(arg: &Argument) -> Self {
        let (rank, shape) = match &arg.ty {
            ArgType::Tensor(t) => (Some(t.rank), t.static_shape.clone()),
            _ => (None, None),
        };
        Self {
            name: arg.name.clone(),
            rank,
            shape,
        }
    }

    fn check_shape(&self, actual: &[usize]) -> PyResult<()> {
        let mismatch = match &self.shape {
            Some(dims) => {
                dims.len() != actual.len()
                    || dims
                        .iter()
                        .zip(actual)
                        .any(|(d, a)| d.is_some_and(|d| d != *a))
            }
            None => self.rank.is_some_and(|r| r != actual.len()),
        };
        if !mismatch {
            return Ok(());
        }
        let expected = match &self.shape {
            Some(dims) => {
                let dims: Vec<String> = dims
                    .iter()
                    .map(|d| d.map_or("?".to_string(), |d| d.to_string()))
                    .collect();
                format!("shape [{}]", dims.join(", "))
            }
            None => format!("{} dims", self.rank.unwrap_or_default()),
        };
        Err(PyValueError::new_err(format!(
            "input '{}': expected {expected}, got shape {actual:?}",
            self.name
        )))
    }
}

/// Accept only float32 ndarrays, with a hint on how to fix anything else.
fn as_f32_array<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<PyReadonlyArrayDyn<'py, f32>> {
    if let Ok(arr) = obj.cast::<PyArrayDyn<f32>>() {
        return Ok(arr.readonly());
    }
    let msg = match obj.cast::<PyUntypedArray>() {
        Ok(arr) => format!(
            "input '{name}': expected a float32 array, got {} (use .astype(np.float32))",
            arr.dtype()
        ),
        Err(_) => format!(
            "input '{name}': expected a numpy array, got {}",
            obj.get_type().name()?
        ),
    };
    Err(PyTypeError::new_err(msg))
}

/// Run one node, turning both op errors and backend panics into a message.
fn run_node(node: &Node, ctx: &mut ExecutionContext) -> OpResult {
    crate::panic::catch(|| dispatch(node, ctx))?
}

#[pyclass]
pub struct OnnxModel {
    nodes: Vec<Node>,
    weights: HashMap<String, FloatPrim>,
    inputs: Vec<InputSpec>,
    output_names: Vec<String>,
    backend: String,
    device: Device,
}

#[pymethods]
impl OnnxModel {
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        inputs: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Vec<Bound<'py, PyArrayDyn<f32>>>> {
        if inputs.len() != self.inputs.len() {
            return Err(PyValueError::new_err(format!(
                "expected {} input(s), got {}",
                self.inputs.len(),
                inputs.len()
            )));
        }

        let mut ctx = ExecutionContext::new(&self.weights, &self.device);
        for (spec, obj) in self.inputs.iter().zip(&inputs) {
            let arr = as_f32_array(obj, &spec.name)?;
            spec.check_shape(arr.shape())?;
            ctx.insert(spec.name.clone(), numpy_to_tensor(&arr, &self.device));
        }

        for node in &self.nodes {
            run_node(node, &mut ctx).map_err(|msg| {
                PyRuntimeError::new_err(format!(
                    "node '{}' ({}): {msg}",
                    node.name(),
                    op_type(node)
                ))
            })?;
        }

        self.output_names
            .iter()
            .map(|name| {
                let t = ctx.get(name).ok_or_else(|| {
                    PyRuntimeError::new_err(format!("model output '{name}' was never computed"))
                })?;
                tensor_to_numpy(py, t)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        let input_names: Vec<&str> = self.inputs.iter().map(|i| i.name.as_str()).collect();
        format!(
            "OnnxModel(inputs={:?}, outputs={:?}, nodes={}, backend={:?})",
            input_names,
            self.output_names,
            self.nodes.len(),
            self.backend
        )
    }
}

pub fn load_onnx(path: &str, backend: &str) -> Result<OnnxModel, String> {
    // backend init (e.g. no usable GPU) fails by panicking, often on a worker thread
    crate::panic::catch(|| load_onnx_on(path, backend)).unwrap_or_else(|msg| {
        Err(format!(
            "failed to load model on backend '{backend}': {msg}"
        ))
    })
}

fn load_onnx_on(path: &str, backend: &str) -> Result<OnnxModel, String> {
    let device = device_for(backend)?;
    let graph = OnnxGraphBuilder::new()
        .parse_file(path)
        .map_err(|e| e.to_string())?;

    // fail fast: a skipped op would only surface later as a missing tensor or wrong output
    let mut unsupported: Vec<String> = graph
        .nodes
        .iter()
        .filter(|n| !is_supported(n))
        .map(op_type)
        .collect();
    if !unsupported.is_empty() {
        unsupported.sort();
        unsupported.dedup();
        return Err(format!(
            "unsupported ONNX op(s): {}",
            unsupported.join(", ")
        ));
    }

    let mut weights = HashMap::new();

    // pass 1: load all named static weights
    for node in &graph.nodes {
        for arg in node.inputs() {
            if let Some((name, tensor)) = load_weight(arg, &device) {
                weights.insert(name, tensor);
            }
        }
    }

    // pass 2: pre-compute BN scale/bias so dispatch is just 2 ops
    for node in &graph.nodes {
        if let Node::BatchNormalization(bn) = node
            && let Some((name, scale, offset)) = precompute_bn(bn, &weights)
        {
            weights.insert(format!("{name}::scale"), scale);
            weights.insert(format!("{name}::offset"), offset);
        }
    }

    // pass 3: fold Conv -> BN pairs so those BNs cost nothing at inference time
    let output_names: Vec<String> = graph.outputs.iter().map(|a| a.name.clone()).collect();
    let nodes = fold_conv_bn(graph.nodes, &mut weights, &output_names);

    Ok(OnnxModel {
        inputs: graph.inputs.iter().map(InputSpec::from_arg).collect(),
        output_names,
        nodes,
        weights,
        backend: backend.to_string(),
        device,
    })
}

// keep in sync with `dispatch`
fn is_supported(node: &Node) -> bool {
    matches!(
        node,
        Node::Relu(_)
            | Node::Sigmoid(_)
            | Node::Tanh(_)
            | Node::Gelu(_)
            | Node::Softmax(_)
            | Node::LogSoftmax(_)
            | Node::Linear(_)
            | Node::Gemm(_)
            | Node::Add(_)
            | Node::Sub(_)
            | Node::Mul(_)
            | Node::Div(_)
            | Node::Reshape(_)
            | Node::Flatten(_)
            | Node::Transpose(_)
            | Node::Conv2d(_)
            | Node::BatchNormalization(_)
            | Node::MaxPool2d(_)
            | Node::AveragePool2d(_)
            | Node::GlobalAveragePool(_)
            | Node::ReduceMean(_)
            | Node::ReduceSum(_)
            | Node::ReduceMax(_)
            | Node::ReduceMin(_)
            | Node::Constant(_)
    )
}

/// Op type of a node (the enum variant name, e.g. "MatMul"), for error messages.
fn op_type(node: &Node) -> String {
    let dbg = format!("{node:?}");
    dbg.split('(').next().unwrap_or(&dbg).to_string()
}

fn dispatch(node: &Node, ctx: &mut ExecutionContext) -> OpResult {
    match node {
        Node::Relu(n) => ops::activation::relu(n, ctx),
        Node::Sigmoid(n) => ops::activation::sigmoid(n, ctx),
        Node::Tanh(n) => ops::activation::tanh(n, ctx),
        Node::Gelu(n) => ops::activation::gelu(n, ctx),
        Node::Softmax(n) => ops::activation::softmax(n, ctx),
        Node::LogSoftmax(n) => ops::activation::log_softmax(n, ctx),
        Node::Linear(n) => ops::linear::linear(n, ctx),
        Node::Gemm(n) => ops::linear::gemm(n, ctx),
        Node::Add(n) => ops::elementwise::add(n, ctx),
        Node::Sub(n) => ops::elementwise::sub(n, ctx),
        Node::Mul(n) => ops::elementwise::mul(n, ctx),
        Node::Div(n) => ops::elementwise::div(n, ctx),
        Node::Reshape(n) => ops::reshape::reshape(n, ctx),
        Node::Flatten(n) => ops::reshape::flatten(n, ctx),
        Node::Transpose(n) => ops::reshape::transpose(n, ctx),
        Node::Conv2d(n) => ops::conv::conv2d(n, ctx),
        Node::BatchNormalization(n) => ops::conv::batch_norm_fused(n, ctx),
        Node::MaxPool2d(n) => ops::pool::max_pool2d(n, ctx),
        Node::AveragePool2d(n) => ops::pool::avg_pool2d(n, ctx),
        Node::GlobalAveragePool(n) => ops::pool::global_avg_pool(n, ctx),
        Node::ReduceMean(n) => ops::reduce::mean(n, ctx),
        Node::ReduceSum(n) => ops::reduce::sum(n, ctx),
        Node::ReduceMax(n) => ops::reduce::max(n, ctx),
        Node::ReduceMin(n) => ops::reduce::min(n, ctx),
        Node::Constant(_) => Ok(()),
        // rejected by `is_supported` at load time
        other => Err(format!("unsupported op '{}'", op_type(other))),
    }
}
