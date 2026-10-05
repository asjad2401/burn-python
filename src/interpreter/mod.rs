// ONNX runtime interpreter:
// parse -> pre-load weights (with BN fusion) -> dispatch nodes on each forward pass

mod context;
pub mod ops;

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

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

use crate::tensor::{B, FloatPrim, default_device, flex_to_numpy, numpy_to_flex};
use context::{ExecutionContext, OpResult};

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
            Some((
                arg.name.clone(),
                B::float_from_data(data, &default_device()),
            ))
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
    let get = |arg: &Argument| -> Option<FloatPrim> {
        if !arg.name.is_empty() {
            weights.get(&arg.name).cloned()
        } else {
            arg.value()
                .map(|d| B::float_from_data(d, &default_device()))
        }
    };

    let gamma = get(&node.inputs[1])?;
    let beta = get(&node.inputs[2])?;
    let mean = get(&node.inputs[3])?;
    let var = get(&node.inputs[4])?;

    let c = gamma.shape().iter().next().copied().unwrap_or(1);

    // scale = gamma / sqrt(var + eps),  shaped [1, C, 1, 1]
    let eps_t = B::float_from_data(TensorData::from([eps]), &default_device());
    let scale_flat = B::float_div(gamma, B::float_sqrt(B::float_add(var, eps_t)));
    // offset = beta - mean * scale,  shaped [1, C, 1, 1]
    let offset_flat = B::float_sub(beta, B::float_mul(mean, scale_flat.clone()));

    // pre-expand to [1, C, 1, 1] so no reshape at inference time
    let shape_4d: Vec<usize> = vec![1, c, 1, 1];
    let scale = B::float_reshape(scale_flat, shape_4d.clone().into());
    let offset = B::float_reshape(offset_flat, shape_4d.into());

    Some((node.name.clone(), scale, offset))
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
    catch_unwind(AssertUnwindSafe(|| dispatch(node, ctx))).unwrap_or_else(|payload| {
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        Err(msg.to_string())
    })
}

#[pyclass]
pub struct OnnxModel {
    nodes: Vec<Node>,
    weights: HashMap<String, FloatPrim>,
    inputs: Vec<InputSpec>,
    output_names: Vec<String>,
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

        let mut ctx = ExecutionContext::new(&self.weights);
        for (spec, obj) in self.inputs.iter().zip(&inputs) {
            let arr = as_f32_array(obj, &spec.name)?;
            spec.check_shape(arr.shape())?;
            ctx.insert(spec.name.clone(), numpy_to_flex(&arr));
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
                flex_to_numpy(py, t)
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        let input_names: Vec<&str> = self.inputs.iter().map(|i| i.name.as_str()).collect();
        format!(
            "OnnxModel(inputs={:?}, outputs={:?}, nodes={})",
            input_names,
            self.output_names,
            self.nodes.len()
        )
    }
}

pub fn load_onnx(path: &str) -> Result<OnnxModel, String> {
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
            if let Some((name, tensor)) = load_weight(arg) {
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

    Ok(OnnxModel {
        inputs: graph.inputs.iter().map(InputSpec::from_arg).collect(),
        output_names: graph.outputs.iter().map(|a| a.name.clone()).collect(),
        nodes: graph.nodes,
        weights,
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
        Node::Constant(_) => Ok(()),
        // rejected by `is_supported` at load time
        other => Err(format!("unsupported op '{}'", op_type(other))),
    }
}
