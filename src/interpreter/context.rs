// Execution context: holds all tensors by name during a forward pass.
// weights (model params) live in the OnnxModel and are referenced here to avoid cloning.

use std::collections::HashMap;

use burn_backend::backend::ops::FloatTensorOps;
use onnx_ir::ir::{Argument, ValueSource};

use crate::tensor::{B, Device, FloatPrim};

/// Key a static/constant argument is pre-loaded under in the weights map.
/// onnx-ir often hands initializers over as anonymous statics (name ""), so those
/// are keyed by their data id instead.
pub fn weight_key(arg: &Argument) -> Option<String> {
    match arg.value_source {
        _ if !arg.name.is_empty() => Some(arg.name.clone()),
        ValueSource::Static(id) => Some(format!("static::{id}")),
        _ => None,
    }
}

/// Ops report failures as a message; the interpreter adds node context.
pub type OpResult = Result<(), String>;

pub struct ExecutionContext<'w> {
    tensors: HashMap<String, FloatPrim>,
    weights: &'w HashMap<String, FloatPrim>,
    device: &'w Device,
}

impl<'w> ExecutionContext<'w> {
    pub fn new(weights: &'w HashMap<String, FloatPrim>, device: &'w Device) -> Self {
        Self {
            tensors: HashMap::new(),
            weights,
            device,
        }
    }

    pub fn get(&self, name: &str) -> Option<FloatPrim> {
        self.tensors
            .get(name)
            .or_else(|| self.weights.get(name))
            .cloned()
    }

    pub fn insert(&mut self, name: String, tensor: FloatPrim) {
        self.tensors.insert(name, tensor);
    }

    /// Resolve any Argument to a FloatPrim:
    ///   - Dynamic/named Constant → look up by name
    ///   - Static → pre-loaded weight (see `weight_key`), else convert inline data
    ///   - Optional/missing → None
    pub fn resolve(&self, arg: &Argument) -> Option<FloatPrim> {
        match arg.value_source {
            ValueSource::Dynamic | ValueSource::Constant => self.get(&arg.name),
            ValueSource::Static(_) => weight_key(arg)
                .and_then(|key| self.weights.get(&key).cloned())
                // not pre-loaded (e.g. non-float data) — convert on the fly
                .or_else(|| arg.value().map(|d| B::float_from_data(d, self.device))),
            ValueSource::Optional => None,
        }
    }

    /// Like `resolve`, but a missing tensor is an error rather than `None`.
    pub fn require(&self, arg: &Argument) -> Result<FloatPrim, String> {
        self.resolve(arg)
            .ok_or_else(|| format!("missing input tensor '{}'", arg.name))
    }
}
