// Execution context: holds all tensors by name during a forward pass.
// weights (model params) live in the OnnxModel and are referenced here to avoid cloning.

use std::collections::HashMap;

use burn_backend::backend::ops::FloatTensorOps;
use onnx_ir::ir::{Argument, ValueSource};

use crate::tensor::{default_device, FloatPrim, B};

pub struct ExecutionContext<'w> {
    tensors: HashMap<String, FloatPrim>,
    weights: &'w HashMap<String, FloatPrim>,
}

impl<'w> ExecutionContext<'w> {
    pub fn new(weights: &'w HashMap<String, FloatPrim>) -> Self {
        Self { tensors: HashMap::new(), weights }
    }

    pub fn get(&self, name: &str) -> Option<FloatPrim> {
        self.tensors.get(name)
            .or_else(|| self.weights.get(name))
            .cloned()
    }

    pub fn insert(&mut self, name: String, tensor: FloatPrim) {
        self.tensors.insert(name, tensor);
    }

    /// Resolve any Argument to a FloatPrim:
    ///   - Dynamic/named Constant → look up by name
    ///   - Static with empty name → convert inline TensorData directly
    ///   - Optional/missing → None
    pub fn resolve(&self, arg: &Argument) -> Option<FloatPrim> {
        match arg.value_source {
            ValueSource::Dynamic | ValueSource::Constant => self.get(&arg.name),
            ValueSource::Static(_) => {
                if !arg.name.is_empty() {
                    // pre-loaded by name
                    if let Some(t) = self.get(&arg.name) {
                        return Some(t);
                    }
                }
                // anonymous inline static — convert on the fly
                arg.value().map(|d| B::float_from_data(d, &default_device()))
            }
            ValueSource::Optional => None,
        }
    }
}
