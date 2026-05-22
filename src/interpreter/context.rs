// Execution context: holds all tensors by name during a forward pass.
// weights (model params) live in the OnnxModel and are referenced here to avoid cloning.

use std::collections::HashMap;
use crate::tensor::FloatPrim;

pub struct ExecutionContext<'w> {
    // intermediate tensors produced during the forward pass
    tensors: HashMap<String, FloatPrim>,
    // read-only reference to pre-loaded model weights
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
}
