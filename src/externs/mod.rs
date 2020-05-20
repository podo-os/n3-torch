mod base;
mod nn;

pub use self::base::{write_param, write_param_default, write_params, ExternModule};
use crate::error::ExternError;

pub fn find_extern_module(name: &str) -> Result<ExternModule, ExternError> {
    match name {
        // activations
        "ReLU" => Ok(Box::new(self::nn::activations::relu::ReLU)),
        "Softmax" => Ok(Box::new(self::nn::activations::softmax::Softmax)),

        // conv
        "Conv2d" => Ok(Box::new(self::nn::conv::Conv2d)),

        // linear
        "Linear" => Ok(Box::new(self::nn::linear::Linear)),

        // tensor
        "Transform" => Ok(Box::new(self::nn::tensor::Transform)),

        _ => Err(ExternError::ModuleNotFound {
            name: name.to_string(),
        }),
    }
}
