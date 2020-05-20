use std::fmt;

use super::super::super::write_param;
use crate::module::TorchModule;

#[derive(Debug)]
pub struct Softmax;

impl super::super::super::base::ExternModuleImpl for Softmax {
    fn prefix(&self) -> &str {
        "nn"
    }

    fn write_params(&self, f: &mut fmt::Formatter<'_>, module: &TorchModule) -> fmt::Result {
        let axis = module.variables["axis"];
        write_param(f, "dim", axis.value.as_ref())
    }
}
