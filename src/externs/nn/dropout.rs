use std::fmt;

use super::super::write_param;
use crate::module::TorchModule;

#[derive(Debug)]
pub struct Dropout;

impl super::super::base::ExternModuleImpl for Dropout {
    fn prefix(&self) -> &str {
        "nn"
    }

    fn write_params(&self, f: &mut fmt::Formatter<'_>, module: &TorchModule) -> fmt::Result {
        let p = module.variables["Probability"];
        write_param(f, "p", p.value.as_ref())
    }
}
