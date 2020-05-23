use std::fmt;

use crate::module::TorchModule;
use crate::writer;

#[derive(Debug)]
pub struct Linear;

impl super::super::base::ExternModuleImpl for Linear {
    fn prefix(&self) -> &str {
        "nn"
    }

    fn write_params(&self, f: &mut fmt::Formatter<'_>, module: &TorchModule) -> fmt::Result {
        let input = &module.input.as_ref().unwrap();
        let input = &input.0[0].0[0];
        writer::write_dim(f, input)?;

        let output = &module.output.as_ref().unwrap();
        let output = &output.0[0].0[0];
        writer::write_dim(f, output)
    }
}
