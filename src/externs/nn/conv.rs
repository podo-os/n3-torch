use std::fmt;

use super::super::{write_param, write_param_default};
use crate::module::TorchModule;
use crate::writer;

use heck::SnakeCase;
use n3_core::Value;

#[derive(Debug)]
pub struct Conv2d;

impl super::super::base::ExternModuleImpl for Conv2d {
    fn prefix(&self) -> &str {
        "nn"
    }

    fn write_params(&self, f: &mut fmt::Formatter<'_>, module: &TorchModule) -> fmt::Result {
        let input = &module.input.as_ref().unwrap();
        let input = &input.0[0].0[0];
        writer::write_dim(f, input)?;

        let output = &module.output.as_ref().unwrap();
        let output = &output.0[0].0[0];
        writer::write_dim(f, output)?;

        for (name, var) in &module.variables {
            let value = var.value.as_ref();
            match name.as_str() {
                "Padding" => {
                    let kernel_size = module.variables["KernelSize"].unwrap_uint().unwrap();
                    let padding = kernel_size / 2;
                    write_param_default(f, &name.to_snake_case(), value, &Value::UInt(padding))?;
                }
                _ => {
                    write_param(f, &name.to_snake_case(), value)?;
                }
            }
        }
        Ok(())
    }
}
