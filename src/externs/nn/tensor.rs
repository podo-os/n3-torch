use std::fmt;

use crate::module::TorchModule;
use crate::writer;

#[derive(Debug)]
pub struct Transform;

impl super::super::base::ExternModuleImpl for Transform {
    fn prefix(&self) -> &str {
        "n3.torch"
    }

    fn write_params(&self, f: &mut fmt::Formatter<'_>, module: &TorchModule) -> fmt::Result {
        let output = &module.output.as_ref().unwrap();
        let output = &output.0[0];
        for dim in &output.0 {
            writer::write_dim(f, dim)?;
        }
        Ok(())
    }
}
