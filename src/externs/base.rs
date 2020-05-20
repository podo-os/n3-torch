use std::fmt;
use std::iter::Iterator;

use crate::module::TorchModule;
use crate::writer;

use n3_core::{Value, Variable};

pub type ExternModule = Box<dyn ExternModuleImpl>;

pub trait ExternModuleImpl: fmt::Debug {
    fn prefix(&self) -> &str;

    fn write_name(&self, f: &mut fmt::Formatter<'_>, name: &str) -> fmt::Result {
        write!(f, "{}.{}", self.prefix(), name)
    }

    fn write_params(&self, f: &mut fmt::Formatter<'_>, module: &TorchModule) -> fmt::Result {
        write_params(f, module.variables.iter())
    }
}

pub fn write_param(f: &mut fmt::Formatter<'_>, name: &str, value: Option<&Value>) -> fmt::Result {
    write!(f, "{}=", name)?;
    writer::write_value(f, value)?;
    write!(f, ", ")
}

pub fn write_param_default(
    f: &mut fmt::Formatter<'_>,
    name: &str,
    value: Option<&Value>,
    default: &Value,
) -> fmt::Result {
    match value {
        Some(value) => unimplemented!(),
        None => Ok(write_param(f, name, Some(default))?),
    }
}

pub fn write_params<'a, I>(f: &mut fmt::Formatter<'_>, iter: I) -> fmt::Result
where
    I: Iterator<Item = (&'a String, &'a &'a Variable)>,
{
    for (name, var) in iter {
        write_param(f, name, var.value.as_ref())?;
    }
    Ok(())
}
