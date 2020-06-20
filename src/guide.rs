use std::fmt;

use crate::error::CompileError;
use crate::externs::{find_extern_module, write_params, ExternModule};
use crate::module::TorchModule;

#[derive(Debug)]
pub enum Guide {
    Extern(ExternModule),
    NonExtern,
}

impl Guide {
    pub fn find_extern_guide(name: &str) -> Result<Self, CompileError> {
        Ok(Self::Extern(find_extern_module(name)?))
    }

    pub fn write_fmt(&self, f: &mut fmt::Formatter<'_>, module: &TorchModule) -> fmt::Result {
        self.write_name(f, module.name)?;
        {
            write!(f, "(")?;
            self.write_params(f, &module)?;
            write!(f, ")")?;
            Ok(())
        }
    }

    pub fn is_extern(&self) -> bool {
        match self {
            Self::Extern(_) => true,
            Self::NonExtern => false,
        }
    }

    fn write_name(&self, f: &mut fmt::Formatter<'_>, name: &str) -> fmt::Result {
        match self {
            Self::Extern(module) => module.write_name(f, name),
            Self::NonExtern => write!(f, "{}", name),
        }
    }

    fn write_params(&self, f: &mut fmt::Formatter<'_>, module: &TorchModule) -> fmt::Result {
        match self {
            Self::Extern(ext) => ext.write_params(f, module),
            Self::NonExtern => Ok(write_params(f, module.variables.iter())?),
        }
    }
}
