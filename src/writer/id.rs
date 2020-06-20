use std::fmt;

use n3_core::{GraphId, GraphIdArg};

pub fn write_id_args(f: &mut fmt::Formatter<'_>, id: &GraphId, num_args: usize) -> fmt::Result {
    for arg in 0..num_args {
        write!(f, "x_")?;
        if arg > 0 {
            write!(f, ", ")?;
        }
        write_id(f, id)?;
        write!(f, "_{}", arg)?;
    }
    Ok(())
}

pub fn write_input_args(f: &mut fmt::Formatter<'_>, arg: Option<u64>) -> fmt::Result {
    write!(f, "input_{}", arg.unwrap_or(0))
}

pub fn write_id_arg(f: &mut fmt::Formatter<'_>, id_arg: &GraphIdArg) -> fmt::Result {
    if id_arg.id.is_input() {
        // FIXME: replace unwrap_or(0)
        write!(f, "input_{}", id_arg.arg.unwrap_or(0))
    } else {
        write_id(f, &id_arg.id)?;
        // FIXME: replace unwrap_or(0)
        write!(f, "_{}", id_arg.arg.unwrap_or(0))
    }
}

pub fn write_id(f: &mut fmt::Formatter<'_>, id: &GraphId) -> fmt::Result {
    write!(f, "{}_{}_{}", id.node, id.pass, id.repeat,)
}
