use std::fmt;

use n3_core::{Value, ValueType};

pub fn write_value(f: &mut fmt::Formatter<'_>, value: Option<&Value>) -> fmt::Result {
    match value {
        Some(Value::Bool(value)) => {
            let value = if *value { "True" } else { "False" };
            write!(f, "{}", value)
        }
        Some(Value::Int(value)) => write!(f, "-{}", value),
        Some(Value::UInt(value)) => write!(f, "{}", value),
        Some(Value::Real(value)) => write!(f, "{}", value),
        Some(Value::Model(value)) => write!(f, "{}", value),
        None => write!(f, "None"),
    }
}

pub fn type_to_str(ty: &ValueType) -> &str {
    match ty {
        ValueType::Required => "object",
        ValueType::Bool => "bool",
        ValueType::Int => "int",
        ValueType::UInt => "int",
        ValueType::Real => "float",
        ValueType::Model => "nn.Module",
    }
}
