use std::fmt;

use heck::CamelCase;
use n3_core::{Dim, DimKey};
use regex::Regex;

pub fn write_dim(f: &mut fmt::Formatter<'_>, dim: &Dim) -> fmt::Result {
    match dim {
        Dim::Key(key) => write_dim_key(f, key),
        Dim::Expr(expr) => {
            let expr = expr.as_str();

            let re_pe: Regex = Regex::new(r"var_(?P<v>\w+)").unwrap();
            let expr = re_pe.replace_all(expr, "self.$v");

            write!(f, "{}, ", expr)
        }
    }
}

fn write_dim_key(f: &mut fmt::Formatter<'_>, key: &DimKey) -> fmt::Result {
    match key {
        DimKey::Placeholder(ph, _) => write!(f, "self._ph_{}, ", ph),
        DimKey::Variable(var) => write!(f, "self.{}, ", var.to_camel_case()),
    }
}
