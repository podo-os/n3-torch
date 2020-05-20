use std::fmt;

use n3_core::{DimKey, Expression};

#[derive(Clone, Debug)]
pub struct Shapes(pub Vec<Shape>);

#[derive(Clone, Debug)]
pub struct Shape(pub Vec<Dim>);

#[derive(Clone, Debug)]
pub enum Dim {
    Placeholder(String),
    Expr(Expression),
}

impl From<Vec<Vec<Expression>>> for Shapes {
    fn from(shapes: Vec<Vec<Expression>>) -> Self {
        Self(shapes.into_iter().map(|s| s.into()).collect())
    }
}

impl From<Vec<Expression>> for Shape {
    fn from(shape: Vec<Expression>) -> Self {
        Self(shape.into_iter().map(|s| s.into()).collect())
    }
}

impl From<Expression> for Dim {
    fn from(dim: Expression) -> Self {
        match DimKey::try_from_expr(&dim) {
            Some(DimKey::Placeholder(ph, _)) => Self::Placeholder(ph),
            // FIXME: 변수가 첨가된 수식
            _ => Self::Expr(dim),
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Placeholder(ph) => write!(f, "{}", ph),
            Self::Expr(expr) => write!(f, "{}", expr.as_str()),
        }
    }
}
