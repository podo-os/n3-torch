use n3_core::Dim;

#[derive(Clone, Debug)]
pub struct Shapes(pub Vec<Shape>);

#[derive(Clone, Debug)]
pub struct Shape(pub Vec<Dim>);

impl From<Vec<Vec<Dim>>> for Shapes {
    fn from(shapes: Vec<Vec<Dim>>) -> Self {
        Self(shapes.into_iter().map(|s| s.into()).collect())
    }
}

impl From<Vec<Dim>> for Shape {
    fn from(shape: Vec<Dim>) -> Self {
        Self(shape)
    }
}
