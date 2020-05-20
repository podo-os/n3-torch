#[derive(Debug)]
pub enum CompileError {
    FrontError { error: n3_core::CompileError },
    ExternError { error: ExternError },
}

#[derive(Debug)]
pub enum ExternError {
    ModuleNotFound { name: String },
    FormatError { error: std::fmt::Error },
}

impl From<n3_core::CompileError> for CompileError {
    fn from(error: n3_core::CompileError) -> Self {
        Self::FrontError { error }
    }
}

impl From<ExternError> for CompileError {
    fn from(error: ExternError) -> Self {
        Self::ExternError { error }
    }
}

impl From<std::fmt::Error> for CompileError {
    fn from(error: std::fmt::Error) -> Self {
        Self::ExternError {
            error: error.into(),
        }
    }
}

impl From<std::fmt::Error> for ExternError {
    fn from(error: std::fmt::Error) -> Self {
        Self::FormatError { error }
    }
}
