use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CausalError {
    #[error("Linfa error: {0}")]
    Linfa(String), // Linfa errors are generic
    #[error("Value error: {0}")]
    ValueError(String),
}

impl From<CausalError> for PyErr {
    fn from(err: CausalError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

impl From<linfa::Error> for CausalError {
    fn from(err: linfa::Error) -> Self {
        CausalError::Linfa(err.to_string())
    }
}
