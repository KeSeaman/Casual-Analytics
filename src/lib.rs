use pyo3::prelude::*;

mod error;
mod foundational_methods;
mod structural_methods;
mod longitudinal_methods;

use foundational_methods::ols_fit;
use structural_methods::iv_fit;
use longitudinal_methods::did_fit;

#[pymodule]
fn _causal_rust_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ols_fit, m)?)?;
    m.add_function(wrap_pyfunction!(iv_fit, m)?)?;
    m.add_function(wrap_pyfunction!(did_fit, m)?)?;
    Ok(())
}
