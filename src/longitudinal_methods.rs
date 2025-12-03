use crate::error::CausalError;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn did_fit(data: &Bound<'_, PyDict>, target: &str, group: &str, time: &str) -> PyResult<f64> {
    let get_col = |name: &str| -> PyResult<Vec<f64>> {
        let item = data.get_item(name)?.ok_or_else(|| CausalError::ValueError(format!("Column {} not found", name)))?;
        item.extract()
    };

    let g_vec = get_col(group)?;
    let t_vec = get_col(time)?;
    let y_vec = get_col(target)?;
    let n = g_vec.len();

    // Create interaction and X matrix
    let mut x_vec = Vec::with_capacity(n * 3);
    for i in 0..n {
        x_vec.push(g_vec[i]);
        x_vec.push(t_vec[i]);
        x_vec.push(g_vec[i] * t_vec[i]); // Interaction
    }

    let x = Array2::from_shape_vec((n, 3), x_vec).unwrap();
    let y = Array2::from_shape_vec((n, 1), y_vec).unwrap().into_shape((n,)).unwrap();

    let dataset = Dataset::new(x, y);
    let model = LinearRegression::default().fit(&dataset).map_err(|e| CausalError::Linfa(e.to_string()))?;

    // Interaction term is index 2
    Ok(model.params()[2])
}
