use crate::error::CausalError;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::{Array2, Axis};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn iv_fit(data: &Bound<'_, PyDict>, target: &str, treatment: &str, instrument: &str) -> PyResult<f64> {
    // Helper to extract column as Array1/Array2
    let get_col = |name: &str| -> PyResult<Vec<f64>> {
        let item = data.get_item(name)?.ok_or_else(|| CausalError::ValueError(format!("Column {} not found", name)))?;
        item.extract()
    };

    let z_vec = get_col(instrument)?;
    let t_vec = get_col(treatment)?;
    let y_vec = get_col(target)?;
    let n = z_vec.len();

    // 1. First Stage: T ~ Z
    let z = Array2::from_shape_vec((n, 1), z_vec).unwrap();
    let t = Array2::from_shape_vec((n, 1), t_vec).unwrap().into_shape((n,)).unwrap();
    
    let dataset_stage1 = Dataset::new(z.clone(), t);
    let model_stage1 = LinearRegression::default().fit(&dataset_stage1).map_err(|e| CausalError::Linfa(e.to_string()))?;
    let t_hat = model_stage1.predict(&z);

    // 2. Second Stage: Y ~ T_hat
    let y = Array2::from_shape_vec((n, 1), y_vec).unwrap().into_shape((n,)).unwrap();
    let t_hat_2d = t_hat.insert_axis(Axis(1));
    
    let dataset_stage2 = Dataset::new(t_hat_2d, y);
    let model_stage2 = LinearRegression::default().fit(&dataset_stage2).map_err(|e| CausalError::Linfa(e.to_string()))?;

    Ok(model_stage2.params()[0])
}
