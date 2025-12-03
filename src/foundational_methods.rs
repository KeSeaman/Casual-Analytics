use crate::error::CausalError;
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::Array2;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
pub fn ols_fit(data: &Bound<'_, PyDict>, target: &str, features: Vec<String>) -> PyResult<f64> {
    // Extract target
    let y_any = data.get_item(target)?.ok_or_else(|| CausalError::ValueError(format!("Column {} not found", target)))?;
    let y_vec: Vec<f64> = y_any.extract()?;
    let y = Array2::from_shape_vec((y_vec.len(), 1), y_vec).map_err(|e| CausalError::ValueError(e.to_string()))?.into_shape((y_any.len()?,)).unwrap();

    // Extract features
    let n_rows = y.len();
    let n_cols = features.len();
    let mut x_vec = Vec::with_capacity(n_rows * n_cols);

    // Column-major to Row-major conversion (or just build row by row? No, dict is col-major)
    // ndarray is row-major by default.
    // We need to construct Array2.
    // Easiest is to extract all cols, then transpose? Or fill Array2.
    
    // Let's extract all feature vectors first
    let mut feature_vecs: Vec<Vec<f64>> = Vec::new();
    for f in &features {
        let f_any = data.get_item(f)?.ok_or_else(|| CausalError::ValueError(format!("Column {} not found", f)))?;
        let f_vec: Vec<f64> = f_any.extract()?;
        if f_vec.len() != n_rows {
            return Err(CausalError::ValueError("Feature length mismatch".into()).into());
        }
        feature_vecs.push(f_vec);
    }

    // Flatten to Array2 (n_rows x n_cols)
    // We iterate rows then cols
    for i in 0..n_rows {
        for j in 0..n_cols {
            x_vec.push(feature_vecs[j][i]);
        }
    }

    let x = Array2::from_shape_vec((n_rows, n_cols), x_vec).map_err(|e| CausalError::ValueError(e.to_string()))?;

    // Fit OLS
    let dataset = Dataset::new(x, y);
    let model = LinearRegression::default().fit(&dataset).map_err(|e| CausalError::Linfa(e.to_string()))?;

    Ok(model.params()[0])
}
