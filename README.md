# Causal Rust Core: High-Performance Causal Inference

A production-ready Causal Inference library with a polyglot architecture combining Rust's performance with Python's data science ecosystem. Implements OLS, IV/2SLS, Difference-in-Differences, and advanced ML-based causal methods.

## 🔬 Overview

This project demonstrates high-performance causal inference, featuring:
- **Multi-Method Support**: OLS, IV/2SLS, DiD, and Causal Forests
- **Rust Backend**: PyO3 bindings for ~10x performance gains
- **EconML Integration**: GPU-accelerated treatment effect estimation
- **Interactive Dashboard**: Marimo notebooks for analysis and visualization

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Causal Inference Stack                        │
├─────────────────────────────────────────────────────────────────┤
│  Frontend    │ Marimo Notebooks (interactive analysis)         │
├──────────────┼──────────────────────────────────────────────────┤
│  ML Layer    │ EconML (Causal Forests), scikit-learn            │
├──────────────┼──────────────────────────────────────────────────┤
│  Data        │ Polars DataFrames, NumPy arrays                  │
├──────────────┼──────────────────────────────────────────────────┤
│  Backend     │ Rust Core (PyO3, matrix ops, estimators)         │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
causal_project_v2/
├── src/                        # Rust backend
│   ├── lib.rs                  # PyO3 module exports
│   ├── foundational_methods.rs # OLS, IV/2SLS implementations
│   ├── longitudinal_methods.rs # DiD implementations
│   ├── structural_methods.rs   # Structural equation methods
│   └── error.rs                # Error handling
├── causal_lib/                 # Python interface
│   ├── causal.py               # Main CausalEstimator class
│   ├── foundational.py         # Python OLS/IV wrappers
│   ├── data_loader.py          # NBER dataset loader
│   ├── data_loader_ihdp.py     # IHDP dataset loader
│   └── visualization.py        # Plotting utilities
├── python/
│   └── causal_rust_core/       # PyO3 Python bindings
├── tests/                      # Test suite
├── notebook.py                 # NBER analysis notebook
└── notebook_ihdp.py            # IHDP analysis notebook
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- Rust toolchain (for backend compilation)
- uv package manager

### 1. Create Environment & Install Dependencies

```bash
uv sync
```

### 2. Build Rust Backend

```bash
source .venv/bin/activate
maturin develop
```

### 3. Run Analysis Notebook

```bash
marimo edit notebook.py
```

## 🔧 Key Components

### Rust Backend (`src/`)

High-performance implementations of causal estimators:

```rust
/// Ordinary Least Squares regression
pub fn ols(y: Vec<f64>, x: Vec<Vec<f64>>) -> OLSResult

/// Instrumental Variable / Two-Stage Least Squares
pub fn iv_2sls(y: Vec<f64>, x: Vec<Vec<f64>>, z: Vec<Vec<f64>>) -> IVResult

/// Difference-in-Differences estimator
pub fn did(before_control: f64, after_control: f64, 
           before_treat: f64, after_treat: f64) -> f64
```

### Python Interface (`causal_lib/`)

Clean Python API wrapping Rust core:

```python
from causal_lib import CausalEstimator

estimator = CausalEstimator()

# OLS regression
result = estimator.ols(y, X)
print(f"Coefficients: {result.coefficients}")
print(f"R-squared: {result.r_squared}")

# IV/2SLS estimation
iv_result = estimator.iv_2sls(y, X, Z)
print(f"Treatment Effect: {iv_result.treatment_effect}")
```

### EconML Integration

GPU-accelerated causal forests for heterogeneous effects:

```python
from econml.dml import CausalForestDML

model = CausalForestDML(n_trees=200)
model.fit(Y, T, X=X, W=W)
treatment_effects = model.effect(X_test)
```

## 📊 Supported Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| **OLS** | Ordinary Least Squares | Baseline regression |
| **IV/2SLS** | Instrumental Variables | Endogeneity correction |
| **DiD** | Difference-in-Differences | Policy evaluation |
| **Causal Forest** | ML-based heterogeneous effects | Individual treatment effects |

## 📈 Example Results

```
=== NBER Dataset Analysis ===
OLS Estimates:
  Coefficient: 0.0532 ± 0.012
  R-squared: 0.847

IV/2SLS Estimates:
  Treatment Effect: 0.092
  First-stage F-stat: 42.3

DiD Estimates:
  ATT: 0.078
  Parallel Trends: ✓ Validated
```

## 🛠️ Development

### Rebuild Backend After Changes

When modifying Rust files in `src/`, recompile:

```bash
maturin develop
```

For release builds with optimizations:

```bash
maturin develop --release
```

### Run Tests

```bash
pytest tests/
```

### Verify Rust Import

```python
from causal_rust_core import ols, iv_2sls, did
print(ols([1.0, 2.0], [[1.0], [2.0]]))
```

## 📝 License

MIT

## 🙏 Acknowledgments

- [EconML](https://github.com/microsoft/EconML) - Heterogeneous treatment effects
- [PyO3](https://pyo3.rs/) - Rust-Python bindings
- [Polars](https://pola.rs/) - Fast DataFrames
- [Marimo](https://marimo.io/) - Reactive notebooks
