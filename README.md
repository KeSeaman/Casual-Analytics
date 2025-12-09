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

## 📊 Results

Results from running the analysis on the **NBER NSW (National Supported Work)** dataset with 722 observations. This classic dataset evaluates a job training program's effect on earnings.

### Foundational Estimates

| Method | Estimate | Description |
|--------|----------|-------------|
| **Neyman (Diff-in-Means)** | $886.30 | Simple difference between treated and control group means. The most straightforward estimator—compares average earnings of those who received training vs. those who didn't. |
| **MLE (OLS)** | $806.51 | Ordinary Least Squares regression controlling for age, education, race, marital status, and prior earnings (re75). Adjusts for observable confounders to reduce bias. |
| **Rubin (IPW)** | $799.45 | Inverse Probability Weighting using propensity scores. Reweights observations to balance treated/control groups on covariates, mimicking a randomized experiment. |

### Interpretation

- **Treatment Effect**: The job training program increased annual earnings by approximately **$800-886**
- **Method Agreement**: All three estimators produce consistent results (~$800-890), suggesting robust findings
- **Bias Correction**: OLS and IPW yield lower estimates than Neyman, indicating some positive selection into treatment
- **Practical Significance**: An ~$800 increase in annual earnings represents meaningful economic impact for the target population

### Advanced ML-Based Estimates

| Method | Estimate | Description |
|--------|----------|-------------|
| **Causal Forest** | $827.26 | ML-based heterogeneous treatment effect estimator (EconML). Uses Random Forests to model outcome and treatment, allowing for individual-level treatment effects. The ATE represents the average across all individuals. |
| **Mediation Total Effect** | $831.04 | Total causal effect decomposed into direct and indirect pathways through a mediator (prior earnings re75). |
| **Mediation ADE (Direct)** | $834.41 | Average Direct Effect—the portion of treatment effect NOT mediated through re75. Shows the direct impact of training on earnings. |
| **Mediation ACME (Indirect)** | -$3.37 | Average Causal Mediation Effect—the portion of effect flowing through the mediator (re75). The small negative value suggests minimal mediation through prior earnings. |

### Key Insights

- **Consistency Across Methods**: All methods (Neyman, OLS, IPW, Causal Forest) estimate treatment effects in the **$800-890 range**
- **Direct vs. Indirect Effects**: Nearly all of the treatment effect is direct ($834), with negligible effect through prior earnings
- **Heterogeneity**: Causal Forest enables exploring treatment effect heterogeneity across subgroups

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
