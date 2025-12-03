import numpy as np
import polars as pl
from typing import List, Tuple, Optional
import causal_rust_core as crc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML

# Intel Patching (Side Effect, but idempotent)
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass

def estimate_iv(df: pl.DataFrame, outcome: str, treatment: str, instrument: str) -> float:
    """
    Instrumental Variables (IV/2SLS) using Rust Core.
    """
    return crc.iv_fit(df, outcome, treatment, instrument)

def estimate_forest(df: pl.DataFrame, outcome: str, treatment: str, covariates: List[str]) -> float:
    """
    Causal Forest using EconML (accelerated by Intel extension).
    Returns the Average Treatment Effect (ATE).
    """
    Y = df.select(outcome).to_numpy().ravel()
    T = df.select(treatment).to_numpy().ravel()
    X = df.select(covariates).to_numpy()
    
    # Configure models (Intel-patched RF)
    # Use Regressor for continuous outcome, Classifier for binary treatment
    model_y = RandomForestRegressor(n_estimators=100, random_state=42)
    model_t = RandomForestClassifier(n_estimators=100, random_state=42)
    
    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=100,
        random_state=42
    )
    
    est.fit(Y, T, X=X)
    return est.ate(X)

def estimate_mediation(df: pl.DataFrame, treatment: str, mediator: str, outcome: str, covariates: List[str]) -> Tuple[float, float, float]:
    """
    Mediation Analysis (Baron-Kenny approach).
    Returns (ACME, ADE, Total Effect).
    ACME: Average Causal Mediation Effect (Indirect)
    ADE: Average Direct Effect
    """
    # 1. Total Effect: T -> Y
    # Y = c*T + e1
    X_total = df.select([treatment] + covariates).to_numpy()
    y_total = df.select(outcome).to_numpy().ravel()
    m_total = LinearRegression().fit(X_total, y_total)
    total_effect = m_total.coef_[0]
    
    # 2. Mediator Model: T -> M
    # M = a*T + e2
    X_med = df.select([treatment] + covariates).to_numpy()
    y_med = df.select(mediator).to_numpy().ravel()
    m_med = LinearRegression().fit(X_med, y_med)
    a_path = m_med.coef_[0]
    
    # 3. Outcome Model: T + M -> Y
    # Y = c'*T + b*M + e3
    X_out = df.select([treatment, mediator] + covariates).to_numpy()
    y_out = df.select(outcome).to_numpy().ravel()
    m_out = LinearRegression().fit(X_out, y_out)
    ade = m_out.coef_[0] # c' (Direct Effect)
    b_path = m_out.coef_[1] # b
    
    acme = a_path * b_path # Indirect Effect
    
    return (acme, ade, total_effect)

# Helper for Linear Regression inside mediation
from sklearn.linear_model import LinearRegression
