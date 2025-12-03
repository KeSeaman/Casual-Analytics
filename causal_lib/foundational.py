import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, LogisticRegression
from typing import Callable, List, Tuple

def estimate_ate_neyman(df: pl.DataFrame, treat_col: str, outcome_col: str) -> float:
    """
    Neyman-Pearson Estimator (Difference-in-Means).
    Pure function.
    """
    treated = df.filter(pl.col(treat_col) == 1)[outcome_col].mean()
    control = df.filter(pl.col(treat_col) == 0)[outcome_col].mean()
    return treated - control

def estimate_ate_mle(df: pl.DataFrame, treat_col: str, outcome_col: str, covariates: List[str]) -> float:
    """
    Maximum Likelihood Estimator (OLS).
    Uses scikit-learn LinearRegression.
    """
    X = df.select([treat_col] + covariates).to_numpy()
    y = df.select(outcome_col).to_numpy().ravel()
    
    model = LinearRegression()
    model.fit(X, y)
    
    # The coefficient for the treatment column (first column) is the ATE
    return model.coef_[0]

def estimate_ate_rubin(df: pl.DataFrame, treat_col: str, outcome_col: str, covariates: List[str]) -> float:
    """
    Rubin Causal Model (Propensity Score Weighting - IPW).
    """
    # 1. Estimate Propensity Scores (e(x))
    X = df.select(covariates).to_numpy()
    T = df.select(treat_col).to_numpy().ravel()
    y = df.select(outcome_col).to_numpy().ravel()
    
    ps_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    ps_model.fit(X, T)
    ps = ps_model.predict_proba(X)[:, 1]
    
    # Avoid division by zero
    ps = np.clip(ps, 0.05, 0.95)
    
    # 2. Calculate IPW ATE
    # ATE = E[Y*T/e(x)] - E[Y*(1-T)/(1-e(x))]
    ipw1 = (y * T) / ps
    ipw0 = (y * (1 - T)) / (1 - ps)
    
    return np.mean(ipw1 - ipw0)

def bootstrap(df: pl.DataFrame, func: Callable[[pl.DataFrame], float], n_boot: int = 100) -> Tuple[float, float, float]:
    """
    Higher-order function for bootstrapping any estimator.
    Returns (mean_estimate, ci_lower, ci_upper).
    """
    estimates = []
    n = len(df)
    
    for _ in range(n_boot):
        # Resample with replacement
        sample = df.sample(n, with_replacement=True, shuffle=True)
        est = func(sample)
        estimates.append(est)
        
    estimates = np.array(estimates)
    return (
        np.mean(estimates),
        np.percentile(estimates, 2.5),
        np.percentile(estimates, 97.5)
    )
