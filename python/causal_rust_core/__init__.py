from ._causal_rust_core import *
from ._causal_rust_core import ols_fit as _ols_fit
from ._causal_rust_core import iv_fit as _iv_fit
from ._causal_rust_core import did_fit as _did_fit
import polars as pl

def ols_fit(df: pl.DataFrame, target: str, features: list[str]) -> float:
    """
    Fit OLS using Rust backend.
    """
    data_dict = df.to_dict(as_series=False)
    return _ols_fit(data_dict, target, features)

def iv_fit(df: pl.DataFrame, target: str, treatment: str, instrument: str) -> float:
    """
    Fit IV/2SLS using Rust backend.
    """
    data_dict = df.to_dict(as_series=False)
    return _iv_fit(data_dict, target, treatment, instrument)

def did_fit(df: pl.DataFrame, target: str, group: str, time: str) -> float:
    """
    Fit DiD using Rust backend.
    """
    data_dict = df.to_dict(as_series=False)
    return _did_fit(data_dict, target, group, time)
