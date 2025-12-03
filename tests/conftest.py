import pytest
import polars as pl
import numpy as np

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n = 1000
    
    # Confounders
    W = rng.normal(0, 1, size=(n, 2))
    
    # Instrument
    Z = rng.binomial(1, 0.5, size=n)
    
    # Treatment
    T_lat = 0.8 * Z + 0.3 * W[:, 0] + rng.normal(0, 0.5, size=n)
    T = (T_lat > 0).astype(float)
    
    # Outcome (True ATE = 2.0)
    Y = 2.0 * T + 0.5 * W[:, 0] + rng.normal(0, 1, size=n)
    
    # DiD Data
    time = rng.binomial(1, 0.5, size=n)
    group = rng.binomial(1, 0.5, size=n)
    # DiD effect = 3.0
    Y_did = 3.0 * (time * group) + 1.0 * time + 0.5 * group + rng.normal(0, 1, size=n)

    df = pl.DataFrame({
        "outcome": Y,
        "treatment": T,
        "instrument": Z,
        "outcome_did": Y_did,
        "time": time,
        "group": group,
        "w0": W[:, 0],
        "w1": W[:, 1],
    })
    return df
