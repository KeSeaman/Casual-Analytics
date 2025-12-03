import pytest
import causal_rust_core as crc
import numpy as np

def test_ols_recovery(synthetic_data):
    # OLS might be biased due to confounding, but we check it runs and returns a reasonable number
    # If we control for W, it should be close to 2.0
    ate = crc.ols_fit(synthetic_data, "outcome", ["treatment", "w0"])
    print(f"OLS ATE: {ate}")
    assert np.isclose(ate, 2.0, atol=0.2)

def test_iv_recovery(synthetic_data):
    # IV should recover 2.0
    ate = crc.iv_fit(synthetic_data, "outcome", "treatment", "instrument")
    print(f"IV ATE: {ate}")
    assert np.isclose(ate, 2.0, atol=0.5) # IV has higher variance

def test_did_recovery(synthetic_data):
    # DiD should recover 3.0
    ate = crc.did_fit(synthetic_data, "outcome_did", "group", "time")
    print(f"DiD ATE: {ate}")
    assert np.isclose(ate, 3.0, atol=0.2)
