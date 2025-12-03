import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import causal_rust_core as crc
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LassoCV
    import time

    # Intel Acceleration
    try:
        from sklearnex import patch_sklearn
        patch_sklearn()
        print("🚀 Intel Acceleration Enabled: Using scikit-learn-intelex")
        gpu_available = True
    except ImportError:
        print("🐢 Intel Extension not found: Using stock Scikit-Learn")
        gpu_available = False

    from sklearn.ensemble import RandomForestRegressor

    return (
        CausalForestDML,
        RandomForestRegressor,
        crc,
        gpu_available,
        mo,
        np,
        pl,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # 🚀 High-Performance Causal Inference Portfolio

    This project demonstrates a hybrid **Rust/Python** architecture for causal inference.

    - **Rust Core**: Ultra-fast OLS, IV/2SLS, and DiD implementations using `linfa` and `polars`.
    - **Python Frontend**: Interactive analysis with `marimo` and `econml`.
    - **GPU Acceleration**: Automatic fallback to `cuml` for Causal Forests.
    """)
    return


@app.cell
def _(np, pl):
    def generate_data(n=100_000):
        rng = np.random.default_rng(42)

        # Confounders
        W = rng.normal(0, 1, size=(n, 5))

        # Instrument (for IV)
        Z = rng.binomial(1, 0.5, size=n)

        # Treatment (endogenous)
        # T depends on Z and W
        T_lat = 0.5 * Z + 0.3 * W[:, 0] + rng.normal(0, 0.5, size=n)
        T = (T_lat > 0).astype(float)

        # Outcome
        # True ATE = 2.0
        Y = 2.0 * T + 0.5 * W[:, 0] + 0.3 * W[:, 1] + rng.normal(0, 1, size=n)

        # Time and Group (for DiD)
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
        })

        # Add confounders
        for i in range(5):
            df = df.with_columns(pl.Series(f"w{i}", W[:, i]))

        return df

    df = generate_data()
    print(f"Generated {len(df):,} rows")
    return (df,)


@app.cell
def _(crc, df, mo, time):
    # Rust OLS
    start_ols = time.time()
    ols_ate = crc.ols_fit(df, "outcome", ["treatment", "w0", "w1", "w2", "w3", "w4"])
    ols_time = (time.time() - start_ols) * 1000

    mo.md(f"**Rust OLS ATE**: {ols_ate:.4f} ({ols_time:.2f} ms)")
    return


@app.cell
def _(crc, df, mo, time):
    # Rust IV
    start_iv = time.time()
    iv_ate = crc.iv_fit(df, "outcome", "treatment", "instrument")
    iv_time = (time.time() - start_iv) * 1000

    mo.md(f"**Rust IV/2SLS ATE**: {iv_ate:.4f} ({iv_time:.2f} ms)")
    return


@app.cell
def _(crc, df, mo, time):
    # Rust DiD
    start_did = time.time()
    did_ate = crc.did_fit(df, "outcome_did", "group", "time")
    did_time = (time.time() - start_did) * 1000

    mo.md(f"**Rust DiD ATE**: {did_ate:.4f} ({did_time:.2f} ms)")
    return


@app.cell
def _(CausalForestDML, RandomForestRegressor, df, gpu_available, mo):
    # EconML Causal Forest

    X = df.select([f"w{i}" for i in range(5)]).to_numpy()
    T = df.select("treatment").to_numpy().ravel()
    Y = df.select("outcome").to_numpy().ravel()

    if gpu_available:
        # Intel extension patches sklearn, so we use the same class
        model_y = RandomForestRegressor(n_estimators=100)
        model_t = RandomForestRegressor(n_estimators=100)
    else:
        model_y = RandomForestRegressor(n_estimators=100)
        model_t = RandomForestRegressor(n_estimators=100)

    est = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        discrete_treatment=True,
        n_estimators=100,
        random_state=42
    )

    est.fit(Y, T, X=X)
    ate = est.ate(X)

    mo.md(f"**EconML Causal Forest ATE**: {ate:.4f}")
    return


if __name__ == "__main__":
    app.run()
