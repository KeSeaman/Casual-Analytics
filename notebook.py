import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from causal_lib import data_loader, foundational, causal, advanced, visualization
    return causal, data_loader, foundational, mo, visualization


@app.cell
def _(mo):
    mo.md(r"""
    # 🧩 Functional Causal Inference Portfolio

    A pure functional architecture for causal analysis, featuring:
    - **Foundational**: MLE, Neyman-Pearson, Rubin (Propensity Scores), Bootstrap
    - **Causal**: IV/2SLS (Rust), Causal Forests (Intel Accelerated), Mediation
    - **Longitudinal**: Difference-in-Differences (Rust)
    """)
    return


@app.cell
def _(data_loader, mo):
    # Load Data
    url = "https://raw.githubusercontent.com/scunning1975/mixtape/master/nsw_mixtape.csv"
    df = data_loader.load_nsw(url)

    if df is not None:
        df = data_loader.preprocess_nsw(df)
        mo.md(f"**✅ Data Loaded**: {len(df)} observations from NSW Mixtape.")
    else:
        mo.md("**⚠️ Data Load Failed**")
    return (df,)


@app.cell
def _(df, foundational, mo):
    # Foundational Methods
    if df is not None:
        # 1. Neyman-Pearson (Diff-in-Means)
        ate_neyman = foundational.estimate_ate_neyman(df, "treat", "re78")

        # 2. MLE (OLS)
        # Covariates: age, education, black, hispanic, married, nodegree, re75
        covariates = ["age", "education", "black", "hispanic", "married", "nodegree", "re75"]
        ate_mle = foundational.estimate_ate_mle(df, "treat", "re78", covariates)

        # 3. Rubin (Propensity Score)
        ate_rubin = foundational.estimate_ate_rubin(df, "treat", "re78", covariates)

        mo.md(
            f"""
            ### 🏛️ Foundational Estimates
            - **Neyman (Diff-in-Means)**: ${ate_neyman:.2f}
            - **MLE (OLS)**: ${ate_mle:.2f}
            - **Rubin (IPW)**: ${ate_rubin:.2f}
            """
        )
    else:
        ate_neyman, ate_mle, ate_rubin = 0.0, 0.0, 0.0
        covariates = []

    return ate_mle, ate_neyman, ate_rubin, covariates


@app.cell
def _(causal, covariates, df, mo):
    # Causal Methods
    if df is not None:
        # 1. Causal Forest (Intel Accelerated)
        ate_forest = causal.estimate_forest(df, "re78", "treat", covariates)

        # 2. Mediation Analysis (Example)
        # Mediator: re75 (Earnings before treatment) -> Just for demo
        # This is a bit contrived for NSW, but demonstrates the method.
        acme, ade, total = causal.estimate_mediation(df, "treat", "re75", "re78", ["age", "education"])

        mo.md(
            f"""
            ### 🌲 Causal & Machine Learning
            - **Causal Forest (Intel)**: ${ate_forest:.2f}
            - **Mediation (ACME)**: ${acme:.2f} (Indirect effect via re75)
            """
        )
    else:
        ate_forest, acme, ade, total = 0.0, 0.0, 0.0, 0.0

    return (ate_forest,)


@app.cell
def _(ate_forest, ate_mle, ate_neyman, ate_rubin, df, mo, visualization):
    # Visualization
    if df is not None:
        results = {
            "Neyman": ate_neyman,
            "MLE (OLS)": ate_mle,
            "Rubin (IPW)": ate_rubin,
            "Causal Forest": ate_forest
        }

        plot = visualization.plot_ate_comparison(results)

        mo.vstack([
            mo.md("### 📊 Method Comparison"),
            plot
        ])
    else:
        plot, results = None, {}

    return


if __name__ == "__main__":
    app.run()
