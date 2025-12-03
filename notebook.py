import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    title=mo.md(r"""
    # 🧩 Functional Causal Inference Portfolio

    A pure functional architecture for causal analysis, featuring:
    - **Foundational**: MLE, Neyman-Pearson, Rubin (Propensity Scores), Bootstrap
    - **Causal**: IV/2SLS (Rust), Causal Forests (Intel Accelerated), Mediation
    - **Longitudinal**: Difference-in-Differences (Rust)
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from causal_lib import data_loader, foundational, causal, advanced, visualization
    return causal, data_loader, foundational, mo, visualization


@app.cell
def _(data_loader, mo):
    # Load Data from NBER (nsw_treated.txt + nsw_control.txt)
    df = data_loader.load_nsw()

    if df is not None:
        df = data_loader.preprocess_nsw(df)
        status = mo.md(f"**✅ Data Loaded**: {len(df)} observations from NBER NSW dataset.")
    else:
        df = None
        status = mo.md("**⚠️ Data Load Failed**")

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

        foundational_output = mo.md(
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
        foundational_output = None

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

        causal_output = mo.md(
            f"""
            ### 🌲 Causal & Machine Learning
            - **Causal Forest (Intel)**: ${ate_forest:.2f}
            - **Mediation (ACME)**: ${acme:.2f} (Indirect effect via re75)
            """
        )
    else:
        ate_forest, acme, ade, total = 0.0, 0.0, 0.0, 0.0
        causal_output = None

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
        viz_output = mo.vstack([
            mo.md("### 📊 Method Comparison"),
            plot
        ])
    else:
        viz_output = None

    return


if __name__ == "__main__":
    app.run()
