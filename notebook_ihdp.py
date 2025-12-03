import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    title=mo.md(r"""
    # 🧩 Functional Causal Inference Portfolio - IHDP Dataset
    
    A pure functional architecture for causal analysis using the **IHDP (Infant Health Development Program)** dataset, featuring:
    - **Foundational**: MLE, Neyman-Pearson, Rubin (Propensity Scores)
    - **Causal**: Causal Forests (Intel Accelerated), Mediation
    - **Dataset**: 25 pre-treatment covariates, binary treatment, continuous outcome
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    from causal_lib import data_loader_ihdp, foundational, causal, visualization
    return causal, data_loader_ihdp, foundational, mo, visualization


@app.cell
def _(data_loader_ihdp, mo):
    # Load IHDP Data (place ihdp.csv in project root or provide path/URL)
    df = data_loader_ihdp.load_ihdp("ihdp.csv")

    if df is not None:
        df = data_loader_ihdp.preprocess_ihdp(df)
        status = mo.md(f"**✅ IHDP Data Loaded**: {len(df)} observations")
    else:
        df = None
        status = mo.md("**⚠️ Data Load Failed - Please place ihdp.csv in project root**")

    return (df,)


@app.cell
def _(df, foundational, mo):
    # Foundational Methods
    if df is not None:
        # Get covariate columns (x1-x25)
        covariates = [col for col in df.columns if col.startswith("x")]
        
        # 1. Neyman-Pearson (Diff-in-Means)
        ate_neyman = foundational.estimate_ate_neyman(df, "treatment", "outcome")

        # 2. MLE (OLS)
        ate_mle = foundational.estimate_ate_mle(df, "treatment", "outcome", covariates)

        # 3. Rubin (Propensity Score)
        ate_rubin = foundational.estimate_ate_rubin(df, "treatment", "outcome", covariates)

        foundational_output = mo.md(
            f"""
            ### 🏛️ Foundational Estimates
            - **Neyman (Diff-in-Means)**: {ate_neyman:.4f}
            - **MLE (OLS)**: {ate_mle:.4f}
            - **Rubin (IPW)**: {ate_rubin:.4f}
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
    if df is not None and len(covariates) > 0:
        # Causal Forest (Intel Accelerated)
        ate_forest = causal.estimate_forest(df, "outcome", "treatment", covariates)

        causal_output = mo.md(
            f"""
            ### 🌲 Causal Forest (Intel Accelerated)
            - **CATE Estimate**: {ate_forest:.4f}
            """
        )
    else:
        ate_forest = 0.0
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
