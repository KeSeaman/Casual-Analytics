import polars as pl
import causal_rust_core as crc

def estimate_did(df: pl.DataFrame, outcome: str, group: str, time: str) -> float:
    """
    Difference-in-Differences (DiD) using Rust Core.
    """
    return crc.did_fit(df, outcome, group, time)
