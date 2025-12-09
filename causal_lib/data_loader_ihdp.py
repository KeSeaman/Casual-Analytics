import polars as pl
from typing import Optional

def load_ihdp(csv_path: str = "ihdp.csv") -> Optional[pl.DataFrame]:
    """
    Loads the IHDP (Infant Health Development Program) dataset.
    
    Schema:
    - treatment: Binary (0/1) indicating intensive intervention
    - outcome (y_factual): Continuous cognitive test score
    - x1-x25: 25 pre-treatment covariates (6 continuous, 19 binary)
    
    Args:
        csv_path: Path to IHDP CSV file (can be URL or local path)
    
    Returns:
        Polars DataFrame or None if loading fails
    """
    try:
        print(f"🔄 Loading IHDP data from: {csv_path}")
        df = pl.read_csv(csv_path)
        
        # Verify expected columns exist
        if "treatment" in df.columns and ("y_factual" in df.columns or "outcome" in df.columns):
            # Rename y_factual to outcome if needed for consistency
            if "y_factual" in df.columns and "outcome" not in df.columns:
                df = df.rename({"y_factual": "outcome"})
            
            print(f"✅ Loaded IHDP Data: {len(df)} observations")
            print(f"   - Treatment: {df['treatment'].sum()} treated, {len(df) - df['treatment'].sum()} control")
            return df
        else:
            print(f"⚠️ IHDP schema not recognized. Expected 'treatment' and 'y_factual'/'outcome' columns.")
            return None
            
    except Exception as e:
        print(f"❌ Failed to load IHDP Data: {e}")
        return None

def preprocess_ihdp(df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocesses the IHDP dataset.
    Ensures correct data types and handles missing values.
    """
    # Drop nulls if any
    df = df.drop_nulls()
    
    # Ensure treatment is integer (0/1)
    if "treatment" in df.columns:
        df = df.with_columns(pl.col("treatment").cast(pl.Int64))
    
    # Ensure outcome is float
    if "outcome" in df.columns:
        df = df.with_columns(pl.col("outcome").cast(pl.Float64))
    
    return df
