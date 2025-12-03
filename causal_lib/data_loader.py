import polars as pl
from typing import Optional

def load_nsw(url: str = "") -> Optional[pl.DataFrame]:
    """
    Loads the NSW dataset from NBER text files (treated and control).
    """
    treated_url = "https://users.nber.org/~rdehejia/data/nsw_treated.txt"
    control_url = "https://users.nber.org/~rdehejia/data/nsw_control.txt"
    
    # Column names based on user description
    # treatment, age, education, black, hispanic, married, nodegree, re75, re78
    columns = ["treat", "age", "education", "black", "hispanic", "married", "nodegree", "re75", "re78"]
    
    try:
        print(f"🔄 Fetching NBER data...")
        
        def load_raw_nber(url):
            # Read as single column (hack with non-existent separator)
            # and parse using regex for multiple spaces
            df_raw = pl.read_csv(url, has_header=False, separator="|", new_columns=["raw"])
            
            # Split by whitespace
            # Extract all non-whitespace sequences
            # This returns a list of strings
            df_parsed = df_raw.select(
                pl.col("raw").str.extract_all(r"\S+").alias("parts")
            )
            
            # Convert list to struct to unnest
            # We know there are 9 columns
            # treatment, age, education, black, hispanic, married, nodegree, re75, re78
            cols = ["treat", "age", "education", "black", "hispanic", "married", "nodegree", "re75", "re78"]
            
            df_struct = df_parsed.select(
                [pl.col("parts").list.get(i).alias(cols[i]) for i in range(len(cols))]
            )
            
            return df_struct

        df_treated = load_raw_nber(treated_url)
        df_control = load_raw_nber(control_url)
        
        # Combine
        df = pl.concat([df_treated, df_control])
        
        # Cast columns to appropriate types
        int_cols = ["treat", "age", "education", "black", "hispanic", "married", "nodegree"]
        float_cols = ["re75", "re78"]
        
        df = df.with_columns([
            pl.col(c).cast(pl.Float64).cast(pl.Int64) for c in int_cols # Cast to float first (scientific notation) then int
        ])
        df = df.with_columns([
            pl.col(c).cast(pl.Float64) for c in float_cols
        ])
        
        print(f"✅ Loaded NBER Data: {len(df)} rows ({len(df_treated)} treated, {len(df_control)} control)")
        return df
        
    except Exception as e:
        print(f"⚠️ Failed to load NBER Data: {e}")
        try:
            print("🔄 Attempting to load local 'nsw.csv'...")
            df = pl.read_csv("nsw.csv")
            print("✅ Loaded local NSW Data")
            return df
        except Exception as e_local:
            print(f"❌ Failed to load local data: {e_local}")
            return None

def preprocess_nsw(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pure function to preprocess the NSW dataset.
    Ensures correct data types and handles missing values if any.
    """
    # Example preprocessing: Ensure columns are float where needed
    # NSW Mixtape schema: treat, re78, etc. usually int or float.
    # Polars handles type inference well, but we can enforce it.
    return df.drop_nulls()
