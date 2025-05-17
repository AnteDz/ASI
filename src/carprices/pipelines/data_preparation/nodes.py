import pandas as pd
import numpy as np

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],
                 errors="ignore")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    if "generation_name" in df.columns:
        df["generation_name"] = df["generation_name"].fillna("unknown")
    df = df[
        (df["year"] > 1900)
        & (df["mileage"] >= 0)
        & (df["vol_engine"] > 0)
        & (df["price"] > 0)
    ]
    return df

def feature_engineering(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    df = df.copy()
    df["age"] = current_year - df["year"]
    df["log_mileage"] = np.log1p(df["mileage"])
    cat_cols = [c for c in ["mark","model","generation_name","fuel","province"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df.drop(columns=["year","mileage"], errors="ignore")
