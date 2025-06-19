import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Tuple, Dict, List

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],
                 errors="ignore")
    return df

def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    df.drop(columns=["province"], errors="ignore", inplace=True)
    df["fuel_type"] = df["fuel"]
    df = df[df["fuel_type"].isin(["Gasoline", "Diesel"])].copy()
    df["fuel_encoded"] = df["fuel_type"].map({"Gasoline": 0, "Diesel": 1})
    df = df[df["price"].between(10000, 300000)]
    df = df[df["mileage"].between(2000, 300000)]
    df = df[df["year"].between(1990, 2025)]
    df.drop_duplicates(inplace=True)
    df["generation_name"] = df["generation_name"].fillna("unknown")
    df["generation_name"] = df["generation_name"].str.replace(r"^gen-", "",
                                                              regex=True)
    freq = df["generation_name"].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    df["generation_name_grouped"] = df["generation_name"].where(
        ~df["generation_name"].isin(rare), "other"
    )
    le = LabelEncoder()
    df["generation_name_encoded"] = le.fit_transform(
        df["generation_name_grouped"]
    )
    df.drop(columns=["generation_name_grouped"], inplace=True)
    return df, le

def extract_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    target = df["price"].copy()
    features = df.drop(columns=["price"])
    return features, target

def create_numerical_features(
    df: pd.DataFrame, current_year: int
) -> pd.DataFrame:
    df = df.copy()
    df["age"] = current_year - df["year"]
    df["mileage_per_year"] = df["mileage"] / df["age"].replace(0, np.nan)
    df["log_mileage"] = np.log1p(df["mileage"])
    return df

def scale_features(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    num_cols = ["age", "mileage", "mileage_per_year", "vol_engine", "log_mileage"]
    scaler = StandardScaler().fit(df[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    return df, scaler

def encode_categoricals(
    df: pd.DataFrame,
    top_marks: int = 20,
    top_cities: int = 30
) -> Tuple[pd.DataFrame, Dict[str, float], List[str], List[str]]:
    df = df.copy()
    top_marks_list = df["mark"].value_counts().index[:top_marks].tolist()
    df["mark_group"] = df["mark"].where(df["mark"].isin(top_marks_list),
                                        "other_mark")
    df = pd.get_dummies(df, columns=["mark_group"], prefix="mark")
    top_cities_list = df["city"].value_counts().index[:top_cities].tolist()
    df["city_group"] = df["city"].where(df["city"].isin(top_cities_list),
                                        "other_city")
    df = pd.get_dummies(df, columns=["city_group"], prefix="city")
    model_te_map = df.groupby("model")["price"].mean().to_dict()
    df["model_te"] = df["model"].map(model_te_map)
    df = pd.get_dummies(df, columns=["generation_name_encoded"], prefix="gen")
    return df, model_te_map, top_marks_list, top_cities_list

def feature_engineering(
    df: pd.DataFrame, current_year: int
) -> pd.DataFrame:
    df_num = create_numerical_features(df, current_year)
    df_scl, _ = scale_features(df_num)
    df_enc, *_ = encode_categoricals(df_scl)
    drop_cols = [
        "year", "mileage", "mark", "model", "city", "generation_name",
        "fuel", "fuel_type"
    ]
    return df_enc.drop(columns=drop_cols, errors="ignore")

def save_preprocessors(
    scaler,
    gen_le: LabelEncoder,
    model_te_map: Dict[str,float],
    top_marks: List[str],
    top_cities: List[str],
    filepath: str = "data/06_models/preprocessors.pkl"
):

    artifacts = {
        "scaler": scaler,
        "gen_le": gen_le,
        "model_te_map": model_te_map,
        "top_marks": top_marks,
        "top_cities": top_cities
    }
    joblib.dump(artifacts, filepath)
