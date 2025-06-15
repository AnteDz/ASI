import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    # Usuwamy dowolne kolumny zaczynające się od "Unnamed"
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")],
                 errors="ignore")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oczyszcza dane samochodów:
      - Usuwa kolumny 'Unnamed: 0' i 'province'
      - Filtruje tylko rekordy z paliwem Gasoline i Diesel, mapuje na 0/1
      - Usuwa outliery cen i przebiegów
      - Filtruje lata między 1990 a 2025
      - Usuwa duplikaty
      - Przetwarza 'generation_name':
          * wypełnia NaN jako 'unknown'
          * usuwa prefix 'gen-'
          * grupuje rzadkie generacje (<1% próby) jako 'other'
          * koduje label encodingiem
    """
    df = df.copy()

    # 1) Drop unwanted columns
    df.drop(columns=["province"], errors="ignore", inplace=True)

    # 2) Keep original fuel for frontend, then filter & encode
    df['fuel_type'] = df['fuel']  # kopiujemy oryginał
    df = df[df['fuel_type'].isin(['Gasoline', 'Diesel'])].copy()
    df['fuel_encoded'] = df['fuel_type'].map({  # kolumna numeryczna do modelu
        'Gasoline': 0,
        'Diesel': 1
    })

    # 3) Filter price and mileage outliers
    df = df[df['price'].between(10000, 300000)]
    df = df[df['mileage'].between(2000, 300000)]

    # 4) Filter production year
    df = df[df['year'].between(1990, 2025)]

    # 5) Drop exact duplicates
    df.drop_duplicates(inplace=True)

    # 6) Process generation_name
    df['generation_name'] = df['generation_name'].fillna('unknown')
    df['generation_name'] = df['generation_name'].str.replace(r'^gen-', '', regex=True)

    # 6a) Group rare categories (<1%)
    freq = df['generation_name'].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    df['generation_name_grouped'] = df['generation_name'].where(
        ~df['generation_name'].isin(rare), 'other'
    )

    # 6b) Label encode grouped generations
    le = LabelEncoder()
    df['generation_name_encoded'] = le.fit_transform(df['generation_name_grouped'])

    # 6c) (opcjonalnie) Zapisz mapping do frontend
    # mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    # pd.Series(mapping).to_csv("data/02_intermediate/generation_mapping.csv")

    # 6d) Drop intermediate column
    df.drop(columns=['generation_name_grouped'], inplace=True)

    return df


def feature_engineering(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    df = df.copy()
    df["age"] = current_year - df["year"]
    df["log_mileage"] = np.log1p(df["mileage"])
    cat_cols = [c for c in ["mark","model","generation_name","fuel","province"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df.drop(columns=["year","mileage"], errors="ignore")
