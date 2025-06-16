import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.drop(columns=["province"], errors="ignore", inplace=True)
    df['fuel_type'] = df['fuel']
    df = df[df['fuel_type'].isin(['Gasoline', 'Diesel'])].copy()
    df['fuel_encoded'] = df['fuel_type'].map({'Gasoline': 0, 'Diesel': 1})
    df = df[df['price'].between(10000, 300000)]
    df = df[df['mileage'].between(2000, 300000)]
    df = df[df['year'].between(1990, 2025)]
    df.drop_duplicates(inplace=True)
    df['generation_name'] = df['generation_name'].fillna('unknown')
    df['generation_name'] = df['generation_name'].str.replace(r'^gen-', '', regex=True)
    freq = df['generation_name'].value_counts(normalize=True)
    rare = freq[freq < 0.01].index
    df['generation_name_grouped'] = df['generation_name'].where(~df['generation_name'].isin(rare), 'other')
    le = LabelEncoder()
    df['generation_name_encoded'] = le.fit_transform(df['generation_name_grouped'])
    df.drop(columns=['generation_name_grouped'], inplace=True)
    return df


def extract_target(df: pd.DataFrame):
    target = df[['price']].copy()
    features = df.drop(columns=['price'])
    return features, target


def create_numerical_features(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    df = df.copy()
    df['age'] = current_year - df['year']
    df['mileage_per_year'] = df['mileage'] / df['age'].replace(0, np.nan)
    df['log_mileage'] = np.log1p(df['mileage'])
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    num_cols = ['age', 'mileage', 'mileage_per_year', 'vol_engine', 'log_mileage']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def encode_categoricals(
    df: pd.DataFrame,
    top_marks: int = 20,
    top_cities: int = 30,
    n_splits: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    df = df.copy()
    top_marks_list = df['mark'].value_counts().index[:top_marks]
    df['mark_group'] = df['mark'].where(df['mark'].isin(top_marks_list), 'other_mark')
    df = pd.get_dummies(df, columns=['mark_group'], prefix='mark')
    top_cities_list = df['city'].value_counts().index[:top_cities]
    df['city_group'] = df['city'].where(df['city'].isin(top_cities_list), 'other_city')
    df = pd.get_dummies(df, columns=['city_group'], prefix='city')
    df['model_te'] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, valid_idx in kf.split(df):
        train, valid = df.iloc[train_idx], df.iloc[valid_idx]
        means = train.groupby('model')['price'].mean()
        df.loc[df.index[valid_idx], 'model_te'] = df.loc[df.index[valid_idx], 'model'].map(means)
    df['model_te'].fillna(df['price'].mean(), inplace=True)
    df = pd.get_dummies(df, columns=['generation_name_encoded'], prefix='gen')
    return df


def feature_engineering(df: pd.DataFrame, current_year: int) -> pd.DataFrame:
    df = create_numerical_features(df, current_year)
    df = scale_features(df)
    df = encode_categoricals(df)
    drop_cols = ['year', 'mileage', 'mark', 'model', 'city', 'generation_name', 'fuel', 'fuel_type']
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')