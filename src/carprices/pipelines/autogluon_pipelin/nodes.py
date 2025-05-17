import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor

def split_data(features_df: pd.DataFrame,
               ag_label: str,
               test_size: float,
               random_state: int):
    X = features_df.drop(columns=[ag_label], errors="ignore")
    y = features_df[ag_label]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_autogluon(features_df: pd.DataFrame,
                    ag_label: str,
                    ag_output_path: str):
    predictor = TabularPredictor(label=ag_label, path=ag_output_path).fit(
        train_data=features_df,
        time_limit=600,
        presets="best_quality"
    )
    return predictor
