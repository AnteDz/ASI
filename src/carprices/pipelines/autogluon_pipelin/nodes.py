import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_data(features_df: pd.DataFrame,
               price_target: pd.DataFrame,
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Dzieli dane na treningowe i testowe.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features_df,
        price_target['price'],
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_autogluon(X_train: pd.DataFrame,
                    y_train: pd.Series,
                    time_limit: int = 600,
                    eval_metric: str = 'rmse',
                    save_path: str = "data/06_models/car_price_predictor") -> TabularPredictor:
    """
    Trenuje model AutoGluon używając tylko LightGBM, CatBoost i XGBoost,
    wyłączając sieci NN poprzez parametr excluded_model_types.
    """
    train_data = X_train.copy()
    train_data['price'] = y_train

    predictor = TabularPredictor(
        label='price',
        eval_metric=eval_metric,
        path=save_path
    ).fit(
        train_data=train_data,
        time_limit=time_limit,
        hyperparameters={
            'GBM': {},
            'CAT': {},
            'XGB': {}
        },
        excluded_model_types=['NN'],  # poprawny parametr
    )
    return predictor


def evaluate_model(predictor: TabularPredictor,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> pd.DataFrame:
    """
    Oblicza MAE, RMSE i R2 na danych testowych.
    """
    y_pred = predictor.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    metrics = pd.DataFrame({
        'metric': ['MAE', 'RMSE', 'R2'],
        'value': [mae, rmse, r2]
    })
    return metrics