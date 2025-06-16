import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_data(features_df: pd.DataFrame,
               price_target: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42):
    """
    Dzieli dane na zbiór treningowy i testowy (hold-out).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features_df,
        price_target,
        test_size=test_size,
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_final_ensemble(X_train: pd.DataFrame,
                         y_train: pd.Series,
                         time_limit: int = 600,
                         eval_metric: str = 'rmse',
                         save_path: str = 'data/07_model_output/car_price_predictor_final') -> TabularPredictor:
    """
    Trenuje WeightedEnsemble (bagging + stacking) na danych treningowych,
    zapisuje predictor pod wskazaną ścieżką.
    """
    train_data = X_train.copy()
    train_data['price'] = y_train.values

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
        num_bag_folds=5,
        num_bag_sets=2,
        num_stack_levels=1,
        excluded_model_types=['NN'],
        refit_full=True
    )
    return predictor


def evaluate_final(predictor: TabularPredictor,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> pd.DataFrame:
    """
    Ewaluacja finalnego predictora na zbiorze testowym.
    Zwraca DataFrame z metrykami MAE, RMSE i R2.
    """
    y_pred = predictor.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2   = r2_score(y_test, y_pred)

    metrics = pd.DataFrame({
        'metric': ['MAE', 'RMSE', 'R2'],
        'value':  [mae, rmse, r2]
    })
    return metrics