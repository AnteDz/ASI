from kedro.pipeline import Pipeline, node
from .nodes import (
    load_data,
    clean_data,
    feature_engineering,
    split_data,
    train_autogluon
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Pipeline dla przetwarzania danych i treningu modelu:
    1. load_data           -> raw_df
    2. clean_data          -> clean_df
    3. feature_engineering -> features_df
    4. split_data          -> X_train, X_test, y_train, y_test
    5. train_autogluon     -> ag_model

    Parametry w conf/base/parameters.yml:
      - csv_path
      - current_year
      - test_size
      - random_state
      - ag_label
      - ag_output_path
    """
    return Pipeline([
        node(
            func=load_data,
            inputs="params:csv_path",
            outputs="raw_df",
            name="load_data_node",
        ),
        node(
            func=clean_data,
            inputs="raw_df",
            outputs="clean_df",
            name="clean_data_node",
        ),
        node(
            func=feature_engineering,
            inputs=["clean_df", "params:current_year"],
            outputs="features_df",
            name="feature_engineering_node",
        ),
        node(
            func=split_data,
            inputs=[
                "features_df",
                "params:ag_label",
                "params:test_size",
                "params:random_state",
            ],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        node(
            func=train_autogluon,
            inputs=["features_df", "params:ag_label", "params:ag_output_path"],
            outputs="ag_model",
            name="train_autogluon_node",
        ),
    ])
