from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_autogluon, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=split_data,
            inputs=["features_df", "price_target"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node"
        ),
        node(
            func=train_autogluon,
            inputs=["X_train", "y_train", "params:time_limit", "params:eval_metric", "params:save_path"],
            outputs="predictor",
            name="train_autogluon_node"
        ),
        node(
            func=evaluate_model,
            inputs=["predictor", "X_test", "y_test"],
            outputs="model_metrics",
            name="evaluate_model_node"
        )
    ])