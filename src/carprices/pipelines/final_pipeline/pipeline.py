from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_final_ensemble, evaluate_final


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=split_data,
            inputs=["features_df", "price_target"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node"
        ),
        node(
            func=train_final_ensemble,
            inputs=["X_train", "y_train", "params:time_limit", "params:eval_metric", "params:save_path_final"],
            outputs="predictor_final",
            name="train_final_ensemble_node"
        ),
        node(
            func=evaluate_final,
            inputs=["predictor_final", "X_test", "y_test"],
            outputs="final_metrics",
            name="evaluate_final_node"
        ),
        node(
            lambda df: df,
            inputs="final_metrics",
            outputs="final_model_metrics",
            name="save_final_metrics_node"
        )
    ])