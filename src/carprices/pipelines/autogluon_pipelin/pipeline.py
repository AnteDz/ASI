from kedro.pipeline import Pipeline, node
from .nodes import split_data, train_autogluon

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(split_data,
             inputs=["features_df",
                     "params:ag_label",
                     "params:test_size",
                     "params:random_state"],
             outputs=["X_train", "X_test", "y_train", "y_test"],
             name="split_data_node"),
        node(train_autogluon,
             inputs=["features_df",
                     "params:ag_label",
                     "params:ag_output_path"],
             outputs="ag_model",
             name="train_autogluon_node"),
    ])
