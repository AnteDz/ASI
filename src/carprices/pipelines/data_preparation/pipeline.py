from kedro.pipeline import Pipeline, node
from .nodes import load_data, clean_data, feature_engineering

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=load_data,
            inputs="params:csv_path",
            outputs="raw_df",
            name="load_data_node"
        ),
        node(
            func=clean_data,
            inputs="raw_df",
            outputs="clean_df",
            name="clean_data_node"
        ),
        node(
            feature_engineering,
            inputs=["clean_df", "params:current_year"],
            outputs="features_df",
            name="feature_engineering_node"
        ),
    ])
