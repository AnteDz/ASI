# src/carprices/pipeline_registry.py
from kedro.pipeline import Pipeline
from .pipelines.data_preparation.pipeline import create_pipeline as prep_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    data_prep: Pipeline = prep_pipeline()
    return {
        "__default__": data_prep,
        "data_preparation": data_prep,
    }
