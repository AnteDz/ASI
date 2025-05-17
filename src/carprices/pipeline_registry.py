# src/carprices/pipeline_registry.py
from kedro.pipeline import Pipeline
from .pipelines.data_preparation.pipeline import create_pipeline as dp_pipeline
from .pipelines.autogluon_pipelin.pipeline import create_pipeline as ag_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    data_prep: Pipeline = dp_pipeline()
    autogluon: Pipeline = ag_pipeline()
    return {
        "__default__": data_prep,              # domyślna
        "data_preparation": data_prep,
        "autogluon_pipeline": autogluon,
        "all": data_prep + autogluon,          # opcjonalnie oba na raz
    }
