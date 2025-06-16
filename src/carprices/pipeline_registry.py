# src/carprices/pipeline_registry.py
from kedro.pipeline import Pipeline
from .pipelines.data_preparation.pipeline import create_pipeline as dp_pipeline
from .pipelines.autogluon_pipelin.pipeline import create_pipeline as ag_pipeline
from .pipelines.final_pipeline.pipeline import create_pipeline as final_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    data_prep: Pipeline = dp_pipeline()
    autogluon: Pipeline = ag_pipeline()
    final_ens: Pipeline = final_pipeline()
    return {
        "__default__": data_prep + autogluon + final_ens,
        "data_preparation": data_prep,
        "autogluon_pipeline": autogluon,
        "final_pipeline": final_ens,
        "all": data_prep + autogluon + final_ens,
    }
