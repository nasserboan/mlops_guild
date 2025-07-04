"""
This is a boilerplate pipeline 'model_pipeline'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_models, eval_models, choose_best_model, optimize_best_model, log_optimized_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(train_models, inputs=["x_train_scaled",
                                   "y_train",
                                   "params:model_pipeline"],
                                   outputs="models"),
        node(eval_models, inputs=["models",
                                 "col_selector",
                                 "x_eval_scaled",
                                 "y_eval"],
                                 outputs="eval_results"),
        node(choose_best_model, inputs="eval_results", outputs="best_model_name"),
        node(optimize_best_model, inputs=["x_train_scaled",
                                          "y_train",
                                          "x_eval_scaled",
                                          "y_eval",
                                          "col_selector",
                                          "best_model_name",
                                          "params:model_pipeline"],
                                          outputs="study"),
        node(log_optimized_model, inputs=["study", "x_train_scaled", "params:model_pipeline"], outputs="optimized_model_name"),
    ])
