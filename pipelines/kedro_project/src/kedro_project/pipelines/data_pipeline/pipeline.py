"""
This is a boilerplate pipeline 'data_pipeline'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import generate_data, split_data, preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(generate_data,
             inputs="params:data_pipeline",
             outputs="raw_data",
             name="generate_data"),
        node(split_data,
             inputs=["raw_data", "params:data_pipeline"],
             outputs=["x_train", "x_eval", "x_test", "y_train", "y_eval", "y_test"],
             name="split_data"),
        node(preprocess_data,
             inputs=["x_train", "x_eval", "params:data_pipeline"],
             outputs=["col_selector", "x_train_scaled", "x_eval_scaled"],
             name="preprocess_data"),
    ])
