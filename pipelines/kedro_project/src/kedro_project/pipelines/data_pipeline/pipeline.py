"""
This is a boilerplate pipeline 'data_pipeline'
generated using Kedro 0.19.14
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import customer_segmentation_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(customer_segmentation_data, None, "raw_data")
    ])
