"""
This is the pipelines that:

1. Generates data (customer segmentation problem, 5 features)
2. Split the data into train, eval, test (80%, 10%, 10%), test is used for final evaluation
3. Generate and fit a preprocessor
4. Transforms the data using the preprocessor
"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def generate_data(params: dict) -> pd.DataFrame:
    n_samples = params["n_samples"]

    data = pd.DataFrame({
        "age": np.random.randint(18, 65, n_samples),
        "income": np.random.randint(12000, 100000, n_samples),
        "spending_score": np.random.randint(0, 100, n_samples),
        "loyalty_score": np.random.randint(0, 100, n_samples),
        "purchase_frequency": np.random.randint(0, 365, n_samples),
        "revenue": np.random.randint(0, 100000, n_samples),
    })

    return data

def split_data(data: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = params["target"]
    train_size = params["train_size"]
    eval_size = params["eval_size"]
    test_size = params["test_size"]
    random_state = params["random_state"]

    x_train, x_test, y_train, y_test = train_test_split(data.drop(target, axis=1),
                                                        data[target],
                                                        train_size=train_size,
                                                        test_size=test_size,
                                                        random_state=random_state)
    x_train, x_eval, y_train, y_eval = train_test_split(x_train,
                                                        y_train,
                                                        train_size=train_size,
                                                        test_size=eval_size,
                                                        random_state=random_state)
    
    logger.info(f"x_train shape: {x_train.shape}")
    logger.info(f"x_eval shape: {x_eval.shape}")
    logger.info(f"x_test shape: {x_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_eval shape: {y_eval.shape}")
    logger.info(f"y_test shape: {y_test.shape}")

    return x_train, x_eval, x_test, y_train, y_eval, y_test

def preprocess_data(x_train: pd.DataFrame, 
                    x_eval: pd.DataFrame, 
                    params: dict) -> tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame]:

    pre_params = params["preprocessor_params"]
    process_cols = pre_params["process_cols"]
    standard_scaler = pre_params["standard_scaler"]

    col_selector = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(**standard_scaler), process_cols),
        ]
    )

    x_train_scaled = col_selector.fit_transform(x_train)
    x_eval_scaled = col_selector.transform(x_eval)

    return col_selector, x_train_scaled, x_eval_scaled