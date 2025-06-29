"""
This is the pipelines that:

1. Generates data (customer segmentation problem, 5 features)
2. Split the data into train, eval, test (80%, 10%, 10%), test is used for final evaluation
3. Generate and fit a preprocessor
4. Transforms the data using the preprocessor
5. Saves all the sets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def customer_segmentation_data() -> pd.DataFrame:
    
    return pd.DataFrame({
        "age": np.random.randint(18, 65, 10000),
        "income": np.random.randint(12000, 100000, 10000),
        "spending_score": np.random.randint(0, 100, 10000),
        "loyalty_score": np.random.randint(0, 100, 10000),
        "purchase_frequency": np.random.randint(0, 365, 10000),
    })