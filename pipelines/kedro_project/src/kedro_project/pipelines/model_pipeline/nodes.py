"""
This is the pipelines that:

1. Train 2 models:
    - Linear Regression
    - LightGBM
    - ANN (PyTorch)
2. Evaluate the models
3. Choose the best model
4. Optimize the best model (with optuna and mlflow)
5. Logs experiments into mlflow
6. Logs the optimized model into mlflow
"""

import logging
import pandas as pd
import mlflow
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.compose import ColumnTransformer
from datetime import datetime
import optuna

logger = logging.getLogger(__name__)

def train_models(x_train: pd.DataFrame, 
                 y_train: pd.DataFrame,
                 params: dict):
    
    mlflow.set_tracking_uri(params["mlflow_params"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow_params"]["experiment_name"])
    
    with mlflow.start_run(run_name="baseline_linear_regression"):
        mlflow.set_tag("model_name", "lr_model")
        mlflow.set_tag(f"baseline_{datetime.now().strftime('%Y-%m-%d')}")   
        logger.info("Traninig Baseline Linear Regression")
        lr_model = LinearRegression(**params["baseline_model_params"]["linear_regression"])
        lr_model.fit(x_train, y_train)
        mlflow.sklearn.log_model(lr_model, "baseline_linear_regression")

    with mlflow.start_run(run_name="baseline_lightgbm"):
        mlflow.set_tag("model_name", "lgbm_model")
        mlflow.set_tag(f"baseline_{datetime.now().strftime('%Y-%m-%d')}")
        logger.info("Training Baseline LightGBM")
        lgbm_model = LGBMRegressor(**params["baseline_model_params"]["lightgbm"])
        lgbm_model.fit(x_train, y_train)
        mlflow.sklearn.log_model(lgbm_model, "baseline_lightgbm")
    
    with mlflow.start_run(run_name="baseline_ann"):
        mlflow.set_tag("model_name", "ann_model")
        mlflow.set_tag(f"baseline_{datetime.now().strftime('%Y-%m-%d')}")
        model_params = params["baseline_model_params"]["ann"]
        batch_size = model_params["batch_size"]
        epochs = model_params["epochs"]
        learning_rate = model_params["learning_rate"]
        hidden_size = model_params["hidden_size"]
        num_layers = model_params["num_layers"]

        x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        dataset = TensorDataset(x_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        ann_model = nn.Sequential()

        ann_model.add_module("input", nn.Linear(x_train.shape[1], hidden_size))
        
        for n in range(num_layers):
            ann_model.add_module(f"layer_{n}", nn.Linear(hidden_size, hidden_size))
            ann_model.add_module(f"relu_{n}", nn.ReLU())
        ann_model.add_module("output", nn.Linear(hidden_size, 1))
        
        optimizer = Adam(ann_model.parameters(), lr=learning_rate)
        
        logger.info("Trainig Baseline ANN")
        for epoch in range(epochs):
            for x_batch, y_batch in dataloader:
                optimizer.zero_grad()
                y_pred = ann_model(x_batch)
                loss = nn.MSELoss()(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        mlflow.pytorch.log_model(ann_model, "baseline_ann")

    return {"lr_model": lr_model, "lgbm_model": lgbm_model, "ann_model": ann_model}

def eval_models(
        models: dict,
        col_selector: ColumnTransformer,
        x_eval: pd.DataFrame,
        y_eval: pd.DataFrame):
    
    eval_results = {}
    for model_name, model in models.items():
        x_eval_scaled = col_selector.transform(x_eval)
        y_pred = model.predict(x_eval_scaled)
        eval_results[model_name] = {
            "rmse": root_mean_squared_error(y_eval, y_pred),
        }
        logger.info(f"{model_name} evaluate successfully, RMSE: {eval_results[model_name]['rmse']}")
    return eval_results

def choose_best_model(eval_results: dict):
    best_model_name = min(eval_results, key=eval_results.get)
    logger.info(f"Best model chosen - {best_model_name}")
    return {"best_model_name": best_model_name}


def objective_function(trial, x_train, y_train, x_eval, y_eval, col_selector, model_name):

    if model_name == "lr_model":
        with mlflow.start_run(run_name=f"lr_model_{trial.number}"):
            mlflow.set_tag("model_name", "lr_model")
            mlflow.set_tag(f"optimization_{datetime.now().strftime('%Y-%m-%d')}")
            max_iter = trial.suggest_float("max_iter", 100, 1000, step=100)
            tol = trial.suggest_float("tol", 1e-4, 1e-2, step=1e-3)
            model = LinearRegression(max_iter=max_iter, tol=tol)
            model.fit(x_train, y_train)
            x_eval_scaled = col_selector.transform(x_eval)
            y_pred = model.predict(x_eval_scaled)
            rmse = root_mean_squared_error(y_eval, y_pred)
            mlflow.log_params(trial.params)
            mlflow.log_metric("rmse", rmse)
            return rmse
    elif model_name == "lgbm_model":
        with mlflow.start_run(run_name=f"lgbm_model_{trial.number}"):
            n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, step=0.01)
            max_depth = trial.suggest_int("max_depth", 10, 100, step=2)
            model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            model.fit(x_train, y_train)
            x_eval_scaled = col_selector.transform(x_eval)
            y_pred = model.predict(x_eval_scaled)
            rmse = root_mean_squared_error(y_eval, y_pred)
            mlflow.log_params(trial.params)
            mlflow.log_metric("rmse", rmse)
            return rmse
    elif model_name == "ann_model":
        with mlflow.start_run(run_name=f"ann_model_{trial.number}"):
            hidden_size = trial.suggest_int("hidden_size", 16, 64, step=2)
            num_layers = trial.suggest_int("num_layers", 2, 10, step=1)
            learning_rate = trial.suggest_float("learning_rate", 0.001, 0.01, step=0.001)
            batch_size = trial.suggest_int("batch_size", 16, 64, step=2)
            epochs = trial.suggest_int("epochs", 100, 1000, step=100)
            
            ## model architecture
            model = nn.Sequential()
            model.add_module("input", nn.Linear(x_train.shape[1], hidden_size))
            for n in range(num_layers):
                model.add_module(f"layer_{n}", nn.Linear(hidden_size, hidden_size))
                model.add_module(f"relu_{n}", nn.ReLU())
            model.add_module("output", nn.Linear(hidden_size, 1))
            
            ## optimizer
            optimizer = Adam(model.parameters(), lr=learning_rate)

            ## loss function
            loss_fn = nn.MSELoss()

            ## training
            x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
            dataset = TensorDataset(x_train_tensor, y_train_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                for x_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    loss.backward()
                    optimizer.step()

            x_eval_scaled = col_selector.transform(x_eval)
            y_pred = model(x_eval_scaled)
            rmse = root_mean_squared_error(y_eval, y_pred)
            mlflow.log_params(trial.params)
            mlflow.log_metric("rmse", rmse)
            return rmse

def optimize_best_model(x_train: pd.DataFrame,
                        y_train: pd.DataFrame,
                        x_eval: pd.DataFrame,
                        y_eval: pd.DataFrame,
                        col_selector: ColumnTransformer,
                        best_model_name: str,
                        params: dict):
    
    opt_params = params["optimize_params"]
    study_name = opt_params["study_name"]
    direction = opt_params["direction"]
    n_trials = opt_params["n_trials"]
    best_model_name = best_model_name["best_model_name"]

    study = optuna.create_study(study_name=study_name, direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective_function(trial, x_train, y_train, x_eval, y_eval, col_selector, best_model_name), n_trials=n_trials)

    return study

def log_optimized_model(study: optuna.Study):
    best_trial = study.best_trial
    best_params = best_trial.params
    best_model_name = best_trial.study.best_params["model_name"]
    if best_model_name == "lr_model":
        best_model = LinearRegression(**best_params)
    elif best_model_name == "lgbm_model":
        best_model = LGBMRegressor(**best_params)
    elif best_model_name == "ann_model":
        best_model = nn.Sequential()
        best_model.add_module("input", nn.Linear(x_train.shape[1], best_params["hidden_size"]))
        for n in range(best_params["num_layers"]):
            best_model.add_module(f"layer_{n}", nn.Linear(best_params["hidden_size"], best_params["hidden_size"]))
            best_model.add_module(f"relu_{n}", nn.ReLU())
        best_model.add_module("output", nn.Linear(best_params["hidden_size"], 1))
        best_model.load_state_dict(best_trial.state_dict)
    else:
        raise ValueError(f"Model {best_model_name} not supported")
    
    mlflow.set_tag("model_name", best_model_name)
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", best_trial.value)
    model_info = mlflow.sklearn.log_model(best_model, "optimized_model")
    logger.info(f"Optimized {best_model_name} w/ best params logged into mlflow: {model_info.model_uri}")
    return best_model_name
