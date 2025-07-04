"""
This is the pipelines that:

1. Train 3 models:
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
from datetime import datetime

import mlflow
import optuna
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from mlflow.tracking import MlflowClient


logger = logging.getLogger(__name__)

def train_models(x_train: pd.DataFrame, 
                 y_train: pd.DataFrame,
                 params: dict):

    mlflow.set_tracking_uri(params["mlflow_params"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow_params"]["experiment_name"])

    with mlflow.start_run(run_name="baseline_svr", nested=True):
        mlflow.set_tag("model_name", "svr_model")
        mlflow.set_tag("baseline_optimization_date", datetime.now().strftime('%Y-%m-%d'))
        logger.info("Training Baseline SVR")
        svr_model = SVR(**params["baseline_model_params"]["svr"])
        svr_model.fit(x_train, y_train)
        mlflow.sklearn.log_model(svr_model, "baseline_svr")

    with mlflow.start_run(run_name="baseline_lightgbm", nested=True):
        mlflow.set_tag("model_name", "lgbm_model")
        mlflow.set_tag("baseline_optimization_date", datetime.now().strftime('%Y-%m-%d'))
        logger.info("Training Baseline LightGBM")
        lgbm_model = LGBMRegressor(**params["baseline_model_params"]["lightgbm"])
        lgbm_model.fit(x_train, y_train)
        mlflow.sklearn.log_model(lgbm_model, "baseline_lightgbm")
    
    with mlflow.start_run(run_name="baseline_ann", nested=True):
        mlflow.set_tag("model_name", "ann_model")
        mlflow.set_tag("baseline_optimization_date", datetime.now().strftime('%Y-%m-%d'))
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

    return {"svr_model": svr_model, "lgbm_model": lgbm_model, "ann_model": ann_model}

def eval_models(
    models: dict,
    col_selector: ColumnTransformer,
    x_eval: pd.DataFrame,
    y_eval: pd.DataFrame
):
    import torch
    eval_results = {}
    for model_name, model in models.items():
        x_eval_scaled = col_selector.transform(x_eval)
        if hasattr(model, "predict"):  # scikit-learn
            y_pred = model.predict(x_eval_scaled)
        else:  # PyTorch
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(x_eval_scaled, dtype=torch.float32)
                y_pred = model(x_tensor).cpu().numpy().flatten()
        eval_results[model_name] = {
            "rmse": root_mean_squared_error(y_eval, y_pred),
        }
        logger.info(f"{model_name} evaluate successfully, RMSE: {eval_results[model_name]['rmse']}")
    return eval_results

def choose_best_model(eval_results: dict):
    best_model_name = min(eval_results, key=lambda k: eval_results[k]["rmse"])
    logger.info(f"Best model chosen - {best_model_name}")
    return {"best_model_name": best_model_name}


def objective_function(trial,
                       x_train,
                       y_train,
                       x_eval,
                       y_eval,
                       col_selector,
                       model_name,
                       params):

    mlflow.set_tracking_uri(params["mlflow_params"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow_params"]["experiment_name"])

    try:
        if model_name == "svr_model":
            kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
            svr_params = {"kernel": kernel}
            if kernel == "poly":
                svr_params["degree"] = trial.suggest_int("degree", 2, 5)
            with mlflow.start_run(run_name=f"svr_model_{trial.number}") as run:
                # Tags
                mlflow.set_tag("model_name", "svr_model")
                mlflow.set_tag("optimization_date", datetime.now().strftime('%Y-%m-%d'))
                mlflow.set_tag("trial_number", trial.number)
                mlflow.set_tag("pipeline_step", "optimize_best_model")
                mlflow.set_tag("experiment_name", params["mlflow_params"]["experiment_name"])
                mlflow.set_tag("author", "nsboan")
                # Parâmetros
                mlflow.log_params(svr_params)
                # Treinamento e logging
                model = SVR(**svr_params)
                model.fit(x_train, y_train)
                mlflow.sklearn.log_model(model, "model")
                # Métricas
                x_eval_scaled = col_selector.transform(x_eval)
                y_pred = model.predict(x_eval_scaled)
                rmse = root_mean_squared_error(y_eval, y_pred)
                mlflow.log_metric("rmse", rmse)
                trial.set_user_attr("mlflow_run_id", run.info.run_id)
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
                x_eval_tensor = torch.tensor(x_eval_scaled, dtype=torch.float32)
                y_pred = model(x_eval_tensor).cpu().numpy().flatten()
                rmse = root_mean_squared_error(y_eval, y_pred)
                mlflow.log_params(trial.params)
                mlflow.log_metric("rmse", rmse)
                trial.set_user_attr("mlflow_run_id", mlflow.last_active_run().info.run_id)
                return rmse
        else:
            raise ValueError(f"Unknown model_name: {model_name}")
    except Exception as e:
        print(f"Trial failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None  # Explicitly return None so Optuna knows it failed

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
    study.optimize(lambda trial: objective_function(trial, x_train, y_train, x_eval, y_eval, col_selector, best_model_name, params), n_trials=n_trials)

    return study

def log_optimized_model(study: optuna.Study, x_train: pd.DataFrame, params: dict):
    best_trial = study.best_trial
    best_params = best_trial.params
    
    # Determinar o tipo de modelo baseado nos parâmetros
    if "kernel" in best_params:
        best_model_name = "svr_model"
        best_model = SVR(**best_params)
    elif "n_estimators" in best_params:
        best_model_name = "lgbm_model"
        best_model = LGBMRegressor(**best_params)
    elif "hidden_size" in best_params:
        best_model_name = "ann_model"
        best_model = nn.Sequential()
        best_model.add_module("input", nn.Linear(x_train.shape[1], best_params["hidden_size"]))
        for n in range(best_params["num_layers"]):
            best_model.add_module(f"layer_{n}", nn.Linear(best_params["hidden_size"], best_params["hidden_size"]))
            best_model.add_module(f"relu_{n}", nn.ReLU())
        best_model.add_module("output", nn.Linear(best_params["hidden_size"], 1))
        # Note: Para ANN, você precisaria treinar o modelo novamente com os melhores parâmetros
    else:
        raise ValueError(f"Could not determine model type from parameters: {best_params}")

    mlflow.set_tracking_uri(params["mlflow_params"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow_params"]["experiment_name"])

    mlflow.set_tag("model_name", best_model_name)
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", best_trial.value)
    model_info = mlflow.sklearn.log_model(best_model, "optimized_model")
    logger.info(f"Optimized {best_model_name} w/ best params logged into mlflow: {model_info.model_uri}")

    # Supondo que você já logou o modelo como "model" no run do melhor trial
    client = MlflowClient()
    run_id = best_trial.user_attrs["mlflow_run_id"]
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, "revenue_prediction_best_model")
    logger.info(f"Registered optimized model into mlflow: {result.name}")

    return best_model_name
