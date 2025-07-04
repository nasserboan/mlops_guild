#!/usr/bin/env python3

import mlflow
import os

def test_mlflow_connection():
    """Testa a conexão com o MLflow"""
    try:
        # Configurar o MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("test_connection")
        
        print("✅ Conectado ao MLflow com sucesso!")
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment: {mlflow.get_experiment_by_name('test_connection')}")
        
        # Tentar criar um run
        with mlflow.start_run(run_name="test_run"):
            mlflow.log_param("test_param", "test_value")
            mlflow.log_metric("test_metric", 0.5)
            print("✅ Run criado com sucesso!")
            
    except Exception as e:
        print(f"❌ Erro ao conectar ao MLflow: {e}")
        print(f"Tipo do erro: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mlflow_connection() 