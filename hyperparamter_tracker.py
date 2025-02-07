import json
import os
import uuid
import datetime
import logging
import joblib
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from concurrent.futures import ThreadPoolExecutor
from model_evaluation import evaluate_model  # Import evaluation function

class HyperparamTracker:
    def __init__(self, log_dir="hyperparam_logs", model_dir="pretrained_models"):
        self.log_dir = log_dir
        self.model_dir = model_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.db_file = os.path.join(self.log_dir, "hyperparameter_history.json")
        self.models = self.load_all_models()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger("HyperparamTracker")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.log_dir, "tracking.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger

    def load_all_models(self):
        models = {}
        for i in range(1, 6):
            model_path = os.path.join(self.model_dir, f"model_{i}.joblib")
            if os.path.exists(model_path):
                models[f"Model_{i}"] = joblib.load(model_path)
            else:
                print(f"Warning: {model_path} not found.")
        return models

    def approve_suggested_hyperparameters(self, model_name, hyperparameters, approval):
        best_hyperparams = self.fetch_hyperparameter_history(model_name=model_name)[-1]["hyperparameters"]
        best_metrics = self.fetch_hyperparameter_history(model_name=model_name)[-1]["metrics"]
        
        metrics = evaluate_model(self.models[model_name], model_name, X_train, Y_train, X_train_balanced, Y_train_balanced, X_test, Y_test)
        
        if metrics["test_rmse"] > best_metrics["test_rmse"]:
            print(f"New hyperparameters performed worse. Rolling back to {best_hyperparams}.")
            return best_hyperparams
        
        if approval:
            self.logger.info(f"Approved new hyperparameters for {model_name}: {hyperparameters}")
            return hyperparameters
        else:
            self.logger.warning(f"Rejected hyperparameters for {model_name}.")
            return None

    def suggest_hyperparameter_tuning(self, model_name, metric="RMSE", max_evals=50, early_stop_rounds=10):
        trials = Trials()
        no_improvement_count = 0
        best_loss = float("inf")

        def objective(params):
            nonlocal no_improvement_count, best_loss
            
            # Handle dataset selection for Model 4
            if model_name == "Model_4":
                X_train_used, Y_train_used = X_train_balanced, Y_train_balanced
            else:
                X_train_used, Y_train_used = X_train, Y_train
            
            loss = evaluate_model(self.models[model_name], model_name, X_train_used, Y_train_used, X_test, Y_test)["test_rmse"]
            
            if loss < best_loss:
                best_loss = loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stop_rounds:
                print("Early stopping triggered. Stopping hyperparameter tuning.")
                return {'loss': best_loss, 'status': STATUS_OK}
            
            return {'loss': loss, 'status': STATUS_OK}

        space = {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        }

        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        best_params['n_estimators'] = int(best_params['n_estimators'])
        
        # Auto-log tuning results
        self.log_hyperparameters(model_name, best_params, {"test_rmse": best_loss})
        self.logger.info(f"Suggested hyperparameters for {model_name}: {best_params}")
        return best_params

    def tune_all_models(self):
        models = ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(self.suggest_hyperparameter_tuning, models)

        for model, result in zip(models, results):
            print(f"Best hyperparameters for {model}: {result}")
