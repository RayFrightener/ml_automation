import json
import os
import uuid
import datetime
import logging
import joblib
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from model_evaluation import evaluate_model  # Import evaluation function

class HyperparamTracker:
    """
    Hyperparameter Tracking System:
    - Logs hyperparameter configurations and performance metrics.
    - Retrieves past hyperparameter runs.
    - Uses Bayesian optimization (Hyperopt) to suggest new configurations.
    - Provides human approval for AI-generated suggestions.
    - Loads and tracks pre-trained models.
    """

    def __init__(self, log_dir="hyperparam_logs", model_dir="pretrained_models"):
        """Initialize the hyperparameter tracker with a logging directory and model directory."""
        self.log_dir = log_dir
        self.model_dir = model_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.db_file = os.path.join(self.log_dir, "hyperparameter_history.json")
        self.models = self.load_all_models()
        
        # Initialize storage if not available
        if not os.path.exists(self.db_file):
            with open(self.db_file, "w") as f:
                json.dump([], f)

        # Setup logging
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """Configure logging system."""
        logger = logging.getLogger("HyperparamTracker")
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(os.path.join(self.log_dir, "tracking.log"))
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        return logger

    def load_all_models(self):
        """Loads all five pre-trained XGBoost models from disk."""
        models = {}
        for i in range(1, 6):
            model_path = os.path.join(self.model_dir, f"model_{i}.joblib")
            if os.path.exists(model_path):
                models[f"Model_{i}"] = joblib.load(model_path)
            else:
                print(f"Warning: {model_path} not found.")
        return models

    def log_hyperparameters(self, model_name, hyperparameters, metrics):
        """Logs hyperparameter configurations and their performance metrics."""
        entry = {
            "run_id": str(uuid.uuid4()),
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Load existing data
        with open(self.db_file, "r") as f:
            data = json.load(f)

        data.append(entry)

        # Save updated log
        with open(self.db_file, "w") as f:
            json.dump(data, f, indent=4)

        self.logger.info(f"Logged hyperparameters for model {model_name}: {hyperparameters}")
        self.logger.info(f"Metrics: {metrics}")

    def fetch_hyperparameter_history(self, model_name=None, start_date=None, end_date=None, filters=None):
        """Retrieves past hyperparameter runs based on filters."""
        with open(self.db_file, "r") as f:
            data = json.load(f)

        if model_name:
            data = [entry for entry in data if entry["model_name"] == model_name]
        if start_date:
            data = [entry for entry in data if entry["timestamp"] >= start_date]
        if end_date:
            data = [entry for entry in data if entry["timestamp"] <= end_date]
        if filters:
            for key, value in filters.items():
                data = [entry for entry in data if entry["hyperparameters"].get(key) == value]

        return data

    def compare_hyperparameter_performance(self, model_name, metric):
        """Ranks historical runs based on a selected performance metric."""
        data = self.fetch_hyperparameter_history(model_name=model_name)
        if not data:
            return None

        df = pd.json_normalize(data)  # Flatten JSON structure
        if f"metrics.{metric}" in df.columns:
            df = df.sort_values(by=f"metrics.{metric}", ascending=True)
        else:
            print(f"Metric {metric} not found.")
        return df

    def suggest_hyperparameter_tuning(self, model_name, metric="RMSE", max_evals=50):
        """Uses Bayesian Optimization to propose new hyperparameter sets."""
        def objective(params):
            """Objective function using actual model evaluation loss."""
            model = self.models.get(model_name)
            if model is None:
                print(f"Error: Model {model_name} not found.")
                return {'loss': float('inf'), 'status': STATUS_OK}

            metrics = evaluate_model(model, self.X_train, self.y_train, self.X_test, self.y_test)
            return {'loss': metrics.get(f'test_{metric.lower()}', float('inf')), 'status': STATUS_OK}

        # Define search space
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        }

        trials = Trials()
        best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        best_params['n_estimators'] = int(best_params['n_estimators'])

        self.logger.info(f"Suggested hyperparameters for {model_name}: {best_params}")
        return best_params

    def approve_suggested_hyperparameters(self, model_name, hyperparameters, approval):
        """Allows human approval for AI-generated hyperparameter suggestions."""
        if approval:
            self.logger.info(f"Approved hyperparameters for {model_name}: {hyperparameters}")
            return hyperparameters
        else:
            self.logger.warning(f"Rejected hyperparameters for {model_name}.")
            return None
