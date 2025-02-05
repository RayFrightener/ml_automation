import json
import os
import uuid
import datetime
import logging
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

class HyperparamTracker:
    """
    Hyperparameter Tracking System:
    - Logs hyperparameter configurations and performance metrics.
    - Enables retrieval of past hyperparameter runs.
    - Uses Bayesian optimization (Hyperopt) to suggest new configurations.
    - Provides human approval for AI-generated suggestions.
    """

    def __init__(self, log_dir="hyperparam_logs"):
        """Initialize the hyperparameter tracker with a logging directory."""
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.db_file = os.path.join(self.log_dir, "hyperparameter_history.json")

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

    def log_hyperparameters(self, model_name, hyperparameters, metrics):
        """
        Logs hyperparameter configurations and their performance metrics.

        Args:
            model_name (str): Identifier for the model.
            hyperparameters (dict): Dictionary of hyperparameter values.
            metrics (dict): Dictionary of evaluation metrics (e.g., RMSE, MSE, MAE).
        """
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
        """
        Retrieves past hyperparameter runs based on filters.

        Args:
            model_name (str, optional): Filter by model name.
            start_date (str, optional): Start date in YYYY-MM-DD format.
            end_date (str, optional): End date in YYYY-MM-DD format.
            filters (dict, optional): Key-value filters for hyperparameters.

        Returns:
            list: Filtered hyperparameter history.
        """
        with open(self.db_file, "r") as f:
            data = json.load(f)

        # Apply filters
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
        """
        Ranks historical runs based on a selected performance metric.

        Args:
            model_name (str): Model identifier.
            metric (str): Metric for ranking (e.g., "RMSE", "MSE", "MAE").

        Returns:
            pd.DataFrame: Sorted dataframe of runs based on the given metric.
        """
        data = self.fetch_hyperparameter_history(model_name=model_name)
        if not data:
            return None

        df = pd.DataFrame(data)
        df = df.sort_values(by=f"metrics.{metric}", ascending=True)
        return df

    def suggest_hyperparameter_tuning(self, model_name, metric="RMSE", max_evals=50):
        """
        Uses Bayesian Optimization to propose new hyperparameter sets.

        Args:
            model_name (str): Model identifier.
            metric (str): Metric to optimize (e.g., "RMSE", "MSE", "MAE").
            max_evals (int): Number of evaluations.

        Returns:
            dict: Suggested hyperparameters.
        """
        def objective(params):
            """Objective function for Bayesian Optimization (simulated loss)."""
            loss = sum(params.values()) / len(params)  # Dummy loss function
            return {'loss': loss, 'status': STATUS_OK}

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

        # Convert discrete values to integers
        best_params['n_estimators'] = int(best_params['n_estimators'])

        self.logger.info(f"Suggested hyperparameters for {model_name}: {best_params}")
        return best_params

    def approve_suggested_hyperparameters(self, model_name, hyperparameters, approval):
        """
        Allows human approval for AI-generated hyperparameter suggestions.

        Args:
            model_name (str): Model identifier.
            hyperparameters (dict): AI-suggested hyperparameter values.
            approval (bool): True if approved, False if rejected.

        Returns:
            dict: Approved hyperparameters or None.
        """
        if approval:
            self.logger.info(f"Approved hyperparameters for {model_name}: {hyperparameters}")
            return hyperparameters
        else:
            self.logger.warning(f"Rejected hyperparameters for {model_name}.")
            return None
