import pandas as pd
import s3fs
import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from concurrent.futures import ProcessPoolExecutor, as_completed
from model_evaluation import evaluate_model, plot_predictions
import datetime
import logging
import json
import numpy as np

class HyperparamTracker:
    def __init__(
        self,
        s3_uri_X_train,
        s3_uri_Y_train,
        s3_uri_X_train_balanced,
        s3_uri_Y_train_balanced,
        s3_uri_X_test,
        s3_uri_Y_test,
        data_version="v1.0",
        model_registry_name_prefix="HomeownerLossModel",
    ):
        """
        For DEMO ONLY:
          - Reads CSV files directly from S3 to build your X_train, Y_train, etc.
          - Ignores advanced ingestion pipeline steps.
        """
        self.X_train = pd.read_csv(s3_uri_X_train)
        self.Y_train = pd.read_csv(s3_uri_Y_train)
        self.X_train_balanced = pd.read_csv(s3_uri_X_train_balanced)
        self.Y_train_balanced = pd.read_csv(s3_uri_Y_train_balanced)
        self.X_test = pd.read_csv(s3_uri_X_test)
        self.Y_test = pd.read_csv(s3_uri_Y_test)

        self.data_version = data_version
        self.model_registry_name_prefix = model_registry_name_prefix

        self.logger = self._setup_logger()
        self.models = self.load_all_models()

        # Optionally, load previous tuning trials from disk to resume
        self.trials = self._load_trials() or Trials()

    def _setup_logger(self):
        logger = logging.getLogger("HyperparamTracker")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def _load_trials(self):
        try:
            with open("trials.json", "r") as f:
                trials_dict = json.load(f)
            self.logger.info("Loaded previous trials from disk.")
            # Conversion to Trials is stubbed for demonstration
        except Exception:
            self.logger.info("No previous trials found. Starting fresh.")
            return None

    def _save_trials(self):
        try:
            with open("trials.json", "w") as f:
                json.dump({}, f)  # Replace with proper serialization if needed
            self.logger.info("Saved trials to disk.")
        except Exception as e:
            self.logger.error("Error saving trials: %s", e)

    def load_all_models(self):
        """
        Loads models from the MLflow Model Registry (Production stage).
        Assumes names like "HomeownerLossModel_Model_1" etc.
        """
        models = {}
        for i in range(1, 6):
            model_name = f"{self.model_registry_name_prefix}_Model_{i}"
            try:
                model_uri = f"models:/{model_name}/Production"
                model = mlflow.pyfunc.load_model(model_uri)
                models[f"Model_{i}"] = model
                self.logger.info("Loaded model from registry: %s", model_name)
            except Exception as e:
                self.logger.error("Error loading model %s: %s", model_name, e)
        return models

    def fetch_hyperparameter_history(self, model_name):
        """
        Queries MLflow to fetch past runs for the given model.
        """
        try:
            runs = mlflow.search_runs(filter_string=f"tags.model_name = '{model_name}'")
            history = []
            for _, row in runs.iterrows():
                history.append({
                    "model_name": model_name,
                    "timestamp": row["start_time"],
                    "hyperparameters": row.get("params", {}),
                    "metrics": row.get("metrics", {})
                })
            return history
        except Exception as e:
            self.logger.error("Error fetching hyperparameter history: %s", e)
            return []

    def send_notification(self, message):
        """
        Stub function to send notifications (e.g., email or Slack).
        """
        self.logger.info("Notification sent: %s", message)

    @staticmethod
    def adjust_search_space(history, model_name):
        """
        Dynamically adjust the hyperparameter search space based on the model type.
        """
        if model_name == "HomeownerLossModel_Model_1":
            return {
                'booster': 'gbtree',
                'n_estimators': hp.choice('n_estimators', [40, 100]),
                'learning_rate': hp.choice('learning_rate', [0.10, 0.20]),
                'max_depth': hp.choice('max_depth', [2, 3, 4]),
                'colsample_bytree': hp.choice('colsample_bytree', [0.80, 0.85]),
                'subsample': hp.choice('subsample', [0.70, 0.80, 0.90]),
                'base_score': 100,
                'reg_alpha': 17,
                'gamma': 0,
                'monotone_constraints': [tuple(np.tile([1, 1, -1], 16))]
            }
        else:
            # For Models 2-5: create a monotone string from a monotone_list.
            monotone_list = [1] * 30  # Adjust this list length as needed.
            monotone_str = "(" + ",".join(str(c) for c in monotone_list) + ")"
            return {
                'booster': 'gbtree',
                'n_estimators': hp.choice('n_estimators', [150, 400, 420]),
                'max_depth': hp.choice('max_depth', [8, 10, 12]),
                'learning_rate': hp.choice('learning_rate', [0.01, 0.015]),
                'subsample': hp.choice('subsample', [0.7, 0.75, 0.8]),
                'colsample_bytree': hp.choice('colsample_bytree', [0.85, 0.9]),
                'reg_alpha': hp.choice('reg_alpha', [0.01, 0.05, 0.1, 1.0]),
                'reg_lambda': hp.choice('reg_lambda', [0.005, 0.01]),
                'min_child_weight': hp.choice('min_child_weight', [1, 2, 3, 4]),
                'gamma': hp.choice('gamma', [0.2, 0.25]),
                'monotone_constraints': [monotone_str]
            }

    @staticmethod
    def _tune_model(model_name, model,
                    X_train, Y_train, X_train_balanced, Y_train_balanced,
                    X_test, Y_test, max_evals, early_stop_rounds, base_trials):
        """
        Perform hyperparameter tuning using Hyperopt with Bayesian optimization.
        """
        import logging
        logger = logging.getLogger(f"Tuning_{model_name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        def objective(params):
            # Choose training data based on model type
            if model_name == "HomeownerLossModel_Model_1":
                X_train_used, Y_train_used = X_train, Y_train
            else:
                X_train_used, Y_train_used = X_train_balanced, Y_train_balanced

            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    metrics = evaluate_model(
                        model,
                        X_train_used,
                        Y_train_used,
                        X_test,
                        Y_test,
                        model_name=model_name,
                        summary_only=True
                    )
                loss = metrics["test_rmse"]
            except Exception as e:
                logger.error("Exception during trial: %s", e)
                loss = float("inf")
            return {"loss": loss, "status": STATUS_OK}

        search_space = HyperparamTracker.adjust_search_space(None, model_name)
        trials = base_trials
        best_loss = float("inf")
        best_params = None
        no_improvement_count = 0

        for i in range(max_evals):
            try:
                result = fmin(
                    fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=len(trials.trials) + 1,
                    trials=trials,
                    verbose=0
                )
                current_loss = min(trials.losses())
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = result
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            except Exception as e:
                logger.error("Error in tuning loop at eval %d: %s", i + 1, e)
                no_improvement_count += 1

            if no_improvement_count >= early_stop_rounds:
                logger.info("Early stopping triggered after %d evaluations.", i + 1)
                break

        if best_params and 'n_estimators' in best_params:
            best_params['n_estimators'] = int(best_params['n_estimators'])

        return best_params, best_loss

    def tune_all_models(self, max_evals=50, early_stop_rounds=10):
        """
        Tune each model and log the best hyperparameters and loss to MLflow.
        """
        models_to_tune = [
            "HomeownerLossModel_Model_1",
            "HomeownerLossModel_Model_2",
            "HomeownerLossModel_Model_3",
            "HomeownerLossModel_Model_4",
            "HomeownerLossModel_Model_5"
        ]
        results = {}
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    HyperparamTracker._tune_model,
                    model_name,
                    self.models[model_name],
                    self.X_train,
                    self.Y_train,
                    self.X_train_balanced,
                    self.Y_train_balanced,
                    self.X_test,
                    self.Y_test,
                    max_evals,
                    early_stop_rounds,
                    self.trials
                ): model_name
                for model_name in models_to_tune if model_name in self.models
            }
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    best_params, best_loss = future.result()
                    with mlflow.start_run():
                        mlflow.set_tag("model_name", model_name)
                        mlflow.log_param("data_version", self.data_version)
                        mlflow.log_params(best_params)
                        mlflow.log_metric("test_rmse", best_loss)
                    self.logger.info("Best hyperparameters for %s: %s (loss=%.4f)", model_name, best_params, best_loss)
                    results[model_name] = best_params
                    self.send_notification(f"New best hyperparameters for {model_name}: {best_params}, loss: {best_loss}")
                except Exception as exc:
                    self.logger.error("Model %s generated an exception: %s", model_name, exc)

        self._save_trials()

        for model_name, params in results.items():
            print(f"Best hyperparameters for {model_name}: {params}")
