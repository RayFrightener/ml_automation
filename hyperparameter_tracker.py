import pandas as pd
import s3fs
import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from concurrent.futures import ProcessPoolExecutor, as_completed
from model_evaluation import evaluate_model  # Your evaluation function
import datetime
import logging
import json

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

        # Example: read from S3 URIs using Pandas
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
            # Convert trials_dict to a Trials() if you want to actually resume
            # This is just a stub
        except Exception:
            self.logger.info("No previous trials found. Starting fresh.")
            return None

    def _save_trials(self):
        try:
            with open("trials.json", "w") as f:
                json.dump({}, f)  # Replace with actual serialization if you want to persist Trials
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
                # Load the Production stage model from the MLflow registry
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

    def adjust_search_space(self, history):
        """
        Dynamically adjust the hyperparameter search space based on historical run data.
        (Stub implementation)
        """
        self.logger.info("Adjusting search space based on history; current implementation is a stub.")
        return {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        }

    def approve_suggested_hyperparameters(self, model_name, hyperparameters, approval):
        """
        Evaluate suggested hyperparams, compare performance to best historical metrics.
        """
        history = self.fetch_hyperparameter_history(model_name)
        if history:
            best_entry = history[-1]
            best_hyperparams = best_entry["hyperparameters"]
            best_metrics = best_entry["metrics"]
        else:
            best_hyperparams = None
            best_metrics = {"test_rmse": float("inf")}

        with mlflow.start_run() as run:
            mlflow.set_tag("model_name", model_name)
            mlflow.log_param("data_version", self.data_version)

            # Model_1 uses original (unbalanced), others use balanced
            if model_name == "HomeownerLossModel_Model_1":
                X_train_used, Y_train_used = self.X_train, self.Y_train
            else:
                X_train_used, Y_train_used = self.X_train_balanced, self.Y_train_balanced

            metrics = evaluate_model(
                self.models[model_name],
                X_train_used,
                Y_train_used,
                self.X_test,
                self.Y_test
            )
            mlflow.log_metrics(metrics)

        if metrics["test_rmse"] > best_metrics.get("test_rmse", float("inf")):
            self.logger.info(
                "New hyperparameters performed worse than previous best. Rolling back to %s.",
                best_hyperparams
            )
            self.send_notification(f"Model {model_name}: new hyperparams not approved (RMSE higher).")
            return best_hyperparams

        if approval:
            self.logger.info("Approved new hyperparameters for %s: %s", model_name, hyperparameters)
            self.send_notification(f"Model {model_name}: new hyperparameters approved.")
            return hyperparameters
        else:
            self.logger.warning("Rejected hyperparameters for %s.", model_name)
            self.send_notification(f"Model {model_name}: hyperparameter suggestion rejected.")
            return None

    @staticmethod
    def _tune_model(model_name, model,
                    X_train, Y_train, X_train_balanced, Y_train_balanced,
                    X_test, Y_test, max_evals, early_stop_rounds, base_trials):
        """
        Perform hyperparameter tuning using Hyperopt + Bayesian optimization.
        """
        import logging
        logger = logging.getLogger(f"Tuning_{model_name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        def objective(params):
            # Model_1 uses unbalanced, everything else uses balanced
            if model_name == "HomeownerLossModel_Model_1":
                X_train_used, Y_train_used = X_train, Y_train
            else:
                X_train_used, Y_train_used = X_train_balanced, Y_train_balanced

            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(params)
                    # Evaluate
                    metrics = evaluate_model(model, X_train_used, Y_train_used, X_test, Y_test)
                    mlflow.log_metrics(metrics)
                loss = metrics["test_rmse"]
            except Exception as e:
                logger.error("Exception during trial: %s", e)
                loss = float("inf")
            return {"loss": loss, "status": STATUS_OK}

        search_space = HyperparamTracker.adjust_search_space(None)

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

        # Convert n_estimators to int if it exists
        if 'n_estimators' in best_params:
            best_params['n_estimators'] = int(best_params['n_estimators'])

        return best_params, best_loss

    def tune_all_models(self, max_evals=50, early_stop_rounds=10):
        """
        Tuning each model. Logs best hyperparams and loss to MLflow.
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
                for model_name in models_to_tune
                if model_name in self.models
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
                    self.send_notification(
                        f"New best hyperparameters for {model_name}: {best_params}, loss: {best_loss}"
                    )
                except Exception as exc:
                    self.logger.error("Model %s generated an exception: %s", model_name, exc)

        # Optionally, save Trials
        self._save_trials()

        # Print final results
        for model_name, params in results.items():
            print(f"Best hyperparameters for {model_name}: {params}")
