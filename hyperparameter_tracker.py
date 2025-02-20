import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from concurrent.futures import ProcessPoolExecutor, as_completed
from model_evaluation import evaluate_model, plot_predictions
import datetime
import logging
import json  # For potential persistence of Trials (resumability)


class HyperparamTracker:
    def __init__(self, X_train, Y_train, X_train_balanced, Y_train_balanced, X_test, Y_test,
                 data_version="v1.0", model_registry_name_prefix="HomeownerLossModel"):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_train_balanced = X_train_balanced
        self.Y_train_balanced = Y_train_balanced
        self.X_test = X_test
        self.Y_test = Y_test
        self.data_version = data_version  # Data versioning context
        self.model_registry_name_prefix = model_registry_name_prefix
        self.logger = self._setup_logger()
        self.models = self.load_all_models()
        # Optionally, load previous tuning trials from disk (stub)
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
        # Stub: implement loading of a persisted Trials object from disk
        try:
            with open("trials.json", "r") as f:
                trials_dict = json.load(f)
            self.logger.info("Loaded previous trials from disk.")
            # Convert trials_dict to a Trials object if needed
            # Return a Trials object reconstructed from trials_dict (implementation-dependent)
        except Exception as e:
            self.logger.info("No previous trials found. Starting fresh.")
            return None

    def _save_trials(self):
        # Stub: implement saving the current Trials object to disk for resumability
        try:
            with open("trials.json", "w") as f:
                json.dump({}, f)  # Replace with actual serialization
            self.logger.info("Saved trials to disk.")
        except Exception as e:
            self.logger.error("Error saving trials: %s", e)

    def load_all_models(self):
        """
        Loads models from the MLflow Model Registry. Assumes models have been
        registered with names like "HomeownerLossModel_Model_1" and are in the "Production" stage.
        """
        models = {}
        for i in range(1, 6):
            model_name = f"{self.model_registry_name_prefix}_Model_{i}"
            try:
                model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
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
        Stub function to send notifications (e.g., email or Slack) when a new best run is found.
        """
        self.logger.info("Notification sent: %s", message)
        # Integrate with your notification system here

    def adjust_search_space(self, history):
        """
        Dynamically adjust the hyperparameter search space based on historical run data.
        Stub implementation – you could narrow ranges if previous runs indicate trends.
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
        Evaluates the suggested hyperparameters and compares the current run’s performance
        against historical best metrics (based on test RMSE). Includes data version context.
        Also logs Actual vs. Predicted plots for training & testing data.
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

            # For Model_4, use the balanced training set; otherwise, use standard.
            if model_name == "Model_4":
                X_train_used, Y_train_used = self.X_train_balanced, self.Y_train_balanced
            else:
                X_train_used, Y_train_used = self.X_train, self.Y_train

            # Evaluate the model, returning predictions so we can plot them
            eval_results = evaluate_model(
                model=self.models[model_name],
                X_train=X_train_used,
                y_train=Y_train_used,
                X_test=self.X_test,
                y_test=self.Y_test,
                log_to_mlflow=False,       # We'll log the metrics manually below
                return_predictions=True
            )

            # Manually log the metrics
            mlflow.log_metrics(eval_results["metrics"])

            # Plot Actual vs. Predicted for training data
            plot_predictions(
                y_true=Y_train_used,
                y_pred=eval_results["train_predictions"],
                title=f"{model_name} (Train): Actual vs. Predicted",
                plot_type="scatter",
                log_to_mlflow=True,
                filename="train_predictions_plot.png"
            )
            # Plot Actual vs. Predicted for testing data
            plot_predictions(
                y_true=self.Y_test,
                y_pred=eval_results["test_predictions"],
                title=f"{model_name} (Test): Actual vs. Predicted",
                plot_type="scatter",
                log_to_mlflow=True,
                filename="test_predictions_plot.png"
            )

        # Compare using a chosen metric; here, test_rmse
        metrics = eval_results["metrics"]
        if metrics["test_rmse"] > best_metrics.get("test_rmse", float("inf")):
            self.logger.info("New hyperparameters performed worse than previous best. Rolling back to %s.", best_hyperparams)
            self.send_notification(f"Model {model_name}: New hyperparameters not approved due to higher RMSE.")
            return best_hyperparams

        if approval:
            self.logger.info("Approved new hyperparameters for %s: %s", model_name, hyperparameters)
            self.send_notification(f"Model {model_name}: New hyperparameters approved.")
            return hyperparameters
        else:
            self.logger.warning("Rejected hyperparameters for %s.", model_name)
            self.send_notification(f"Model {model_name}: Hyperparameter suggestion rejected.")
            return None

    @staticmethod
    def _tune_model(model_name, model,
                    X_train, Y_train, X_train_balanced, Y_train_balanced,
                    X_test, Y_test, max_evals, early_stop_rounds, base_trials):
        """
        Performs hyperparameter tuning using Hyperopt with Bayesian Optimization.
        Each trial is logged as a nested MLflow run.
        """
        import logging
        logger = logging.getLogger(f"Tuning_{model_name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        def objective(params):
            if model_name == "Model_4":
                X_train_used, Y_train_used = X_train_balanced, Y_train_balanced
            else:
                X_train_used, Y_train_used = X_train, Y_train

            # Log each trial as a nested MLflow run.
            try:
                with mlflow.start_run(nested=True) as run:
                    mlflow.log_params(params)
                    # Evaluate model on this trial
                    eval_results = evaluate_model(
                        model,
                        X_train_used,
                        Y_train_used,
                        X_test,
                        Y_test,
                        log_to_mlflow=True,  # We'll log metrics manually
                        return_predictions=False
                    )
                    mlflow.log_metrics(eval_results["metrics"])
                loss = eval_results["metrics"]["test_rmse"]
            except Exception as e:
                logger.error("Exception during trial: %s", e)
                loss = float("inf")
            return {"loss": loss, "status": STATUS_OK}

        # Adjust the search space dynamically (stub)
        search_space = HyperparamTracker.adjust_search_space(None)

        # Use base_trials for resumability
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

        best_params['n_estimators'] = int(best_params['n_estimators'])
        return best_params, best_loss

    def tune_all_models(self, max_evals=50, early_stop_rounds=10):
        """
        Tunes all models concurrently. For each model, the best hyperparameters and corresponding loss
        are logged to MLflow.
        """
        models_to_tune = ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"]
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
                    with mlflow.start_run() as run:
                        mlflow.set_tag("model_name", model_name)
                        mlflow.log_param("data_version", self.data_version)
                        mlflow.log_params(best_params)
                        mlflow.log_metric("test_rmse", best_loss)
                    self.logger.info("Best hyperparameters for %s: %s with loss %s", model_name, best_params, best_loss)
                    results[model_name] = best_params
                    # Send a notification about the new best run
                    self.send_notification(f"New best hyperparameters for {model_name}: {best_params}, loss: {best_loss}")
                except Exception as exc:
                    self.logger.error("Model %s generated an exception: %s", model_name, exc)

        # Optionally, save the updated Trials object to disk for resumability.
        self._save_trials()

        for model_name, params in results.items():
            print(f"Best hyperparameters for {model_name}: {params}")
