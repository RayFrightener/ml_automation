import json
import os
import datetime
import logging
import joblib
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from concurrent.futures import ProcessPoolExecutor, as_completed
from model_evaluation import evaluate_model  # Import your evaluation function
from multiprocessing import Lock

# Global lock for safe JSON file access (for processes)
json_lock = Lock()


class HyperparamTracker:
    def __init__(self, X_train, Y_train, X_train_balanced, Y_train_balanced, X_test, Y_test,
                 log_dir="hyperparam_logs", model_dir="pretrained_models"):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_train_balanced = X_train_balanced
        self.Y_train_balanced = Y_train_balanced
        self.X_test = X_test
        self.Y_test = Y_test

        self.log_dir = log_dir
        self.model_dir = model_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.db_file = os.path.join(self.log_dir, "hyperparameter_history.json")

        # Setup logger before loading models
        self.logger = self._setup_logger()
        self.models = self.load_all_models()

    def _setup_logger(self):
        logger = logging.getLogger("HyperparamTracker")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.FileHandler(os.path.join(self.log_dir, "tracking.log"))
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
        return logger

    def load_all_models(self):
        models = {}
        # Load models from model_dir (models must be joblib-serializable)
        for i in range(1, 6):
            model_path = os.path.join(self.model_dir, f"model_{i}.joblib")
            if os.path.exists(model_path):
                try:
                    models[f"Model_{i}"] = joblib.load(model_path)
                except Exception as e:
                    self.logger.error("Error loading %s: %s", model_path, e)
            else:
                self.logger.warning("Warning: %s not found.", model_path)
        return models

    def fetch_hyperparameter_history(self, model_name):
        try:
            with json_lock:
                if os.path.exists(self.db_file):
                    with open(self.db_file, "r") as f:
                        data = json.load(f)
                else:
                    data = []
            history = [entry for entry in data if entry["model_name"] == model_name]
            return history
        except Exception as e:
            self.logger.error("Error reading hyperparameter history: %s", e)
            return []

    def log_hyperparameters(self, model_name, hyperparameters, metrics):
        entry = {
            "model_name": model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "hyperparameters": hyperparameters,
            "metrics": metrics
        }
        try:
            with json_lock:
                if os.path.exists(self.db_file):
                    with open(self.db_file, "r") as f:
                        data = json.load(f)
                else:
                    data = []
                data.append(entry)
                with open(self.db_file, "w") as f:
                    json.dump(data, f, indent=4)
        except Exception as e:
            self.logger.error("Error logging hyperparameters: %s", e)

    def approve_suggested_hyperparameters(self, model_name, hyperparameters, approval):
        history = self.fetch_hyperparameter_history(model_name)
        if history:
            best_entry = history[-1]
            best_hyperparams = best_entry["hyperparameters"]
            best_metrics = best_entry["metrics"]
        else:
            best_hyperparams = None
            best_metrics = {"test_rmse": float("inf")}

        # Evaluate new hyperparameters on current data
        metrics = evaluate_model(
            self.models[model_name],
            model_name,
            self.X_train,
            self.Y_train,
            self.X_train_balanced,
            self.Y_train_balanced,
            self.X_test,
            self.Y_test,
        )
        if metrics["test_rmse"] > best_metrics.get("test_rmse", float("inf")):
            self.logger.info("New hyperparameters performed worse than previous best. Rolling back to %s.", best_hyperparams)
            return best_hyperparams

        if approval:
            self.logger.info("Approved new hyperparameters for %s: %s", model_name, hyperparameters)
            return hyperparameters
        else:
            self.logger.warning("Rejected hyperparameters for %s.", model_name)
            return None

    @staticmethod
    def _tune_model(model_name, model,
                    X_train, Y_train, X_train_balanced, Y_train_balanced,
                    X_test, Y_test, max_evals, early_stop_rounds):
        """
        This static method performs hyperparameter tuning for a single model.
        It uses an iterative approach to call Hyperopt until no improvement is seen
        for `early_stop_rounds` evaluations.
        """
        # Set up a simple process-local logger.
        import logging
        logger = logging.getLogger(f"Tuning_{model_name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        def objective(params):
            # For Model_4 use the balanced training set.
            if model_name == "Model_4":
                X_train_used, Y_train_used = X_train_balanced, Y_train_balanced
            else:
                X_train_used, Y_train_used = X_train, Y_train

            metrics = evaluate_model(model, model_name, X_train_used, Y_train_used, X_test, Y_test)
            loss = metrics["test_rmse"]
            return {"loss": loss, "status": STATUS_OK}

        # Define the hyperparameter search space.
        space = {
            'learning_rate': hp.uniform('learning_rate', 0.001, 0.1),
            'max_depth': hp.choice('max_depth', [3, 5, 7, 9]),
            'n_estimators': hp.quniform('n_estimators', 50, 300, 1),
            'subsample': hp.uniform('subsample', 0.5, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        }

        trials = Trials()
        best_loss = float("inf")
        best_params = None
        no_improvement_count = 0

        # Iteratively increase max_evals by one and check for improvement.
        for i in range(max_evals):
            result = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=i + 1, trials=trials, verbose=0)
            current_loss = min(trials.losses())
            if current_loss < best_loss:
                best_loss = current_loss
                best_params = result
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stop_rounds:
                logger.info("Early stopping triggered after %d evaluations.", i + 1)
                break

        # Ensure n_estimators is an integer.
        best_params['n_estimators'] = int(best_params['n_estimators'])
        return best_params, best_loss

    def tune_all_models(self, max_evals=50, early_stop_rounds=10):
        """
        Tunes all models concurrently using a process pool.
        For each model, the best hyperparameters and the corresponding loss are logged.
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
                ): model_name
                for model_name in models_to_tune if model_name in self.models
            }
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    best_params, best_loss = future.result()
                    self.log_hyperparameters(model_name, best_params, {"test_rmse": best_loss})
                    self.logger.info("Best hyperparameters for %s: %s with loss %s", model_name, best_params, best_loss)
                    results[model_name] = best_params
                except Exception as exc:
                    self.logger.error("Model %s generated an exception: %s", model_name, exc)

        for model_name, params in results.items():
            print(f"Best hyperparameters for {model_name}: {params}")
