import os
import json
import shutil
import tempfile
import unittest
import joblib
from hyperopt import STATUS_OK

# --- Dummy implementations for testing --- #
def dummy_evaluate_model(model, model_name, X_train, Y_train, *args):
    """
    A dummy evaluate_model that computes a simple loss based on the model's hyperparameter.
    This function accepts additional parameters so it works both for tuning (6 arguments)
    and for approval (8 arguments). For simplicity, it returns the 'learning_rate' value as the loss.
    Lower values are considered better.
    """
    lr = model.params.get("learning_rate", 0.05)
    return {"test_rmse": lr}

class DummyModel:
    """
    A simple dummy model that supports setting parameters.
    """
    def __init__(self):
        self.params = {}
    def set_params(self, **params):
        self.params.update(params)
        return self

# --- End of dummy implementations --- #

# Import the hyperparameter_tracker module.
import hyperparameter_tracker
# Monkey-patch the evaluate_model function used in hyperparameter_tracker with our dummy.
hyperparameter_tracker.evaluate_model = dummy_evaluate_model
from hyperparameter_tracker import HyperparamTracker

class TestHyperparamTracker(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for logs and models.
        self.temp_log_dir = tempfile.mkdtemp()
        self.temp_model_dir = tempfile.mkdtemp()

        # Create dummy training and testing data.
        self.X_train = [[0]] * 10
        self.Y_train = [0] * 10
        self.X_train_balanced = [[0]] * 10
        self.Y_train_balanced = [0] * 10
        self.X_test = [[0]] * 5
        self.Y_test = [0] * 5

        # Create and save five dummy models (Model_1 to Model_5).
        for i in range(1, 6):
            model = DummyModel()
            model_path = os.path.join(self.temp_model_dir, f"model_{i}.joblib")
            joblib.dump(model, model_path)

        # Instantiate the tracker with the dummy data and temporary directories.
        self.tracker = HyperparamTracker(
            self.X_train, self.Y_train,
            self.X_train_balanced, self.Y_train_balanced,
            self.X_test, self.Y_test,
            log_dir=self.temp_log_dir,
            model_dir=self.temp_model_dir
        )

    def tearDown(self):
        # Clean up temporary directories.
        shutil.rmtree(self.temp_log_dir)
        shutil.rmtree(self.temp_model_dir)

    def test_load_all_models(self):
        """Test that all five models are loaded correctly."""
        self.assertEqual(len(self.tracker.models), 5, "Not all models were loaded.")

    def test_log_and_fetch_hyperparameters(self):
        """Test that hyperparameter logging and fetching work as intended."""
        # Initially, the history for Model_1 should be empty.
        history = self.tracker.fetch_hyperparameter_history("Model_1")
        self.assertEqual(len(history), 0, "History should be empty initially.")
        
        # Log a dummy hyperparameter configuration.
        hyperparams = {
            "learning_rate": 0.01,
            "max_depth": 5,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        metrics = {"test_rmse": 0.01}
        self.tracker.log_hyperparameters("Model_1", hyperparams, metrics)
        
        # Fetch the history again and verify it now contains one entry.
        history = self.tracker.fetch_hyperparameter_history("Model_1")
        self.assertEqual(len(history), 1, "History should have one entry after logging.")

    def test_tune_all_models(self):
        """Test that tuning runs for all models and logs hyperparameter history entries."""
        # Run tuning for all models with a small evaluation budget.
        self.tracker.tune_all_models(max_evals=10, early_stop_rounds=3)
        
        # Verify that the hyperparameter history JSON file exists.
        history_file = os.path.join(self.temp_log_dir, "hyperparameter_history.json")
        self.assertTrue(os.path.exists(history_file), "History file should exist after tuning.")
        
        # Load the history and check that each model has at least one logged entry.
        with open(history_file, "r") as f:
            history = json.load(f)
        for i in range(1, 6):
            model_name = f"Model_{i}"
            entries = [entry for entry in history if entry["model_name"] == model_name]
            self.assertGreater(len(entries), 0, f"{model_name} should have at least one tuning entry.")

    def test_approve_suggested_hyperparameters(self):
        """Test that approve_suggested_hyperparameters correctly approves or rejects candidates."""
        # Log a best configuration for Model_1.
        best_params = {
            "learning_rate": 0.005,
            "max_depth": 5,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        best_metrics = {"test_rmse": 0.005}
        self.tracker.log_hyperparameters("Model_1", best_params, best_metrics)
        
        # Create a candidate that is worse (learning_rate = 0.02, hence higher loss).
        candidate = {
            "learning_rate": 0.02,
            "max_depth": 7,
            "n_estimators": 150,
            "subsample": 0.7,
            "colsample_bytree": 0.7
        }
        approved = self.tracker.approve_suggested_hyperparameters("Model_1", candidate, approval=True)
        self.assertEqual(approved, best_params,
                         "A candidate with worse performance should be rejected in favor of the best logged hyperparameters.")
        
        # Now create a candidate that is better (learning_rate = 0.001, lower loss).
        candidate = {
            "learning_rate": 0.001,
            "max_depth": 7,
            "n_estimators": 150,
            "subsample": 0.7,
            "colsample_bytree": 0.7
        }
        approved = self.tracker.approve_suggested_hyperparameters("Model_1", candidate, approval=True)
        self.assertEqual(approved, candidate,
                         "A candidate with better performance should be approved.")

if __name__ == "__main__":
    unittest.main()
