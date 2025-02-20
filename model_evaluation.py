import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import mlflow

logger = logging.getLogger("ModelEvaluation")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

def evaluate_model(model, X_train, y_train, X_test, y_test,
                   log_to_mlflow=True, return_predictions=False,
                   model_name=None, summary_only=False,
                   target_transformer=None, sample_weight=None):
    """
    Evaluate the model and return performance metrics.
    
    For Models 2–5, if a target_transformer is provided, the test targets and predictions
    are inverse transformed to their original scale. Sample weights are applied if provided.
    
    Parameters:
    - model: Trained model.
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Testing features.
    - y_test: Testing labels.
    - log_to_mlflow: Flag to log metrics to MLflow.
    - return_predictions: If True, also return predictions.
    - model_name: Optional model name. If provided and not equal to "HomeownerLossModel_Model_1",
                  inverse transformation is applied to test data if target_transformer is not None.
    - summary_only: If True, log only test RMSE and R² with specific keys.
    - target_transformer: Optional transformer to inverse transform targets (applied for Models 2–5).
    - sample_weight: Optional sample weights for metric calculations.
    
    Returns:
    - Dictionary containing performance metrics.
    """
    # Evaluate on training data (without inverse transformation)
    start_time = time.time()
    train_predictions = model.predict(X_train)
    train_latency = time.time() - start_time

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions, sample_weight=sample_weight))
    train_mae = mean_absolute_error(y_train, train_predictions, sample_weight=sample_weight)
    train_mse = mean_squared_error(y_train, train_predictions, sample_weight=sample_weight)
    train_r2 = r2_score(y_train, train_predictions, sample_weight=sample_weight)
    train_mape = mean_absolute_percentage_error(y_train, train_predictions, sample_weight=sample_weight)
    
    # Evaluate on testing data
    start_time = time.time()
    test_predictions = model.predict(X_test)
    test_latency = time.time() - start_time

    # For Models 2–5, apply inverse transformation if target_transformer is provided
    if target_transformer is not None and (model_name is not None and model_name != "HomeownerLossModel_Model_1"):
        y_test_array = np.array(y_test).reshape(-1, 1)
        y_pred_array = np.array(test_predictions).reshape(-1, 1)
        y_test_exp = target_transformer.inverse_transform(y_test_array).flatten()
        y_pred_exp = target_transformer.inverse_transform(y_pred_array).flatten()
    else:
        y_test_exp = np.array(y_test)
        y_pred_exp = np.array(test_predictions)

    test_rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp, sample_weight=sample_weight))
    test_mae = mean_absolute_error(y_test_exp, y_pred_exp, sample_weight=sample_weight)
    test_mse = mean_squared_error(y_test_exp, y_pred_exp, sample_weight=sample_weight)
    test_r2 = r2_score(y_test_exp, y_pred_exp, sample_weight=sample_weight)
    test_mape = mean_absolute_percentage_error(y_test_exp, y_pred_exp, sample_weight=sample_weight)

    # Multiply MAPE by 100 to convert to percentage
    train_mape *= 100
    test_mape *= 100

    metrics = {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mape": train_mape,
        "test_mape": test_mape,
        "train_latency_sec": train_latency,
        "test_latency_sec": test_latency
    }
    
    logger.info(
        "Training Metrics: RMSE=%.4f, MAE=%.4f, MSE=%.4f, R2=%.4f, MAPE=%.4f%%, Latency=%.4f sec",
        train_rmse, train_mae, train_mse, train_r2, train_mape, train_latency
    )
    logger.info(
        "Testing Metrics: RMSE=%.4f, MAE=%.4f, MSE=%.4f, R2=%.4f, MAPE=%.4f%%, Latency=%.4f sec",
        test_rmse, test_mae, test_mse, test_r2, test_mape, test_latency
    )
    
    if log_to_mlflow and mlflow.active_run():
        if summary_only and model_name is not None:
            if model_name == "HomeownerLossModel_Model_1":
                mlflow.log_metric("rmse", test_rmse)
                mlflow.log_metric("r2", test_r2)
            else:
                mlflow.log_metric("rmse_top", test_rmse)
                mlflow.log_metric("r2_top", test_r2)
        else:
            mlflow.log_metrics(metrics)
    
    if return_predictions:
        return {
            "metrics": metrics,
            "train_predictions": train_predictions,
            "test_predictions": test_predictions,
            "y_test_exp": y_test_exp,
            "y_pred_exp": y_pred_exp
        }
    else:
        return metrics

def plot_predictions(y_true, y_pred, title="Actual vs. Predict",
                     plot_type="scatter", log_to_mlflow=True, filename="predictions_plot.png"):
    """
    Plot predictions vs. actual values and optionally log the plot as an MLflow artifact.
    
    Parameters:
    - y_true: Actual target values.
    - y_pred: Predicted values.
    - title: Plot title.
    - plot_type: "scatter" or "line".
    - log_to_mlflow: If True, log the figure to MLflow.
    - filename: Filename for the saved artifact.
    """
    plt.figure(figsize=(10, 6))
    
    if plot_type == "scatter":
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([np.min(y_true), np.max(y_true)],
                 [np.min(y_true), np.max(y_true)], 'k--', lw=2)
    elif plot_type == "line":
        plt.plot(y_true, marker='o', linestyle='-', label='Actual')
        plt.plot(y_pred, marker='o', linestyle='-', label='Predicted')
        plt.legend()
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    
    if log_to_mlflow and mlflow.active_run():
        plt.savefig(filename)
        mlflow.log_artifact(filename, artifact_path="plots")
        logger.info("Logged prediction plot to MLflow as artifact: %s", filename)
    
    plt.show()
