import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    log_to_mlflow=True,
    return_predictions=False
):
    """
    Evaluate the model and return performance metrics.
    
    Parameters:
    - model: Trained model.
    - X_train: Training features.
    - y_train: Training labels.
    - X_test: Testing features.
    - y_test: Testing labels.
    - log_to_mlflow: Boolean flag to log metrics to MLflow.
    - return_predictions: If True, also return train & test predictions.

    Returns:
    - A dictionary containing performance metrics.
      If return_predictions=True, also includes 'train_predictions' and 'test_predictions'.
    """
    # Evaluate on training data
    start_time = time.time()
    train_predictions = model.predict(X_train)
    train_latency = time.time() - start_time

    # Calculate training metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    train_mae = mean_absolute_error(y_train, train_predictions)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    train_mape = mean_absolute_percentage_error(y_train, train_predictions)
    
    # Evaluate on testing data
    start_time = time.time()
    test_predictions = model.predict(X_test)
    test_latency = time.time() - start_time

    # Calculate testing metrics
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    test_mape = mean_absolute_percentage_error(y_test, test_predictions)
    
    # Aggregate metrics
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
    
    # Log metrics using logger
    logger.info(
        "Training Metrics: RMSE=%.4f, MAE=%.4f, MSE=%.4f, R2=%.4f, MAPE=%.4f, Latency=%.4f sec",
        train_rmse, train_mae, train_mse, train_r2, train_mape, train_latency
    )
    logger.info(
        "Testing Metrics: RMSE=%.4f, MAE=%.4f, MSE=%.4f, R2=%.4f, MAPE=%.4f, Latency=%.4f sec",
        test_rmse, test_mae, test_mse, test_r2, test_mape, test_latency
    )
    
    # Optionally log to MLflow if requested and run is active
    if log_to_mlflow and mlflow.active_run():
        mlflow.log_metrics(metrics)

    # If user wants predictions, return them in addition to metrics
    if return_predictions:
        return {
            "metrics": metrics,
            "train_predictions": train_predictions,
            "test_predictions": test_predictions
        }
    else:
        return metrics

def plot_predictions(
    y_true,
    y_pred,
    title="Actual vs. Predict",
    plot_type="scatter",
    log_to_mlflow=True,
    filename="predictions_plot.png"
):
    """
    Plot predictions vs actual values and optionally log the plot as an MLflow artifact.
    
    Parameters:
    - y_true: Actual values.
    - y_pred: Predicted values.
    - title: Plot title.
    - plot_type: Type of plot ("scatter" or "line").
    - log_to_mlflow: Boolean flag to log the figure to MLflow.
    - filename: Filename for the saved figure artifact.
    """
    plt.figure(figsize=(10, 6))
    
    if plot_type == "scatter":
        sns.scatterplot(x=y_true, y=y_pred)
        # Diagonal line
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
