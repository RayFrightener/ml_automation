from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the model and print performance metrics.
    
    Parameters:
    - model: Trained model
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels
    
    Returns:
    - Dictionary containing performance metrics

    When to Use the evaluate_model Function?

    For Initial Training: 
    You can use this function right after you initially train your model 
    to get an idea of its performance on both the training and testing datasets.

    After Hyperparameter Tuning: 
    If you have already tuned your model
    you can use this function to evaluate the model's performance.
    """
    # Make predictions on the training data
    train_predictions = model.predict(X_train)
    
    # Calculate RMSE for Training Data
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    
    # Make predictions on the testing data
    test_predictions = model.predict(X_test)
    
    # Calculate RMSE for Testing Data
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    
    # Calculate other metrics
    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    train_mape = mean_absolute_percentage_error(y_train, train_predictions)
    test_mape = mean_absolute_percentage_error(y_test, test_predictions)
    
    # Print metrics
    print(f"Training RMSE: {train_rmse}")
    print(f"Testing RMSE: {test_rmse}")
    print(f"Training MAE: {train_mae}")
    print(f"Testing MAE: {test_mae}")
    print(f"Training MSE: {train_mse}")
    print(f"Testing MSE: {test_mse}")
    print(f"Training R2: {train_r2}")
    print(f"Testing R2: {test_r2}")
    print(f"Training MAPE: {train_mape}")
    print(f"Testing MAPE: {test_mape}")
    
    # Return metrics as a dictionary
    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mape": train_mape,
        "test_mape": test_mape
    }

def plot_predictions(y_true, y_pred, title="Predictions vs Actual", plot_type="scatter"):
    """
    Plot predictions vs actual values.
    
    Parameters:
    - y_true: Actual values
    - y_pred: Predicted values
    - title: Plot title
    - plot_type: Type of plot ("scatter" or "line")
    """
    plt.figure(figsize=(10, 6))
    
    if plot_type == "scatter":
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    elif plot_type == "line":
        plt.plot(y_true, marker='o', linestyle='-', label='Actual')
        plt.plot(y_pred, marker='o', linestyle='-', label='Predicted')
        plt.legend()
    
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()