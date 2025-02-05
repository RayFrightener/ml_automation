from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_train, y_train, X_test, y_test, sample_weight=None):
    """
    Evaluate the model and print performance metrics.
    
    Parameters:
    - model: Trained model
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - y_test: Testing labels
    - sample_weight: Sample weights for training data (optional)
    
    Returns:
    - Dictionary containing performance metrics
    """
    # Fit the model on the training data
    model.fit(X=X_train, y=y_train, sample_weight=sample_weight)
    
    # Make predictions on the training data
    train_predictions = model.predict(X_train)
    
    # Calculate RMSE for Training Data
    train_rmse = mean_squared_error(y_train, train_predictions, squared=False)
    
    # Make predictions on the testing data
    test_predictions = model.predict(X_test)
    
    # Calculate RMSE for Testing Data
    test_rmse = mean_squared_error(y_test, test_predictions, squared=False)
    
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

def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
    """
    Plot predictions vs actual values.
    
    Parameters:
    - y_true: Actual values
    - y_pred: Predicted values
    - title: Plot title
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()