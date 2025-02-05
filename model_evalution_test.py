import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from model_evaluation import evaluate_model, plot_predictions

# Create synthetic data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y = X @ np.array([1.5, -2.0, 1.0, 0.5, 2.0]) + np.random.randn(100) * 0.5  # Linear combination with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

# Print the evaluation metrics
print(metrics)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Plot predictions vs actual values
plot_predictions(y_test, y_pred, title="Predictions vs Actual (Synthetic Data)")