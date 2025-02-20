#!/usr/bin/env python
"""
run_pipeline.py

This is the main entry point for running the Homeowner Loss History Prediction
pipeline. It initializes the HyperparamTracker with the S3 URIs (temporary solution),
runs hyperparameter tuning for the available models, and then performs model evaluation.
The logged runs can be reviewed in the MLflow UI, where you can use the model registry
features to manually approve or reject hyperparameter suggestions if desired.
"""

import mlflow
from hyperparameter_tracker import HyperparamTracker
from model_evaluation import plot_predictions
import argparse

def main():
    # Example S3 URI for demonstration; in practice, these should be actual S3 paths.
    s3_uri = "s3://grangeprojectbucket/ut_loss_history_1.csv"
    
    # Initialize MLflow experiment (set an experiment name)
    mlflow.set_experiment("HomeownerLossHistoryPrediction")
    
    # Instantiate the HyperparamTracker with the same URI for train/test for demo purposes.
    tracker = HyperparamTracker(
        s3_uri_X_train=s3_uri,
        s3_uri_Y_train=s3_uri,
        s3_uri_X_train_balanced=s3_uri,
        s3_uri_Y_train_balanced=s3_uri,
        s3_uri_X_test=s3_uri,
        s3_uri_Y_test=s3_uri,
        data_version="v1.0",
        model_registry_name_prefix="HomeownerLossModel"
    )
    
    # Run hyperparameter tuning for all available models.
    # This will log the tuning runs and best parameters to MLflow.
    tracker.tune_all_models(max_evals=50, early_stop_rounds=10)
    
    # For demonstration, you may want to evaluate a particular model and plot predictions.
    # Here we pick Model 1 for example purposes.
    model_key = "Model_1"
    if model_key in tracker.models:
        model = tracker.models[model_key]
        # Start an MLflow run for evaluation
        with mlflow.start_run(run_name="ModelEvaluation_" + model_key):
            # Evaluate model (note: this uses the updated evaluate_model function)
            # For Model 1, no inverse transform is applied.
            metrics = tracker.logger.info("Evaluating " + model_key)
            eval_metrics = tracker.__class__.evaluate_model(
                model,
                tracker.X_train,
                tracker.Y_train,
                tracker.X_test,
                tracker.Y_test,
                log_to_mlflow=True,
                return_predictions=False,
                model_name=f"{tracker.model_registry_name_prefix}_{model_key}",
                summary_only=True,
                target_transformer=None,  # Not used for Model 1
                sample_weight=None         # Provide weights if available
            )
            # Optionally, plot the Actual vs. Predicted graph.
            predictions = model.predict(tracker.X_test)
            plot_predictions(tracker.Y_test, predictions, title=f"{model_key}: Actual vs. Predicted")
    
    # Note: The human-in-the-loop approval (accept/reject hyperparameter suggestion)
    # can be implemented externally. You can review runs in the MLflow UI and then
    # manually invoke the approval function (e.g., tracker.approve_suggested_hyperparameters)
    # using the MLflow API or via a custom dashboard.
    print("Pipeline run complete. Please review the MLflow UI for details and perform any manual approval if necessary.")

if __name__ == "__main__":
    main()
