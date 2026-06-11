from tsla_preprocessing import (
    load_tsla_data,
    handle_missing_values,
    select_target_feature,
    scale_series,
    create_time_series_sequences,
    split_train_test,
    evaluate_and_plot,
)
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf


def calculate_accuracy_metrics(y_true, y_pred):
    """Calculate comprehensive accuracy metrics for model evaluation."""
    # Ensure proper shapes
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2,
        'MAPE': mape
    }


def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "TSLA.csv")
    df = load_tsla_data(dataset_path)
    df = handle_missing_values(df)
    target_df = select_target_feature(df, target_column="Adj Close")

    scaled_values, scaler = scale_series(target_df)
    window_size = 60
    forecast_horizon = 1

    X, y = create_time_series_sequences(
        scaled_values, window_size=window_size, forecast_horizon=forecast_horizon
    )

    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio=0.8)

    # Evaluate saved models
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    model_results = []
    
    print("=" * 80)
    print("MODEL EVALUATION REPORT - TESLA STOCK PRICE PREDICTION")
    print("=" * 80)
    print()
    
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".h5") or model_file.endswith(".keras"):
            model_path = os.path.join(models_dir, model_file)
            model_name = os.path.splitext(model_file)[0]
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Make predictions
            preds = model.predict(X_test, verbose=0)
            preds = np.asarray(preds).reshape(-1, 1)
            y_test_arr = np.asarray(y_test).reshape(-1, 1)
            
            # Inverse transform to get actual prices
            preds_inv = scaler.inverse_transform(preds)
            y_true_inv = scaler.inverse_transform(y_test_arr)
            
            # Calculate accuracy metrics
            metrics = calculate_accuracy_metrics(y_true_inv, preds_inv)
            
            # Store results
            result = {
                'Model': model_name,
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2_Score': metrics['R2_Score'],
                'MAPE': metrics['MAPE']
            }
            model_results.append(result)
            
            # Print model accuracy
            print(f"📊 Model: {model_name.upper()}")
            print("-" * 80)
            print(f"  Mean Squared Error (MSE):        {metrics['MSE']:.6f}")
            print(f"  Root Mean Squared Error (RMSE):  ${metrics['RMSE']:.2f}")
            print(f"  Mean Absolute Error (MAE):       ${metrics['MAE']:.2f}")
            print(f"  R² Score:                        {metrics['R2_Score']:.4f}")
            print(f"  Mean Absolute Percentage Error:  {metrics['MAPE']:.2f}%")
            print()
            
            # Also generate the plot
            evaluate_and_plot(model_path, X_test, y_test, scaler, model_name)
    
    # Summary table
    if model_results:
        print("=" * 80)
        print("ACCURACY SUMMARY TABLE")
        print("=" * 80)
        results_df = pd.DataFrame(model_results)
        print(results_df.to_string(index=False))
        print("=" * 80)
        
        # Find best model by R² score
        best_model = results_df.loc[results_df['R2_Score'].idxmax()]
        print(f"\n🏆 Best Model (by R² Score): {best_model['Model']}")
        print(f"   R² Score: {best_model['R2_Score']:.4f}")
        print(f"   RMSE: ${best_model['RMSE']:.2f}")
        print()


if __name__ == "__main__":
    main()
