# ===========================================
# SECTION 1: IMPORTING REQUIRED MODULES
# ===========================================

# Import preprocessing functions from the external module
# These functions handle data loading, cleaning, scaling, and sequence creation
from tsla_preprocessing import (
    load_tsla_data,           # Loads Tesla stock CSV and parses dates
    handle_missing_values,    # Fills missing values using forward/backward fill
    select_target_feature,    # Extracts the 'Adj Close' column for prediction
    scale_series,             # Normalizes data to [0, 1] range using MinMaxScaler
    create_time_series_sequences,  # Creates sliding window sequences (60 days input -> 1 day output)
    split_train_test,         # Splits data chronologically (80% train, 20% test)
    evaluate_and_plot,        # Generates actual vs predicted price plots
)

import os                     # For file and directory operations (scanning models folder)
import numpy as np            # For numerical operations on arrays
import pandas as pd           # For creating summary DataFrames
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Evaluation metrics
import tensorflow as tf       # For loading saved Keras models


# ===========================================
# SECTION 2: METRICS CALCULATION FUNCTION
# ===========================================

def calculate_accuracy_metrics(y_true, y_pred):
    """
    Calculate comprehensive accuracy metrics for model evaluation.
    
    This function computes multiple metrics to evaluate prediction quality:
    - MSE: Mean Squared Error (penalizes large errors heavily)
    - RMSE: Root Mean Squared Error (error in same units as price - dollars)
    - MAE: Mean Absolute Error (average absolute error in dollars)
    - R² Score: Coefficient of determination (how well model explains variance)
    - MAPE: Mean Absolute Percentage Error (percentage-based error, scale-independent)
    
    Args:
        y_true: Actual values (original dollar scale)
        y_pred: Predicted values (original dollar scale)
        
    Returns:
        Dictionary containing all five accuracy metrics
    """
    # Ensure both arrays are 2D with shape (n_samples, 1) for consistent operations
    y_true = np.asarray(y_true).reshape(-1, 1)
    y_pred = np.asarray(y_pred).reshape(-1, 1)
    
    # Calculate Mean Squared Error - average of squared differences
    # Larger errors are penalized more due to squaring
    mse = mean_squared_error(y_true, y_pred)
    
    # Calculate Root Mean Squared Error - square root of MSE
    # Returns error in the same unit as price (dollars) - easy to interpret
    rmse = np.sqrt(mse)
    
    # Calculate Mean Absolute Error - average of absolute differences
    # Less sensitive to outliers than MSE
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate R² Score (Coefficient of Determination)
    # Range: 0 to 1 (1 = perfect prediction, 0 = same as predicting mean)
    # Example: R² = 0.94 means model explains 94% of price variance
    r2 = r2_score(y_true, y_pred)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Computes percentage error for each prediction, then averages
    # Formula: |(Actual - Predicted) / Actual| * 100
    # Add small epsilon to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Return all metrics as a dictionary for easy access
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2_Score': r2,
        'MAPE': mape
    }


# ===========================================
# SECTION 3: MAIN EXECUTION FUNCTION
# ===========================================

def main():
    """
    Main function that orchestrates the entire evaluation pipeline.
    
    Steps:
    1. Load and preprocess the Tesla stock data
    2. Scale data and create sequences
    3. Split into train/test sets
    4. Load all saved models from 'models' directory
    5. Evaluate each model using multiple accuracy metrics
    6. Generate comparison table and identify best model
    7. Create prediction plots for each model
    """
    
    # ===== STEP 1: DATA LOADING AND PREPROCESSING =====
    
    # Define the path to the Tesla CSV file (located in same directory as this script)
    dataset_path = os.path.join(os.path.dirname(__file__), "TSLA.csv")
    
    # Load raw data from CSV, parse dates, sort chronologically
    df = load_tsla_data(dataset_path)
    
    # Handle missing values using forward fill then backward fill
    # This ensures no gaps in the time series
    df = handle_missing_values(df)
    
    # Select only the target column for prediction (Adjusted Close price)
    # Adjusted Close accounts for stock splits and dividends
    target_df = select_target_feature(df, target_column="Adj Close")
    
    # Scale the data to [0, 1] range - helps neural networks converge faster
    # Save the scaler to inverse transform predictions back to dollar values
    scaled_values, scaler = scale_series(target_df)
    
    # ===== STEP 2: SEQUENCE CREATION =====
    
    # Define window size: use 60 past days to predict the next day
    window_size = 60
    
    # Define forecast horizon: predict 1 day ahead
    forecast_horizon = 1
    
    # Create input-output pairs for supervised learning
    # X: 60-day sequences, y: next day's price
    X, y = create_time_series_sequences(
        scaled_values,
        window_size=window_size,
        forecast_horizon=forecast_horizon
    )
    
    # Split data chronologically: 80% for training, 20% for testing
    # IMPORTANT: No random shuffling to preserve temporal order
    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio=0.8)
    
    # ===== STEP 3: MODEL EVALUATION =====
    
    # Define the directory where trained models are stored
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # List to store evaluation results for all models
    model_results = []
    
    # Print header for the evaluation report
    print("=" * 80)
    print("MODEL EVALUATION REPORT - TESLA STOCK PRICE PREDICTION")
    print("=" * 80)
    print()
    
    # Loop through all model files in the models directory
    # Supports both .h5 (older Keras format) and .keras (newer format)
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".h5") or model_file.endswith(".keras"):
            
            # Construct full path to the model file
            model_path = os.path.join(models_dir, model_file)
            
            # Extract model name without extension (e.g., 'tsla_lstm' from 'tsla_lstm.keras')
            model_name = os.path.splitext(model_file)[0]
            
            # ===== STEP 3.1: LOAD AND PREDICT =====
            
            # Load the saved Keras model from disk
            model = tf.keras.models.load_model(model_path)
            
            # Make predictions on the test set
            # verbose=0 suppresses progress bars for cleaner output
            preds = model.predict(X_test, verbose=0)
            
            # Ensure predictions are in correct shape (n_samples, 1)
            preds = np.asarray(preds).reshape(-1, 1)
            y_test_arr = np.asarray(y_test).reshape(-1, 1)
            
            # Inverse transform: convert from scaled [0,1] back to actual dollar prices
            # This makes metrics interpretable (e.g., error is $5.50, not 0.03)
            preds_inv = scaler.inverse_transform(preds)
            y_true_inv = scaler.inverse_transform(y_test_arr)
            
            # ===== STEP 3.2: CALCULATE METRICS =====
            
            # Calculate all five accuracy metrics
            metrics = calculate_accuracy_metrics(y_true_inv, preds_inv)
            
            # Store results in a dictionary for later comparison
            result = {
                'Model': model_name,
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'R2_Score': metrics['R2_Score'],
                'MAPE': metrics['MAPE']
            }
            model_results.append(result)
            
            # ===== STEP 3.3: PRINT INDIVIDUAL MODEL RESULTS =====
            
            print(f"📊 Model: {model_name.upper()}")
            print("-" * 80)
            print(f"  Mean Squared Error (MSE):        {metrics['MSE']:.6f}")
            print(f"  Root Mean Squared Error (RMSE):  ${metrics['RMSE']:.2f}")
            print(f"  Mean Absolute Error (MAE):       ${metrics['MAE']:.2f}")
            print(f"  R² Score:                        {metrics['R2_Score']:.4f}")
            print(f"  Mean Absolute Percentage Error:  {metrics['MAPE']:.2f}%")
            print()
            
            # ===== STEP 3.4: GENERATE PLOT =====
            
            # Create and save actual vs predicted price plot
            # Plot saved as 'models/{model_name}_prediction.png'
            evaluate_and_plot(model_path, X_test, y_test, scaler, model_name)
    
    # ===== STEP 4: SUMMARY AND COMPARISON =====
    
    # If at least one model was evaluated, create comparison summary
    if model_results:
        print("=" * 80)
        print("ACCURACY SUMMARY TABLE")
        print("=" * 80)
        
        # Convert results list to pandas DataFrame for nice table formatting
        results_df = pd.DataFrame(model_results)
        
        # Print the summary table without row numbers (index=False)
        print(results_df.to_string(index=False))
        print("=" * 80)
        
        # Find the best model based on highest R² Score
        # R² close to 1.0 indicates excellent prediction accuracy
        best_model = results_df.loc[results_df['R2_Score'].idxmax()]
        
        print(f"\n🏆 Best Model (by R² Score): {best_model['Model']}")
        print(f"   R² Score: {best_model['R2_Score']:.4f}")
        print(f"   RMSE: ${best_model['RMSE']:.2f}")
        print()


# ===========================================
# SECTION 4: SCRIPT ENTRY POINT
# ===========================================

# This ensures the main() function only runs when this script is executed directly
# (not when imported as a module by another script)
if __name__ == "__main__":
    main()
