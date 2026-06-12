# ===========================================
# SECTION 1: IMPORTING REQUIRED LIBRARIES
# ===========================================

import os  # For file and directory operations (saving models, checking file paths)
from typing import Tuple  # For type hints (specifies function return types)
import numpy as np  # For numerical operations on arrays
import pandas as pd  # For data manipulation and reading CSV files
from sklearn.preprocessing import MinMaxScaler  # For normalizing data to [0,1] range
import tensorflow as tf  # Main deep learning framework
from tensorflow.keras import Sequential  # For creating sequential neural network models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # For training optimization
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN  # Neural network layers
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
from sklearn.metrics import mean_squared_error  # For evaluating model performance


# ===========================================
# SECTION 2: DATA LOADING FUNCTIONS
# ===========================================

def load_tsla_data(csv_path: str) -> pd.DataFrame:
    """
    Load Tesla stock data from CSV file and prepare date index.
    
    Args:
        csv_path: Path to the TSLA.csv file
        
    Returns:
        DataFrame with Date as index, sorted chronologically
    """
    # Check if the file exists at the given path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    # Read CSV file and parse the 'Date' column as datetime objects
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    
    # Sort by date (oldest to newest) - CRITICAL for time series
    df.sort_values("Date", inplace=True)
    
    # Set the Date column as the DataFrame index for time-based operations
    df.set_index("Date", inplace=True)
    
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Print basic dataset information for exploratory data analysis.
    
    Args:
        df: DataFrame containing stock data
    """
    print("=== Dataset preview ===")
    print(df.head())  # Show first 5 rows
    print("\n=== Data types ===")
    print(df.dtypes)  # Show data type of each column
    print("\n=== Missing values ===")
    print(df.isna().sum())  # Count missing values per column
    print("\n=== Descriptive statistics ===")
    print(df.describe())  # Statistical summary (mean, std, min, max, etc.)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in time-series data.
    
    Uses forward fill (carry previous value forward) then backward fill
    to ensure no gaps in the sequence.
    
    Args:
        df: DataFrame that may contain missing values
        
    Returns:
        DataFrame with all missing values filled
    """
    # If no missing values exist, return original DataFrame
    if df.isna().sum().sum() == 0:
        return df
    
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Forward fill: replace NaN with previous valid value
    df = df.ffill()
    
    # Backward fill: replace any remaining NaN with next valid value
    df = df.bfill()
    
    return df


def select_target_feature(df: pd.DataFrame, target_column: str = "Adj Close") -> pd.DataFrame:
    """
    Select the target feature (column) for prediction.
    
    Adjusted Close is used because it accounts for stock splits and dividends.
    
    Args:
        df: DataFrame containing all stock data
        target_column: Name of the column to predict (default: "Adj Close")
        
    Returns:
        DataFrame with only the target column
    """
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")
    
    # Extract only the target column as a new DataFrame
    target_df = df[[target_column]].copy()
    
    return target_df


# ===========================================
# SECTION 3: DATA PREPROCESSING FUNCTIONS
# ===========================================

def scale_series(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize the time series data to the range [0, 1].
    
    Neural networks learn better when input values are in a small range.
    Scaling to [0,1] helps gradient descent converge faster.
    
    Args:
        df: DataFrame with the target column
        
    Returns:
        - scaled: NumPy array of normalized values
        - scaler: Fitted scaler object (for inverse transformation later)
    """
    # Initialize MinMaxScaler to scale values between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Reshape from (n_samples,) to (n_samples, 1) as required by sklearn
    # Fit the scaler on the data and transform in one step
    scaled = scaler.fit_transform(df.values.reshape(-1, 1))
    
    return scaled, scaler


def create_time_series_sequences(
    scaled_values: np.ndarray,
    window_size: int = 60,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series into supervised learning format.
    
    Creates overlapping input-output pairs:
    - Input (X): 'window_size' consecutive past days
    - Output (y): Next 'forecast_horizon' days
    
    Example with window_size=3, forecast_horizon=1:
    [Day1, Day2, Day3] -> Day4
    [Day2, Day3, Day4] -> Day5
    
    Args:
        scaled_values: Normalized price array of shape (n_samples, 1)
        window_size: Number of past days to use as input
        forecast_horizon: Number of future days to predict
        
    Returns:
        - X: Input sequences of shape (samples, window_size, 1)
        - y: Target values of shape (samples, 1)
    """
    X, y = [], []  # Empty lists to store sequences
    n_samples = len(scaled_values)  # Total number of data points
    
    # Loop through the data to create overlapping sequences
    # Start index goes from 0 to (last possible start position)
    for start_idx in range(n_samples - window_size - forecast_horizon + 1):
        # Define the end index of the input window
        end_idx = start_idx + window_size
        
        # Add input sequence (window_size consecutive days)
        X.append(scaled_values[start_idx:end_idx, 0])
        
        # Add target value (next forecast_horizon days)
        y.append(scaled_values[end_idx:end_idx + forecast_horizon, 0])
    
    # Convert lists to numpy arrays for efficient computation
    X = np.array(X)
    y = np.array(y)
    
    # For single-step prediction, reshape y to (samples, 1)
    if y.shape[1] == 1:
        y = y.reshape(-1, 1)
    
    # Reshape X to 3D format required by RNN/LSTM:
    # (samples, timesteps, features)
    # Here, features = 1 because we only use price (univariate time series)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets chronologically.
    
    IMPORTANT: No random shuffling is performed because time series data
    must maintain chronological order. We cannot train on future data.
    
    Args:
        X: Input sequences
        y: Target values
        train_ratio: Proportion of data to use for training (0.8 = 80%)
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Validate that train_ratio is between 0 and 1
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")
    
    # Calculate the split index (e.g., 80% of data)
    split_index = int(len(X) * train_ratio)
    
    # First split_index samples go to training, rest to testing
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    return X_train, X_test, y_train, y_test


# ===========================================
# SECTION 4: MODEL BUILDING FUNCTIONS
# ===========================================

def build_sequence_model(
    model_type: str,
    input_shape: Tuple[int, int],
    units: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
) -> tf.keras.Model:
    """
    Build and compile a SimpleRNN or LSTM model for time series prediction.
    
    Model architecture:
    - Recurrent layer (SimpleRNN or LSTM) with 'units' memory cells
    - Dropout layer to prevent overfitting (randomly disables 20% of neurons)
    - Dense output layer with linear activation (for regression)
    
    Args:
        model_type: 'simplernn' or 'lstm'
        input_shape: Shape of input data (window_size, n_features)
        units: Number of memory cells in the recurrent layer
        dropout_rate: Fraction of neurons to randomly disable (0.2 = 20%)
        learning_rate: Step size for gradient descent optimization
        optimizer_name: 'adam' (adaptive) or 'sgd' (standard)
        
    Returns:
        Compiled Keras model ready for training
    """
    # Choose the recurrent layer based on model_type
    if model_type.lower() == "simplernn":
        # SimpleRNN: Basic recurrent network, prone to vanishing gradient
        recurrent_layer = SimpleRNN(units, return_sequences=False, input_shape=input_shape)
    elif model_type.lower() == "lstm":
        # LSTM: Advanced with forget/input/output gates, handles long-term dependencies
        recurrent_layer = LSTM(units, return_sequences=False, input_shape=input_shape)
    else:
        raise ValueError("model_type must be 'simplernn' or 'lstm'.")
    
    # Build the sequential model (layers stacked in order)
    model = Sequential([
        recurrent_layer,      # RNN or LSTM layer
        Dropout(dropout_rate), # Dropout for regularization
        Dense(1, activation="linear"),  # Output layer for price prediction
    ])
    
    # Configure the optimizer
    if optimizer_name.lower() == "adam":
        # Adam: Adaptive Moment Estimation (adjusts learning rate automatically)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        # SGD: Standard Stochastic Gradient Descent (fixed learning rate)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sgd'.")
    
    # Compile the model with loss function and metrics
    # MSE (Mean Squared Error) penalizes large errors more heavily
    model.compile(
        loss="mean_squared_error",
        optimizer=optimizer,
        metrics=["mean_squared_error"]
    )
    
    return model


# ===========================================
# SECTION 5: MODEL TRAINING FUNCTIONS
# ===========================================

def train_sequence_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    model_dir: str = "models",
) -> Tuple[tf.keras.callbacks.History, str]:
    """
    Train the model with early stopping and automatic checkpointing.
    
    Early stopping prevents overfitting by stopping when validation loss stops improving.
    ModelCheckpoint saves only the best model (lowest validation loss).
    
    Args:
        model: Compiled Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data (for monitoring during training)
        model_name: Name to save the model as
        epochs: Maximum number of training epochs
        batch_size: Number of samples per gradient update
        patience: How many epochs to wait before early stopping
        model_dir: Directory to save trained models
        
    Returns:
        - history: Training history object (loss values per epoch)
        - model_path: Path where the best model was saved
    """
    # Create the models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    
    # Remove existing model file if it exists (to avoid conflicts)
    if os.path.exists(model_path):
        try:
            os.remove(model_path)
        except OSError as exc:
            raise OSError(
                f"Unable to remove existing checkpoint file at {model_path}: {exc}"
            ) from exc
    
    # Define training callbacks
    callbacks = [
        # EarlyStopping: Stop training if val_loss doesn't improve for 'patience' epochs
        # restore_best_weights: Revert to the best model when stopping
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        # ModelCheckpoint: Save the model only when val_loss improves
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,  # Only save if this epoch has better val_loss
            verbose=1,
        ),
    ]
    
    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),  # Data to evaluate after each epoch
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,  # 2 = one line per epoch
    )
    
    return history, model_path


# ===========================================
# SECTION 6: EVALUATION & VISUALIZATION
# ===========================================

def evaluate_and_plot(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
    model_name: str
) -> float:
    """
    Load saved model, make predictions, evaluate with MSE, and plot results.
    
    Critical step: Inverse transform predictions back to original dollar scale
    so results are interpretable (e.g., $150.50 instead of 0.67).
    
    Args:
        model_path: Path to saved .keras model file
        X_test, y_test: Test data (scaled)
        scaler: Fitted MinMaxScaler for inverse transformation
        model_name: Name for labeling the plot
        
    Returns:
        MSE (Mean Squared Error) in original dollar scale
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Skipping evaluation for {model_name}.")
        return float("nan")
    
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    
    # Make predictions on test data
    preds = model.predict(X_test)
    
    # Ensure arrays are 2D with shape (n_samples, 1)
    preds = np.asarray(preds).reshape(-1, 1)
    y_test_arr = np.asarray(y_test).reshape(-1, 1)
    
    # Inverse transform: Convert from scaled [0,1] back to actual dollar values
    preds_inv = scaler.inverse_transform(preds)
    y_true_inv = scaler.inverse_transform(y_test_arr)
    
    # Calculate MSE on original dollar scale (interpretable metric)
    mse = mean_squared_error(y_true_inv, preds_inv)
    
    # Create and save the comparison plot
    plt.figure(figsize=(10, 5))  # 10 inches wide, 5 inches tall
    
    # Plot actual vs predicted values
    plt.plot(y_true_inv, label="Actual", linewidth=1.5)
    plt.plot(preds_inv, label=f"Predicted ({model_name})", linewidth=1.5, alpha=0.8)
    
    # Add labels, title, and legend
    plt.title(f"Actual vs Predicted - {model_name} (MSE={mse:.4f})")
    plt.xlabel("Test sample index")
    plt.ylabel("Price (USD)")
    plt.legend()
    
    # Save the plot as PNG file
    plot_path = os.path.join("models", f"{model_name}_prediction.png")
    plt.tight_layout()  # Adjust spacing to prevent cutting off labels
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    
    print(f"Saved prediction plot to: {plot_path}")
    print(f"{model_name} Test MSE: {mse:.6f}")
    
    return mse


# ===========================================
# SECTION 7: MAIN EXECUTION PIPELINE
# ===========================================

if __name__ == "__main__":
    """
    This is the main entry point of the script.
    Orchestrates the entire pipeline from data loading to model comparison.
    """
    
    # Step 1: Define dataset path (expects TSLA.csv in the same directory as this script)
    dataset_path = os.path.join(os.path.dirname(__file__), "TSLA.csv")
    
    # Step 2: Load and explore the data
    df = load_tsla_data(dataset_path)
    explore_data(df)  # Print dataset information
    
    # Step 3: Preprocess data
    df = handle_missing_values(df)  # Fill any missing values
    target_df = select_target_feature(df, target_column="Adj Close")  # Use Adjusted Close price
    
    # Step 4: Scale the data to [0, 1] range
    scaled_values, scaler = scale_series(target_df)
    
    # Step 5: Define sequence parameters
    window_size = 60  # Use 60 days of past data to predict
    forecast_horizon = 1  # Predict next day's price
    
    # Step 6: Create input-output sequences
    X, y = create_time_series_sequences(
        scaled_values,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
    )
    
    # Step 7: Split into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio=0.8)
    
    # Step 8: Further split training data into train (80%) and validation (20%)
    val_ratio = 0.2  # Use 20% of training data for validation
    val_split = int(len(X_train) * (1 - val_ratio))
    X_train_sub, X_val = X_train[:val_split], X_train[val_split:]
    y_train_sub, y_val = y_train[:val_split], y_train[val_split:]
    
    # Print dataset shapes for verification
    print(f"\nCreated sequences: X={X.shape}, y={y.shape}")
    print(f"Training split: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Validation split: X_val={X_val.shape}, y_val={y_val.shape}")
    
    # Step 9: Train both models (SimpleRNN and LSTM)
    model_types = ["simplernn", "lstm"]
    saved_model_paths = []  # List to store (model_type, model_path) tuples
    
    for model_type in model_types:
        print(f"\nBuilding and training {model_type.upper()} model...")
        
        # Build the model
        model = build_sequence_model(
            model_type=model_type,
            input_shape=(window_size, 1),
            units=50,  # 50 memory cells
            dropout_rate=0.2,  # 20% dropout for regularization
            learning_rate=0.001,
            optimizer_name="adam",
        )
        
        # Display model architecture summary
        model.summary()
        
        # Train the model
        history, model_path = train_sequence_model(
            model,
            X_train_sub,
            y_train_sub,
            X_val,
            y_val,
            model_name=f"tsla_{model_type}",
            epochs=50,  # Maximum epochs (early stopping will likely stop earlier)
            batch_size=32,
            patience=10,  # Stop if no improvement for 10 epochs
            model_dir="models",
        )
        
        print(f"{model_type.upper()} training complete. Best model saved to: {model_path}")
        saved_model_paths.append((model_type, model_path))
    
    # Step 10: Evaluate both models on the test set
    model_results = []
    for model_type, model_path in saved_model_paths:
        model_name = f"tsla_{model_type}"
        mse = evaluate_and_plot(model_path, X_test, y_test, scaler, model_name)
        model_results.append((model_name, mse))
    
    # Step 11: Compare and display results
    print("\nModel comparison results:")
    for model_name, mse in model_results:
        print(f"  {model_name}: MSE = {mse:.6f}")
    
    # Identify the best performing model (lowest MSE)
    if len(model_results) >= 2:
        best_model = min(model_results, key=lambda item: item[1])
        print(f"\nBest performing model: {best_model[0]} with MSE={best_model[1]:.6f}")
        
        # Print limitations and suggestions for improvement
        print("This comparison shows which model better captured the test-set price trend under the current setup.")
        print("Limitations: both models use only past adjusted-close values, so they may miss market-moving news, volume shifts, and macroeconomic effects.")
        print("Improvements: add technical indicators, trading volume, sentiment features, or macroeconomic variables, and consider a larger dataset or ensemble methods.")
    else:
        print("\nInsufficient models were evaluated to compare performance.")
    
    print("\nPreprocessing, training, and evaluation are complete.")
