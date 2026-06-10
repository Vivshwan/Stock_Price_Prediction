import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, LSTM, SimpleRNN
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error



def load_tsla_data(csv_path: str) -> pd.DataFrame:
    """Load Tesla stock data from CSV and parse the date index."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Date"] )
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df


def explore_data(df: pd.DataFrame) -> None:
    """Print basic dataset information and missing-value summary."""
    print("=== Dataset preview ===")
    print(df.head())
    print("\n=== Data types ===")
    print(df.dtypes)
    print("\n=== Missing values ===")
    print(df.isna().sum())
    print("\n=== Descriptive statistics ===")
    print(df.describe())


def select_target_feature(df: pd.DataFrame, target_column: str = "Adj Close") -> pd.DataFrame:
    """Select the target feature and keep the time index."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset.")

    target_df = df[[target_column]].copy()
    return target_df


def scale_series(df: pd.DataFrame) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale the target series to the range [0, 1]."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df.values.reshape(-1, 1))
    return scaled, scaler


def create_time_series_sequences(
    scaled_values: np.ndarray,
    window_size: int = 60,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create input-output pairs for time-series forecasting.

    Args:
        scaled_values: Array of shape (n_samples, 1) containing normalized values.
        window_size: Number of past days used as input.
        forecast_horizon: Number of days ahead to predict.

    Returns:
        Tuple of `X` and `y` arrays suitable for RNN/LSTM training.
    """
    X, y = [], []
    n_samples = len(scaled_values)

    for start_idx in range(n_samples - window_size - forecast_horizon + 1):
        end_idx = start_idx + window_size
        X.append(scaled_values[start_idx:end_idx, 0])
        y.append(scaled_values[end_idx:end_idx + forecast_horizon, 0])

    X = np.array(X)
    y = np.array(y)

    # For single-step forecasts, reshape y to (n_samples, 1)
    if y.shape[1] == 1:
        y = y.reshape(-1, 1)

    # Reshape X to (samples, window_size, 1) for RNN/LSTM input
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


def split_train_test(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split input-output pairs into train and test sets."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in a time-series aware manner.

    We forward-fill and then backward-fill remaining gaps to preserve continuity.
    """
    if df.isna().sum().sum() == 0:
        return df

    df = df.copy()
    df = df.ffill()
    df = df.bfill()
    return df


def build_sequence_model(
    model_type: str,
    input_shape: Tuple[int, int],
    units: int = 50,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
) -> tf.keras.Model:
    """Build and compile a SimpleRNN or LSTM model.

    Args:
        model_type: 'simplernn' or 'lstm'.
        input_shape: Shape of the input sequence (window_size, n_features).
        units: Number of recurrent units.
        dropout_rate: Dropout fraction between 0 and 1.
        learning_rate: Learning rate for the optimizer.
        optimizer_name: 'adam' or 'sgd'.

    Returns:
        Compiled Keras model.
    """
    if model_type.lower() == "simplernn":
        recurrent_layer = SimpleRNN(units, return_sequences=False, input_shape=input_shape)
    elif model_type.lower() == "lstm":
        recurrent_layer = LSTM(units, return_sequences=False, input_shape=input_shape)
    else:
        raise ValueError("model_type must be 'simplernn' or 'lstm'.")

    model = Sequential([
        recurrent_layer,
        Dropout(dropout_rate),
        Dense(1, activation="linear"),
    ])

    if optimizer_name.lower() == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("optimizer_name must be 'adam' or 'sgd'.")

    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mean_squared_error"])
    return model


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
    """Train the model with early stopping and checkpointing."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.keras")

    if os.path.exists(model_path):
        try:
            os.remove(model_path)
        except OSError as exc:
            raise OSError(
                f"Unable to remove existing checkpoint file at {model_path}: {exc}"
            ) from exc

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    return history, model_path


def evaluate_and_plot(model_path: str, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler, model_name: str) -> float:
    """Load a saved Keras model, predict on X_test, compute MSE, and plot results.

    Returns the MSE on the unscaled values.
    """
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}. Skipping evaluation for {model_name}.")
        return float("nan")

    model = tf.keras.models.load_model(model_path)
    preds = model.predict(X_test)

    # Ensure shapes are (n,1)
    preds = np.asarray(preds).reshape(-1, 1)
    y_test_arr = np.asarray(y_test).reshape(-1, 1)

    preds_inv = scaler.inverse_transform(preds)
    y_true_inv = scaler.inverse_transform(y_test_arr)

    mse = mean_squared_error(y_true_inv, preds_inv)

    plt.figure(figsize=(10, 5))
    plt.plot(y_true_inv, label="Actual")
    plt.plot(preds_inv, label=f"Predicted ({model_name})")
    plt.title(f"Actual vs Predicted - {model_name} (MSE={mse:.4f})")
    plt.xlabel("Test sample index")
    plt.ylabel("Price")
    plt.legend()
    plot_path = os.path.join("models", f"{model_name}_prediction.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved prediction plot to: {plot_path}")
    print(f"{model_name} Test MSE: {mse:.6f}")
    return mse



if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "TSLA.csv")

    df = load_tsla_data(dataset_path)
    explore_data(df)

    df = handle_missing_values(df)
    target_df = select_target_feature(df, target_column="Adj Close")

    scaled_values, scaler = scale_series(target_df)
    window_size = 60
    forecast_horizon = 1

    X, y = create_time_series_sequences(
        scaled_values,
        window_size=window_size,
        forecast_horizon=forecast_horizon,
    )

    X_train, X_test, y_train, y_test = split_train_test(X, y, train_ratio=0.8)

    val_ratio = 0.2
    val_split = int(len(X_train) * (1 - val_ratio))
    X_train_sub, X_val = X_train[:val_split], X_train[val_split:]
    y_train_sub, y_val = y_train[:val_split], y_train[val_split:]

    print(f"\nCreated sequences: X={X.shape}, y={y.shape}")
    print(f"Training split: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Validation split: X_val={X_val.shape}, y_val={y_val.shape}")

    model_types = ["simplernn", "lstm"]
    saved_model_paths = []
    for model_type in model_types:
        print(f"\nBuilding and training {model_type.upper()} model...")
        model = build_sequence_model(
            model_type=model_type,
            input_shape=(window_size, 1),
            units=50,
            dropout_rate=0.2,
            learning_rate=0.001,
            optimizer_name="adam",
        )

        model.summary()

        history, model_path = train_sequence_model(
            model,
            X_train_sub,
            y_train_sub,
            X_val,
            y_val,
            model_name=f"tsla_{model_type}",
            epochs=50,
            batch_size=32,
            patience=10,
            model_dir="models",
        )
        print(f"{model_type.upper()} training complete. Best model saved to: {model_path}")
        saved_model_paths.append((model_type, model_path))

    # Evaluate saved models on the test set (inverse-scaled)
    model_results = []
    for model_type, model_path in saved_model_paths:
        model_name = f"tsla_{model_type}"
        mse = evaluate_and_plot(model_path, X_test, y_test, scaler, model_name)
        model_results.append((model_name, mse))

    print("\nModel comparison results:")
    for model_name, mse in model_results:
        print(f"  {model_name}: MSE = {mse:.6f}")

    if len(model_results) >= 2:
        best_model = min(model_results, key=lambda item: item[1])
        print(f"\nBest performing model: {best_model[0]} with MSE={best_model[1]:.6f}")
        print("This comparison shows which model better captured the test-set price trend under the current setup.")
        print("Limitations: both models use only past adjusted-close values, so they may miss market-moving news, volume shifts, and macroeconomic effects.")
        print("Improvements: add technical indicators, trading volume, sentiment features, or macroeconomic variables, and consider a larger dataset or ensemble methods.")
    else:
        print("\nInsufficient models were evaluated to compare performance.")

    print("\nPreprocessing, training, and evaluation are complete.")
