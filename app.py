import os
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

from tsla_preprocessing import (
    create_time_series_sequences,
    handle_missing_values,
    load_tsla_data,
    scale_series,
    select_target_feature,
)

MODEL_DIR = "models"
DEFAULT_WINDOW_SIZE = 60
CSV_PATH = "TSLA.csv"


def get_model_files(model_dir: str) -> Dict[str, str]:
    return {
        os.path.splitext(name)[0]: os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith(".keras")
    }


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = load_tsla_data(path)
    df = handle_missing_values(df)
    return df


@st.cache_resource
def load_keras_model(model_path: str) -> tf.keras.Model:
    return load_model(model_path)


def prepare_prediction_input(df: pd.DataFrame, window_size: int = DEFAULT_WINDOW_SIZE):
    target_df = select_target_feature(df, target_column="Adj Close")
    scaled, scaler = scale_series(target_df)
    X, _ = create_time_series_sequences(scaled, window_size=window_size, forecast_horizon=1)
    last_window = X[-1]
    return last_window.reshape(1, window_size, 1), scaler, target_df


def predict_next_value(model: tf.keras.Model, input_data: np.ndarray, scaler) -> float:
    prediction_scaled = model.predict(input_data, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    return float(prediction.flatten()[0])


def main():
    st.set_page_config(page_title="Tesla Stock Forecast", layout="wide")

    st.title("Tesla Stock Price Forecast")
    st.markdown(
        "Use saved models to forecast the next Tesla adjusted close price based on historical TSLA data."
    )

    if not os.path.exists(CSV_PATH):
        st.error(f"Dataset not found at {CSV_PATH}. Please place `TSLA.csv` in the project root.")
        return

    df = load_data(CSV_PATH)
    model_files = get_model_files(MODEL_DIR)

    if not model_files:
        st.error(f"No `.keras` models were found in `{MODEL_DIR}`. Add trained models and reload.")
        return

    model_name = st.selectbox("Select model", list(model_files.keys()))
    window_size = st.slider("Window size (days)", min_value=10, max_value=120, value=DEFAULT_WINDOW_SIZE)
    st.write("### Latest dataset preview")
    st.dataframe(df.tail(10)[["Adj Close"]])

    if st.button("Predict next adjusted close"):
        if len(df) < window_size + 1:
            st.error(
                f"Not enough data to create a {window_size}-day input window. "
                f"Dataset length is {len(df)}."
            )
            return

        model = load_keras_model(model_files[model_name])
        input_window, scaler, target_df = prepare_prediction_input(df, window_size=window_size)
        predicted_price = predict_next_value(model, input_window, scaler)

        st.success(f"Next predicted Tesla adjusted close: ${predicted_price:,.2f}")

        historical = target_df[-window_size:].copy()
        historical["Predicted Next Close"] = np.nan
        next_index = historical.index[-1] + pd.Timedelta(days=1)
        historical.loc[next_index] = [np.nan, predicted_price]

        st.line_chart(historical.rename(columns={"Adj Close": "Historical Close"}))

    st.markdown("---")
    st.write("#### Available models")
    st.write(list(model_files.keys()))


if __name__ == "__main__":
    main()
