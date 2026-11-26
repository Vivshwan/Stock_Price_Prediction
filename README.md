# Tesla Stock Price Forecast

A Streamlit dashboard for exploring Tesla (TSLA) stock price history and generating short-term adjusted closing price forecasts. It uses the bundled `TSLA.csv` dataset, which covers June 2010 through February 2020.

## Problem Statement

Historical price charts show how a stock has moved. This project lets learners test how LSTM and SimpleRNN models use recent adjusted closing prices to estimate subsequent values. The dashboard puts the forecasts alongside the historical series so users can inspect the model output in context.

## Key Features

- Choose between trained LSTM and SimpleRNN models and adjust the historical lookback. Start with 60 observations, the window used in training.
- Generate a 1- to 14-day forecast. Each predicted price feeds the next step, and forecast dates advance by calendar day from the last date in the CSV.
- Explore historical prices and forecasts on interactive charts, review price and volatility metrics, and download the source CSV.
- Forecast as a guest or create an account to save runs. Signed-in users can open **View > Forecast history** to review the model, lookback, horizon, and predicted prices for each run.
- Train both models locally and generate actual-versus-predicted plots for the test split.

## Tech Stack

| Technology | Role |
| --- | --- |
| Python 3.13 | Runs the training pipeline and dashboard. |
| TensorFlow 2.21.0 and Keras | Train, save, and load the LSTM and SimpleRNN models. |
| NumPy 2.4.6 and pandas 3.0.3 | Process the historical CSV and prepare price sequences and forecast tables. |
| scikit-learn 1.9.0 | Scales adjusted prices and calculates model evaluation metrics. |
| Streamlit 1.58.0 and Plotly 6.8.0 | Provide dashboard controls, interactive charts, and CSV downloads. |
| Matplotlib 3.10.9 | Saves model evaluation plots during training. |
| SQLite | Stores local accounts and forecast history using Python's built-in `sqlite3` module. |

## Setup and Build

Install Git and Python 3.13. On Windows, use PowerShell with the Python launcher (`py`). The commands call the virtual environment's Python directly, so activation is optional.

1. Clone the repository and enter the project directory. If you already have a checkout, open its root directory, where `app.py` and `TSLA.csv` are located.

   ```powershell
   git clone https://github.com/Vivshwan/Stock_Price_Prediction.git
   Set-Location Stock_Price_Prediction
   ```

2. Create a virtual environment and install the pinned dependencies.

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Optionally retrain the models. This repository includes `models/tsla_lstm.keras` and `models/tsla_simplernn.keras`, so you can go straight to **Run** after installing the dependencies. To regenerate the models and evaluation plots, use the training command below.

   ```powershell
   .\.venv\Scripts\python.exe tsla_preprocessing.py
   ```

   The script reads `TSLA.csv` and writes both `.keras` files and their evaluation plots to `models/`. The Streamlit app requires no separate compilation step.

The app creates `forecast_accounts.sqlite3` in the project directory. To store it somewhere else, set `FORECAST_DATABASE_PATH` to a file path in an existing writable directory before launching the app.

## Run

From the project root, start the dashboard.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. In the sidebar, choose a model and forecast horizon, then select **Predict Future Close Prices**.
