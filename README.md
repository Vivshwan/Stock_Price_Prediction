# Tesla Stock Price Forecast

A Streamlit app for Tesla (`TSLA`) adjusted-close price forecasting using saved Keras models.

## Project structure

- `app.py` - Streamlit dashboard for loading data, selecting a trained model, and predicting the next closing price.
- `forecast_store.py` - SQLite account authentication and per-user forecast history.
- `tsla_preprocessing.py` - Data loading, preprocessing, scaling, sequence creation, and model training utilities.
- `TSLA.csv` - Historical Tesla stock price data used by the app.
- `models/` - Directory containing trained model files such as `tsla_lstm.keras` and `tsla_simplernn.keras`.
- `requirements.txt` - Python dependency list.
- `.venv/` - Local virtual environment for the project.

## Requirements

The project uses Python and the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
- `streamlit`
- `plotly`

## Setup

1. Open PowerShell in the project root.
2. Activate the existing virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

## Run the Streamlit app

Execute:

```powershell
streamlit run app.py
```

Then open the provided local URL in your browser.

## Accounts and forecast history

Use **Create account** in the sidebar to register with an email and a password of at least 12 characters. Returning users select **Sign in**. Each forecast you run while signed in appears under **View → Forecast history**, with its model, lookback window, forecast horizon, and daily predicted prices. Select **Sign out** to end your session. Guest forecasting is available without saving history.

The app creates `forecast_accounts.sqlite3` in the project root on first run. SQLite stores account email addresses, salted password hashes, and forecast history. To use an existing SQLite database or another location, set `FORECAST_DATABASE_PATH` before starting Streamlit. The parent directory must exist and be writable by the app. The app creates the `forecast_users` and `forecast_runs` tables if needed.

```powershell
$env:FORECAST_DATABASE_PATH = 'D:\data\forecast_accounts.sqlite3'
streamlit run app.py
```

## Notes

- Make sure `TSLA.csv` is present in the project root.
- Make sure your `models/` directory contains trained `.keras` or `.h5` model files.
- If you need to retrain models, use `tsla_preprocessing.py` as a reference for data preprocessing and training.

## Troubleshooting

- If the app fails due to missing dependencies, reinstall with:

```powershell
pip install -r requirements.txt
```

- If the existing `.venv` was copied from another location and fails to start, create a fresh virtual environment using Python 3.13 and reinstall the requirements into it.
- If the app cannot find data, start Streamlit from the project root and confirm `TSLA.csv` is there.
