# Tesla Stock Price Forecast

A Streamlit app for Tesla (`TSLA`) adjusted-close price forecasting using saved Keras models.

## Project structure

- `app.py` - Streamlit dashboard for loading data, selecting a trained model, and predicting the next closing price.
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

1. Open PowerShell in the project root .
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

## Notes

- Make sure `TSLA.csv` is present in the project root.
- Make sure your `models/` directory contains trained `.keras` or `.h5` model files.
- If you need to retrain models, use `tsla_preprocessing.py` as a reference for data preprocessing and training.

## Troubleshooting

- If the app fails due to missing dependencies, reinstall with:

```powershell
pip install -r requirements.txt
```

- If the app cannot find data, confirm `TSLA.csv` is in root directory.
