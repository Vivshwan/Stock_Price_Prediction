# ===========================================
# SECTION 1: IMPORTING REQUIRED LIBRARIES
# ===========================================

import os  # For file and directory operations (checking model files, paths)
from typing import Dict  # For type hints (specifies dictionary return type)
from datetime import datetime, timedelta  # For handling date calculations in forecasts

import numpy as np  # For numerical operations on arrays
import pandas as pd  # For data manipulation and DataFrame operations
import streamlit as st  # Web app framework for creating interactive dashboards
import tensorflow as tf  # Deep learning framework for loading and running models
from tensorflow.keras.models import load_model  # Specifically for loading saved Keras models
import plotly.graph_objects as go  # For creating interactive charts (Plotly)
from plotly.subplots import make_subplots  # For creating multiple subplots in one figure

# Import preprocessing functions from external module
from tsla_preprocessing import (
    create_time_series_sequences,  # Creates sliding window sequences for prediction
    handle_missing_values,  # Fills missing values in time series
    load_tsla_data,  # Loads Tesla stock CSV and parses dates
    scale_series,  # Normalizes data to [0,1] range
    select_target_feature,  # Extracts the 'Adj Close' column for prediction
)

# ===========================================
# SECTION 2: CONSTANTS & CONFIGURATION
# ===========================================

MODEL_DIR = "models"  # Directory where trained models are stored
DEFAULT_WINDOW_SIZE = 60  # Default number of past days to use for prediction
CSV_PATH = "TSLA.csv"  # Path to the Tesla stock data file

# Page configuration for Streamlit app (must be first Streamlit command)
st.set_page_config(
    page_title="Tesla Stock Forecast",  # Title shown in browser tab
    page_icon="🚀",  # Emoji icon for browser tab
    layout="wide",  # Use wide layout (full width of screen)
    initial_sidebar_state="expanded"  # Sidebar starts open
)

# ===========================================
# SECTION 3: CUSTOM CSS STYLING
# ===========================================

# Custom CSS to make the app look professional and visually appealing
st.markdown("""
<style>
    /* Main header styling - gradient background with rounded corners */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Prediction card styling - purple gradient for forecast results */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Large prediction value text */
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Metric cards - white cards with colored left border */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar header styling */
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Custom button styling - gradient background with hover effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    /* Button hover effect - slight lift and shadow */
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Success message styling */
    .stAlert {
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Dataframe container styling - rounded corners and shadow */
    .dataframe-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Footer styling - gradient background */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)  # unsafe_allow_html=True allows HTML/CSS to be rendered


# ===========================================
# SECTION 4: HELPER FUNCTIONS
# ===========================================

def get_model_files(model_dir: str) -> Dict[str, str]:
    """
    Get all model files from the models directory.
    
    Scans the specified directory for .keras and .h5 model files.
    Returns a dictionary mapping model names (without extension) to full file paths.
    
    Args:
        model_dir: Path to the directory containing trained models
        
    Returns:
        Dictionary: {model_name: full_file_path}
        Example: {'tsla_lstm': 'models/tsla_lstm.keras'}
    """
    # Check if the models directory exists
    if not os.path.exists(model_dir):
        return {}  # Return empty dictionary if directory not found
    
    # Dictionary comprehension to collect all .keras and .h5 files
    return {
        os.path.splitext(name)[0]: os.path.join(model_dir, name)  # Remove extension for key
        for name in os.listdir(model_dir)  # Loop through all files in directory
        if name.endswith(".keras") or name.endswith(".h5")  # Only keep model files
    }


@st.cache_data  # Cache this function's output to avoid reloading data repeatedly
def load_data(path: str) -> pd.DataFrame:
    """
    Load and preprocess TSLA data.
    
    Cached decorator ensures data is loaded only once per session,
    significantly improving performance when rerunning the app.
    
    Args:
        path: File path to the TSLA CSV file
        
    Returns:
        DataFrame with processed stock data
    """
    df = load_tsla_data(path)  # Load raw data from CSV
    df = handle_missing_values(df)  # Fill any missing values
    return df


@st.cache_resource  # Cache the loaded model in memory (not just data)
def load_keras_model(model_path: str) -> tf.keras.Model:
    """
    Load Keras model with caching.
    
    Models can be large, so caching prevents reloading the same model
    every time the user interacts with the app.
    
    Args:
        model_path: Full path to the saved model file (.keras or .h5)
        
    Returns:
        Loaded TensorFlow/Keras model
    """
    return load_model(model_path, compile=False)  # compile=False speeds up loading


def prepare_prediction_input(df: pd.DataFrame, window_size: int = DEFAULT_WINDOW_SIZE):
    """
    Prepare input data for prediction.
    
    Steps:
    1. Extract the target column (Adjusted Close)
    2. Scale the data to [0,1] range
    3. Create sequences using the same method as training
    4. Get the most recent window for prediction
    
    Args:
        df: DataFrame with stock data
        window_size: Number of past days to use for prediction
        
    Returns:
        - last_window: Most recent window of data (shape: 1, window_size, 1)
        - scaler: Fitted scaler for inverse transformation
        - target_df: DataFrame with only the target column
    """
    # Extract only the Adjusted Close column for prediction
    target_df = select_target_feature(df, target_column="Adj Close")
    
    # Scale the data to [0,1] range (same scaling as during training)
    scaled, scaler = scale_series(target_df)
    
    # Create sequences (X = input sequences, y = target values - not needed for prediction)
    X, _ = create_time_series_sequences(scaled, window_size=window_size, forecast_horizon=1)
    
    # Get the most recent complete window (last window_size days)
    last_window = X[-1]
    
    # Reshape to (1, window_size, 1) - batch size of 1 for single prediction
    return last_window.reshape(1, window_size, 1), scaler, target_df


def predict_future_values(
    model: tf.keras.Model,
    input_data: np.ndarray,
    scaler,
    forecast_horizon: int = 1,
) -> np.ndarray:
    """
    Predict multiple future values using recursive forecasting.
    
    How it works:
    - Predict next day's price using current window
    - Add that prediction to the window (slide window forward)
    - Use the new window to predict the next day
    - Repeat for the desired forecast horizon
    
    This is called "recursive forecasting" because each prediction
    becomes input for the next prediction.
    
    Args:
        model: Trained Keras model
        input_data: Initial input window (shape: 1, window_size, 1)
        scaler: Fitted scaler for inverse transformation
        forecast_horizon: Number of future days to predict
        
    Returns:
        Array of predicted prices in original dollar scale
    """
    predictions = []  # List to store predicted values
    current_window = input_data.copy()  # Start with the initial window
    
    # Predict one day at a time for the requested horizon
    for _ in range(forecast_horizon):
        # Make prediction for next day (returns scaled value)
        prediction_scaled = model.predict(current_window, verbose=0)
        
        # Convert from scaled [0,1] back to actual dollar amount
        prediction_value = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        predictions.append(prediction_value)
        
        # Slide the window: remove oldest day, add the new prediction
        # First, reshape the prediction to match window format (1, 1, 1)
        next_input = prediction_scaled.reshape(1, 1, 1)
        
        # Concatenate: keep all but first day + add new prediction
        current_window = np.concatenate([current_window[:, 1:, :], next_input], axis=1)
    
    return np.array(predictions)


def create_price_chart(df: pd.DataFrame, historical_window: pd.Series, predicted_series: pd.Series = None):
    """
    Create an interactive price chart using Plotly.
    
    Creates a two-panel chart:
    - Top panel: Stock price history + predictions (line chart)
    - Bottom panel: Daily returns percentage (bar chart)
    
    Args:
        df: DataFrame with full stock history
        historical_window: Last N days of actual prices (for context)
        predicted_series: Series of predicted future prices
        
    Returns:
        Plotly figure object for rendering in Streamlit
    """
    
    # Create subplots: 2 rows, 1 column with different heights
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("📈 Tesla Stock Price History", "📊 Daily Returns (%)"),
        vertical_spacing=0.12,  # Space between the two charts
        row_heights=[0.7, 0.3]  # Top chart takes 70% height, bottom 30%
    )
    
    # ===== TOP PANEL: Historical Price Line =====
    fig.add_trace(
        go.Scatter(
            x=df.index,  # Dates on x-axis
            y=df['Adj Close'],  # Adjusted close prices on y-axis
            mode='lines',  # Line chart
            name='Historical Close',
            line=dict(color='#2a5298', width=2),  # Dark blue line
            fill='tozeroy',  # Fill area under the line
            fillcolor='rgba(42, 82, 152, 0.1)'  # Light blue fill with transparency
        ),
        row=1, col=1
    )
    
    # ===== TOP PANEL: Predicted Prices (if provided) =====
    if predicted_series is not None and historical_window is not None:
        # Generate future dates for predictions (assumes consecutive trading days)
        future_dates = [
            historical_window.index[-1] + pd.Timedelta(days=i)
            for i in range(1, len(predicted_series) + 1)
        ]

        # Add predicted line with dashed style and star markers
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=predicted_series.values,
                mode='lines+markers',  # Both line and markers
                name='Predicted Close',
                line=dict(color='#ff6b6b', width=2, dash='dash'),  # Red dashed line
                marker=dict(size=10, color='#ff6b6b', symbol='star'),  # Star markers
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Predicted Close: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add a connecting line between last actual and first predicted
        fig.add_trace(
            go.Scatter(
                x=[historical_window.index[-1], future_dates[0]],
                y=[historical_window.iloc[-1, 0] if hasattr(historical_window.iloc[-1], '__getitem__') else historical_window.iloc[-1], predicted_series.values[0]],
                mode='lines',
                name='Forecast Bridge',
                line=dict(color='#ff6b6b', width=2, dash='dot'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # ===== BOTTOM PANEL: Daily Returns (Percentage Change) =====
    returns = df['Adj Close'].pct_change() * 100  # Convert to percentage
    # Color bars: red for negative returns, green for positive returns
    colors = ['#ff4444' if x < 0 else '#00c851' for x in returns.fillna(0)]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=returns,
            name='Daily Returns %',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add horizontal line at y=0 (reference line for positive/negative)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # ===== LAYOUT CONFIGURATION =====
    fig.update_layout(
        height=600,  # Total chart height in pixels
        showlegend=True,
        hovermode='x unified',  # Show all values at the same x coordinate
        template='plotly_white',  # Clean white background
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=1.02,  # Position above the chart
            xanchor="right",
            x=1
        )
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
    
    return fig


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate key metrics for display in the dashboard.
    
    Computes various statistics to give users quick insights about the stock.
    
    Args:
        df: DataFrame with stock data
        
    Returns:
        Dictionary containing all calculated metrics
    """
    latest_price = df['Adj Close'].iloc[-1]  # Most recent closing price
    prev_price = df['Adj Close'].iloc[-2]    # Previous day's closing price
    
    # Calculate price change (absolute and percentage)
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    # Calculate overall statistics
    max_price = df['Adj Close'].max()  # Highest price in dataset
    min_price = df['Adj Close'].min()  # Lowest price in dataset
    avg_price = df['Adj Close'].mean()  # Average price
    
    # Volatility = standard deviation of daily returns (annualized not needed here)
    volatility = df['Adj Close'].pct_change().std() * 100
    
    return {
        'latest_price': latest_price,
        'price_change': price_change,
        'price_change_pct': price_change_pct,
        'max_price': max_price,
        'min_price': min_price,
        'avg_price': avg_price,
        'volatility': volatility
    }


# ===========================================
# SECTION 5: MAIN APP FUNCTION
# ===========================================

def main():
    """
    Main function that builds and runs the Streamlit dashboard.
    
    This function orchestrates the entire user interface:
    - Sidebar configuration
    - Metrics display
    - Interactive charts
    - Prediction functionality
    - Data download
    """
    
    # ===== HEADER SECTION =====
    # Display styled header with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Tesla Stock Price Forecast</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">AI-powered predictions using Deep Learning (LSTM/SimpleRNN)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ===== SIDEBAR CONFIGURATION =====
    with st.sidebar:
        # Sidebar header
        st.markdown("""
        <div class="sidebar-header">
            <h3>⚙️ Configuration Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if dataset exists
        if not os.path.exists(CSV_PATH):
            st.error(f"❌ Dataset not found at `{CSV_PATH}`")
            st.info("Please place `TSLA.csv` in the project root directory.")
            return  # Stop execution if data is missing
        
        # Load data (cached)
        df = load_data(CSV_PATH)
        
        # Get available models from the models directory
        model_files = get_model_files(MODEL_DIR)
        
        # Show error if no models found
        if not model_files:
            st.error(f"❌ No models found in `{MODEL_DIR}`")
            st.info("Please add trained `.keras` or `.h5` models to the models directory.")
            return
        
        # ===== MODEL SELECTION DROPDOWN =====
        st.markdown("### 🤖 Model Selection")
        model_name = st.selectbox(
            "Choose your prediction model",
            options=list(model_files.keys()),
            format_func=lambda x: f"🧠 {x.upper().replace('_', ' ')}",  # Nicely formatted display
            help="LSTM models generally perform better for time series prediction"
        )
        
        # ===== WINDOW SIZE SLIDER =====
        st.markdown("### 📅 Lookback Window")
        window_size = st.slider(
            "Days to look back for prediction",
            min_value=10,  # Minimum 10 days
            max_value=120,  # Maximum 120 days (~6 months)
            value=DEFAULT_WINDOW_SIZE,
            step=5,
            help="Larger windows capture more historical context but may include irrelevant patterns"
        )

        # ===== FORECAST HORIZON SLIDER =====
        st.markdown("### 📈 Forecast Horizon")
        forecast_horizon = st.slider(
            "Days to predict ahead",
            min_value=1,
            max_value=14,  # Maximum 2 weeks
            value=3,
            step=1,
            help="Predict the next N trading days using recursive forecasting"
        )
        
        # ===== MODEL INFORMATION DISPLAY =====
        st.markdown("---")
        st.markdown("### ℹ️ Model Information")
        st.info(f"""
        **Model Name:** `{model_name}`  
        **Window Size:** {window_size} days  
        **Forecast Horizon:** {forecast_horizon} days  
        **Data Points:** {len(df)} days of historical data
        """)
        
        st.markdown("---")
        # Optional disclaimer (commented out)
        # st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. Not financial advice.")
    
    # ===== MAIN CONTENT AREA =====
    
    # Calculate metrics for display
    metrics = calculate_metrics(df)
    
    # ===== METRICS DISPLAY (4 columns) =====
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Current Price Card
        st.markdown(f"""
        <div class="metric-card">
            <h4>💰 Current Price</h4>
            <h2 style="color: #2a5298;">${metrics['latest_price']:,.2f}</h2>
            <span style="color: {'#00c851' if metrics['price_change'] >= 0 else '#ff4444'}">
                {'▲' if metrics['price_change'] >= 0 else '▼'} {abs(metrics['price_change_pct']):.2f}%
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Period High Card
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Period High</h4>
            <h3>${metrics['max_price']:,.2f}</h3>
            <small>All-time high in dataset</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Period Low Card
        st.markdown(f"""
        <div class="metric-card">
            <h4>📉 Period Low</h4>
            <h3>${metrics['min_price']:,.2f}</h3>
            <small>All-time low in dataset</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Volatility Card
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Volatility</h4>
            <h3>{metrics['volatility']:.2f}%</h3>
            <small>Daily standard deviation</small>
        </div>
        """, unsafe_allow_html=True)
    
    # ===== INTERACTIVE CHART =====
    st.markdown("---")
    st.markdown("### 📊 Stock Price Analysis")
    fig = create_price_chart(df, None, None)  # Initial chart without predictions
    st.plotly_chart(fig, use_container_width=True)  # Render chart full width

    # ===== PREDICTION SECTION =====
    st.markdown("---")
    st.markdown("### 🎯 Prediction Center")
    st.markdown("Ready to forecast the next closing price?")
    
    # Check if we have enough data for the selected window size
    if len(df) < window_size + 1:
        st.warning(f"⚠️ Insufficient data for {window_size}-day window")
        st.error(f"Need {window_size + 1} days, have {len(df)} days")
        return
    
    # ===== PREDICTION BUTTON =====
    if st.button("🔮 **Predict Future Close Prices**", use_container_width=True):
        # Show spinner while prediction is running
        with st.spinner("🧠 AI is analyzing market patterns..."):
            try:
                # Load the selected model (cached)
                model = load_keras_model(model_files[model_name])
                
                # Prepare input data for prediction
                input_window, scaler, target_df = prepare_prediction_input(df, window_size=window_size)
                
                # Generate predictions for the requested horizon
                predicted_values = predict_future_values(
                    model,
                    input_window,
                    scaler,
                    forecast_horizon=forecast_horizon,
                )

                # Create date range for predicted values
                forecast_dates = [
                    target_df.index[-1] + pd.Timedelta(days=i)
                    for i in range(1, forecast_horizon + 1)
                ]
                predicted_series = pd.Series(predicted_values, index=forecast_dates)

                # ===== DISPLAY PREDICTION RESULTS =====
                # Large prediction card
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>📊 Predicted Close Prices</h3>
                    <div class="prediction-value">{predicted_series.iloc[0]:,.2f} → {predicted_series.iloc[-1]:,.2f}</div>
                    <p>Forecast for <strong>{forecast_horizon}</strong> days using <strong>{model_name}</strong></p>
                </div>
                """, unsafe_allow_html=True)

                # Show trend summary (up/down/flat)
                price_diff = predicted_series.iloc[-1] - metrics['latest_price']
                diff_pct = (price_diff / metrics['latest_price']) * 100

                if price_diff > 0:
                    st.success(f"📈 Forecast trend up by ${price_diff:.2f} ({diff_pct:.2f}%)")
                elif price_diff < 0:
                    st.error(f"📉 Forecast trend down by ${abs(price_diff):.2f} ({abs(diff_pct):.2f}%)")
                else:
                    st.info("📊 Forecast trend remains flat")

                # Display detailed forecast table
                forecast_df = pd.DataFrame({
                    "Forecast Date": predicted_series.index.strftime('%Y-%m-%d'),
                    "Predicted Close": predicted_series.values,
                })
                # Format prices as currency in the table
                st.table(forecast_df.assign(**{"Predicted Close": forecast_df["Predicted Close"].map("${:,.2f}".format)}))

                # Update chart with predictions
                historical_window = target_df[-window_size:]  # Last window_size days for context
                updated_fig = create_price_chart(df, historical_window, predicted_series)
                st.plotly_chart(updated_fig, use_container_width=True)

            except Exception as e:
                # Handle any prediction errors gracefully
                st.error(f"Prediction failed: {str(e)}")
                st.info("Please check if your model file is valid and compatible.")
    
    # ===== DATASET DOWNLOAD SECTION =====
    st.markdown("---")
    st.markdown("### 📥 Download Dataset")
    
    # Provide download button for the CSV file
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "rb") as file:
            csv_data = file.read()
        st.download_button(
            label="📊 Download TSLA.csv",
            data=csv_data,
            file_name="TSLA.csv",
            mime="text/csv",
            help="Download the complete Tesla stock historical dataset"
        )
        st.info(f"✅ Dataset loaded: `{CSV_PATH}` ({len(df)} trading days)")
    else:
        st.error(f"❌ Dataset not found: `{CSV_PATH}`")
    
    # ===== FOOTER =====
    st.markdown("""
    <hr>
    <small>Built with Streamlit • TensorFlow • Tesla Stock Data</small>
    """, unsafe_allow_html=True)


# ===========================================
# SECTION 6: SCRIPT ENTRY POINT
# ===========================================

if __name__ == "__main__":
    main()  # Run the main function when script is executed
