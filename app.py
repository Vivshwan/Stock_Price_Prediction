"""
Tesla Stock Price Forecast App
================================
A modular Streamlit application for predicting Tesla stock prices 
using deep learning models (LSTM, SimpleRNN, CNN-LSTM).
"""

import os
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import base64

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsla_preprocessing import (
    create_time_series_sequences,
    handle_missing_values,
    load_tsla_data,
    scale_series,
    select_target_feature,
)


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

@dataclass
class AppConfig:
    """Application configuration settings"""
    # File paths
    model_dir: str = "models"
    csv_path: str = "TSLA.csv"
    
    # Model settings
    default_window_size: int = 60
    min_window_size: int = 10
    max_window_size: int = 120
    window_step: int = 5
    
    # Display settings
    recent_data_days: int = 10
    chart_height: int = 600
    
    # Target column
    target_column: str = "Adj Close"


config = AppConfig()

# Streamlit page configuration
st.set_page_config(
    page_title="Tesla Stock Forecast",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# BACKGROUND IMAGE FUNCTION
# ============================================================================

def set_tesla_background():
    """Set a Tesla-themed background image"""
    
    # Tesla-inspired gradient background with carbon fiber texture effect
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f0f23 100%);
            position: relative;
        }
        
        /* Carbon fiber texture overlay */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: radial-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1px);
            background-size: 30px 30px;
            pointer-events: none;
            z-index: 0;
        }
        
        /* Tesla logo watermark */
        .stApp::after {
            content: "⚡ TESLA";
            position: fixed;
            bottom: 20px;
            right: 20px;
            font-size: 40px;
            font-weight: bold;
            color: rgba(255, 255, 255, 0.03);
            font-family: monospace;
            pointer-events: none;
            z-index: 0;
            letter-spacing: 5px;
        }
        
        /* Main content container - semi-transparent */
        .main .block-container {
            background: rgba(255, 255, 255, 0.92);
            border-radius: 20px;
            padding: 2rem;
            backdrop-filter: blur(0px);
            z-index: 1;
        }
        
        /* Sidebar styling - glass morphism */
        [data-testid="stSidebar"] {
            background: rgba(10, 10, 20, 0.85);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label {
            color: #cccccc !important;
        }
        
        [data-testid="stSidebar"] .stSelectbox > div {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        
        /* Custom header */
        .app-header {
            background: linear-gradient(135deg, rgba(0,0,0,0.8) 0%, rgba(50,50,80,0.9) 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(5px);
        }
        
        .app-header h1 {
            margin: 0;
            font-size: 2.5rem;
            background: linear-gradient(135deg, #e0e0e0 0%, #ff4444 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .app-header p {
            margin: 0.5rem 0 0 0;
            opacity: 0.8;
        }
        
        /* Prediction Card */
        .prediction-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
            padding: 1.5rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin: 1rem 0;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .prediction-label {
            font-size: 1rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }
        
        .prediction-value {
            font-size: 3rem;
            font-weight: bold;
            margin: 0.5rem 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
            border-left: 4px solid #ff4444;
            transition: transform 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-title {
            font-size: 0.8rem;
            color: #666;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .positive { color: #00c851; }
        .negative { color: #ff4444; }
        
        /* Footer */
        .app-footer {
            text-align: center;
            padding: 1.5rem;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 20px;
            margin-top: 2rem;
            color: white;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        /* Button */
        .stButton > button {
            background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
            color: white;
            font-weight: bold;
            padding: 0.75rem;
            border-radius: 10px;
            width: 100%;
            transition: all 0.3s ease;
            border: none;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(255, 68, 68, 0.4);
        }
        
        /* Info Box */
        .info-box {
            background: rgba(30, 30, 50, 0.9);
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #ff4444;
            margin: 1rem 0;
            color: white;
        }
        
        /* Dataframe styling */
        .dataframe-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 0.5rem;
        }
        
        /* Section divider */
        .section-divider {
            margin: 1.5rem 0;
            border-top: 2px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Tesla electric accent */
        .tesla-accent {
            color: #ff4444;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_stock_data(path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess TSLA stock data"""
    try:
        df = load_tsla_data(path)
        df = handle_missing_values(df)
        return df
    except Exception as e:
        st.error(f"⚠️ Failed to load data: {str(e)}")
        return None


@st.cache_resource
def load_keras_model(model_path: str) -> Optional[tf.keras.Model]:
    """Load Keras model with caching"""
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {str(e)}")
        return None


def get_model_files(model_dir: str) -> Dict[str, str]:
    """Get all model files from the models directory"""
    if not os.path.exists(model_dir):
        return {}
    
    model_extensions = (".keras", ".h5")
    return {
        os.path.splitext(name)[0]: os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith(model_extensions)
    }


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def prepare_prediction_input(
    df: pd.DataFrame, 
    window_size: int = 60
) -> Tuple[np.ndarray, object, pd.Series]:
    """Prepare input data for prediction"""
    target_df = select_target_feature(df, target_column="Adj Close")
    scaled, scaler = scale_series(target_df)
    X, _ = create_time_series_sequences(scaled, window_size=window_size, forecast_horizon=1)
    last_window = X[-1]
    return last_window.reshape(1, window_size, 1), scaler, target_df


def predict_next_value(
    model: tf.keras.Model, 
    input_data: np.ndarray, 
    scaler
) -> float:
    """Predict the next value using the model"""
    prediction_scaled = model.predict(input_data, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    return float(prediction.flatten()[0])


def validate_prediction_conditions(df: pd.DataFrame, window_size: int) -> Tuple[bool, str]:
    """Validate if conditions are met for prediction"""
    if df is None or len(df) == 0:
        return False, "No data available for prediction"
    
    if len(df) < window_size + 1:
        return False, f"Need {window_size + 1} days of data, but only have {len(df)} days"
    
    return True, "Ready for prediction"


# ============================================================================
# METRICS CALCULATIONS
# ============================================================================

@dataclass
class StockMetrics:
    """Stock metrics data container"""
    latest_price: float
    price_change: float
    price_change_pct: float
    max_price: float
    min_price: float
    avg_price: float
    volatility: float


def calculate_stock_metrics(df: pd.DataFrame) -> StockMetrics:
    """Calculate key stock metrics for display"""
    latest_price = df['Adj Close'].iloc[-1]
    prev_price = df['Adj Close'].iloc[-2] if len(df) > 1 else latest_price
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
    
    return StockMetrics(
        latest_price=latest_price,
        price_change=price_change,
        price_change_pct=price_change_pct,
        max_price=df['Adj Close'].max(),
        min_price=df['Adj Close'].min(),
        avg_price=df['Adj Close'].mean(),
        volatility=df['Adj Close'].pct_change().std() * 100
    )


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_price_chart(
    df: pd.DataFrame, 
    historical_window: Optional[pd.Series] = None, 
    predicted_price: Optional[float] = None
) -> go.Figure:
    """Create an interactive price chart using Plotly with Tesla theme"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("⚡ TESLA STOCK PRICE HISTORY", "📊 DAILY RETURNS (%)"),
        vertical_spacing=0.12,
        row_heights=[0.7, 0.3]
    )
    
    # Historical price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Adj Close'],
            mode='lines',
            name='Historical Close',
            line=dict(color='#ff4444', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add predicted point if available
    if predicted_price and historical_window is not None:
        next_date = historical_window.index[-1] + pd.Timedelta(days=1)
        
        # Predicted point marker
        fig.add_trace(
            go.Scatter(
                x=[next_date],
                y=[predicted_price],
                mode='markers',
                name='🎯 Predicted Next Close',
                marker=dict(size=20, color='#ff4444', symbol='star', 
                           line=dict(color='white', width=2)),
                hovertemplate=f'<b>Predicted Price:</b> ${predicted_price:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Trend line
        fig.add_trace(
            go.Scatter(
                x=[historical_window.index[-1], next_date],
                y=[historical_window.iloc[-1], predicted_price],
                mode='lines',
                name='📈 Trend Line',
                line=dict(color='#ff8888', width=2, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Daily returns bar chart
    returns = df['Adj Close'].pct_change() * 100
    colors = ['#ff4444' if x < 0 else '#00c851' for x in returns.fillna(0)]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=returns,
            name='Daily Returns',
            marker_color=colors,
            opacity=0.7,
            hovertemplate='<b>Date:</b> %{x}<br><b>Return:</b> %{y:.2f}%<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Horizontal line at 0
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Layout configuration with Tesla theme
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(0,0,0,0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text="Returns (%)", row=2, col=1, gridcolor='rgba(255,255,255,0.1)')
    
    return fig


def display_prediction_result(predicted_price: float, current_price: float, model_name: str) -> None:
    """Display prediction result with styling"""
    price_diff = predicted_price - current_price
    diff_pct = (price_diff / current_price) * 100 if current_price != 0 else 0
    
    # Prediction card
    direction = "UP" if price_diff > 0 else "DOWN" if price_diff < 0 else "STABLE"
    direction_emoji = "📈" if price_diff > 0 else "📉" if price_diff < 0 else "➡️"
    
    st.markdown(f"""
    <div class="prediction-card">
        <div class="prediction-label">⚡ AI PREDICTION RESULT</div>
        <div class="prediction-value">${predicted_price:,.2f}</div>
        <div class="prediction-label">{direction_emoji} Predicted {direction} {direction_emoji}</div>
        <div class="prediction-model">Model: {model_name}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Price movement indicator
    if price_diff > 0:
        st.success(f"🚀 Predicted **UP** by ${price_diff:.2f} (+{diff_pct:.2f}%)")
    elif price_diff < 0:
        st.error(f"⚠️ Predicted **DOWN** by ${abs(price_diff):.2f} ({diff_pct:.2f}%)")
    else:
        st.info("📊 Predicted to remain stable")


def display_metrics_row(metrics: StockMetrics) -> None:
    """Display metrics in a row of columns"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Current Price</div>
            <div class="metric-value">${metrics.latest_price:,.2f}</div>
            <div class="metric-change {'positive' if metrics.price_change >= 0 else 'negative'}">
                {'▲' if metrics.price_change >= 0 else '▼'} {abs(metrics.price_change_pct):.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📈 Period High</div>
            <div class="metric-value">${metrics.max_price:,.2f}</div>
            <div class="metric-change">All-time high</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">📉 Period Low</div>
            <div class="metric-value">${metrics.min_price:,.2f}</div>
            <div class="metric-change">All-time low</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ Volatility</div>
            <div class="metric-value">{metrics.volatility:.2f}%</div>
            <div class="metric-change">Daily std dev</div>
        </div>
        """, unsafe_allow_html=True)


def display_recent_data(df: pd.DataFrame, days: int = 10) -> None:
    """Display recent data table"""
    st.markdown("### 📋 Recent Market Data")
    
    recent_df = df.tail(days)[["Adj Close"]].copy()
    recent_df.index = recent_df.index.strftime('%Y-%m-%d')
    recent_df.columns = ["Adjusted Close Price"]
    
    # Add change column
    recent_df["Change %"] = df["Adj Close"].tail(days).pct_change().mul(100).round(2)
    
    # Color formatting
    def color_change(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #00c851'
            elif val < 0:
                return 'color: #ff4444'
        return ''
    
    st.dataframe(
        recent_df.style.applymap(color_change, subset=['Change %']),
        use_container_width=True,
        height=400
    )


# ============================================================================
# SIDEBAR COMPONENTS
# ============================================================================

def render_sidebar(df: pd.DataFrame) -> Tuple[Optional[str], Optional[int]]:
    """Render sidebar and return selected model and window size"""
    
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>⚙️ CONFIGURATION</h3>
            <p style="font-size: 0.8rem; opacity: 0.8;">Tesla Stock Predictor</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        model_files = get_model_files(config.model_dir)
        
        if not model_files:
            st.error("❌ No models found")
            st.info(f"Place `.keras` files in `{config.model_dir}/`")
            return None, None
        
        st.markdown("### 🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose prediction model",
            options=list(model_files.keys()),
            format_func=lambda x: f"⚡ {x.upper().replace('_', ' ')}",
            help="LSTM models generally perform better for time series prediction"
        )
        
        # Window size selection
        st.markdown("### 📅 Lookback Window")
        window_size = st.slider(
            "Days to look back",
            min_value=config.min_window_size,
            max_value=config.max_window_size,
            value=config.default_window_size,
            step=config.window_step,
            help="Larger windows capture more historical context"
        )
        
        # Model information
        st.markdown("---")
        st.markdown("### ℹ️ Model Info")
        st.info(f"""
        **Model:** `{selected_model}`  
        **Window:** {window_size} days  
        **Data points:** {len(df)} days
        """)
        
        st.markdown("---")
        st.caption("⚠️ **Disclaimer:** Educational purposes only. Not financial advice.")
        st.caption("⚡ **Tesla Stock Predictor v1.0**")
        
        return selected_model, window_size


def render_footer(model_files: Dict[str, str]) -> None:
    """Render footer with available models"""
    st.markdown("""
    <div class="app-footer">
        <h4>🤖 Available Models</h4>
    """, unsafe_allow_html=True)
    
    # Display models in columns
    model_names = list(model_files.keys())
    cols = st.columns(min(len(model_names), 4))
    
    for idx, model in enumerate(model_names[:4]):
        with cols[idx % len(cols)]:
            st.markdown(f"✅ `{model}`")
    
    st.markdown("""
    <hr style="margin: 1rem 0;">
    <small>Built with ❤️ using Streamlit • TensorFlow • Tesla Stock Data</small><br>
    <small style="opacity: 0.7;">© 2024 Tesla Stock Forecast | AI-Powered Predictions</small>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main() -> None:
    """Main application entry point"""
    
    # Apply Tesla-themed background
    set_tesla_background()
    
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🚀 TESLA STOCK PRICE FORECAST</h1>
        <p>AI-powered predictions using Deep Learning (LSTM / SimpleRNN / CNN-LSTM)</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">⚡ Powered by TensorFlow & Streamlit ⚡</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_stock_data(config.csv_path)
    
    if df is None:
        st.error("⚠️ Unable to load Tesla stock data. Please check your CSV file.")
        return
    
    # Render sidebar and get selections
    selected_model, window_size = render_sidebar(df)
    
    if selected_model is None:
        return
    
    # Get model files
    model_files = get_model_files(config.model_dir)
    
    # Calculate and display metrics
    metrics = calculate_stock_metrics(df)
    display_metrics_row(metrics)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Chart and prediction section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 📊 Stock Price Analysis")
        # Create and display chart
        fig = create_price_chart(df, None, None)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Prediction Center")
        st.markdown('<div class="info-box">⚡ Ready to forecast the next closing price using AI.</div>', 
                   unsafe_allow_html=True)
        
        # Validate conditions
        is_valid, message = validate_prediction_conditions(df, window_size)
        
        if not is_valid:
            st.warning(message)
            return
        
        # Prediction button
        if st.button("🔮 PREDICT NEXT CLOSE PRICE"):
            with st.spinner("🧠 AI is analyzing market patterns..."):
                try:
                    # Load model
                    model_path = model_files[selected_model]
                    model = load_keras_model(model_path)
                    
                    if model is None:
                        return
                    
                    # Make prediction
                    input_window, scaler, target_df = prepare_prediction_input(df, window_size=window_size)
                    predicted_price = predict_next_value(model, input_window, scaler)
                    
                    # Display prediction
                    display_prediction_result(predicted_price, metrics.latest_price, selected_model)
                    
                    # Update chart with prediction
                    historical_window = target_df[-window_size:]
                    updated_fig = create_price_chart(df, historical_window, predicted_price)
                    st.plotly_chart(updated_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"⚠️ Prediction failed: {str(e)}")
                    st.info("Please check if your model file is valid and compatible.")
    
    # Recent data section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    display_recent_data(df, config.recent_data_days)
    
    # Footer
    render_footer(model_files)


if __name__ == "__main__":
    main()
