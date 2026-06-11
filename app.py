import os
from typing import Dict
from datetime import datetime, timedelta

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

MODEL_DIR = "models"
DEFAULT_WINDOW_SIZE = 60
CSV_PATH = "TSLA.csv"

# Page configuration
st.set_page_config(
    page_title="Tesla Stock Forecast",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Prediction card styling */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    /* Sidebar styling */
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Success message styling */
    .stAlert {
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Dataframe styling */
    .dataframe-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 20px;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def get_model_files(model_dir: str) -> Dict[str, str]:
    """Get all model files from the models directory"""
    if not os.path.exists(model_dir):
        return {}
    return {
        os.path.splitext(name)[0]: os.path.join(model_dir, name)
        for name in os.listdir(model_dir)
        if name.endswith(".keras") or name.endswith(".h5")
    }


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and preprocess TSLA data"""
    df = load_tsla_data(path)
    df = handle_missing_values(df)
    return df


@st.cache_resource
def load_keras_model(model_path: str) -> tf.keras.Model:
    """Load Keras model with caching"""
    return load_model(model_path, compile=False)


def prepare_prediction_input(df: pd.DataFrame, window_size: int = DEFAULT_WINDOW_SIZE):
    """Prepare input data for prediction"""
    target_df = select_target_feature(df, target_column="Adj Close")
    scaled, scaler = scale_series(target_df)
    X, _ = create_time_series_sequences(scaled, window_size=window_size, forecast_horizon=1)
    last_window = X[-1]
    return last_window.reshape(1, window_size, 1), scaler, target_df


def predict_next_value(model: tf.keras.Model, input_data: np.ndarray, scaler) -> float:
    """Predict the next value using the model"""
    prediction_scaled = model.predict(input_data, verbose=0)
    prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
    return float(prediction.flatten()[0])


def create_price_chart(df: pd.DataFrame, historical_window: pd.Series, predicted_price: float = None):
    """Create an interactive price chart using Plotly"""
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("📈 Tesla Stock Price History", "📊 Daily Returns (%)"),
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
            line=dict(color='#2a5298', width=2),
            fill='tozeroy',
            fillcolor='rgba(42, 82, 152, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add predicted point if available
    if predicted_price and historical_window is not None:
        next_date = historical_window.index[-1] + pd.Timedelta(days=1)
        fig.add_trace(
            go.Scatter(
                x=[next_date],
                y=[predicted_price],
                mode='markers',
                name='Predicted Next Close',
                marker=dict(
                    size=20,
                    color='#ff6b6b',
                    symbol='star',
                    line=dict(color='white', width=2)
                ),
                hovertemplate=f'<b>Predicted Price:</b> ${predicted_price:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add connecting line to historical data
        fig.add_trace(
            go.Scatter(
                x=[historical_window.index[-1], next_date],
                y=[historical_window.iloc[-1], predicted_price],
                mode='lines',
                name='Trend Line',
                line=dict(color='#ff6b6b', width=2, dash='dash'),
                opacity=0.7
            ),
            row=1, col=1
        )
    
    # Calculate and add daily returns
    returns = df['Adj Close'].pct_change() * 100
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
    
    # Add horizontal line at 0 for returns
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
    
    return fig


def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate key metrics for display"""
    latest_price = df['Adj Close'].iloc[-1]
    prev_price = df['Adj Close'].iloc[-2]
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    max_price = df['Adj Close'].max()
    min_price = df['Adj Close'].min()
    avg_price = df['Adj Close'].mean()
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


def main():
    # Header section with gradient background
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Tesla Stock Price Forecast</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">AI-powered predictions using Deep Learning (LSTM/SimpleRNN)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>⚙️ Configuration Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if data exists
        if not os.path.exists(CSV_PATH):
            st.error(f"❌ Dataset not found at `{CSV_PATH}`")
            st.info("Please place `TSLA.csv` in the project root directory.")
            return
        
        # Load data
        df = load_data(CSV_PATH)
        
        # Get available models
        model_files = get_model_files(MODEL_DIR)
        
        if not model_files:
            st.error(f"❌ No models found in `{MODEL_DIR}`")
            st.info("Please add trained `.keras` or `.h5` models to the models directory.")
            return
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        model_name = st.selectbox(
            "Choose your prediction model",
            options=list(model_files.keys()),
            format_func=lambda x: f"🧠 {x.upper().replace('_', ' ')}",
            help="LSTM models generally perform better for time series prediction"
        )
        
        # Window size selection
        st.markdown("### 📅 Lookback Window")
        window_size = st.slider(
            "Days to look back for prediction",
            min_value=10,
            max_value=120,
            value=DEFAULT_WINDOW_SIZE,
            step=5,
            help="Larger windows capture more historical context but may include irrelevant patterns"
        )
        
        # Display model info
        st.markdown("---")
        st.markdown("### ℹ️ Model Information")
        st.info(f"""
        **Model Name:** `{model_name}`  
        **Window Size:** {window_size} days  
        **Data Points:** {len(df)} days of historical data
        """)
        
        st.markdown("---")
        #st.caption("⚠️ **Disclaimer:** This tool is for educational purposes only. Not financial advice.")
    
    # Main content area
    # Calculate metrics
    metrics = calculate_metrics(df)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
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
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Period High</h4>
            <h3>${metrics['max_price']:,.2f}</h3>
            <small>All-time high in dataset</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📉 Period Low</h4>
            <h3>${metrics['min_price']:,.2f}</h3>
            <small>All-time low in dataset</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Volatility</h4>
            <h3>{metrics['volatility']:.2f}%</h3>
            <small>Daily standard deviation</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive chart
    st.markdown("---")
    chart_col1, chart_col2 = st.columns([3, 1])
    
    with chart_col1:
        st.markdown("### 📊 Stock Price Analysis")
        # Display chart without prediction initially
        fig = create_price_chart(df, None, None)
        st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.markdown("### 🎯 Prediction Center")
        st.markdown("Ready to forecast the next closing price?")
        
        # Data sufficiency check
        if len(df) < window_size + 1:
            st.warning(f"⚠️ Insufficient data for {window_size}-day window")
            st.error(f"Need {window_size + 1} days, have {len(df)} days")
            return
        
        # Prediction button
        if st.button("🔮 **Predict Next Close Price**", use_container_width=True):
            with st.spinner("🧠 AI is analyzing market patterns..."):
                try:
                    # Load model and make prediction
                    model = load_keras_model(model_files[model_name])
                    input_window, scaler, target_df = prepare_prediction_input(df, window_size=window_size)
                    predicted_price = predict_next_value(model, input_window, scaler)
                    
                    # Display prediction in a beautiful card
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>📊 Predicted Next Close</h3>
                        <div class="prediction-value">${predicted_price:,.2f}</div>
                        <p>Based on {window_size} days of historical data using <strong>{model_name}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show comparison with current price
                    price_diff = predicted_price - metrics['latest_price']
                    diff_pct = (price_diff / metrics['latest_price']) * 100
                    
                    if price_diff > 0:
                        st.success(f"📈 Predicted **UP** by ${price_diff:.2f} ({diff_pct:.2f}%)")
                    elif price_diff < 0:
                        st.error(f"📉 Predicted **DOWN** by ${abs(price_diff):.2f} ({abs(diff_pct):.2f}%)")
                    else:
                        st.info("📊 Predicted to remain stable")
                    
                    # Update chart with prediction
                    historical_window = target_df[-window_size:]
                    updated_fig = create_price_chart(df, historical_window, predicted_price)
                    st.plotly_chart(updated_fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.info("Please check if your model file is valid and compatible.")
    
    # Recent data table
    st.markdown("---")
    st.markdown("### 📋 Recent Historical Data")
    
    # Create styled dataframe
    recent_df = df.tail(10)[["Adj Close"]].copy()
    recent_df.index = recent_df.index.strftime('%Y-%m-%d')
    recent_df.columns = ["Adjusted Close Price"]
    
    # Add change column
    recent_df["Change %"] = df["Adj Close"].tail(10).pct_change().mul(100).round(2)
    
    # Display dataframe with custom styling
    st.dataframe(
        recent_df,
        use_container_width=True,
        height=400
    )
    
    # Footer with model information
    st.markdown("""
    <div class="footer">
        <h5>🤖 Available Models</h5>
    """, unsafe_allow_html=True)
    
    # Display available models in columns
    model_cols = st.columns(min(len(model_files), 4))
    for idx, model in enumerate(list(model_files.keys())[:4]):
        with model_cols[idx % len(model_cols)]:
            st.markdown(f"✅ `{model}`")
    
    st.markdown("""
    <hr>
    <small>Built with Streamlit • TensorFlow • Tesla Stock Data</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
