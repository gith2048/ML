import streamlit as st
import pandas as pd
import numpy as np
from model import compute_rsi  
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# Load data
data = pd.read_csv('stock_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)
# Select target column first
target_column = st.selectbox("Select the target column for prediction", options=data.columns.tolist())

# Then drop it from the features
data = data.dropna(subset=[target_column])  # Remove rows with NaN in the target column
X = data.drop(columns=[target_column])
y = data[target_column]

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model (you can replace this with a more sophisticated one)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on validation data
val_predictions = model.predict(X_val)
mse = mean_squared_error(y_val, val_predictions)
st.write(f"Validation Mean : {mse}")

# Sidebar for user input to select forecast horizon
st.sidebar.header("Forecast Settings")
future_days = st.sidebar.slider("Select Future Prediction Days (1-30)", min_value=1, max_value=30, value=7)

# Predict future stock prices for the selected period
last_known_date = data.index[-1]
future_dates = [last_known_date + timedelta(days=i) for i in range(1, future_days + 1)]
future_predictions = model.predict(X.tail(future_days))

# Plotting
st.header(f"{target_column} Stock Price Prediction")
st.write(f"Forecasting the next {future_days} days")

# Create figure
fig = go.Figure()

# Plot historical data (last 2-3 years)
years_back = 3  # Adjust to show more years if needed
start_date = last_known_date - timedelta(days=365 * years_back)
historical_data = data[target_column].loc[start_date:]
fig.add_trace(go.Scatter(
    x=historical_data.index,
    y=historical_data,
    mode='lines',
    name='Historical Data',
    line=dict(color='blue')
))

# Plot future predictions
fig.add_trace(go.Scatter(
    x=future_dates,
    y=future_predictions,
    mode='lines+markers',
    name='Future Prediction',
    line=dict(color='red', dash='dash')
))

# Customize layout
fig.update_layout(
    title=f"{target_column} Stock Price Prediction (Next {future_days} Days)",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_white"
)

# Show plot
st.plotly_chart(fig, use_container_width=True)
data = pd.read_csv('stock_data.csv', parse_dates=['Date'])
data.set_index('Date', inplace=True)
# Compute RSI
rsi = compute_rsi(data[target_column])

# --- RSI Plotting ---
st.subheader(f"{target_column} RSI (Relative Strength Index)")

fig_rsi = go.Figure()

fig_rsi.add_trace(go.Scatter(
    x=data.index,
    y=rsi,
    mode='lines',
    name='RSI',
    line=dict(color='purple')
))

fig_rsi.add_hline(y=70, line=dict(color='red', dash='dash'), annotation_text='Overbought', annotation_position="top right")
fig_rsi.add_hline(y=30, line=dict(color='green', dash='dash'), annotation_text='Oversold', annotation_position="bottom right")

fig_rsi.add_trace(go.Scatter(
    x=data.index[rsi > 70],
    y=rsi[rsi > 70],
    mode='markers',
    marker=dict(color='red', size=6),
    name='Overbought'
))
fig_rsi.add_trace(go.Scatter(
    x=data.index[rsi < 30],
    y=rsi[rsi < 30],
    mode='markers',
    marker=dict(color='green', size=6),
    name='Oversold'
))

fig_rsi.update_layout(
    title=f"{target_column} RSI Indicator",
    xaxis_title="Date",
    yaxis_title="RSI",
    template="plotly_white",
    yaxis=dict(range=[0, 100])
)

st.plotly_chart(fig_rsi, use_container_width=True)
recent_data = historical_data[-90:]
price_change = recent_data[-1] - recent_data[0]
percentage_change = (price_change / recent_data[0]) * 100

if price_change > 0:
    trend_direction = "upward"
    trend_emoji = "📈"
elif price_change < 0:
    trend_direction = "downward"
    trend_emoji = "📉"
else:
    trend_direction = "sideways"
    trend_emoji = "➡"

# 2. RSI INTERPRETATION (based on last RSI value)
latest_rsi = rsi.dropna().iloc[-1]  # This should now work because rsi is defined
if latest_rsi > 70:
    rsi_comment = "The asset appears to be overbought, which may indicate a potential price correction."
    rsi_emoji = "🔺"
elif latest_rsi < 30:
    rsi_comment = "The asset appears to be oversold, which may signal a potential rebound."
    rsi_emoji = "🔻"
else:
    rsi_comment = "The asset is in a neutral RSI range, suggesting no strong momentum signal."
    rsi_emoji = "⚖"

# Display in Streamlit
st.markdown(f"### Trend Summary {trend_emoji}")
st.write(f"Over the past 90 days, {target_column} has shown an {trend_direction} trend, moving from {recent_data[0]:.2f} to {recent_data[-1]:.2f} ({percentage_change:.2f}%).")
st.write(f"The model predicts continued movement in this direction over the next {future_days} days.")

st.markdown(f"### RSI Analysis {rsi_emoji}")
st.write(f"Current RSI value: {latest_rsi:.2f}")
st.write(rsi_comment)