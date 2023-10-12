"""
Dow Jones Prediction and Analysis Script

Description:
    This script analyzes historical data for the Dow Jones Industrial Average from Yahoo Finance.
    It uses a Long Short-Term Memory (LSTM) neural network to predict future closing prices, displays
    significant historical events on the timeline, and visualizes other relevant statistics and indicators
    such as moving averages, trading volume, and volatility (Average True Range).

Usage:
    python TrendAnalysis.py

Created by: Raymond Beecham
Date: October 11, 2023
Version: 4.0
Contact: rb102@my.tamuct.edu

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - yfinance
    - scikit-learn
    - tensorflow
    - datetime

Notes:
    - Modify TICKER to analyze different stocks or indices.
    - Adjust TIME_STEP and ATR_PERIOD constants for tuning the LSTM and ATR calculations.
    - Add or modify dates in 'important_dates_and_events' dictionary for additional annotations on the plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

# Constants
TICKER = "^DJI"
START_DATE = "1992-10-01"
TIME_STEP = 100
ATR_PERIOD = 141

# Dates and events plotting
important_dates_and_events = {
"1999-03-29": "10,000+ Close",
"2001-09-17": "Reopening after 9/11",
"2007-07-19": "14,000+ Close",
"2008-10-13": "Financial Crisis Low",
"2013-03-05": "All-time High",
"2020-03-23": "COVID-19 Low"
}

def fetch_data(ticker, start_date):
    """Download the stock data from Yahoo Finance."""
    current_date = datetime.today().strftime('%Y-%m-%d')
    return yf.download(ticker, start=start_date, end=current_date)

def compute_average_true_range(data, period=ATR_PERIOD):
    """Calculate Average True Range for volatility analysis."""
    data['H-L'] = abs(data['High'] - data['Low'])
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    TR = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return TR.rolling(period).mean()

def calculate_statistics(data):
    """Compute and print statistics for the dataset."""
    # Historical Highs and Lows
    high_date = data['High'].idxmax()
    low_date = data['Low'].idxmin()
    print(f"Historical High: {data['High'].max()} on {high_date.strftime('%Y-%m-%d')}")
    print(f"Historical Low: {data['Low'].min()} on {low_date.strftime('%Y-%m-%d')}")
    
    # Statistical Analysis
    print(f"Mean Closing Price: {data['Close'].mean()}")
    print(f"Median Closing Price: {data['Close'].median()}")
    print(f"Standard Deviation of Closing Price: {data['Close'].std()}")
    print(f"Variance of Closing Price: {data['Close'].var()}")
    
    # Return and Drawdown Calculation
    data['Daily Return'] = data['Close'].pct_change()
    print(f"Total Return: {(data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1:.2%}")
    data['Peak'] = data['Close'].cummax()
    data['Drawdown'] = (data['Close'] - data['Peak']) / data['Peak']
    print(f"Max Drawdown: {data['Drawdown'].min():.2%}")

def get_closest_date(target_date, available_dates):
    """Get the closest date to the target date from a list of available dates."""
    return min(available_dates, key=lambda d: abs(d - target_date))


def lstm_prediction(prices, time_step=TIME_STEP):
    """Predict the next closing price using LSTM model."""
    # Scaling and data preparation
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
    X, y = create_dataset(prices_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # LSTM Model
    model = build_lstm_model(time_step)
    
    # Train Model
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    
    # Predict Next Closing Price
    test_data = prices_scaled[-time_step:].reshape(1, time_step, 1)
    predicted_price = model.predict(test_data)
    return scaler.inverse_transform(predicted_price)[0][0]

def build_lstm_model(time_step):
    """Construct the LSTM model."""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def create_dataset(data, time_step=1):
    """Prepare data for LSTM training."""
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        dataX.append(data[i:(i + time_step)])
        dataY.append(data[i + time_step])
    return np.array(dataX), np.array(dataY)

def get_next_business_day(last_date):
    """Determine the next business day after the given date."""
    next_date = last_date + pd.Timedelta(days=2)
    while next_date.weekday() > 4:  # 5: Saturday, 6: Sunday
        next_date += pd.Timedelta(days=1)
    return next_date

def plot_data(data, predicted_date, predicted_next_closing):
    """Visualize the historical data and predictions."""
    # Resample to monthly data for plotting
    monthly_data = data.resample('M').mean()

    plt.figure(figsize=(12, 10))

    # Subplot 1: Closing Price with Moving Averages
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(monthly_data['Close'], label='Closing Price', color='blue')
    plt.plot(monthly_data['Close'].rolling(window=6).mean(), 'g--', label='6-months SMA')
    plt.plot(monthly_data['Close'].rolling(window=12).mean(), 'r--', label='12-months SMA')
    plt.title("Dow Jones Industrial Average Trend Analysis")
    plt.ylabel("Price")
    plt.scatter(predicted_date, predicted_next_closing, color='red', label='Predicted Closing Price', zorder=5)
    annotation_text = f"{predicted_date.strftime('%Y-%m-%d')}: ${predicted_next_closing:.2f}"
    plt.annotate(annotation_text, (predicted_date, predicted_next_closing), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color='red')
    plt.legend()
    plt.grid(True)

    # Dates and events plotting
    for date, event in important_dates_and_events.items():
        closest_date = get_closest_date(pd.Timestamp(date), monthly_data.index)
        ax1.axvline(closest_date, color='k', linestyle='--', alpha=0.7)
        ax1.annotate(event, (closest_date, monthly_data['Close'][closest_date]), 
                     textcoords="offset points", xytext=(-0,-30), ha='center', fontsize=5, color='black')

    # Subplot 2: Volume
    plt.subplot(3, 1, 2)
    plt.bar(monthly_data.index, monthly_data['Volume'], color='gray', width=pd.Timedelta(days=20), label='Volume')
    plt.title("Volume Analysis")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid(True)
    plt.ticklabel_format(axis="y", style="plain")

    # Subplot 3: ATR - Volatility
    plt.subplot(3, 1, 3)
    plt.plot(monthly_data['ATR'], label='Average True Range', color='red')
    plt.title("Volatility Analysis")
    plt.xlabel("Date")
    plt.ylabel("Average True Range(ATR)")
    plt.legend()
    plt.grid(True)

def main():
    # 1. Fetch the data
    data = fetch_data(TICKER, START_DATE)

    # 2. Calculate statistics
    calculate_statistics(data)

    # 3. Data Analysis: Add ATR to the data
    data['ATR'] = compute_average_true_range(data)

    # 4. Make a prediction
    predicted_next_closing = lstm_prediction(data['Close'].values)
    predicted_date = get_next_business_day(data.index[-1])
    print(f"Predicted closing price for {predicted_date.strftime('%Y-%m-%d')}: ${predicted_next_closing:.2f}")

    # 5. Resample the data to monthly frequency
    monthly_data = data.resample('M').mean()

    # 6. Plot the data
    plot_data(data, predicted_date, predicted_next_closing)

    # Adjusting the spacing between subplots
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout()
    plt.show()

# Call the main function
main()