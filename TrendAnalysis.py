import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

# Download Data From Yahoo Finance
ticker = "^DJI"
current_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(ticker, start="1992-10-01", end=current_date)

# Functions
# Average True Range Calculation
def average_true_range(data, period=14):
    data['H-L'] = abs(data['High'] - data['Low'])
    data['H-PC'] = abs(data['High'] - data['Close'].shift(1))
    data['L-PC'] = abs(data['Low'] - data['Close'].shift(1))
    TR = data[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return TR.rolling(period).mean()

# Closest Date Calculation
def get_closest_date(target_date, available_dates):
    """Get the closest date to the target date from a list of available dates."""
    return min(available_dates, key=lambda d: abs(d - target_date))

# LSTM Data Preparation
def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        dataX.append(data[i:(i + time_step)])
        dataY.append(data[i + time_step])
    return np.array(dataX), np.array(dataY)

# LSTM Prediction
def lstm_prediction(prices):
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1))
    TIME_STEP = 100
    X, y = create_dataset(prices_scaled, TIME_STEP)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # LSTM Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(TIME_STEP, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train Model
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:], y[train_size:]
    model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)
    
    # Predict Next Closing Price
    test_data = prices_scaled[-TIME_STEP:].reshape(1, TIME_STEP, 1)
    predicted_price = model.predict(test_data)
    predicted_price = scaler.inverse_transform(predicted_price)
    return predicted_price[0][0]

# Volatility Analysis (ATR Calculation)
data['ATR'] = average_true_range(data)

# Historical Highs and Lows
high_date = data['High'].idxmax()
print(f"Historical High: {data['High'].max()} on {high_date.strftime('%Y-%m-%d')}")
low_date = data['Low'].idxmin()
print(f"Historical Low: {data['Low'].min()} on {low_date.strftime('%Y-%m-%d')}")

# Statistical Analysis of the Mean, Median, Standard Deviation, and Variance
print(f"Mean Closing Price: {data['Close'].mean()}")
print(f"Median Closing Price: {data['Close'].median()}")
print(f"Standard Deviation of Closing Price: {data['Close'].std()}")
print(f"Variance of Closing Price: {data['Close'].var()}")

# Return Calculation
data['Daily Return'] = data['Close'].pct_change()
print(f"Total Return: {(data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1:.2%}")

# Drawdown Analysis
data['Peak'] = data['Close'].cummax()
data['Drawdown'] = (data['Close'] - data['Peak']) / data['Peak']
print(f"Max Drawdown: {data['Drawdown'].min():.2%}")

# --Only used this to analyze my volume data. Feel free to uncomment and use it.--
# Volume Analysis
# volume_stats = data['Volume'].describe()
# print(volume_stats)
# high_volume = data['Volume'][data['Volume'] > data['Volume'].quantile(0.95)]
# low_volume = data['Volume'][data['Volume'] < data['Volume'].quantile(0.05)]
# print("High Volume Dates:", high_volume)
# print("Low Volume Dates:", low_volume)
# correlation = data['Volume'].corr(data['Daily Return'])
# print("Correlation between Volume and Daily Returns:", correlation)

# Predict the next closing price while skipping weekends for the predicted date. 
# The prediction is based on LSTM model and the result is printed.
predicted_next_closing = lstm_prediction(data['Close'].values)
predicted_date = data.index[-1]
for _ in range(1):  # moving to the next day
    predicted_date += pd.Timedelta(days=1)
    while predicted_date.weekday() > 4:  # 5: Saturday, 6: Sunday
        predicted_date += pd.Timedelta(days=1)

print(f"Predicted closing price for {predicted_date.strftime('%Y-%m-%d')}: ${predicted_next_closing:.2f}")



# Resample to monthly data for plotting
monthly_data = data.resample('M').mean()

# Plotting
plt.figure(figsize=(12, 10))  # Increased the size slightly for clarity

# Subplot 1: Closing Price with Moving Averages
plt.subplot(3, 1, 1)
plt.plot(monthly_data['Close'], label='Closing Price', color='blue')
plt.plot(monthly_data['Close'].rolling(window=6).mean(), 'g--', label='6-months SMA')
plt.plot(monthly_data['Close'].rolling(window=12).mean(), 'r--', label='12-months SMA')
plt.title("Dow Jones Industrial Average Trend Analysis")
plt.ylabel("Price")
plt.scatter(predicted_date, predicted_next_closing, color='red', label='Predicted Closing Price', zorder=5) 
annotation_text = f"{predicted_date.strftime('%Y-%m-%d')}: ${predicted_next_closing:.2f}"
plt.annotate(annotation_text, (predicted_date, predicted_next_closing), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')
plt.legend()
plt.grid(True)

# Dates and Events
important_dates_and_events = {
    "1999-03-29": "10,000+ Close",
    "2001-09-17": "Reopening after 9/11",
    "2007-07-19": "14,000+ Close",
    "2008-10-13": "Financial Crisis Low",
    "2013-03-05": "All-time High",
    "2020-03-23": "COVID-19 Low",

}
# Plotting the dates and events
for date, event in important_dates_and_events.items():
    closest_date = get_closest_date(pd.Timestamp(date), monthly_data.index)
    plt.axvline(closest_date, color='k', linestyle='--', alpha=0.7)
    plt.annotate(event, (closest_date, monthly_data['Close'][closest_date]), 
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

# --Decided against Dispersion--
# # Dispersion
# plt.figure(figsize=(10, 6))
# plt.hist(data['Close'], bins=50, color='lightblue', edgecolor='black')
# plt.axvline(data['Close'].mean(), color='red', linestyle='dashed', linewidth=1)
# plt.title('Distribution of Closing Prices with Mean Indicated')
# plt.xlabel('Closing Price')
# plt.ylabel('Frequency')
# plt.legend(['Mean', 'Closing Price'])

# Adjusting the spacing between subplots
plt.subplots_adjust(hspace=0.4)
plt.tight_layout()
plt.show()
