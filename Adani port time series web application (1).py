#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Adani stock time series (1).csv')


# In[3]:


df.info()


# In[4]:


df.head()


# In[5]:


df['Close'].dtypes


# In[6]:


df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Close'].isna().sum()


# In[44]:


df['Log_Close'] = np.log(df['Close'])


# In[46]:


df['Log_Close_Diff'] = df['Log_Close'].diff()


# In[48]:


df[['Date', 'Close', 'Log_Close', 'Log_Close_Diff']].head()


# In[50]:


plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price')
plt.title('Adani Ports Stock Price Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[52]:


# 20-day and 50-day moving averages
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA50'] = df['Close'].rolling(window=50).mean()

plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['MA20'], label='20-Day MA', linestyle='--')
plt.plot(df['MA50'], label='50-Day MA', linestyle='--')
plt.title('Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[54]:


print(df.columns)


# In[56]:


df.columns = df.columns.str.strip()  # remove spaces
df.rename(columns={'date': 'Date'}, inplace=True)


# In[58]:


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)


# In[60]:


df.info()


# In[81]:


# ✅ Step 2: Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# ✅ Step 3: Drop rows with invalid dates (if any)
df.dropna(subset=['Date'], inplace=True)

# ✅ Step 4: Set Date as index
df.set_index('Date', inplace=True)

# ✅ Step 5: Resample monthly (mean Close price)
monthly = df['Close'].resample('M').mean()

# ✅ Step 6: Show first few results
print(monthly.head())


# In[83]:


import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(monthly, marker='o', label='Monthly Avg Close')
plt.title('Monthly Stock Price Trend (Seasonality)')
plt.xlabel('Month')
plt.ylabel('Average Close Price')
plt.legend()
plt.show()


# In[85]:


yearly = df['Close'].resample('Y').mean()

plt.figure(figsize=(10,5))
plt.plot(yearly, marker='o', label='Yearly Avg Close')
plt.title('Yearly Stock Price Pattern')
plt.xlabel('Year')
plt.ylabel('Average Close Price')
plt.legend()
plt.show()


# In[89]:


df['Log_Close'] = np.log(df['Close'])
df['Log_Close_Diff'] = df['Log_Close'].diff()

plt.figure(figsize=(14,6))
plt.plot(df['Log_Close_Diff'], label='Log Returns')
plt.title('Log Returns of Adani Ports Stock')
plt.xlabel('Date')
plt.ylabel('Log Return')
plt.legend()
plt.show()


# In[91]:


#ARIMA MODEL


# In[93]:


from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


# In[95]:


df['Close'] = df['Close'].astype(float)
ts = df['Close']


# In[97]:


result = adfuller(ts)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] > 0.05:
    print("Time series is non-stationary. Differencing needed.")
else:
    print("Time series is stationary. ARIMA can be applied directly.")


# In[101]:


ts_diff = ts.diff().dropna()


# In[103]:


plt.figure(figsize=(12,5))
plot_acf(ts_diff, lags=40)
plt.show()

plt.figure(figsize=(12,5))
plot_pacf(ts_diff, lags=40)
plt.show()


# In[105]:


df = df.asfreq('B')  # 'B' = business day frequency


# In[107]:


# Example: ARIMA(1,1,1) → p=1, d=1, q=1
model = ARIMA(ts, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())


# In[109]:


forecast = model_fit.get_forecast(steps=30)
forecast_index = pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=30, freq='B')
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Actual')
plt.plot(forecast_index, forecast_values, label='Forecast', color='red')
plt.fill_between(forecast_index,
                 forecast_ci.iloc[:,0],
                 forecast_ci.iloc[:,1],
                 color='pink', alpha=0.3)
plt.title('ARIMA Forecast for Adani Ports Stock')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[111]:


#SARIMA MODEL


# In[113]:


from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima


# In[115]:


# Ensure Date column exists and is datetime
df['Date'] = pd.to_datetime(df.index) if 'Date' not in df.columns else pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Make sure the index has a frequency
df = df.asfreq('B')  # 'B' = Business day frequency

# Use Close price
ts = df['Close']


# In[117]:


# If you are using Close price directly
ts = df['Close'].copy()

# Drop missing values
ts = ts.dropna()

# OR fill missing values (optional)
# ts = ts.fillna(method='ffill')  # forward fill
# ts = ts.fillna(method='bfill')  # backward fill

# Now run ADF test
from statsmodels.tsa.stattools import adfuller
result = adfuller(ts)
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# In[119]:


result = adfuller(ts)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# If p-value > 0.05, series is non-stationary → differencing needed


# In[121]:


import pmdarima as pm

# Automatically find best SARIMA parameters
smodel = pm.auto_arima(ts,
                       seasonal=True,
                       m=5,      # weekly seasonality (adjust based on data)
                       stepwise=True,
                       trace=True,
                       suppress_warnings=True)
print(smodel.summary())


# In[123]:


model = SARIMAX(ts,
                order=(1,1,1),
                seasonal_order=(1,1,1,5),
                enforce_stationarity=False,
                enforce_invertibility=False)
model_fit = model.fit()
print(model_fit.summary())


# In[125]:


forecast_steps = 30  # Next 30 business days
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=ts.index[-1]+pd.Timedelta(days=1),
                               periods=forecast_steps, freq='B')
forecast_values = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(ts, label='Actual')
plt.plot(forecast_index, forecast_values, color='red', label='Forecast')
plt.fill_between(forecast_index,
                 forecast_ci.iloc[:,0],
                 forecast_ci.iloc[:,1],
                 color='pink', alpha=0.3)
plt.title('SARIMA Forecast for Adani Ports Stock')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[127]:


#PROPHET MODEL


# In[129]:




# In[131]:


import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


# In[133]:


# Ensure Date column exists and is datetime
df['Date'] = pd.to_datetime(df.index) if 'Date' not in df.columns else pd.to_datetime(df['Date'])

# Create Prophet DataFrame
prophet_df = df[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})

# Handle missing values
prophet_df = prophet_df.dropna()


# In[135]:


m = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)

# Fit the model
m.fit(prophet_df)


# In[137]:


# Forecast next 30 days
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# View forecast
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[139]:


# Plot forecast
fig1 = m.plot(forecast)
plt.title('Prophet Forecast for Adani Ports')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()

# Plot components: trend, weekly, yearly seasonality
fig2 = m.plot_components(forecast)
plt.show()


# In[141]:


#LSTM


# In[143]:


get_ipython().system('pip install tensorflow')


# In[145]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[147]:


# Use Close price
data = df[['Close']].values

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)


# In[149]:


X = []
y = []
time_steps = 60  # number of past days to look at

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i-time_steps:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Reshape X for LSTM [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


# In[151]:


model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')



# In[153]:


model.summary()


# In[155]:


history = model.fit(X, y, epochs=25, batch_size=32, validation_split=0.2, verbose=1)


# In[157]:


# Prepare test data (use last 60 days)
last_60_days = scaled_data[-time_steps:]
X_test = np.array([last_60_days])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
print("Predicted Next Close Price:", predicted_price[0][0])


# In[159]:


plt.figure(figsize=(12,6))
plt.plot(df.index[-len(y):], scaler.inverse_transform(y.reshape(-1, 1)), label='Actual Price', color='blue')
plt.plot(df.index[-len(y):], scaler.inverse_transform(model.predict(X)), label='LSTM Predicted', color='red')
plt.title('Adani Ports Stock Price Prediction (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[161]:


print("Model built and compiled successfully ✅")


# In[163]:


#MODEL EVALUATION


# In[165]:


train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - 60:]  # include last 60 points for continuity

X_train, y_train = [], []
for i in range(60, len(train_data)):
    X_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# In[167]:


history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)


# In[169]:


X_test = []
y_test = scaled_data[train_size:, 0]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[171]:


predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # scale back to original values
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[173]:


y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[175]:


predictions = scaler.inverse_transform(predictions)  # should already be 2D


# In[177]:


# Check shapes
print("y_test_actual shape:", y_test_actual.shape)
print("predictions shape:", predictions.shape)

# Make them equal
min_len = min(len(y_test_actual), len(predictions))
y_test_actual = y_test_actual[:min_len]
predictions = predictions[:min_len]


# In[179]:


mask = ~np.isnan(y_test_actual.flatten()) & ~np.isnan(predictions.flatten())
y_test_actual = y_test_actual[mask]
predictions = predictions[mask]


# In[181]:


time_steps = 60
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_steps:]  # include last 60 points from train

# Create X_test and y_test
X_test, y_test = [], []
for i in range(time_steps, len(test_data)):
    X_test.append(test_data[i-time_steps:i, 0])
    y_test.append(test_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

# Check lengths
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Reshape for LSTM
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# In[183]:


predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))


# In[185]:


# -------------------------------
# LSTM for Adani Ports Stock Price
# -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1️⃣ Prepare Data
df = pd.read_csv('Adani stock time series (1).csv')  # replace with your file path
df['Date'] = pd.to_datetime(df['Date'])  # ensure date is datetime
df = df.sort_values('Date')
data = df[['Close']].values  # using Close price

# Scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# 2️⃣ Train-Test Split
time_steps = 60
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - time_steps:]  # include last 60 points from train

# Create sequences
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# 3️⃣ Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()  # view model architecture

# 4️⃣ Train Model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 5️⃣ Predict & Inverse Scale
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# 6️⃣ Evaluate Model
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)
mape = np.mean(np.abs((y_test_actual - predictions)/y_test_actual)) * 100

print("\n📊 Model Evaluation Metrics:")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.2f}%")

# 7️⃣ Plot Actual vs Predicted
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.title('Adani Ports Stock Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# 8️⃣ Plot Training Loss
plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Training Performance')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[187]:


#FORECASTING & FUTURE PREDICTION


# In[189]:


# -------------------------------
# Forecast Next 30 Days
# -------------------------------

future_days = 30
last_sequence = scaled_data[-time_steps:]  # last 60 days from your dataset
forecast_scaled = []

current_seq = last_sequence.copy()

for _ in range(future_days):
    # reshape for LSTM [1, time_steps, 1]
    input_seq = current_seq.reshape((1, time_steps, 1))
    pred = model.predict(input_seq)[0,0]  # predicted scaled value
    forecast_scaled.append(pred)

    # update sequence by appending prediction and removing first element
    current_seq = np.append(current_seq[1:], pred)

# Inverse scale predictions to original price
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))

# Prepare dates for plotting
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

# Plot forecast
plt.figure(figsize=(12,6))
plt.plot(df['Date'], df['Close'], label='Historical Close Price')
plt.plot(future_dates, forecast, label='30-Day Forecast', marker='o')
plt.title('Adani Ports Stock Price 30-Day Forecast (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[191]:


# -------------------------------
# Forecast Next 90 Days
# -------------------------------

future_days = 90
last_sequence = scaled_data[-time_steps:]  # last 60 days from your dataset
forecast_scaled = []

current_seq = last_sequence.copy()

for _ in range(future_days):
    # reshape for LSTM [1, time_steps, 1]
    input_seq = current_seq.reshape((1, time_steps, 1))
    pred = model.predict(input_seq)[0,0]  # predicted scaled value
    forecast_scaled.append(pred)

    # update sequence by appending prediction and removing first element
    current_seq = np.append(current_seq[1:], pred)

# Inverse scale predictions to original price
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))

# Prepare dates for plotting
last_date = df['Date'].iloc[-1]
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days)

# Plot forecast
plt.figure(figsize=(14,7))
plt.plot(df['Date'], df['Close'], label='Historical Close Price')
plt.plot(future_dates, forecast, label='90-Day Forecast', marker='o')
plt.title('Adani Ports Stock Price 90-Day Forecast (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[193]:


#PREDICTING FUTURE FOR NEXT 6 MONTHS


# In[195]:


# -------------------------------
# Forecast Next 6 Months (~132 Trading Days)
# -------------------------------

future_days = 132  # approx 6 months
last_sequence = scaled_data[-time_steps:]  # last 60 days from your dataset
forecast_scaled = []

current_seq = last_sequence.copy()

for _ in range(future_days):
    # reshape for LSTM [1, time_steps, 1]
    input_seq = current_seq.reshape((1, time_steps, 1))
    pred = model.predict(input_seq)[0,0]  # predicted scaled value
    forecast_scaled.append(pred)

    # update sequence by appending prediction and removing first element
    current_seq = np.append(current_seq[1:], pred)

# Inverse scale predictions to original price
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))

# Prepare dates for plotting
last_date = df['Date'].iloc[-1]
future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=future_days)  # business days

# Plot forecast
plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Close'], label='Historical Close Price')
plt.plot(future_dates, forecast, label='6-Month Forecast', marker='o', markersize=3)
plt.title('Adani Ports Stock Price 6-Month Forecast (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[209]:


plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Close'], label='Historical Close Price', color='blue')
plt.plot(future_dates_array, forecast, label='6-Month Forecast', color='red')
plt.title('Adani Ports Stock Price 6-Month Forecast (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[204]:


import numpy as np

window_size = 5

# Calculate rough confidence interval
std_dev = np.std(forecast)
forecast_upper = forecast + 1.96 * std_dev
forecast_lower = forecast - 1.96 * std_dev

# Smooth the forecast and intervals
forecast_smooth = pd.Series(forecast.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_upper_smooth = pd.Series(forecast_upper.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_lower_smooth = pd.Series(forecast_lower.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()

# Convert future_dates to numpy array
future_dates_array = future_dates.to_numpy()

# Plot
plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Close'], label='Historical Close Price', color='blue')
plt.plot(future_dates_array, forecast_smooth, label='6-Month Forecast (Smoothed)', color='red')
plt.fill_between(future_dates_array, forecast_lower_smooth, forecast_upper_smooth, color='red', alpha=0.2, label='95% Confidence Interval')
plt.title('Adani Ports Stock Price 6-Month Forecast with Smoothed Curve (LSTM)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[211]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# -------------------------------
# Historical Data
# -------------------------------
dates = df['Date']
prices = df['Close']

# -------------------------------
# ARIMA Forecast (assume forecast_arima exists)
# -------------------------------
# Example: forecast_arima is a numpy array of future prices
# forecast_arima_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=len(forecast_arima))
# Replace with your actual ARIMA forecast

# -------------------------------
# SARIMA Forecast (assume forecast_sarima exists)
# -------------------------------
# Example: forecast_sarima is a numpy array of future prices
# forecast_sarima_dates = pd.date_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=len(forecast_sarima))
# Replace with your actual SARIMA forecast

# -------------------------------
# Prophet Forecast (assume forecast_prophet_df exists)
# -------------------------------
# forecast_prophet_df has 'ds' and 'yhat' columns
# prophet_dates = forecast_prophet_df['ds']
# prophet_forecast = forecast_prophet_df['yhat']

# -------------------------------
# LSTM Forecast (6-month)
# -------------------------------
future_days = 132
future_dates = pd.bdate_range(dates.iloc[-1] + pd.Timedelta(days=1), periods=future_days)

# Smoothed LSTM forecast and confidence intervals
window_size = 5
forecast_smooth = pd.Series(forecast.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_upper_smooth = pd.Series(forecast_upper.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()
forecast_lower_smooth = pd.Series(forecast_lower.flatten()).rolling(window=window_size, min_periods=1, center=True).mean()

# -------------------------------
# Plot Combined Forecasts
# -------------------------------
plt.figure(figsize=(18,8))
plt.plot(dates, prices, label='Historical Close Price', color='black')

# Uncomment and replace with your actual forecasts
# plt.plot(forecast_arima_dates, forecast_arima, label='ARIMA Forecast', color='blue')
# plt.plot(forecast_sarima_dates, forecast_sarima, label='SARIMA Forecast', color='green')
# plt.plot(prophet_dates, prophet_forecast, label='Prophet Forecast', color='orange')

plt.plot(future_dates.to_numpy(), forecast_smooth, label='LSTM Forecast (Smoothed)', color='red')
plt.fill_between(future_dates.to_numpy(), forecast_lower_smooth, forecast_upper_smooth, color='red', alpha=0.2, label='LSTM 95% CI')

plt.title('Adani Ports Stock Price Forecast Comparison')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[213]:


pip install mplfinance


# In[215]:


import mplfinance as mpf
import pandas as pd
import numpy as np

# -------------------------------
# Prepare Historical Data for Candles
# -------------------------------
df_candle = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()
df_candle.set_index('Date', inplace=True)

# -------------------------------
# Prepare Forecast Data for Candles (LSTM)
# -------------------------------
forecast_candle = pd.DataFrame({
    'Open': forecast_smooth.shift(1).fillna(forecast_smooth.iloc[0]),  # previous close as open
    'High': forecast_upper_smooth,
    'Low': forecast_lower_smooth,
    'Close': forecast_smooth
}, index=future_dates)

# -------------------------------
# Combine Historical + Forecast
# -------------------------------
combined = pd.concat([df_candle, forecast_candle])

# -------------------------------
# Limit to last 1 year historical + 6-month forecast
# -------------------------------
subset = combined[-(252 + len(future_dates)):]  # 252 trading days ≈ 1 year

# -------------------------------
# Plot Candlestick Chart
# -------------------------------
mpf.plot(
    subset,
    type='candle',       # red/green candles
    style='charles',
    figsize=(16,8),
    title='Adani Ports Stock Price with LSTM 6-Month Forecast',
    ylabel='Price',
    volume=False
)


# In[219]:


last_sequence = np.append(last_sequence[:,1:,:], np.array(next_price).reshape(1,1,1), axis=1)


# In[225]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------
# 1. Prepare Data
# -------------------------
# Use only 'Close' prices
data = df['Close'].values.reshape(-1,1)

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
data_scaled = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60  # number of past days used to predict next day
X, y = create_sequences(data_scaled, SEQ_LENGTH)

# Split into train/test if needed (optional)
# Here, we use all data for training
# -------------------------
# 2. Build LSTM Model
# -------------------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X, y, epochs=50, batch_size=32, verbose=1)

# -------------------------
# 3. Predict Future 30 Days
# -------------------------
future_days = 30
predicted_prices = []
last_sequence = data_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)

for _ in range(future_days):
    next_price = model.predict(last_sequence)[0,0]
    predicted_prices.append(next_price)

    # Append predicted price (reshaped) and remove first value
    last_sequence = np.append(last_sequence[:,1:,:], np.array(next_price).reshape(1,1,1), axis=1)


# Inverse transform to get actual prices
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1,1))

# -------------------------
# 4. Prepare Dates for Future Prediction
# -------------------------
future_dates = pd.date_range(start=df['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)

# -------------------------
# 5. Plot
# -------------------------
plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Close'], label='Historical Close Price', color='green')
plt.plot(future_dates, predicted_prices, label='30-Day Forecast', color='red')
plt.title('Stock Price 30-Day Forecast with LSTM')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()


# In[19]:


pip install streamlit


# In[24]:


get_ipython().system('streamlit run forecast_app.py')


# In[26]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

st.title("📊 Time Series Forecasting Dashboard")
st.write("Compare ARIMA, SARIMA, Prophet, and LSTM forecasts on your dataset")

# -------------------------
# UPLOAD SECTION
# -------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=True)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------
    # COLUMN SELECTION
    # -------------------------
    date_col = st.selectbox("Select the Date column", df.columns)
    value_col = st.selectbox("Select the Value column", df.columns)

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, value_col]].dropna()
    df = df.sort_values(by=date_col)
    df = df.rename(columns={date_col: "ds", value_col: "y"})

    st.line_chart(df.set_index("ds")["y"])

    # -------------------------
    # FORECAST PERIOD
    # -------------------------
    periods = st.slider("📅 Forecast periods (days)", 30, 365, 90)

    # -------------------------
    # MODEL SELECTION
    # -------------------------
    model_choice = st.radio("Choose Forecasting Model", ["ARIMA", "SARIMA", "Prophet", "LSTM"])

    # -------------------------
    # FORECASTING
    # -------------------------
    if st.button("🚀 Run Forecast"):
        if model_choice == "ARIMA":
            st.subheader("🔮 ARIMA Forecast")
            model = ARIMA(df["y"], order=(5, 1, 0))
            fitted = model.fit()
            forecast = fitted.forecast(steps=periods)
            forecast_index = pd.date_range(df["ds"].iloc[-1], periods=periods+1, freq='D')[1:]
            forecast_df = pd.DataFrame({"ds": forecast_index, "Forecast": forecast.values})
            st.line_chart(pd.concat([df.set_index("ds")["y"], forecast_df.set_index("ds")["Forecast"]], axis=1))

        elif model_choice == "SARIMA":
            st.subheader("🔮 SARIMA Forecast")
            model = SARIMAX(df["y"], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            fitted = model.fit(disp=False)
            forecast = fitted.forecast(steps=periods)
            forecast_index = pd.date_range(df["ds"].iloc[-1], periods=periods+1, freq='D')[1:]
            forecast_df = pd.DataFrame({"ds": forecast_index, "Forecast": forecast.values})
            st.line_chart(pd.concat([df.set_index("ds")["y"], forecast_df.set_index("ds")["Forecast"]], axis=1))

        elif model_choice == "Prophet":
            st.subheader("🔮 Prophet Forecast")
            m = Prophet()
            m.fit(df)
            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)
            fig = m.plot(forecast)
            st.pyplot(fig)

        elif model_choice == "LSTM":
            st.subheader("🔮 LSTM Forecast")
            data = df["y"].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled = scaler.fit_transform(data)

            def create_dataset(series, look_back=30):
                X, y = [], []
                for i in range(len(series) - look_back):
                    X.append(series[i:i+look_back])
                    y.append(series[i+look_back])
                return np.array(X), np.array(y)

            look_back = 30
            X, y = create_dataset(scaled, look_back)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))

            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
            model.add(LSTM(50))
            model.add(Dense(1))
            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(X, y, epochs=10, batch_size=16, verbose=0)

            # Forecast
            last_seq = scaled[-look_back:]
            forecast = []
            for _ in range(periods):
                pred = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)
                forecast.append(pred[0][0])
                last_seq = np.append(last_seq[1:], pred)[-look_back:]

            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            forecast_index = pd.date_range(df["ds"].iloc[-1], periods=periods+1, freq='D')[1:]
            forecast_df = pd.DataFrame({"ds": forecast_index, "Forecast": forecast.flatten()})
            st.line_chart(pd.concat([df.set_index("ds")["y"], forecast_df.set_index("ds")["Forecast"]], axis=1))

else:
    st.info("👆 Upload a CSV file to start forecasting.")


# In[ ]:





# In[ ]:




