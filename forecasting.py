import pandas as pd
import numpy as np
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def prophet_lstm_forecast(df, date_col, value_col, n_days=7):
    """
    Forecasts future values using Prophet for trend/seasonality,
    then LSTM to model the residuals, then combines them.
    Returns forecast_df with columns: Date, Prophet_Forecast, LSTM_Residual, Final_Forecast
    """
    df = df.copy()
    df = df[[date_col, value_col]].dropna()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    # --- Prophet forecast ---
    df_prophet = df.rename(columns={date_col: "ds", value_col: "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)
    forecast = forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Prophet_Forecast"})
    forecast = forecast.set_index("Date")
    
    # --- Compute residuals on train set ---
    df_merged = df.set_index(date_col).join(forecast, how='left')
    df_merged = df_merged[~df_merged[value_col].isnull()]
    df_merged["Residual"] = df_merged[value_col] - df_merged["Prophet_Forecast"]
    
    # --- LSTM on residuals ---
    residual_series = df_merged["Residual"].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    res_scaled = scaler.fit_transform(residual_series.reshape(-1, 1))
    look_back = 7
    X, y = [], []
    for i in range(len(res_scaled) - look_back):
        X.append(res_scaled[i:i+look_back, 0])
        y.append(res_scaled[i+look_back, 0])
    if len(X) < 1:
        # Not enough data for LSTM, fallback to Prophet only
        forecast = forecast.reset_index()
        forecast["LSTM_Residual"] = 0
        forecast["Final_Forecast"] = forecast["Prophet_Forecast"]
        return forecast[-n_days:]
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    lstm = Sequential([
        LSTM(32, input_shape=(look_back, 1)),
        Dense(1)
    ])
    lstm.compile(loss="mse", optimizer="adam")
    lstm.fit(X, y, epochs=30, batch_size=1, verbose=0)
    # Predict future residuals
    last_seq = res_scaled[-look_back:].reshape(1, look_back, 1)
    future_residuals = []
    for _ in range(n_days):
        pred = lstm.predict(last_seq)[0][0]
        future_residuals.append(pred)
        last_seq = np.append(last_seq.flatten()[1:], pred).reshape(1, look_back, 1)
    future_residuals = scaler.inverse_transform(np.array(future_residuals).reshape(-1, 1)).flatten()
    
    # --- Combine Prophet + LSTM residuals ---
    forecast = forecast.reset_index()
    forecast["LSTM_Residual"] = 0.0
    forecast["Final_Forecast"] = forecast["Prophet_Forecast"]
    forecast.loc[forecast.index[-n_days:], "LSTM_Residual"] = future_residuals
    forecast.loc[forecast.index[-n_days:], "Final_Forecast"] = (
        forecast.loc[forecast.index[-n_days:], "Prophet_Forecast"].values + future_residuals
    )
    return forecast[-n_days:]
