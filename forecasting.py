import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def linear_regression_forecast(df, date_column, value_column, n_periods, freq="D"):
    """
    Forecast for hourly/daily/weekly/monthly periods.
    - For hourly: X = [0, 1, ..., n-1]
    - For daily/weekly/monthly: use date ordinals
    """
    df = df.copy()
    df = df[[date_column, value_column]].dropna()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)

    if len(df) < 2:
        return []

    if freq == "H":
        # Predict next n hours, sequentially
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[value_column].values
        model = LinearRegression().fit(X, y)
        future_X = np.arange(len(df), len(df) + n_periods).reshape(-1, 1)
        preds = model.predict(future_X)
        last_time = df[date_column].iloc[-1]
        future_times = [last_time + pd.Timedelta(hours=i+1) for i in range(n_periods)]
        return [(future_times[i], float(preds[i])) for i in range(n_periods)]
    else:
        # For D/W/M: use date ordinal for X
        df['DateOrdinal'] = df[date_column].map(pd.Timestamp.toordinal)
        X = df[['DateOrdinal']]
        y = df[value_column]
        model = LinearRegression().fit(X, y)
        last_date = df[date_column].max()
        preds = []
        for i in range(1, n_periods+1):
            if freq == "D":
                next_date = last_date + pd.Timedelta(days=i)
            elif freq == "W":
                next_date = last_date + pd.Timedelta(weeks=i)
            elif freq == "M":
                next_date = (last_date + pd.DateOffset(months=i)).replace(day=1)
            else:
                next_date = last_date + pd.Timedelta(days=i)
            pred = model.predict([[next_date.toordinal()]])[0]
            preds.append((next_date, float(pred)))
        return preds

def prepare_lstm_data(series, n_steps):
    scaler = MinMaxScaler()
    series = np.asarray(series).reshape(-1, 1)
    scaled = scaler.fit_transform(series)
    X, y = [], []
    for i in range(len(scaled) - n_steps):
        X.append(scaled[i:i+n_steps])
        y.append(scaled[i+n_steps])
    if not X:
        return None, None, None
    return np.array(X), np.array(y), scaler

def lstm_forecast(df, value_column, n_forecast=7, n_steps=14, freq="D"):
    df = df.copy()
    # Use index as time if possible
    if isinstance(df.index, pd.DatetimeIndex):
        series = df[value_column].values
        time_index = df.index
    else:
        series = df[value_column].values
        time_index = pd.to_datetime(df.iloc[:, 0])

    # Adjust n_steps for short series
    effective_n_steps = min(n_steps, max(1, len(series)//2))
    if len(series) < effective_n_steps + 1:
        return []

    X, y, scaler = prepare_lstm_data(series, effective_n_steps)
    if X is None or len(X) == 0:
        return []
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)
    forecast = []
    curr_seq = X[-1]
    for _ in range(n_forecast):
        pred = model.predict(curr_seq[np.newaxis, ...], verbose=0)
        forecast.append(pred[0,0])
        curr_seq = np.vstack([curr_seq[1:], pred])
    forecast_values = scaler.inverse_transform(np.array(forecast).reshape(-1,1)).flatten()

    # Get correct date index
    last_date = time_index[-1]
    if freq == "H":
        dates = [last_date + pd.Timedelta(hours=i+1) for i in range(n_forecast)]
    elif freq == "W":
        dates = [last_date + pd.Timedelta(weeks=i+1) for i in range(n_forecast)]
    elif freq == "M":
        dates = [(last_date + pd.DateOffset(months=i+1)).replace(day=1) for i in range(n_forecast)]
    else:
        dates = [last_date + pd.Timedelta(days=i+1) for i in range(n_forecast)]
    return [(dates[i], float(forecast_values[i])) for i in range(n_forecast)]
