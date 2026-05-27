import pandas as pd
import numpy as np

def calculate_ma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()

def calculate_rma(data, window):
    return data.ewm(alpha=1 / window, adjust=False).mean()
import pandas as pd

def calculate_rsi(data, window=14):
    data = pd.Series(data, dtype=float)
    if len(data) < window:
        return pd.Series([50] * len(data), index=data.index, name='RSI')
    delta = data.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    rsi = pd.Series(index=data.index, dtype=float)
    avg_gain = gains.iloc[1:window + 1].mean()
    avg_loss = losses.iloc[1:window + 1].mean()
    if avg_loss == 0:
        rsi.iloc[window] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi.iloc[window] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(window + 1, len(data)):
        current_gain = gains.iloc[i]
        current_loss = losses.iloc[i]
        avg_gain = (avg_gain * (window - 1) + current_gain) / window
        avg_loss = (avg_loss * (window - 1) + current_loss) / window
        if avg_loss == 0:
            rsi.iloc[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi.iloc[i] = 100.0 - 100.0 / (1.0 + rs)
    rsi.ffill(inplace=True)
    rsi.fillna(50, inplace=True)
    return rsi

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    ema_short = calculate_ema(data, short_window)
    ema_long = calculate_ema(data, long_window)
    macd = ema_short - ema_long
    signal = calculate_ema(macd, signal_window)
    return (macd, signal)

def calculate_obv(data_close, data_volume):
    obv = [0]
    for i in range(1, len(data_close)):
        if data_close.iloc[i] > data_close.iloc[i - 1]:
            obv.append(obv[-1] + data_volume.iloc[i])
        elif data_close.iloc[i] < data_close.iloc[i - 1]:
            obv.append(obv[-1] - data_volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=data_close.index)

def calculate_vwap(data_close, data_volume):
    cumulative_volume = data_volume.cumsum()
    cumulative_price_volume = (data_close * data_volume).cumsum()
    vwap = cumulative_price_volume / cumulative_volume
    return vwap
