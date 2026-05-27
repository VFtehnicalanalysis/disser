import pandas as pd
from TA.base_methods import calculate_ema, calculate_rsi, calculate_macd

def crossover_strategy(df, price_column='close', indicator_column='SMA_50'):
    signals = pd.Series(index=df.index, dtype=object)
    for i in range(1, len(df)):
        if df[price_column].iloc[i] > df[indicator_column].iloc[i] and df[price_column].iloc[i - 1] <= df[indicator_column].iloc[i - 1]:
            signals.iloc[i] = 'buy'
        elif df[price_column].iloc[i] < df[indicator_column].iloc[i] and df[price_column].iloc[i - 1] >= df[indicator_column].iloc[i - 1]:
            signals.iloc[i] = 'sell'
    return signals

def ema_cross_strategy(df, short_ema='EMA_21', long_ema='EMA_50'):
    signals = pd.Series(index=df.index, dtype=object)
    for i in range(1, len(df)):
        if df[short_ema].iloc[i] > df[long_ema].iloc[i] and df[short_ema].iloc[i - 1] <= df[long_ema].iloc[i - 1]:
            signals.iloc[i] = 'buy'
        elif df[short_ema].iloc[i] < df[long_ema].iloc[i] and df[short_ema].iloc[i - 1] >= df[long_ema].iloc[i - 1]:
            signals.iloc[i] = 'sell'
    return signals

def ma_cross_strategy(df, short_ma='SMA_50', long_ma='SMA_100'):
    signals = pd.Series(index=df.index, dtype=object)
    for i in range(1, len(df)):
        if df[short_ma].iloc[i] > df[long_ma].iloc[i] and df[short_ma].iloc[i - 1] <= df[long_ma].iloc[i - 1]:
            signals.iloc[i] = 'buy'
        elif df[short_ma].iloc[i] < df[long_ma].iloc[i] and df[short_ma].iloc[i - 1] >= df[long_ma].iloc[i - 1]:
            signals.iloc[i] = 'sell'
    return signals

def rsi_strategy(df, rsi_column='RSI_14', buy_level=30, sell_level=70):
    signals = pd.Series(index=df.index, dtype=object)
    for i in range(1, len(df)):
        if df[rsi_column].iloc[i] > buy_level and df[rsi_column].iloc[i - 1] <= buy_level:
            signals.iloc[i] = 'buy'
        elif df[rsi_column].iloc[i] < sell_level and df[rsi_column].iloc[i - 1] >= sell_level:
            signals.iloc[i] = 'sell'
    return signals

def macd_strategy(df, macd_column='MACD', signal_column='MACD_Signal'):
    signals = pd.Series(index=df.index, dtype=object)
    for i in range(1, len(df)):
        if df[macd_column].iloc[i] > df[signal_column].iloc[i] and df[macd_column].iloc[i - 1] <= df[signal_column].iloc[i - 1]:
            signals.iloc[i] = 'buy'
        elif df[macd_column].iloc[i] < df[signal_column].iloc[i] and df[macd_column].iloc[i - 1] >= df[signal_column].iloc[i - 1]:
            signals.iloc[i] = 'sell'
    return signals
