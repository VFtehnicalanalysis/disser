import numpy as np
import pandas as pd
from .base import Strategy
from TA.base_methods import calculate_ema, calculate_rsi, calculate_macd

def calculate_pivot_high(series, left, right):
    pivots = pd.Series(np.nan, index=series.index)
    for i in range(left, len(series) - right):
        window = series.iloc[i - left:i + right + 1]
        if series.iloc[i] == window.max():
            pivots.iloc[i] = series.iloc[i]
    return pivots

def calculate_pivot_low(series, left, right):
    pivots = pd.Series(np.nan, index=series.index)
    for i in range(left, len(series) - right):
        window = series.iloc[i - left:i + right + 1]
        if series.iloc[i] == window.min():
            pivots.iloc[i] = series.iloc[i]
    return pivots

def calculate_htf_data(df, htf_freq):
    df_htf = df.resample(htf_freq).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    return df_htf

def long_trend_func(ema_HTF_short, ema_HTF_long, ema_LTF_short, ema_LTF_long):
    if ema_HTF_short < ema_HTF_long and ema_LTF_short < ema_LTF_long:
        return True
    if ema_HTF_short > ema_HTF_long and ema_LTF_short < ema_LTF_long and (ema_LTF_short > ema_HTF_short):
        return True
    return False

def short_trend_func(ema_HTF_short, ema_HTF_long, ema_LTF_short, current_close):
    if ema_HTF_short > ema_HTF_long and (ema_LTF_short < ema_HTF_short and current_close < ema_HTF_short or ema_LTF_short > ema_HTF_long):
        return True
    return False

class EMACrossoverStrategy(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, short_ema=50, long_ema=100, pyramiding=1, allowshort=True, debug=False):
        name = f'EMA_Crossover_{short_ema}_{long_ema}_{allowshort}'
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.short_ema = short_ema
        self.long_ema = long_ema
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        self.data['EMA_short'] = calculate_ema(self.data['close'], self.short_ema)
        self.data['EMA_long'] = calculate_ema(self.data['close'], self.long_ema)
        self.data['signal'] = None
        upward_cross = (self.data['EMA_short'].shift(1) < self.data['EMA_long'].shift(1)) & (self.data['EMA_short'] > self.data['EMA_long'])
        downward_cross = (self.data['EMA_short'].shift(1) > self.data['EMA_long'].shift(1)) & (self.data['EMA_short'] < self.data['EMA_long'])
        self.data.loc[upward_cross, 'signal'] = np.where(self.allowshort, 'close short; long', 'long')
        self.data.loc[downward_cross, 'signal'] = np.where(self.allowshort, 'close long; short', 'close long')
        self.signals_df = self.data[['signal', 'close']].dropna()
        if self.debug:
            self.debug_output()

    def debug_output(self):
        filename = f'DEBUG_{self.ticker}_{self.name}.csv'
        cols = ['open', 'high', 'low', 'close', 'EMA_short', 'EMA_long', 'signal']
        self.data[cols].to_csv(filename)
        print(f'Debug output saved to {filename}')

class LongTermHold(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, debug=False):
        super().__init__('LongTermHold', ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=False)
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        self.data['signal'] = None
        if not self.data.empty:
            first_index = self.data.index[0]
            last_index = self.data.index[-1]
            self.data.loc[first_index, 'signal'] = 'long'
            self.data.loc[last_index, 'signal'] = 'close long'
        self.signals_df = self.data[['signal', 'close']].dropna()
        if self.debug:
            self.debug_output()

    def debug_output(self):
        filename = f'DEBUG_{self.ticker}_{self.name}.csv'
        self.data.to_csv(filename)
        print(f'Debug output saved to {filename}')

class RSIStrategy(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, rsi_lower, rsi_upper, rsi_window, pyramiding=1, allowshort=True, debug=False):
        name = f'RSI_{rsi_lower}_{rsi_upper}_{rsi_window}'
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.rsi_lower = rsi_lower
        self.rsi_upper = rsi_upper
        self.rsi_window = rsi_window
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        rsi_col = f'RSI_{self.rsi_window}'
        if rsi_col not in self.data.columns:
            self.data[rsi_col] = calculate_rsi(self.data['close'], self.rsi_window)
        self.data['signal'] = None
        buy_condition = (self.data[rsi_col].shift(1) < self.rsi_lower) & (self.data[rsi_col] >= self.rsi_lower)
        sell_condition = (self.data[rsi_col].shift(1) > self.rsi_upper) & (self.data[rsi_col] <= self.rsi_upper)
        self.data.loc[buy_condition, 'signal'] = 'buy'
        if self.allowshort:
            self.data.loc[sell_condition, 'signal'] = 'short'
        else:
            self.data.loc[sell_condition, 'signal'] = 'sell'
        self.signals_df = self.data[['signal', 'close']].dropna()
        if self.debug:
            self.debug_output()

    def debug_output(self):
        filename = f'DEBUG_{self.ticker}_{self.name}.csv'
        self.data.to_csv(filename)
        print(f'Debug output saved to {filename}')

class MACDHistogramStrategy(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, debug=False):
        super().__init__('MACDHistogram', ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        self.data['hist'] = self.data['MACD'] - self.data['MACD_Signal']
        self.data['signal'] = None
        buy_condition = (self.data['hist'].shift(1) < 0) & (self.data['hist'] > 0)
        sell_condition = (self.data['hist'].shift(1) > 0) & (self.data['hist'] < 0)
        self.data.loc[buy_condition, 'signal'] = 'buy'
        if self.allowshort:
            self.data.loc[sell_condition, 'signal'] = 'short'
        else:
            self.data.loc[sell_condition, 'signal'] = 'sell'
        self.signals_df = self.data[['signal', 'close']].dropna()
        if self.debug:
            self.debug_output()

    def debug_output(self):
        filename = f'DEBUG_{self.ticker}_{self.name}.csv'
        self.data.to_csv(filename)
        print(f'Debug output saved to {filename}')

class TrendlineBreakoutStrategy(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, debug=False):
        super().__init__('TrendlineBreakout', ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        signals = []
        if len(self.data) < 200:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])
            return
        for current_time in self.data.index[200:]:
            window = self.data.loc[:current_time].tail(200)
            n = len(window)
            x = np.arange(n)
            highs = window['high'].values
            a_d, b_d = np.polyfit(x, highs, 1)
            trend_down = a_d * (n - 1) + b_d
            tol_down = 0.02 * trend_down
            touches_down = np.sum(np.abs(highs - (a_d * x + b_d)) < tol_down)
            lows = window['low'].values
            a_u, b_u = np.polyfit(x, lows, 1)
            trend_up = a_u * (n - 1) + b_u
            tol_up = 0.02 * trend_up
            touches_up = np.sum(np.abs(lows - (a_u * x + b_u)) < tol_up)
            candle = self.data.loc[current_time]
            if touches_down >= 3 and candle['close'] > trend_down:
                signals.append((current_time, 'buy', candle['close']))
            elif touches_up >= 3 and candle['close'] < trend_up:
                signals.append((current_time, 'sell', candle['close']))
        if signals:
            df_signals = pd.DataFrame(signals, columns=['time', 'signal', 'close'])
            df_signals.set_index('time', inplace=True)
            self.signals_df = df_signals
        else:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])
        if self.debug:
            self.debug_output()

    def debug_output(self):
        filename = f'DEBUG_{self.ticker}_{self.name}.csv'
        self.data.to_csv(filename)
        print(f'Debug output saved to {filename}')

class TrendlineRetestStrategy(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, debug=False):
        super().__init__('TrendlineRetest', ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        signals = []
        if len(self.data) < 200:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])
            return
        prev_breakout = None
        for current_time in self.data.index[200:]:
            window = self.data.loc[:current_time].tail(200)
            n = len(window)
            x = np.arange(n)
            highs = window['high'].values
            a_d, b_d = np.polyfit(x, highs, 1)
            trend_down = a_d * (n - 1) + b_d
            tol_down = 0.02 * trend_down
            touches_down = np.sum(np.abs(highs - (a_d * x + b_d)) < tol_down)
            lows = window['low'].values
            a_u, b_u = np.polyfit(x, lows, 1)
            trend_up = a_u * (n - 1) + b_u
            tol_up = 0.02 * trend_up
            touches_up = np.sum(np.abs(lows - (a_u * x + b_u)) < tol_up)
            candle = self.data.loc[current_time]
            if touches_down >= 3 and candle['close'] > trend_down:
                prev_breakout = ('buy', trend_down, tol_down)
                continue
            elif touches_up >= 3 and candle['close'] < trend_up:
                prev_breakout = ('sell', trend_up, tol_up)
                continue
            if prev_breakout is not None:
                bp_type, trend_val, tol = prev_breakout
                if bp_type == 'buy' and abs(candle['close'] - trend_val) <= tol:
                    signals.append((current_time, 'buy', candle['close']))
                    prev_breakout = None
                elif bp_type == 'sell' and abs(candle['close'] - trend_val) <= tol:
                    signals.append((current_time, 'sell', candle['close']))
                    prev_breakout = None
        if signals:
            df_signals = pd.DataFrame(signals, columns=['time', 'signal', 'close'])
            df_signals.set_index('time', inplace=True)
            self.signals_df = df_signals
        else:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])
        if self.debug:
            self.debug_output()

    def debug_output(self):
        filename = f'DEBUG_{self.ticker}_{self.name}.csv'
        self.data.to_csv(filename)
        print(f'Debug output saved to {filename}')
