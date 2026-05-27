import numpy as np
import pandas as pd
from .base import Strategy
from TA.base_methods import calculate_ema, calculate_rsi, calculate_macd, calculate_obv, calculate_vwap

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

def _compute_atr(data, window=14):
    high = data['high']
    low = data['low']
    prev_close = data['close'].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()

def _detect_swing_pivots(highs, lows, window):
    n = len(highs)
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    if n < 2 * window + 1:
        return (is_high, is_low)
    for i in range(window, n - window):
        h = highs[i]
        l = lows[i]
        left_h = highs[i - window:i]
        right_h = highs[i + 1:i + window + 1]
        if h > left_h.max() and h >= right_h.max():
            is_high[i] = True
        left_l = lows[i - window:i]
        right_l = lows[i + 1:i + window + 1]
        if l < left_l.min() and l <= right_l.min():
            is_low[i] = True
    return (is_high, is_low)

def _cluster_horizontal_levels(pivot_x, pivot_prices, tol, min_touches):
    if len(pivot_prices) == 0:
        return []
    order = np.argsort(pivot_prices)
    sorted_prices = pivot_prices[order]
    levels = []
    cluster = [sorted_prices[0]]
    for p in sorted_prices[1:]:
        if p - cluster[-1] <= tol:
            cluster.append(p)
        else:
            if len(cluster) >= min_touches:
                levels.append(float(np.mean(cluster)))
            cluster = [p]
    if len(cluster) >= min_touches:
        levels.append(float(np.mean(cluster)))
    return levels

def _find_diagonal_lines_pivot(pivot_x, pivot_prices, tol, min_touches, kind):
    n = len(pivot_x)
    if n < 2 or n < min_touches:
        return []
    lines = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = (pivot_x[i], pivot_prices[i])
            x2, y2 = (pivot_x[j], pivot_prices[j])
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            line_at_pivots = slope * pivot_x + intercept
            if kind == 'resistance':
                if np.any(pivot_prices > line_at_pivots + tol):
                    continue
                touches = int(np.sum(np.abs(pivot_prices - line_at_pivots) <= tol))
            else:
                if np.any(pivot_prices < line_at_pivots - tol):
                    continue
                touches = int(np.sum(np.abs(pivot_prices - line_at_pivots) <= tol))
            if touches >= min_touches:
                lines.append((float(slope), float(intercept)))
    return lines

def _fit_diagonal_line_ols(pivot_x, pivot_prices, min_touches):
    if len(pivot_x) < min_touches:
        return []
    slope, intercept = np.polyfit(pivot_x, pivot_prices, 1)
    return [(float(slope), float(intercept))]

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

    def debug_output(self):
        filename = f'DEBUG_{self.ticker}_{self.name}.csv'
        self.data[['open', 'high', 'low', 'close', 'EMA_short', 'EMA_long', 'signal']].to_csv(filename)

class LongTermHold(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, debug=False):
        super().__init__('LongTermHold', ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=False)
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        self.data['signal'] = None
        if not self.data.empty:
            self.data.loc[self.data.index[0], 'signal'] = 'long'
            self.data.loc[self.data.index[-1], 'signal'] = 'close long'
        self.signals_df = self.data[['signal', 'close']].dropna()

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

class MACDHistogramStrategy(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, macd_fast=12, macd_slow=26, macd_signal=9, pyramiding=1, allowshort=True, debug=False):
        name = f'MACD_{macd_fast}_{macd_slow}_{macd_signal}'
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        macd_line, signal_line = calculate_macd(self.data['close'], short_window=self.macd_fast, long_window=self.macd_slow, signal_window=self.macd_signal)
        self.data['_macd'] = macd_line
        self.data['_macd_signal'] = signal_line
        self.data['hist'] = self.data['_macd'] - self.data['_macd_signal']
        self.data['signal'] = None
        buy_condition = (self.data['hist'].shift(1) < 0) & (self.data['hist'] > 0)
        sell_condition = (self.data['hist'].shift(1) > 0) & (self.data['hist'] < 0)
        self.data.loc[buy_condition, 'signal'] = 'buy'
        if self.allowshort:
            self.data.loc[sell_condition, 'signal'] = 'short'
        else:
            self.data.loc[sell_condition, 'signal'] = 'sell'
        self.signals_df = self.data[['signal', 'close']].dropna()

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

class _BreakoutBase(Strategy):

    def __init__(self, name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, lookback=200, pivot_window=5, min_touches=3, atr_window=14, tolerance_atr_mult=0.5, cluster_atr_mult=0.5, confirmation_bars=1, cooldown_bars=5, volume_confirmation=False, volume_window=20, volume_mult=1.5, include_horizontal=True, include_diagonal=True):
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.lookback = lookback
        self.pivot_window = pivot_window
        self.min_touches = min_touches
        self.atr_window = atr_window
        self.tolerance_atr_mult = tolerance_atr_mult
        self.cluster_atr_mult = cluster_atr_mult
        self.confirmation_bars = confirmation_bars
        self.cooldown_bars = cooldown_bars
        self.volume_confirmation = volume_confirmation
        self.volume_window = volume_window
        self.volume_mult = volume_mult
        self.include_horizontal = include_horizontal
        self.include_diagonal = include_diagonal

    def _build_diagonal_lines(self, pivot_x, pivot_prices, tol, kind):
        raise NotImplementedError

    def _emit(self, signals, last_signal_bar, side, t, time_t, close_t, volumes, vol_avg):
        if t - last_signal_bar[side] < self.cooldown_bars:
            return False
        if self.volume_confirmation:
            v_t = volumes[t]
            v_avg = vol_avg[t]
            if not (v_avg > 0 and v_t > self.volume_mult * v_avg):
                return False
        signals.append((time_t, side, close_t))
        last_signal_bar[side] = t
        return True

    def generate_signals(self):
        self.load_data()
        df = self.data
        n = len(df)
        empty = pd.DataFrame(columns=['signal', 'close'])
        if n < max(self.lookback + self.pivot_window + 2, self.atr_window + 2):
            self.signals_df = empty
            return
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        closes = df['close'].values.astype(float)
        volumes = df['volume'].values.astype(float) if 'volume' in df.columns else np.zeros(n)
        atr = _compute_atr(df, self.atr_window).values
        is_ph, is_pl = _detect_swing_pivots(highs, lows, self.pivot_window)
        vol_avg = pd.Series(volumes).rolling(self.volume_window, min_periods=1).mean().values
        signals = []
        pending = {'buy': None, 'sell': None}
        last_signal_bar = {'buy': -10 ** 9, 'sell': -10 ** 9}
        for t in range(self.lookback, n):
            atr_t = atr[t]
            if np.isnan(atr_t) or atr_t <= 0:
                continue
            tol = self.tolerance_atr_mult * atr_t
            cluster_tol = self.cluster_atr_mult * atr_t
            close_t = closes[t]
            close_prev = closes[t - 1]
            time_t = df.index[t]
            for side in ('buy', 'sell'):
                p = pending[side]
                if p is None:
                    continue
                level_t = p['slope'] * t + p['intercept']
                still = close_t > level_t if side == 'buy' else close_t < level_t
                if still:
                    p['bars_confirmed'] += 1
                    if p['bars_confirmed'] >= self.confirmation_bars:
                        self._emit(signals, last_signal_bar, side, t, time_t, close_t, volumes, vol_avg)
                        pending[side] = None
                else:
                    pending[side] = None
            wstart = t - self.lookback
            wend = t - self.pivot_window
            if wend <= wstart:
                continue
            mask_h = is_ph[wstart:wend]
            mask_l = is_pl[wstart:wend]
            window_x = np.arange(wstart, wend)
            ph_x = window_x[mask_h].astype(float)
            ph_p = highs[wstart:wend][mask_h]
            pl_x = window_x[mask_l].astype(float)
            pl_p = lows[wstart:wend][mask_l]
            res_levels = []
            sup_levels = []
            if self.include_horizontal:
                for price in _cluster_horizontal_levels(ph_x, ph_p, cluster_tol, self.min_touches):
                    res_levels.append((0.0, price))
                for price in _cluster_horizontal_levels(pl_x, pl_p, cluster_tol, self.min_touches):
                    sup_levels.append((0.0, price))
            if self.include_diagonal:
                res_levels.extend(self._build_diagonal_lines(ph_x, ph_p, tol, 'resistance'))
                sup_levels.extend(self._build_diagonal_lines(pl_x, pl_p, tol, 'support'))
            if pending['buy'] is None:
                broken = []
                for slope, intercept in res_levels:
                    v_t = slope * t + intercept
                    v_prev = slope * (t - 1) + intercept
                    if close_prev <= v_prev and close_t > v_t:
                        broken.append((slope, intercept, v_t))
                if broken:
                    broken.sort(key=lambda x: x[2])
                    slope, intercept, _ = broken[0]
                    if self.confirmation_bars <= 1:
                        self._emit(signals, last_signal_bar, 'buy', t, time_t, close_t, volumes, vol_avg)
                    else:
                        pending['buy'] = {'slope': slope, 'intercept': intercept, 'bars_confirmed': 1}
            if pending['sell'] is None:
                broken = []
                for slope, intercept in sup_levels:
                    v_t = slope * t + intercept
                    v_prev = slope * (t - 1) + intercept
                    if close_prev >= v_prev and close_t < v_t:
                        broken.append((slope, intercept, v_t))
                if broken:
                    broken.sort(key=lambda x: -x[2])
                    slope, intercept, _ = broken[0]
                    if self.confirmation_bars <= 1:
                        self._emit(signals, last_signal_bar, 'sell', t, time_t, close_t, volumes, vol_avg)
                    else:
                        pending['sell'] = {'slope': slope, 'intercept': intercept, 'bars_confirmed': 1}
        if signals:
            df_signals = pd.DataFrame(signals, columns=['time', 'signal', 'close'])
            df_signals.set_index('time', inplace=True)
            self.signals_df = df_signals
        else:
            self.signals_df = empty

class BreakoutPivotStrategy(_BreakoutBase):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, **kwargs):
        super().__init__('BreakoutPivot', ticker, timeframe, initial_account_size, entry_percentage, **kwargs)

    def _build_diagonal_lines(self, pivot_x, pivot_prices, tol, kind):
        return _find_diagonal_lines_pivot(pivot_x, pivot_prices, tol, self.min_touches, kind)

class BreakoutOLSPivotStrategy(_BreakoutBase):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, **kwargs):
        super().__init__('BreakoutOLSPivot', ticker, timeframe, initial_account_size, entry_percentage, **kwargs)

    def _build_diagonal_lines(self, pivot_x, pivot_prices, tol, kind):
        return _fit_diagonal_line_ols(pivot_x, pivot_prices, self.min_touches)

class _RetestBase(Strategy):

    def __init__(self, name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, lookback=200, pivot_window=5, min_touches=3, atr_window=14, tolerance_atr_mult=0.5, cluster_atr_mult=0.5, confirmation_bars=1, cooldown_bars=5, volume_confirmation=False, volume_window=20, volume_mult=1.5, include_horizontal=True, include_diagonal=True, retest_window=20, min_bars_after_breakout=2, cancel_atr_mult=1.0):
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.lookback = lookback
        self.pivot_window = pivot_window
        self.min_touches = min_touches
        self.atr_window = atr_window
        self.tolerance_atr_mult = tolerance_atr_mult
        self.cluster_atr_mult = cluster_atr_mult
        self.confirmation_bars = confirmation_bars
        self.cooldown_bars = cooldown_bars
        self.volume_confirmation = volume_confirmation
        self.volume_window = volume_window
        self.volume_mult = volume_mult
        self.include_horizontal = include_horizontal
        self.include_diagonal = include_diagonal
        self.retest_window = retest_window
        self.min_bars_after_breakout = min_bars_after_breakout
        self.cancel_atr_mult = cancel_atr_mult

    def _build_diagonal_lines(self, pivot_x, pivot_prices, tol, kind):
        raise NotImplementedError

    def _emit(self, signals, last_signal_bar, side, t, time_t, price, volumes, vol_avg):
        if t - last_signal_bar[side] < self.cooldown_bars:
            return False
        if self.volume_confirmation:
            v_t = volumes[t]
            v_avg = vol_avg[t]
            if not (v_avg > 0 and v_t > self.volume_mult * v_avg):
                return False
        signals.append((time_t, side, price))
        last_signal_bar[side] = t
        return True

    def _compute_levels_at(self, t, atr_t, highs, lows, is_ph, is_pl):
        wstart = t - self.lookback
        wend = t - self.pivot_window
        if wend <= wstart:
            return ([], [])
        tol = self.tolerance_atr_mult * atr_t
        cluster_tol = self.cluster_atr_mult * atr_t
        mask_h = is_ph[wstart:wend]
        mask_l = is_pl[wstart:wend]
        window_x = np.arange(wstart, wend)
        ph_x = window_x[mask_h].astype(float)
        ph_p = highs[wstart:wend][mask_h]
        pl_x = window_x[mask_l].astype(float)
        pl_p = lows[wstart:wend][mask_l]
        res = []
        sup = []
        if self.include_horizontal:
            for price in _cluster_horizontal_levels(ph_x, ph_p, cluster_tol, self.min_touches):
                res.append((0.0, price))
            for price in _cluster_horizontal_levels(pl_x, pl_p, cluster_tol, self.min_touches):
                sup.append((0.0, price))
        if self.include_diagonal:
            res.extend(self._build_diagonal_lines(ph_x, ph_p, tol, 'resistance'))
            sup.extend(self._build_diagonal_lines(pl_x, pl_p, tol, 'support'))
        return (res, sup)

    def generate_signals(self):
        self.load_data()
        df = self.data
        n = len(df)
        empty = pd.DataFrame(columns=['signal', 'close'])
        if n < max(self.lookback + self.pivot_window + 2, self.atr_window + 2):
            self.signals_df = empty
            return
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        closes = df['close'].values.astype(float)
        volumes = df['volume'].values.astype(float) if 'volume' in df.columns else np.zeros(n)
        atr = _compute_atr(df, self.atr_window).values
        is_ph, is_pl = _detect_swing_pivots(highs, lows, self.pivot_window)
        vol_avg = pd.Series(volumes).rolling(self.volume_window, min_periods=1).mean().values
        signals = []
        pending_breakout = {'buy': None, 'sell': None}
        pending_retest = {'buy': None, 'sell': None}
        last_signal_bar = {'buy': -10 ** 9, 'sell': -10 ** 9}
        for t in range(self.lookback, n):
            atr_t = atr[t]
            if np.isnan(atr_t) or atr_t <= 0:
                continue
            tol = self.tolerance_atr_mult * atr_t
            close_t = closes[t]
            close_prev = closes[t - 1]
            high_t = highs[t]
            low_t = lows[t]
            time_t = df.index[t]
            for side in ('buy', 'sell'):
                p = pending_breakout[side]
                if p is None:
                    continue
                level_t = p['slope'] * t + p['intercept']
                still = close_t > level_t if side == 'buy' else close_t < level_t
                if still:
                    p['bars_confirmed'] += 1
                    if p['bars_confirmed'] >= self.confirmation_bars:
                        pending_retest[side] = {'slope': p['slope'], 'intercept': p['intercept'], 'breakout_bar': t}
                        pending_breakout[side] = None
                else:
                    pending_breakout[side] = None
            for side in ('buy', 'sell'):
                r = pending_retest[side]
                if r is None:
                    continue
                age = t - r['breakout_bar']
                if age > self.retest_window:
                    pending_retest[side] = None
                    continue
                level_t = r['slope'] * t + r['intercept']
                if side == 'buy' and close_t < level_t - self.cancel_atr_mult * atr_t:
                    pending_retest[side] = None
                    continue
                if side == 'sell' and close_t > level_t + self.cancel_atr_mult * atr_t:
                    pending_retest[side] = None
                    continue
                if age >= self.min_bars_after_breakout:
                    if side == 'buy':
                        if low_t <= level_t + tol and close_t > level_t - tol:
                            self._emit(signals, last_signal_bar, 'buy', t, time_t, close_t, volumes, vol_avg)
                            pending_retest[side] = None
                    elif high_t >= level_t - tol and close_t < level_t + tol:
                        self._emit(signals, last_signal_bar, 'sell', t, time_t, close_t, volumes, vol_avg)
                        pending_retest[side] = None
            res_levels, sup_levels = self._compute_levels_at(t, atr_t, highs, lows, is_ph, is_pl)
            if pending_breakout['buy'] is None and pending_retest['buy'] is None:
                broken = []
                for slope, intercept in res_levels:
                    v_t = slope * t + intercept
                    v_prev = slope * (t - 1) + intercept
                    if close_prev <= v_prev and close_t > v_t:
                        broken.append((slope, intercept, v_t))
                if broken:
                    broken.sort(key=lambda x: x[2])
                    slope, intercept, _ = broken[0]
                    if self.confirmation_bars <= 1:
                        pending_retest['buy'] = {'slope': slope, 'intercept': intercept, 'breakout_bar': t}
                    else:
                        pending_breakout['buy'] = {'slope': slope, 'intercept': intercept, 'bars_confirmed': 1}
            if pending_breakout['sell'] is None and pending_retest['sell'] is None:
                broken = []
                for slope, intercept in sup_levels:
                    v_t = slope * t + intercept
                    v_prev = slope * (t - 1) + intercept
                    if close_prev >= v_prev and close_t < v_t:
                        broken.append((slope, intercept, v_t))
                if broken:
                    broken.sort(key=lambda x: -x[2])
                    slope, intercept, _ = broken[0]
                    if self.confirmation_bars <= 1:
                        pending_retest['sell'] = {'slope': slope, 'intercept': intercept, 'breakout_bar': t}
                    else:
                        pending_breakout['sell'] = {'slope': slope, 'intercept': intercept, 'bars_confirmed': 1}
        if signals:
            df_signals = pd.DataFrame(signals, columns=['time', 'signal', 'close'])
            df_signals.set_index('time', inplace=True)
            self.signals_df = df_signals
        else:
            self.signals_df = empty

class RetestPivotStrategy(_RetestBase):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, **kwargs):
        super().__init__('RetestPivot', ticker, timeframe, initial_account_size, entry_percentage, **kwargs)

    def _build_diagonal_lines(self, pivot_x, pivot_prices, tol, kind):
        return _find_diagonal_lines_pivot(pivot_x, pivot_prices, tol, self.min_touches, kind)

class RetestOLSPivotStrategy(_RetestBase):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, **kwargs):
        super().__init__('RetestOLSPivot', ticker, timeframe, initial_account_size, entry_percentage, **kwargs)

    def _build_diagonal_lines(self, pivot_x, pivot_prices, tol, kind):
        return _fit_diagonal_line_ols(pivot_x, pivot_prices, self.min_touches)

class OBVSignalStrategy(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, obv_ema_window=14, confirmation_bars=1, pyramiding=1, allowshort=True, debug=False):
        name = f'OBV_Signal_{obv_ema_window}'
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.obv_ema_window = int(obv_ema_window)
        self.confirmation_bars = max(1, int(confirmation_bars))
        self.debug = debug

    def generate_signals(self):
        self.load_data()
        close = self.data['close']
        volume = self.data['volume']
        obv = calculate_obv(close, volume)
        obv_ema = calculate_ema(obv, self.obv_ema_window)
        self.data['_obv'] = obv
        self.data['_obv_ema'] = obv_ema
        diff = obv - obv_ema
        raw_buy = (diff.shift(1) <= 0) & (diff > 0)
        raw_sell = (diff.shift(1) >= 0) & (diff < 0)
        if self.confirmation_bars > 1:
            buy = raw_buy.rolling(self.confirmation_bars, min_periods=1).max().astype(bool)
            sell = raw_sell.rolling(self.confirmation_bars, min_periods=1).max().astype(bool)
        else:
            buy = raw_buy
            sell = raw_sell
        self.data['signal'] = None
        self.data.loc[buy, 'signal'] = 'buy'
        if self.allowshort:
            self.data.loc[sell, 'signal'] = 'short'
        else:
            self.data.loc[sell, 'signal'] = 'sell'
        self.signals_df = self.data[['signal', 'close']].dropna()

class VWAPSignalStrategy(Strategy):

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, vwap_window=100, vwap_tol=0.005, vwap_mode='rolling', pyramiding=1, allowshort=True, debug=False):
        if vwap_mode not in ('rolling', 'yearly'):
            raise ValueError("vwap_mode must be 'rolling' or 'yearly'")
        name = f'VWAP_Signal_{vwap_mode}_{vwap_window}_{int(round(vwap_tol * 1000))}'
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.vwap_window = int(vwap_window)
        self.vwap_tol = float(vwap_tol)
        self.vwap_mode = vwap_mode
        self.debug = debug

    def _compute_vwap(self, close, volume):
        if self.vwap_mode == 'yearly':
            parts = []
            for year, group in close.groupby(close.index.year):
                v_group = volume.loc[group.index]
                pv = (group * v_group).cumsum()
                vv = v_group.cumsum()
                parts.append(pv / vv.replace(0, np.nan))
            return pd.concat(parts).sort_index()
        win = self.vwap_window
        pv = (close * volume).rolling(win, min_periods=max(1, win // 2)).sum()
        vv = volume.rolling(win, min_periods=max(1, win // 2)).sum()
        return pv / vv.replace(0, np.nan)

    def generate_signals(self):
        self.load_data()
        close = self.data['close']
        volume = self.data['volume']
        vwap = self._compute_vwap(close, volume)
        self.data['_vwap'] = vwap
        ratio = close / vwap
        buy = (ratio.shift(1) <= 1 + self.vwap_tol) & (ratio > 1 + self.vwap_tol)
        sell = (ratio.shift(1) >= 1 - self.vwap_tol) & (ratio < 1 - self.vwap_tol)
        self.data['signal'] = None
        self.data.loc[buy, 'signal'] = 'buy'
        if self.allowshort:
            self.data.loc[sell, 'signal'] = 'short'
        else:
            self.data.loc[sell, 'signal'] = 'sell'
        self.signals_df = self.data[['signal', 'close']].dropna()

def _candle_body(o, c):
    return abs(c - o)

def _candle_range(h, l):
    return h - l

def _upper_wick(o, h, c):
    return h - max(o, c)

def _lower_wick(o, l, c):
    return min(o, c) - l

def _is_bullish_engulfing(po, ph, pl, pc, co, ch, cl, cc):
    return pc < po and cc > co and (co <= pc) and (cc >= po)

def _is_bearish_engulfing(po, ph, pl, pc, co, ch, cl, cc):
    return pc > po and cc < co and (co >= pc) and (cc <= po)

def _is_hammer(o, h, l, c, body_ratio=2.0, body_min_frac=0.1):
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return False
    lower = min(o, c) - l
    upper = h - max(o, c)
    return lower >= body * body_ratio and upper < body and (body / rng >= body_min_frac)

def _is_shooting_star(o, h, l, c, body_ratio=2.0, body_min_frac=0.1):
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return False
    lower = min(o, c) - l
    upper = h - max(o, c)
    return upper >= body * body_ratio and lower < body and (body / rng >= body_min_frac)

def _is_doji(o, h, l, c, tol=0.1):
    body = abs(c - o)
    rng = h - l
    if rng <= 0:
        return False
    return body / rng < tol

def _is_bullish_harami(po, ph, pl, pc, co, ch, cl, cc):
    return pc < po and cc > co and (ch <= po) and (cl >= pc)

def _is_bearish_harami(po, ph, pl, pc, co, ch, cl, cc):
    return pc > po and cc < co and (ch <= pc) and (cl >= po)

class _CandleBaseStrategy(Strategy):
    PATTERN_SET = ()
    NAME_PREFIX = 'Candle'
    EMIT_MODE = 'raw'

    def __init__(self, ticker, timeframe, initial_account_size, entry_percentage, doji_tol=0.1, wick_body_ratio=2.0, ensemble_window=5, min_votes=2, pyramiding=1, allowshort=True, debug=False):
        if self.EMIT_MODE == 'ensemble':
            name = f'{self.NAME_PREFIX}_win{ensemble_window}_v{min_votes}'
        else:
            name = self.NAME_PREFIX
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.doji_tol = float(doji_tol)
        self.wick_body_ratio = float(wick_body_ratio)
        self.ensemble_window = max(1, int(ensemble_window))
        self.min_votes = max(1, int(min_votes))
        self.debug = debug

    def _detect_patterns(self):
        o = self.data['open'].values
        h = self.data['high'].values
        l = self.data['low'].values
        c = self.data['close'].values
        n = len(o)
        bull = np.zeros(n, dtype=bool)
        bear = np.zeros(n, dtype=bool)
        pats = self.PATTERN_SET
        wbr = self.wick_body_ratio
        dtol = self.doji_tol
        for i in range(1, n):
            po, ph, pl, pc = (o[i - 1], h[i - 1], l[i - 1], c[i - 1])
            co, ch, cl, cc = (o[i], h[i], l[i], c[i])
            if 'engulfing' in pats:
                if _is_bullish_engulfing(po, ph, pl, pc, co, ch, cl, cc):
                    bull[i] = True
                if _is_bearish_engulfing(po, ph, pl, pc, co, ch, cl, cc):
                    bear[i] = True
            if 'hammer' in pats:
                if _is_hammer(co, ch, cl, cc, body_ratio=wbr):
                    bull[i] = True
                if _is_shooting_star(co, ch, cl, cc, body_ratio=wbr):
                    bear[i] = True
            if 'harami' in pats:
                if _is_bullish_harami(po, ph, pl, pc, co, ch, cl, cc):
                    bull[i] = True
                if _is_bearish_harami(po, ph, pl, pc, co, ch, cl, cc):
                    bear[i] = True
            if 'doji' in pats and _is_doji(co, ch, cl, cc, tol=dtol):
                if pc < po:
                    bull[i] = True
                elif pc > po:
                    bear[i] = True
        idx = self.data.index
        return (pd.Series(bull, index=idx), pd.Series(bear, index=idx))

    @staticmethod
    def _rising_edge(series_bool):
        prev = series_bool.shift(1, fill_value=False)
        return series_bool & ~prev

    def generate_signals(self):
        self.load_data()
        bull, bear = self._detect_patterns()
        if self.EMIT_MODE == 'ensemble':
            bull_count = bull.rolling(self.ensemble_window, min_periods=1).sum()
            bear_count = bear.rolling(self.ensemble_window, min_periods=1).sum()
            buy = self._rising_edge(bull_count >= self.min_votes)
            sell = self._rising_edge(bear_count >= self.min_votes)
        else:
            buy = bull
            sell = bear
        self.data['signal'] = None
        self.data.loc[buy, 'signal'] = 'buy'
        if self.allowshort:
            self.data.loc[sell, 'signal'] = 'short'
        else:
            self.data.loc[sell, 'signal'] = 'sell'
        self.signals_df = self.data[['signal', 'close']].dropna()

class EngulfingStrategy(_CandleBaseStrategy):
    PATTERN_SET = ('engulfing',)
    NAME_PREFIX = 'Engulfing'
    EMIT_MODE = 'raw'

class HammerDojiStrategy(_CandleBaseStrategy):
    PATTERN_SET = ('hammer', 'doji')
    NAME_PREFIX = 'HammerDoji'
    EMIT_MODE = 'raw'

class HaramiStrategy(_CandleBaseStrategy):
    PATTERN_SET = ('harami',)
    NAME_PREFIX = 'Harami'
    EMIT_MODE = 'raw'

class CandleEnsembleStrategy(_CandleBaseStrategy):
    PATTERN_SET = ('engulfing', 'hammer', 'harami', 'doji')
    NAME_PREFIX = 'CandleEnsemble'
    EMIT_MODE = 'ensemble'
