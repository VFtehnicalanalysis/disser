import os
import pandas as pd
import numpy as np
import ta
from .base import Strategy

class MyStrategy(Strategy):

    def __init__(self, name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, partial_entry_pct=100, partial_close_pct=100, htf=None, rsi_period=14, rsi_low=30, rsi_high=70, is_rsi_changer=False, rsi_changer=0, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=False, macd_fast=12, macd_slow=26, macd_signal=9, is_using_trend_analysis=True, check_info='BUY & SELL', smart_stop=5.0, smart_stop_activation=10.0, is_smart_stop_activated=True, sell_count=10, is_using_stops=True, stop_l_percent=3.0, stop_s_percent=1.0, is_using_trailing_stop=True, TRS_percent=3.0, is_using_take_profits=True, TP_l_percent=6.0, TP_s_percent=3.0, is_price_step=True, price_step=0.1):
        super().__init__(name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort, partial_entry_pct, partial_close_pct)
        self.htf = htf if htf is not None else timeframe
        self.rsi_period = rsi_period
        self.rsi_low = rsi_low
        self.rsi_high = rsi_high
        self.is_rsi_changer = is_rsi_changer
        self.rsi_changer = rsi_changer
        self.L_short_period = L_short_period
        self.L_long_period = L_long_period
        self.H_short_period = H_short_period
        self.H_long_period = H_long_period
        self.pivot_period = pivot_period
        self.is_using_macd = is_using_macd
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.is_using_trend_analysis = is_using_trend_analysis
        self.check_info = check_info
        self.smart_stop = smart_stop
        self.smart_stop_activation = smart_stop_activation
        self.is_smart_stop_activated = is_smart_stop_activated
        self.sell_count = sell_count
        self.is_using_stops = is_using_stops
        self.stop_l_percent = stop_l_percent
        self.stop_s_percent = stop_s_percent
        self.is_using_trailing_stop = is_using_trailing_stop
        self.TRS_percent = TRS_percent
        self.is_using_take_profits = is_using_take_profits
        self.TP_l_percent = TP_l_percent
        self.TP_s_percent = TP_s_percent
        self.is_price_step = is_price_step
        self.price_step = price_step

    def round_quantity(self, quantity):
        if self.is_price_step:
            factor = 1 / self.price_step
            return int(np.floor(quantity * factor) / factor)
        return int(quantity)

    def load_data(self):
        if self.timeframe == '1h':
            file_path = f'data/{self.ticker}_hourly_data_new.csv'
        elif self.timeframe == '1D':
            file_path = f'data/{self.ticker}_data_new.csv'
        else:
            raise ValueError("Неверный таймфрейм. Допустимые значения: '1h' или '1D'.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Файл данных не найден: {file_path}')
        df = pd.read_csv(file_path)
        df['begin'] = pd.to_datetime(df['begin'])
        df.set_index('begin', inplace=True)
        df.sort_index(inplace=True)
        if getattr(self, 'use_date_filter', False) and getattr(self, 'backtest_start_date', None) is not None:
            df = df[df.index >= self.backtest_start_date]
        self.data = df
        print(f'Данные загружены из {file_path}. Количество баров: {len(self.data)}')

    def generate_signals(self):
        df = self.data.copy()
        pp = self.pivot_period
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
        df['EMA_LTF_short'] = ta.trend.EMAIndicator(df['close'], window=self.L_short_period).ema_indicator()
        df['EMA_LTF_long'] = ta.trend.EMAIndicator(df['close'], window=self.L_long_period).ema_indicator()
        if self.htf != self.timeframe:
            if self.timeframe == '1h' and self.htf == '1D':
                rule = 'D'
            elif self.timeframe == '1D' and self.htf == '1W':
                rule = 'W'
            else:
                rule = self.htf
            df_htf = df.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
            df_htf['EMA_HTF_short'] = ta.trend.EMAIndicator(df_htf['close'], window=self.H_short_period).ema_indicator()
            df_htf['EMA_HTF_long'] = ta.trend.EMAIndicator(df_htf['close'], window=self.H_long_period).ema_indicator()
            df_htf_shifted = df_htf[['close', 'EMA_HTF_short', 'EMA_HTF_long']].shift(1)
            df_htf_shifted = df_htf_shifted.rename(columns={'close': 'close_htf'})
            df = pd.merge_asof(df.sort_index(), df_htf_shifted.sort_index(), left_index=True, right_index=True, direction='backward')
        else:
            df['close_htf'] = df['close']
            df['EMA_HTF_short'] = ta.trend.EMAIndicator(df['close'], window=self.H_short_period).ema_indicator()
            df['EMA_HTF_long'] = ta.trend.EMAIndicator(df['close'], window=self.H_long_period).ema_indicator()
        if self.is_using_macd:
            macd_ind = ta.trend.MACD(df['close'], window_fast=self.macd_fast, window_slow=self.macd_slow, window_sign=self.macd_signal)
            df['MACD_line'] = macd_ind.macd()
        else:
            df['MACD_line'] = 0.0
        if self.is_rsi_changer:
            bearish_htf = df['EMA_HTF_short'] < df['EMA_HTF_long']
            df['true_rsi_low'] = np.where(bearish_htf, self.rsi_low - self.rsi_changer, self.rsi_low + self.rsi_changer)
            df['true_rsi_high'] = np.where(bearish_htf, self.rsi_high - self.rsi_changer, self.rsi_high + self.rsi_changer)
        else:
            df['true_rsi_low'] = float(self.rsi_low)
            df['true_rsi_high'] = float(self.rsi_high)
        w = 2 * pp + 1
        df['pivot_high_val'] = df['close'].rolling(window=w).apply(lambda x: x.iloc[pp] if x.values.argmax() == pp else np.nan, raw=False)
        df['pivot_low_val'] = df['close'].rolling(window=w).apply(lambda x: x.iloc[pp] if x.values.argmin() == pp else np.nan, raw=False)
        closes = df['close'].values
        rsi_arr = df['RSI'].values
        macd_arr = df['MACD_line'].values
        ph_arr = df['pivot_high_val'].values
        pl_arr = df['pivot_low_val'].values
        ema_ltf_s = df['EMA_LTF_short'].values
        ema_ltf_l = df['EMA_LTF_long'].values
        ema_htf_s = df['EMA_HTF_short'].values
        ema_htf_l = df['EMA_HTF_long'].values
        rsi_low_arr = df['true_rsi_low'].values
        rsi_high_arr = df['true_rsi_high'].values
        n = len(df)
        signals = [''] * n
        for T in range(n):
            if np.isnan(rsi_arr[T]) or np.isnan(ema_ltf_s[T]) or np.isnan(ema_htf_s[T]):
                continue
            cur_close_T = closes[T]
            cur_rsi_T = rsi_arr[T]
            true_rsi_h = rsi_high_arr[T]
            true_rsi_l = rsi_low_arr[T]
            long_trend = False
            if ema_htf_s[T] < ema_htf_l[T] and ema_ltf_s[T] < ema_ltf_l[T]:
                long_trend = True
            if ema_htf_s[T] > ema_htf_l[T] and ema_ltf_s[T] < ema_ltf_l[T] and (ema_ltf_s[T] > ema_htf_s[T]):
                long_trend = True
            short_trend = False
            if ema_htf_s[T] > ema_htf_l[T]:
                if ema_ltf_s[T] < ema_htf_s[T] and cur_close_T < ema_htf_s[T] or ema_ltf_s[T] > ema_ltf_l[T]:
                    short_trend = True
            signal = ''
            if not np.isnan(ph_arr[T]) and self.check_info != 'BUY':
                cur_pivot_val = ph_arr[T]
                cur_pivot_idx = T - pp
                length_up = 0
                for x in range(1, 41):
                    bar_x_back = T - x
                    if bar_x_back < 0:
                        break
                    if np.isnan(ph_arr[bar_x_back]):
                        continue
                    prev_pivot_val = ph_arr[bar_x_back]
                    prev_pivot_idx = bar_x_back - pp
                    rsi_at_cur_pivot = rsi_arr[cur_pivot_idx]
                    rsi_at_prev_pivot = rsi_arr[prev_pivot_idx]
                    if prev_pivot_val < cur_pivot_val and rsi_at_prev_pivot > rsi_at_cur_pivot and (rsi_at_prev_pivot >= true_rsi_h):
                        length_up = x
                        if x == 0:
                            continue
                        tg_rsi = (rsi_arr[T - x] - rsi_arr[T]) / x
                        tg_price = (closes[prev_pivot_idx] - closes[cur_pivot_idx]) / x
                        tg_macd = (macd_arr[T - x] - macd_arr[T]) / x
                        valid = True
                        for y in range(1, length_up):
                            inter_idx = T - y - pp
                            if inter_idx < 0:
                                break
                            interp_price = closes[cur_pivot_idx] + tg_price * y
                            interp_rsi = rsi_arr[cur_pivot_idx] + tg_rsi * y
                            interp_macd = macd_arr[cur_pivot_idx] + tg_macd * y
                            broken = False
                            if closes[inter_idx] > interp_price:
                                broken = True
                            if not broken and rsi_arr[inter_idx] > interp_rsi:
                                broken = True
                            if not broken and self.is_using_macd and (macd_arr[inter_idx] > interp_macd):
                                broken = True
                            if broken:
                                valid = False
                                length_up = 0
                                break
                        if valid and length_up != 0:
                            break
                if length_up != 0:
                    if not self.is_using_trend_analysis or short_trend:
                        if self.is_using_trend_analysis and ema_ltf_s[T] < ema_htf_s[T] and (ema_ltf_s[T] < ema_ltf_l[T]):
                            signal = 'short'
                        else:
                            signal = 'sell'
            if signal == '' and (not np.isnan(pl_arr[T])) and (self.check_info != 'SELL'):
                cur_pivot_val = pl_arr[T]
                cur_pivot_idx = T - pp
                length_down = 0
                for x in range(1, 41):
                    bar_x_back = T - x
                    if bar_x_back < 0:
                        break
                    if np.isnan(pl_arr[bar_x_back]):
                        continue
                    prev_pivot_val = pl_arr[bar_x_back]
                    prev_pivot_idx = bar_x_back - pp
                    rsi_at_cur_pivot = rsi_arr[cur_pivot_idx]
                    rsi_at_prev_pivot = rsi_arr[prev_pivot_idx]
                    if prev_pivot_val > cur_pivot_val and rsi_at_prev_pivot < rsi_at_cur_pivot and (rsi_at_prev_pivot <= true_rsi_l):
                        length_down = x
                        tg_rsi = (rsi_arr[T] - rsi_arr[T - x]) / x
                        tg_price = (closes[cur_pivot_idx] - closes[prev_pivot_idx]) / x
                        tg_macd = (macd_arr[T] - macd_arr[T - x]) / x
                        valid = True
                        for y in range(1, length_down):
                            inter_idx = T - y - pp
                            if inter_idx < 0:
                                break
                            interp_price = closes[cur_pivot_idx] - tg_price * y
                            interp_rsi = rsi_arr[cur_pivot_idx] - tg_rsi * y
                            interp_macd = macd_arr[cur_pivot_idx] - tg_macd * y
                            broken = False
                            if closes[inter_idx] < interp_price:
                                broken = True
                            if not broken and rsi_arr[inter_idx] < interp_rsi:
                                broken = True
                            if not broken and self.is_using_macd and (macd_arr[inter_idx] < interp_macd):
                                broken = True
                            if broken:
                                valid = False
                                length_down = 0
                                break
                        if valid and length_down != 0:
                            break
                if length_down != 0:
                    if not self.is_using_trend_analysis or long_trend:
                        signal = 'long'
            signals[T] = signal
        df['signal'] = signals
        self.signals_df = df[['signal', 'close']]
        print(f'Сигналы сгенерированы (PineScript-точный порт без lookahead): {sum((1 for s in signals if s))} сигналов из {n} баров.')

    def round_quantity(self, quantity):
        if self.is_price_step:
            factor = 1 / self.price_step
            return int(np.floor(quantity * factor) / factor)
        return int(quantity)

    def enter_long(self, entry_time, price, comment=''):
        qty = self.current_account_size * (self.entry_percentage / 100) / price
        trade = super().enter_long(entry_time, price, comment)
        trade.quantity = self.round_quantity(qty)
        trade.remaining_quantity = trade.quantity
        trade.stop_activation_price = price * (1 + self.smart_stop_activation / 100)
        trade.stop_price = price * (1 + self.smart_stop / 100)
        trade.stop_active = False
        return trade

    def enter_short(self, entry_time, price, comment=''):
        qty = self.current_account_size * (self.entry_percentage / 100) / price
        trade = super().enter_short(entry_time, price, comment)
        trade.quantity = self.round_quantity(qty)
        trade.remaining_quantity = trade.quantity
        trade.stop_activation_price = price * (1 - self.smart_stop_activation / 100)
        trade.stop_price = price * (1 - self.smart_stop / 100)
        trade.stop_active = False
        return trade

    def close_long_trade(self, trade, exit_time, price, comment='', close_qty=None):
        if close_qty is None or close_qty >= trade.remaining_quantity:
            profit = trade.close_trade(exit_time, price, comment if comment else 'close long')
            trade.remaining_quantity = 0
        else:
            profit = (price - trade.entry_price) * close_qty
            trade.remaining_quantity -= close_qty
            trade.comment = f'partial close ({close_qty} contracts)'
        self.current_account_size += profit
        self.current_profit += profit
        return profit

    def close_short_trade(self, trade, exit_time, price, comment='', close_qty=None):
        if close_qty is None or close_qty >= trade.remaining_quantity:
            profit = trade.close_trade(exit_time, price, comment if comment else 'close short')
            trade.remaining_quantity = 0
        else:
            profit = (trade.entry_price - price) * close_qty
            trade.remaining_quantity -= close_qty
            trade.comment = f'partial close ({close_qty} contracts)'
        self.current_account_size += profit
        self.current_profit += profit
        return profit

    def execute_trades(self):
        open_long = []
        open_short = []
        equity_series = []
        for current_time, candle in self.data.iterrows():
            current_price = candle['close']
            self.update_open_trades(current_time, current_price)
            open_long = [t for t in open_long if not t.is_closed]
            open_short = [t for t in open_short if not t.is_closed]
            if current_time in self.signals_df.index:
                raw_signal = self.signals_df.loc[current_time, 'signal'].lower().strip()
                if raw_signal:
                    signals = [s.strip() for s in raw_signal.split(';') if s.strip() != '']
                    for sig in signals:
                        if sig == 'long':
                            if self.allowshort and open_short:
                                for trade in open_short.copy():
                                    self.close_short_trade(trade, current_time, current_price, comment='close short (signal long)')
                                open_short.clear()
                            if sum((1 for t in open_long if not t.is_closed)) < self.pyramiding:
                                trade = self.enter_long(current_time, current_price, comment='open long (signal)')
                                open_long.append(trade)
                        elif sig == 'short':
                            if open_long:
                                for trade in open_long.copy():
                                    self.close_long_trade(trade, current_time, current_price, comment='close long (signal short)')
                                open_long.clear()
                            if sum((1 for t in open_short if not t.is_closed)) < self.pyramiding:
                                trade = self.enter_short(current_time, current_price, comment='open short (signal)')
                                open_short.append(trade)
                        elif sig == 'sell':
                            for trade in open_long.copy():
                                close_qty = int(trade.remaining_quantity * (self.sell_count / 100))
                                if close_qty < 1 and trade.remaining_quantity > 0:
                                    close_qty = trade.remaining_quantity
                                self.close_long_trade(trade, current_time, current_price, comment='partial close long (divergence sell)', close_qty=close_qty)
                                if trade.remaining_quantity == 0:
                                    open_long.remove(trade)
                        elif sig == 'sell_short':
                            for trade in open_short.copy():
                                close_qty = int(trade.remaining_quantity * (self.partial_close_pct / 100))
                                self.close_short_trade(trade, current_time, current_price, comment='partial close short', close_qty=close_qty)
                                if trade.remaining_quantity == 0:
                                    open_short.remove(trade)
            unrealized = 0
            for t in open_long + open_short:
                if not t.is_closed:
                    if t.direction == 'long':
                        unrealized += (current_price - t.entry_price) * t.remaining_quantity
                    elif t.direction == 'short':
                        unrealized += (t.entry_price - current_price) * t.remaining_quantity
            equity_series.append({'time': current_time, 'equity': self.current_account_size + unrealized})
        last_time = self.data.index[-1]
        last_price = self.data['close'].iloc[-1]
        for trade in open_long + open_short:
            if not trade.is_closed:
                self.close_long_trade(trade, last_time, last_price, comment='close at end') if trade.direction == 'long' else self.close_short_trade(trade, last_time, last_price, comment='close at end')
        self._equity_from_execution = pd.DataFrame(equity_series).set_index('time')
        if not self._equity_from_execution.empty:
            self._equity_from_execution.iloc[0] = self.initial_account_size

    def update_open_trades(self, current_time, current_price):
        for trade in self.trades:
            if trade.is_closed:
                continue
            if trade.direction == 'long':
                trade.unrealized_return = (current_price - trade.entry_price) * trade.remaining_quantity
            elif trade.direction == 'short':
                trade.unrealized_return = (trade.entry_price - current_price) * trade.remaining_quantity
            if self.is_smart_stop_activated:
                if not trade.stop_active:
                    if trade.direction == 'long' and current_price > trade.stop_activation_price or (trade.direction == 'short' and current_price < trade.stop_activation_price):
                        trade.stop_active = True
                if trade.stop_active:
                    if trade.direction == 'long' and current_price < trade.stop_price or (trade.direction == 'short' and current_price > trade.stop_price):
                        self.close_long_trade(trade, current_time, current_price, comment='smart stop') if trade.direction == 'long' else self.close_short_trade(trade, current_time, current_price, comment='smart stop')
            if self.is_using_stops:
                stop_level = trade.entry_price * (1 - self.stop_l_percent / 100) if trade.direction == 'long' else trade.entry_price * (1 + self.stop_s_percent / 100)
                if trade.direction == 'long' and current_price < stop_level or (trade.direction == 'short' and current_price > stop_level):
                    self.close_long_trade(trade, current_time, current_price, comment='standard stop') if trade.direction == 'long' else self.close_short_trade(trade, current_time, current_price, comment='standard stop')
            if self.is_using_take_profits:
                tp_level = trade.entry_price * (1 + self.TP_l_percent / 100) if trade.direction == 'long' else trade.entry_price * (1 - self.TP_s_percent / 100)
                if trade.direction == 'long' and current_price > tp_level or (trade.direction == 'short' and current_price < tp_level):
                    self.close_long_trade(trade, current_time, current_price, comment='take profit') if trade.direction == 'long' else self.close_short_trade(trade, current_time, current_price, comment='take profit')
            if self.is_using_trailing_stop:
                trailing_stop = current_price - current_price * self.TRS_percent / 100 if trade.direction == 'long' else current_price + current_price * self.TRS_percent / 100
                if trade.direction == 'long' and trailing_stop > trade.stop_price or (trade.direction == 'short' and trailing_stop < trade.stop_price):
                    trade.stop_price = trailing_stop
                if trade.direction == 'long' and current_price < trade.stop_price or (trade.direction == 'short' and current_price > trade.stop_price):
                    self.close_long_trade(trade, current_time, current_price, comment='trailing stop') if trade.direction == 'long' else self.close_short_trade(trade, current_time, current_price, comment='trailing stop')

    def get_equity_curve(self):
        if hasattr(self, '_equity_from_execution') and self._equity_from_execution is not None:
            return self._equity_from_execution
        return super().get_equity_curve()

    def finalize(self):
        self.equity_df = self.get_equity_curve()
        print('Equity curve finalized. Total bars:', len(self.equity_df))

    def debug_print_dfs(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        print('\n==== DataFrame сигналов (signals_df) ====')
        if self.signals_df is not None and (not self.signals_df.empty):
            print(self.signals_df)
        else:
            print('signals_df пуст или не сгенерирован.')
        print('\n==== DataFrame сделок (get_trades_df()) ====')
        trades_df = self.get_trades_df()
        if not trades_df.empty:
            print(trades_df)
        else:
            print('Сделки отсутствуют или нет закрытых сделок.')
if __name__ == '__main__':
    strat = MyStrategy(name='VF Strategy', ticker='SBER', timeframe='1D', initial_account_size=100000, entry_percentage=10, pyramiding=1, allowshort=True, partial_entry_pct=100, partial_close_pct=50, htf='1W', rsi_period=14, rsi_low=30, rsi_high=70, is_rsi_changer=True, rsi_changer=5, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=True, macd_fast=12, macd_slow=26, macd_signal=9, smart_stop=5.0, smart_stop_activation=10.0, is_smart_stop_activated=True, sell_count=10, is_using_stops=True, stop_l_percent=3.0, stop_s_percent=1.0, is_using_trailing_stop=True, TRS_percent=3.0, is_using_take_profits=True, TP_l_percent=6.0, TP_s_percent=3.0, is_price_step=True, price_step=0.1)
    strat.load_data()
    strat.generate_signals()
    strat.execute_trades()
    strat.debug_print_dfs()
