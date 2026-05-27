import pandas as pd
import os
import numpy as np

class Trade:

    def __init__(self, entry_time, direction, entry_price, account_size, entry_percentage, comment=''):
        self.entry_time = entry_time
        self.direction = direction
        self.entry_price = entry_price
        self.quantity = int(account_size * (entry_percentage / 100) / entry_price)
        self.remaining_quantity = self.quantity
        self.exit_time = None
        self.exit_price = None
        self.is_closed = False
        self.result = 0
        self.profit_accounted = 0
        self.comment = comment
        self.trade_id = None
        self.trade_type = None
        self.realized_return = 0
        self.unrealized_return = 0

    def close_trade(self, exit_time, exit_price, comment='', close_qty=None):
        if close_qty is None or close_qty >= self.remaining_quantity:
            close_qty = self.remaining_quantity
            self.remaining_quantity = 0
            self.is_closed = True
        else:
            self.remaining_quantity -= close_qty
        if self.direction == 'long':
            partial_result = (exit_price - self.entry_price) * close_qty
        elif self.direction == 'short':
            partial_result = (self.entry_price - exit_price) * close_qty
        else:
            partial_result = 0
        self.result += partial_result
        self.exit_time = exit_time
        self.exit_price = exit_price
        if comment:
            self.comment = comment
        incremental_profit = self.result - self.profit_accounted
        self.profit_accounted = self.result
        self.realized_return = self.result
        return incremental_profit

class Strategy:

    def __init__(self, name, ticker, timeframe, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, partial_entry_pct=100, partial_close_pct=100):
        self.name = name
        self.ticker = ticker
        self.timeframe = timeframe
        self.initial_account_size = initial_account_size
        self.current_account_size = initial_account_size
        self.unrealized_profit = 0
        self.entry_percentage = entry_percentage
        self.pyramiding = pyramiding
        self.allowshort = allowshort
        self.partial_entry_pct = partial_entry_pct
        self.partial_close_pct = partial_close_pct
        self.signals_df = None
        self.trades = []
        self.current_profit = 0
        self.trade_counter = 0

    @property
    def all_trades(self):
        return self.trades

    def load_data(self):
        if self.timeframe == '1h':
            file_path = f'data/{self.ticker}_hourly_data_new.csv'
        elif self.timeframe == '1D':
            file_path = f'data/{self.ticker}_data_new.csv'
        else:
            raise ValueError("Неверный таймфрейм. Допустимые значения: '1h' или '1D'.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Файл данных не найден: {file_path}')
        self.data = pd.read_csv(file_path)
        self.data['begin'] = pd.to_datetime(self.data['begin'])
        self.data.set_index('begin', inplace=True)
        self.data.sort_index(inplace=True)

    def generate_signals(self):
        raise NotImplementedError('Метод generate_signals должен быть реализован в дочернем классе.')

    @staticmethod
    def _normalize_signals(raw_signal):
        if raw_signal is None:
            return []
        try:
            if pd.isna(raw_signal):
                return []
        except (TypeError, ValueError):
            pass
        s = str(raw_signal).strip()
        if not s or s.lower() == 'nan':
            return []
        tokens = [t.strip() for t in s.lower().split(';') if t.strip()]
        normalized = []
        valid = {'long', 'short', 'close long', 'close short'}
        for t in tokens:
            if t == 'buy':
                normalized.append('long')
            elif t == 'sell':
                normalized.append('close long')
            elif t in valid:
                normalized.append(t)
        return normalized

    def enter_long(self, entry_time, price, comment=''):
        full_qty = int(self.current_account_size * (self.entry_percentage / 100) / price)
        qty = int(full_qty * (self.partial_entry_pct / 100))
        trade = Trade(entry_time, 'long', price, self.current_account_size, self.entry_percentage, comment)
        trade.quantity = qty
        trade.remaining_quantity = qty
        trade.trade_type = 'open_long'
        self.trade_counter += 1
        trade.trade_id = self.trade_counter
        self.trades.append(trade)
        return trade

    def enter_short(self, entry_time, price, comment=''):
        full_qty = int(self.current_account_size * (self.entry_percentage / 100) / price)
        qty = int(full_qty * (self.partial_entry_pct / 100))
        trade = Trade(entry_time, 'short', price, self.current_account_size, self.entry_percentage, comment)
        trade.quantity = qty
        trade.remaining_quantity = qty
        trade.trade_type = 'open_short'
        self.trade_counter += 1
        trade.trade_id = self.trade_counter
        self.trades.append(trade)
        return trade

    def close_long_fifo(self, target_qty, exit_time, price, comment='close long'):
        incremental_total = 0
        for trade in sorted([t for t in self.trades if not t.is_closed and t.direction == 'long'], key=lambda t: t.entry_time):
            if target_qty <= 0:
                break
            available = trade.remaining_quantity
            if available <= 0:
                continue
            to_close = min(available, target_qty)
            profit = trade.close_trade(exit_time, price, comment, close_qty=to_close)
            incremental_total += profit
            target_qty -= to_close
        return incremental_total

    def close_short_fifo(self, target_qty, exit_time, price, comment='close short'):
        incremental_total = 0
        for trade in sorted([t for t in self.trades if not t.is_closed and t.direction == 'short'], key=lambda t: t.entry_time):
            if target_qty <= 0:
                break
            available = trade.remaining_quantity
            if available <= 0:
                continue
            to_close = min(available, target_qty)
            profit = trade.close_trade(exit_time, price, comment, close_qty=to_close)
            incremental_total += profit
            target_qty -= to_close
        return incremental_total

    def execute_trades(self):
        open_long = []
        open_short = []
        for current_time, row in self.signals_df.iterrows():
            signals = self._normalize_signals(row['signal'])
            price = row['close']
            for sig in signals:
                if sig == 'long':
                    if self.allowshort and open_short:
                        total_short = sum((t.remaining_quantity for t in open_short if not t.is_closed))
                        if total_short > 0:
                            profit = self.close_short_fifo(total_short, current_time, price, comment='close short (flip)')
                            self.current_account_size += profit
                            self.current_profit += profit
                        open_short = []
                    if sum((1 for t in open_long if not t.is_closed)) < self.pyramiding:
                        trade = self.enter_long(current_time, price, comment='open long (signal)')
                        open_long.append(trade)
                elif sig == 'close long':
                    total_long = sum((t.remaining_quantity for t in open_long if not t.is_closed))
                    target_close = int(total_long * (self.partial_close_pct / 100))
                    if target_close == 0 and total_long > 0:
                        target_close = total_long
                    if total_long > 0:
                        profit = self.close_long_fifo(target_close, current_time, price, comment='close long')
                        self.current_account_size += profit
                        self.current_profit += profit
                    open_long = [t for t in open_long if not t.is_closed]
                elif sig == 'short':
                    if open_long:
                        total_long = sum((t.remaining_quantity for t in open_long if not t.is_closed))
                        if total_long > 0:
                            profit = self.close_long_fifo(total_long, current_time, price, comment='close long (flip)')
                            self.current_account_size += profit
                            self.current_profit += profit
                        open_long = []
                    if sum((1 for t in open_short if not t.is_closed)) < self.pyramiding:
                        trade = self.enter_short(current_time, price, comment='open short (signal)')
                        open_short.append(trade)
                elif sig == 'close short':
                    total_short = sum((t.remaining_quantity for t in open_short if not t.is_closed))
                    target_close = int(total_short * (self.partial_close_pct / 100))
                    if target_close == 0 and total_short > 0:
                        target_close = total_short
                    if total_short > 0:
                        profit = self.close_short_fifo(target_close, current_time, price, comment='close short')
                        self.current_account_size += profit
                        self.current_profit += profit
                    open_short = [t for t in open_short if not t.is_closed]
        last_time = self.data.index[-1]
        last_price = self.data['close'].iloc[-1]
        for trade in open_long:
            if not trade.is_closed:
                profit = trade.close_trade(last_time, last_price, 'close long at end')
                self.current_account_size += profit
                self.current_profit += profit
        for trade in open_short:
            if not trade.is_closed:
                profit = trade.close_trade(last_time, last_price, 'close short at end')
                self.current_account_size += profit
                self.current_profit += profit

    def print_results(self):
        print(f'\nРезультаты стратегии: {self.name} ({self.ticker}, {self.timeframe})')
        print('=' * 50)
        print(f'Начальный размер счета: {self.initial_account_size:.2f}')
        print(f'Текущий размер счета: {self.current_account_size:.2f}')
        print(f'Общая прибыль: {self.current_profit:.2f}')
        print('\nСделки:')
        print(f"{'№ сделки':<10} {'Время входа':<20} {'Тип':<12} {'Количество':<10} {'Прибыль/Убыток':<15} {'Комментарий':<25}")
        print('-' * 80)

    def save_results(self):
        if self.signals_df is None:
            raise ValueError('Сигналы не сгенерированы. Сначала вызовите generate_signals.')
        filename = f'{self.ticker}_{self.name}_{self.timeframe}.csv'
        if os.path.exists(filename):
            os.remove(filename)
        self.signals_df.to_csv(filename)
        print(f'Результаты сохранены в файл: {filename}')

    def get_trades_df(self):
        records = []
        for trade in self.trades:
            if trade.is_closed:
                fmt = '%Y-%m-%d' if self.timeframe == '1D' else '%Y-%m-%d %H:%M'
                holding = (trade.exit_time - trade.entry_time).days if self.timeframe == '1D' else (trade.exit_time - trade.entry_time).total_seconds() / 3600.0
                records.append({'№ сделки': trade.trade_id, 'Дата': trade.entry_time.strftime(fmt), 'Цена': trade.entry_price, 'Сигнал': trade.trade_type if trade.trade_type is not None else 'open ' + ('long' if trade.direction == 'long' else 'short'), 'Количество': trade.quantity, 'Прибыль/Убыток': trade.result, 'Время удержания': round(holding, 2), 'Комментарий': trade.comment, 'raw_date': trade.entry_time})
        df = pd.DataFrame(records)
        if not df.empty:
            df['Дата'] = pd.to_datetime(df['Дата'])
            df = df.sort_values('Дата')
        return df

    def get_stats(self):
        total_signals = len(self.signals_df) if self.signals_df is not None else 0
        closed_trades = [t for t in self.trades if t.is_closed]
        total_trades = len(closed_trades)
        profitable = [t for t in closed_trades if t.result > 0]
        losing = [t for t in closed_trades if t.result < 0]
        pct_profitable = len(profitable) / total_trades * 100 if total_trades > 0 else 0
        avg_profit = round(sum((t.result for t in profitable)) / len(profitable), 2) if profitable else 0
        avg_loss = round(sum((t.result for t in losing)) / len(losing), 2) if losing else 0
        if self.timeframe == '1D':
            holding_times = [(t.exit_time - t.entry_time).days for t in closed_trades]
            avg_holding_days = round(sum(holding_times) / len(holding_times), 2) if holding_times else 0
            avg_holding_hours = None
        else:
            holding_times = [(t.exit_time - t.entry_time).total_seconds() / 3600.0 for t in closed_trades]
            avg_holding_days = round(sum((ht / 24.0 for ht in holding_times)) / len(holding_times), 2) if holding_times else 0
            avg_holding_hours = round(sum(holding_times) / len(holding_times), 2) if holding_times else 0
        if not hasattr(self, 'equity_df') or self.equity_df is None or self.equity_df.empty:
            equity_series = pd.Series([self.initial_account_size])
        else:
            equity_series = self.equity_df['equity']
        running_max = equity_series.cummax()
        drawdowns = (running_max - equity_series) / running_max * 100
        max_drawdown = drawdowns.max()
        returns = equity_series.pct_change().dropna()
        if self.timeframe == '1D':
            scale = np.sqrt(252)
        elif self.timeframe == '1h':
            scale = np.sqrt(252 * 6.5)
        else:
            scale = 1.0
        sharpe = np.nan
        if returns.std() != 0:
            sharpe = returns.mean() / returns.std() * scale
        downside = returns[returns < 0]
        sortino = np.nan
        if downside.std() != 0:
            sortino = returns.mean() / downside.std() * scale
        std_return = returns.std()
        reward_risk = np.nan
        annual_return = returns.mean() * scale
        if max_drawdown > 0:
            reward_risk = annual_return / (max_drawdown / 100)
        pct_growth = round(self.current_profit / self.initial_account_size * 100, 2)
        stats = {'Стратегия': self.name, 'Таймфрейм': self.timeframe, 'Количество сигналов': total_signals, 'Количество сделок': total_trades, 'Прибыльных сделок': len(profitable), 'Убыточных сделок': len(losing), 'Процент прибыльных сделок': round(pct_profitable, 2), 'Средняя прибыль': avg_profit, 'Средний убыток': avg_loss, 'Среднее время удержания (дней)': avg_holding_days, 'Среднее время удержания (часов)': avg_holding_hours, 'Макс. просадка (%)': round(max_drawdown, 2), 'Макс. прибыль по сделке': max([t.result for t in closed_trades], default=0), 'Макс. убыток по сделке': min([t.result for t in closed_trades], default=0), 'Процент роста': pct_growth, 'Sharpe': round(sharpe, 2) if not np.isnan(sharpe) else None, 'Sortino': round(sortino, 2) if not np.isnan(sortino) else None, 'Std. return': round(std_return, 4) if not np.isnan(std_return) else None, 'Reward/Risk': round(reward_risk, 2) if not np.isnan(reward_risk) else None}
        return stats

    def get_equity_curve(self):
        equity_series = []
        local_trades = []
        cash = self.initial_account_size
        for current_time, candle in self.data.iterrows():
            price = candle['close']
            if current_time in self.signals_df.index:
                raw_signal_value = self.signals_df.loc[current_time, 'signal']
                if hasattr(raw_signal_value, 'iloc'):
                    raw_signal_value = raw_signal_value.iloc[0]
                signals = self._normalize_signals(raw_signal_value)
                for sig in signals:
                    if sig == 'long':
                        if self.allowshort:
                            for t in local_trades.copy():
                                if t.direction == 'short' and (not t.is_closed):
                                    profit = t.close_trade(current_time, price, 'close short')
                                    cash += profit
                                    local_trades.remove(t)
                        if sum((1 for t in local_trades if t.direction == 'long' and (not t.is_closed))) < self.pyramiding:
                            new_trade = Trade(current_time, 'long', price, cash, self.entry_percentage, comment='open long')
                            local_trades.append(new_trade)
                    elif sig == 'close long':
                        for t in local_trades.copy():
                            if t.direction == 'long' and (not t.is_closed):
                                profit = t.close_trade(current_time, price, 'close long')
                                cash += profit
                                local_trades.remove(t)
                    elif sig == 'short':
                        for t in local_trades.copy():
                            if t.direction == 'long' and (not t.is_closed):
                                profit = t.close_trade(current_time, price, 'close long')
                                cash += profit
                        local_trades = [t for t in local_trades if not (t.direction == 'long' and t.is_closed)]
                        if sum((1 for t in local_trades if t.direction == 'short' and (not t.is_closed))) < self.pyramiding:
                            new_trade = Trade(current_time, 'short', price, cash, self.entry_percentage, comment='open short')
                            local_trades.append(new_trade)
                    elif sig == 'close short':
                        for t in local_trades.copy():
                            if t.direction == 'short' and (not t.is_closed):
                                profit = t.close_trade(current_time, price, 'close short')
                                cash += profit
                                local_trades.remove(t)
            unrealized_profit = 0
            for t in local_trades:
                if not t.is_closed:
                    if t.direction == 'long':
                        unrealized_profit += (price - t.entry_price) * t.quantity
                    elif t.direction == 'short':
                        unrealized_profit += (t.entry_price - price) * t.quantity
            total_equity = cash + unrealized_profit
            equity_series.append({'time': current_time, 'equity': total_equity})
        equity_df = pd.DataFrame(equity_series).set_index('time')
        if not equity_df.empty:
            equity_df.iloc[0] = self.initial_account_size
        return equity_df
