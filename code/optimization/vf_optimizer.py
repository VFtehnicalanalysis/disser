import os
import sys
import time
import json
import argparse
import warnings
from copy import deepcopy
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
from strategies.vf import MyStrategy
STATIC_PARAMS = dict(initial_account_size=1000000, entry_percentage=50.0, is_using_trend_analysis=True, check_info='BUY & SELL', is_rsi_changer=False, rsi_changer=0, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, is_price_step=False)

def get_search_space(trial: optuna.Trial) -> dict:
    rsi_period = trial.suggest_int('rsi_period', 7, 30)
    rsi_low = trial.suggest_int('rsi_low', 15, 40)
    rsi_high = trial.suggest_int('rsi_high', 60, 85)
    pivot_period = trial.suggest_int('pivot_period', 2, 10)
    L_short = trial.suggest_int('L_short_period', 20, 150)
    L_long = trial.suggest_int('L_long_period', L_short + 20, 300)
    H_short = trial.suggest_int('H_short_period', 20, 150)
    H_long = trial.suggest_int('H_long_period', H_short + 20, 300)
    is_using_macd = trial.suggest_categorical('is_using_macd', [True, False])
    is_smart_stop_activated = trial.suggest_categorical('is_smart_stop_activated', [True, False])
    smart_stop = trial.suggest_float('smart_stop', 1.0, 10.0, step=0.5)
    smart_stop_activation = trial.suggest_float('smart_stop_activation', 5.0, 25.0, step=1.0)
    sell_count = trial.suggest_int('sell_count', 5, 30)
    pyramiding = trial.suggest_int('pyramiding', 1, 3)
    allowshort = trial.suggest_categorical('allowshort', [True, False])
    return dict(rsi_period=rsi_period, rsi_low=rsi_low, rsi_high=rsi_high, pivot_period=pivot_period, L_short_period=L_short, L_long_period=L_long, H_short_period=H_short, H_long_period=H_long, is_using_macd=is_using_macd, macd_fast=12, macd_slow=26, macd_signal=9, is_smart_stop_activated=is_smart_stop_activated, smart_stop=smart_stop, smart_stop_activation=smart_stop_activation, sell_count=sell_count, pyramiding=pyramiding, allowshort=allowshort)

def composite_score(stats: dict) -> float:
    n_trades = stats.get('Количество сделок', 0)
    if n_trades < 1:
        return -1000000000.0
    sharpe = stats.get('Sharpe', None)
    profit_pct = stats.get('Процент роста', 0)
    max_dd = stats.get('Макс. просадка (%)', 0)
    if sharpe is None or pd.isna(sharpe):
        return -1000000000.0
    if max_dd <= 0.01:
        max_dd = 0.01
    score = sharpe * (profit_pct / max_dd)
    if pd.isna(score) or np.isinf(score):
        return -1000000000.0
    return float(score)
import io
import contextlib

@contextlib.contextmanager
def _suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout

def run_strategy_on_slice(ticker: str, timeframe: str, params: dict, data_slice: pd.DataFrame, htf_value: str, quiet: bool=True):
    full_params = dict(STATIC_PARAMS)
    full_params.update(params)
    try:
        strat = MyStrategy(name='opt_trial', ticker=ticker, timeframe=timeframe, htf=htf_value, **full_params)
    except Exception:
        return (None, None)
    strat.data = data_slice
    try:
        if quiet:
            with _suppress_stdout():
                strat.generate_signals()
                strat.execute_trades()
                strat.finalize()
        else:
            strat.generate_signals()
            strat.execute_trades()
            strat.finalize()
        stats = strat.get_stats()
    except Exception:
        return (None, None)
    return (stats, strat)

class WalkForwardOptimizer:

    def __init__(self, ticker: str, timeframe: str, htf_value: str, folds: list, n_trials: int=200, output_dir: str='results/vf_strategy'):
        self.ticker = ticker
        self.timeframe = timeframe
        self.htf_value = htf_value
        self.folds = folds
        self.n_trials = n_trials
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        if timeframe == '1D':
            data_file = f'data/{ticker}_data_new.csv'
        else:
            data_file = f'data/{ticker}_hourly_data_new.csv'
        df = pd.read_csv(data_file)
        df['begin'] = pd.to_datetime(df['begin'])
        df.set_index('begin', inplace=True)
        df.sort_index(inplace=True)
        self.full_data = df
        print(f'[{ticker} {timeframe}] Загружено {len(df)} баров: {df.index[0].date()} .. {df.index[-1].date()}')

    def slice_data(self, start: str, end: str) -> pd.DataFrame:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        return self.full_data.loc[(self.full_data.index >= s) & (self.full_data.index < e)].copy()

    def make_objective(self, train_data: pd.DataFrame):

        def objective(trial: optuna.Trial) -> float:
            params = get_search_space(trial)
            stats, _ = run_strategy_on_slice(self.ticker, self.timeframe, params, train_data, self.htf_value)
            if stats is None:
                return -1000000000.0
            return composite_score(stats)
        return objective

    def run_fold(self, fold_id: int, train_start: str, train_end: str, val_start: str, val_end: str) -> dict:
        print(f"\n{'─' * 70}")
        print(f'FOLD {fold_id}: train [{train_start} .. {train_end}], val [{val_start} .. {val_end}]')
        print(f"{'─' * 70}")
        train_data = self.slice_data(train_start, train_end)
        val_data = self.slice_data(val_start, val_end)
        if len(train_data) < 200 or len(val_data) < 50:
            print(f'  Недостаточно данных (train={len(train_data)}, val={len(val_data)})')
            return None
        print(f'  Train баров: {len(train_data)}, Val баров: {len(val_data)}')
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        t0 = time.time()
        study.optimize(self.make_objective(train_data), n_trials=self.n_trials, show_progress_bar=False, n_jobs=1)
        elapsed = time.time() - t0
        best_params = study.best_params
        best_score = study.best_value
        print(f'  Optuna завершено за {elapsed:.1f}s, best score = {best_score:.4f}')
        print(f"  Best params: rsi[{best_params['rsi_low']},{best_params['rsi_high']}],{best_params['rsi_period']}p, L[{best_params['L_short_period']}/{best_params['L_long_period']}], H[{best_params['H_short_period']}/{best_params['H_long_period']}], piv={best_params['pivot_period']}, MACD={best_params['is_using_macd']}, SS={best_params['is_smart_stop_activated']}({best_params['smart_stop']}/{best_params['smart_stop_activation']}), sell={best_params['sell_count']}, pyr={best_params['pyramiding']}, short={best_params['allowshort']}")
        train_stats, _ = run_strategy_on_slice(self.ticker, self.timeframe, best_params, train_data, self.htf_value)
        val_stats, _ = run_strategy_on_slice(self.ticker, self.timeframe, best_params, val_data, self.htf_value)
        if train_stats is None or val_stats is None:
            print(f'  Ошибка при финальном прогоне fold')
            return None
        print(f"  IS  (train): {train_stats.get('Количество сделок', 0):>4} trades, WinRate={train_stats.get('Процент прибыльных сделок', 0):.1f}%, Profit={train_stats.get('Процент роста', 0):.1f}%, MaxDD={train_stats.get('Макс. просадка (%)', 0):.1f}%, Sharpe={train_stats.get('Sharpe', '—')}")
        print(f"  OOS (val):   {val_stats.get('Количество сделок', 0):>4} trades, WinRate={val_stats.get('Процент прибыльных сделок', 0):.1f}%, Profit={val_stats.get('Процент роста', 0):.1f}%, MaxDD={val_stats.get('Макс. просадка (%)', 0):.1f}%, Sharpe={val_stats.get('Sharpe', '—')}")
        return {'fold_id': fold_id, 'train_start': train_start, 'train_end': train_end, 'val_start': val_start, 'val_end': val_end, 'best_params': best_params, 'best_train_score': best_score, 'train_stats': train_stats, 'val_stats': val_stats, 'elapsed_sec': elapsed}

    def run(self) -> dict:
        print(f"\n{'#' * 70}")
        print(f'#  WALK-FORWARD OPTIMIZATION: {self.ticker} {self.timeframe}')
        print(f'#  HTF={self.htf_value}, n_trials={self.n_trials}, folds={len(self.folds)}')
        print(f"{'#' * 70}")
        results = []
        total_t0 = time.time()
        for i, (ts, te, vs, ve) in enumerate(self.folds, 1):
            r = self.run_fold(i, ts, te, vs, ve)
            if r is not None:
                results.append(r)
        total_elapsed = time.time() - total_t0
        print(f'\nTotal optimization time: {total_elapsed / 60:.1f} min')
        out = {'ticker': self.ticker, 'timeframe': self.timeframe, 'htf': self.htf_value, 'n_trials_per_fold': self.n_trials, 'folds': results, 'total_time_sec': total_elapsed}
        json_path = os.path.join(self.output_dir, f'{self.ticker}_optimization_{self.timeframe}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        print(f'Saved: {json_path}')
        rows = []
        for r in results:
            row = {'fold': r['fold_id'], 'train_period': f"{r['train_start']} .. {r['train_end']}", 'val_period': f"{r['val_start']} .. {r['val_end']}", 'train_score': round(r['best_train_score'], 4), 'train_trades': r['train_stats'].get('Количество сделок', 0), 'train_winrate': r['train_stats'].get('Процент прибыльных сделок', 0), 'train_profit_pct': r['train_stats'].get('Процент роста', 0), 'train_maxdd_pct': r['train_stats'].get('Макс. просадка (%)', 0), 'train_sharpe': r['train_stats'].get('Sharpe', None), 'val_trades': r['val_stats'].get('Количество сделок', 0), 'val_winrate': r['val_stats'].get('Процент прибыльных сделок', 0), 'val_profit_pct': r['val_stats'].get('Процент роста', 0), 'val_maxdd_pct': r['val_stats'].get('Макс. просадка (%)', 0), 'val_sharpe': r['val_stats'].get('Sharpe', None), **{f'p_{k}': v for k, v in r['best_params'].items()}}
            rows.append(row)
        csv_path = os.path.join(self.output_dir, f'{self.ticker}_optimization_{self.timeframe}.csv')
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'Saved: {csv_path}')
        return out

def make_default_folds_for_data(data: pd.DataFrame, n_folds: int=3) -> list:
    start = data.index[0]
    end = data.index[-1]
    total_days = (end - start).days
    train_initial_days = total_days // 3
    remaining_days = total_days - train_initial_days
    val_window_days = remaining_days // n_folds
    folds = []
    train_start = start
    for i in range(n_folds):
        val_start = start + pd.Timedelta(days=train_initial_days + i * val_window_days)
        val_end = val_start + pd.Timedelta(days=val_window_days)
        if i == n_folds - 1:
            val_end = end + pd.Timedelta(days=1)
        train_end = val_start
        folds.append((train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d'), val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d')))
    return folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--folds', type=int, default=3)
    args = parser.parse_args()
    htf_value = '1W' if args.timeframe == '1D' else '1D'
    if args.timeframe == '1D':
        data_file = f'data/{args.ticker}_data_new.csv'
    else:
        data_file = f'data/{args.ticker}_hourly_data_new.csv'
    df = pd.read_csv(data_file)
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    folds = make_default_folds_for_data(df, args.folds)
    print(f'Walk-forward folds:')
    for i, f in enumerate(folds, 1):
        print(f'  Fold {i}: train [{f[0]} .. {f[1]}], val [{f[2]} .. {f[3]}]')
    opt = WalkForwardOptimizer(ticker=args.ticker, timeframe=args.timeframe, htf_value=htf_value, folds=folds, n_trials=args.trials)
    opt.run()
if __name__ == '__main__':
    main()
