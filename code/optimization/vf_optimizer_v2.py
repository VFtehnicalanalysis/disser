import os
import sys
import time
import json
import argparse
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import NSGAIISampler
from .vf_optimizer import run_strategy_on_slice, composite_score, _suppress_stdout
optuna.logging.set_verbosity(optuna.logging.WARNING)

def make_reduced_search_space(trial: optuna.Trial, fixed_params: dict) -> dict:
    rsi_period = trial.suggest_int('rsi_period', 7, 30)
    sell_count = trial.suggest_int('sell_count', 5, 30)
    H_short = trial.suggest_int('H_short_period', 20, 150)
    pivot_period = trial.suggest_int('pivot_period', 2, 10)
    rsi_low = trial.suggest_int('rsi_low', 15, 40)
    rsi_high = trial.suggest_int('rsi_high', 60, 85)
    L_short = trial.suggest_int('L_short_period', 20, 150)
    pyramiding = trial.suggest_int('pyramiding', 1, 4)
    entry_percentage = round(100.0 / pyramiding, 2)
    L_long_default = fixed_params.get('L_long_period', 200)
    H_long_default = fixed_params.get('H_long_period', 200)
    L_long = max(L_long_default, L_short + 30)
    H_long = max(H_long_default, H_short + 30)
    params = dict(fixed_params)
    params.update(dict(rsi_period=rsi_period, rsi_low=rsi_low, rsi_high=rsi_high, pivot_period=pivot_period, L_short_period=L_short, L_long_period=L_long, H_short_period=H_short, H_long_period=H_long, sell_count=sell_count, pyramiding=pyramiding, entry_percentage=entry_percentage))
    return params
DEFAULT_FIXED_PARAMS = dict(is_using_macd=False, macd_fast=12, macd_slow=26, macd_signal=9, is_smart_stop_activated=True, smart_stop=6.0, smart_stop_activation=17.0, allowshort=False, L_long_period=200, H_long_period=296)

def make_default_folds(data: pd.DataFrame, n_folds: int=3) -> list:
    start = data.index[0]
    end = data.index[-1]
    total_days = (end - start).days
    train_initial = total_days // 3
    remaining = total_days - train_initial
    val_window = remaining // n_folds
    folds = []
    for i in range(n_folds):
        val_start = start + pd.Timedelta(days=train_initial + i * val_window)
        val_end = val_start + pd.Timedelta(days=val_window) if i < n_folds - 1 else end + pd.Timedelta(days=1)
        train_end = val_start
        folds.append((start, train_end, val_start, val_end))
    return folds

class MultiObjectiveOptimizer:

    def __init__(self, ticker: str, timeframe: str, htf: str, folds: list, n_trials: int=400, fixed_params: dict=None, output_dir: str='results/vf_strategy'):
        self.ticker = ticker
        self.timeframe = timeframe
        self.htf = htf
        self.folds = folds
        self.n_trials = n_trials
        self.fixed_params = fixed_params if fixed_params else DEFAULT_FIXED_PARAMS
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
        self.train_slices = []
        self.val_slices = []
        for ts, te, vs, ve in folds:
            train = df.loc[(df.index >= ts) & (df.index < te)].copy()
            val = df.loc[(df.index >= vs) & (df.index < ve)].copy()
            self.train_slices.append(train)
            self.val_slices.append(val)

    def evaluate_trial(self, params: dict) -> tuple:
        sharpes = []
        maxdds = []
        n_valid_folds = 0
        for val_data in self.val_slices:
            stats, _ = run_strategy_on_slice(self.ticker, self.timeframe, params, val_data, self.htf)
            if stats is None or stats.get('Количество сделок', 0) < 1:
                continue
            sharpe = stats.get('Sharpe', None)
            maxdd = stats.get('Макс. просадка (%)', 0)
            if sharpe is None or pd.isna(sharpe):
                continue
            sharpes.append(float(sharpe))
            maxdds.append(float(maxdd))
            n_valid_folds += 1
        if n_valid_folds < 2:
            return (-10.0, 100.0, 10.0)
        avg_sharpe = float(np.mean(sharpes))
        avg_maxdd = float(np.mean(maxdds))
        std_sharpe = float(np.std(sharpes))
        return (avg_sharpe, avg_maxdd, std_sharpe)

    def make_objective(self):

        def objective(trial: optuna.Trial):
            params = make_reduced_search_space(trial, self.fixed_params)
            return self.evaluate_trial(params)
        return objective

    def run(self) -> dict:
        print(f"\n{'#' * 70}")
        print(f'#  MULTI-OBJECTIVE NSGA-II OPTIMIZATION')
        print(f'#  {self.ticker} {self.timeframe}, htf={self.htf}, trials={self.n_trials}')
        print(f'#  Objectives: max(avg_Sharpe), min(avg_MaxDD), min(std_Sharpe)')
        print(f'#  Walk-forward folds: {len(self.folds)} val periods')
        print(f"{'#' * 70}\n")
        for i, (ts, te, vs, ve) in enumerate(self.folds, 1):
            print(f'  Fold {i}: train [{ts.date()} .. {te.date()}], val [{vs.date()} .. {ve.date()}]')
        pop_size = max(20, int(np.sqrt(self.n_trials)))
        sampler = NSGAIISampler(population_size=pop_size, seed=42)
        study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'], sampler=sampler)
        t0 = time.time()
        study.optimize(self.make_objective(), n_trials=self.n_trials, show_progress_bar=False)
        elapsed = time.time() - t0
        print(f'\nЗавершено за {elapsed / 60:.1f} мин ({self.n_trials} trials, ~{elapsed / self.n_trials:.1f}s/trial)')
        pareto_trials = study.best_trials
        print(f'\nPareto-оптимальных решений: {len(pareto_trials)}')
        all_trials_data = []
        for t in study.trials:
            row = {'trial': t.number}
            row.update(t.params)
            if t.values is not None and len(t.values) == 3:
                row['avg_sharpe'] = t.values[0]
                row['avg_maxdd'] = t.values[1]
                row['std_sharpe'] = t.values[2]
            else:
                row['avg_sharpe'] = None
                row['avg_maxdd'] = None
                row['std_sharpe'] = None
            row['is_pareto'] = t.number in [pt.number for pt in pareto_trials]
            all_trials_data.append(row)
        all_trials_df = pd.DataFrame(all_trials_data)
        csv_path = os.path.join(self.output_dir, f'{self.ticker}_v2_all_trials_{self.timeframe}.csv')
        all_trials_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'Saved all trials: {csv_path}')
        pareto_data = []
        for pt in pareto_trials:
            row = {'trial': pt.number}
            row.update(pt.params)
            row['avg_sharpe'] = pt.values[0]
            row['avg_maxdd'] = pt.values[1]
            row['std_sharpe'] = pt.values[2]
            pareto_data.append(row)
        pareto_df = pd.DataFrame(pareto_data).sort_values('avg_sharpe', ascending=False)
        pareto_path = os.path.join(self.output_dir, f'{self.ticker}_v2_pareto_{self.timeframe}.csv')
        pareto_df.to_csv(pareto_path, index=False, encoding='utf-8-sig')
        print(f'Saved pareto solutions: {pareto_path}')
        out = {'ticker': self.ticker, 'timeframe': self.timeframe, 'htf': self.htf, 'n_trials': self.n_trials, 'fixed_params': self.fixed_params, 'elapsed_sec': elapsed, 'n_pareto': len(pareto_trials), 'pareto_trials': [{'trial': pt.number, 'params': pt.params, 'avg_sharpe': pt.values[0], 'avg_maxdd': pt.values[1], 'std_sharpe': pt.values[2]} for pt in pareto_trials]}
        json_path = os.path.join(self.output_dir, f'{self.ticker}_v2_optimization_{self.timeframe}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=str)
        print(f'Saved JSON: {json_path}')
        print(f"\n{'=' * 70}\nТОП-5 PARETO-ОПТИМАЛЬНЫХ РЕШЕНИЙ (по avg_Sharpe)\n{'=' * 70}")
        print(f"{'Trial':>6} {'avgSh':>8} {'avgDD%':>8} {'stdSh':>7} | params")
        for _, row in pareto_df.head(5).iterrows():
            params_parts = []
            for k in ['rsi_period', 'rsi_low', 'rsi_high', 'pivot_period', 'L_short_period', 'H_short_period', 'sell_count', 'pyramiding']:
                if k in row.index and (not pd.isna(row[k])):
                    params_parts.append(f"{k.replace('_period', '')}={int(row[k])}")
            params_str = ', '.join(params_parts)
            print(f"{int(row['trial']):>6} {row['avg_sharpe']:>8.3f} {row['avg_maxdd']:>8.2f} {row['std_sharpe']:>7.3f} | {params_str}")
        return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--trials', type=int, default=400, help='Total trials for NSGA-II (recommended: 400-800)')
    parser.add_argument('--folds', type=int, default=3)
    args = parser.parse_args()
    htf = '1W' if args.timeframe == '1D' else '1D'
    if args.timeframe == '1D':
        data_file = f'data/{args.ticker}_data_new.csv'
    else:
        data_file = f'data/{args.ticker}_hourly_data_new.csv'
    df = pd.read_csv(data_file)
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df.sort_index(inplace=True)
    folds = make_default_folds(df, args.folds)
    sensitivity_file = f'results/vf_strategy/{args.ticker}_importance_{args.timeframe}.json'
    fixed = dict(DEFAULT_FIXED_PARAMS)
    if os.path.exists(sensitivity_file):
        with open(sensitivity_file, 'r', encoding='utf-8') as f:
            sens = json.load(f)
        bp = sens['best_params']
        for key in ['rsi_period', 'is_using_macd', 'is_smart_stop_activated', 'smart_stop', 'smart_stop_activation', 'sell_count', 'pyramiding', 'allowshort']:
            if key in bp:
                fixed[key] = bp[key]
        print(f'Используем fixed params из {sensitivity_file}:')
        for k in ['rsi_period', 'is_using_macd', 'is_smart_stop_activated', 'smart_stop', 'smart_stop_activation', 'sell_count', 'pyramiding', 'allowshort']:
            print(f'  {k} = {fixed.get(k)}')
    opt = MultiObjectiveOptimizer(ticker=args.ticker, timeframe=args.timeframe, htf=htf, folds=folds, n_trials=args.trials, fixed_params=fixed)
    opt.run()
if __name__ == '__main__':
    main()
