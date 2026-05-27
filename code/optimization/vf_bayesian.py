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
from optuna.samplers import TPESampler
from .vf_optimizer import run_strategy_on_slice, composite_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

def full_search_space(trial: optuna.Trial) -> dict:
    rsi_period = trial.suggest_int('rsi_period', 7, 30)
    rsi_low = trial.suggest_int('rsi_low', 15, 40)
    rsi_high = trial.suggest_int('rsi_high', 60, 85)
    pivot_period = trial.suggest_int('pivot_period', 2, 10)
    L_short = trial.suggest_int('L_short_period', 20, 150)
    L_long_offset = trial.suggest_int('L_long_offset', 20, 200)
    L_long = L_short + L_long_offset
    H_short = trial.suggest_int('H_short_period', 20, 150)
    H_long_offset = trial.suggest_int('H_long_offset', 20, 200)
    H_long = H_short + H_long_offset
    is_using_macd = trial.suggest_categorical('is_using_macd', [True, False])
    is_smart_stop_activated = trial.suggest_categorical('is_smart_stop_activated', [True, False])
    smart_stop = trial.suggest_float('smart_stop', 1.0, 10.0, step=0.5)
    smart_stop_activation = trial.suggest_float('smart_stop_activation', 5.0, 25.0, step=1.0)
    sell_count = trial.suggest_int('sell_count', 5, 30)
    pyramiding = trial.suggest_int('pyramiding', 1, 4)
    entry_percentage = round(100.0 / pyramiding, 2)
    allowshort = trial.suggest_categorical('allowshort', [True, False])
    return dict(rsi_period=rsi_period, rsi_low=rsi_low, rsi_high=rsi_high, pivot_period=pivot_period, L_short_period=L_short, L_long_period=L_long, H_short_period=H_short, H_long_period=H_long, is_using_macd=is_using_macd, macd_fast=12, macd_slow=26, macd_signal=9, is_smart_stop_activated=is_smart_stop_activated, smart_stop=smart_stop, smart_stop_activation=smart_stop_activation, sell_count=sell_count, pyramiding=pyramiding, entry_percentage=entry_percentage, allowshort=allowshort)

def warm_start_from_pareto(study: optuna.Study, pareto_csv: str, top_n: int=3):
    if not os.path.exists(pareto_csv):
        print(f'  Warm-start пропущен: {pareto_csv} не найден')
        return
    df = pd.read_csv(pareto_csv)
    sort_col = 'avg_sharpe' if 'avg_sharpe' in df.columns else 'sharpe'
    df = df.sort_values(sort_col, ascending=False).head(top_n)
    for _, row in df.iterrows():
        params = {}
        for k in ['rsi_period', 'rsi_low', 'rsi_high', 'pivot_period', 'L_short_period', 'H_short_period', 'sell_count', 'pyramiding']:
            if k in row and (not pd.isna(row[k])):
                params[k] = int(row[k])
        if 'L_long_period' in row and (not pd.isna(row['L_long_period'])):
            params['L_long_offset'] = int(row['L_long_period']) - params.get('L_short_period', 100)
            if params['L_long_offset'] < 20:
                params['L_long_offset'] = 20
            if params['L_long_offset'] > 200:
                params['L_long_offset'] = 200
        else:
            params['L_long_offset'] = 100
        if 'H_long_period' in row and (not pd.isna(row['H_long_period'])):
            params['H_long_offset'] = int(row['H_long_period']) - params.get('H_short_period', 100)
            if params['H_long_offset'] < 20:
                params['H_long_offset'] = 20
            if params['H_long_offset'] > 200:
                params['H_long_offset'] = 200
        else:
            params['H_long_offset'] = 100
        params['is_using_macd'] = False
        params['is_smart_stop_activated'] = True
        params['smart_stop'] = 6.0
        params['smart_stop_activation'] = 17.0
        params['allowshort'] = False
        try:
            study.enqueue_trial(params)
        except Exception as e:
            print(f'  Warm-start trial skipped: {e}')
    print(f'  Warm-start: {top_n} trials из {pareto_csv}')

def run_bayesian_optimization(ticker: str, timeframe: str, n_trials: int=500, output_dir: str='results/vf_strategy'):
    htf = '1W' if timeframe == '1D' else '1D'
    if timeframe == '1D':
        data_file = f'data/{ticker}_data_new.csv'
    else:
        data_file = f'data/{ticker}_hourly_data_new.csv'
    df = pd.read_csv(data_file)
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df.sort_index(inplace=True)
    print(f"\n{'#' * 70}")
    print(f'#  BAYESIAN OPTIMIZATION (TPE) — FULL SEARCH SPACE')
    print(f'#  {ticker} {timeframe}, htf={htf}, trials={n_trials}')
    print(f'#  Objective: composite Sharpe × (Profit% / MaxDD%)')
    print(f'#  All 14 parameters free; warm-start from V2 Pareto')
    print(f'#  Data: {len(df)} bars')
    print(f"{'#' * 70}")

    def objective(trial: optuna.Trial) -> float:
        params = full_search_space(trial)
        stats, _ = run_strategy_on_slice(ticker, timeframe, params, df, htf)
        if stats is None:
            return -1000000000.0
        return composite_score(stats)
    sampler = TPESampler(seed=42, n_startup_trials=30, prior_weight=1.0)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    pareto_csv = f'{output_dir}/{ticker}_v2_pareto_{timeframe}.csv'
    warm_start_from_pareto(study, pareto_csv, top_n=3)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    print(f'\nЗавершено за {elapsed:.1f}s ({elapsed / n_trials:.2f}s/trial)')
    print(f'Best composite score: {study.best_value:.4f}')
    print(f'Best params:')
    for k, v in study.best_params.items():
        print(f'  {k} = {v}')
    convergence = []
    best_so_far = -1000000000.0
    for t in study.trials:
        if t.value is not None:
            if t.value > best_so_far:
                best_so_far = t.value
        convergence.append({'trial': t.number, 'value': t.value, 'best_so_far': best_so_far, **t.params})
    conv_df = pd.DataFrame(convergence)
    csv_path = os.path.join(output_dir, f'{ticker}_bayesian_full_{timeframe}.csv')
    conv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Saved: {csv_path}')
    best_params = dict(study.best_params)
    best_params['L_long_period'] = best_params['L_short_period'] + best_params['L_long_offset']
    best_params['H_long_period'] = best_params['H_short_period'] + best_params['H_long_offset']
    best_params['entry_percentage'] = round(100.0 / best_params['pyramiding'], 2)
    out = {'ticker': ticker, 'timeframe': timeframe, 'method': 'Bayesian TPE (Optuna) with warm-start', 'n_trials': n_trials, 'elapsed_sec': elapsed, 'best_score': study.best_value, 'best_params': best_params}
    json_path = os.path.join(output_dir, f'{ticker}_bayesian_full_{timeframe}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'Saved: {json_path}')
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--output', default='results/vf_strategy')
    args = parser.parse_args()
    run_bayesian_optimization(args.ticker, args.timeframe, args.trials, args.output)
if __name__ == '__main__':
    main()
