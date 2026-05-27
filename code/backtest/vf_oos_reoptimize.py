import os
import sys
import time
import json
import argparse
import random
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
from optimization.vf_optimizer import run_strategy_on_slice, composite_score
from optimization.vf_bayesian import full_search_space, warm_start_from_pareto
from optimization.vf_gradient import PARAM_BOUNDS, STEP_SIZES, clip_param, evaluate, random_start_point, load_warm_start, load_importance, gradient_descent
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_split(ticker: str, timeframe: str, split_date: str):
    if timeframe == '1D':
        file = f'data/{ticker}_data_new.csv'
    else:
        file = f'data/{ticker}_hourly_data_new.csv'
    df = pd.read_csv(file)
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df.sort_index(inplace=True)
    split_ts = pd.Timestamp(split_date)
    is_data = df[df.index < split_ts].copy()
    oos_data = df[df.index >= split_ts].copy()
    return (df, is_data, oos_data)

def run_bayesian_is(ticker: str, timeframe: str, htf: str, is_data: pd.DataFrame, n_trials: int, output_dir: str) -> dict:
    print(f"\n{'=' * 70}\nBAYESIAN TPE — оптимизация на IS\n{'=' * 70}")
    print(f'IS: {len(is_data)} bars ({is_data.index[0].date()}..{is_data.index[-1].date()})')

    def objective(trial):
        params = full_search_space(trial)
        stats, _ = run_strategy_on_slice(ticker, timeframe, params, is_data, htf)
        if stats is None:
            return -1000000000.0
        return composite_score(stats)
    sampler = TPESampler(seed=42, n_startup_trials=30)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    pareto_csv = f'{output_dir}/{ticker}_v2_pareto_{timeframe}.csv'
    warm_start_from_pareto(study, pareto_csv, top_n=3)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0
    print(f'Завершено за {elapsed:.1f}s. Best IS score: {study.best_value:.4f}')
    best_params = dict(study.best_params)
    best_params['L_long_period'] = best_params['L_short_period'] + best_params.get('L_long_offset', 100)
    best_params['H_long_period'] = best_params['H_short_period'] + best_params.get('H_long_offset', 100)
    best_params['entry_percentage'] = round(100.0 / best_params['pyramiding'], 2)
    return {'best_params': best_params, 'best_score_IS': study.best_value, 'elapsed': elapsed}

def run_gradient_is(ticker: str, timeframe: str, htf: str, is_data: pd.DataFrame, max_iter: int, n_starts: int, output_dir: str) -> dict:
    print(f"\n{'=' * 70}\nGRADIENT DESCENT — оптимизация на IS\n{'=' * 70}")
    print(f'IS: {len(is_data)} bars')
    importance = load_importance(ticker, timeframe, output_dir)
    random.seed(42)
    start_points = [('warm_from_V2', load_warm_start(ticker, timeframe, output_dir))]
    for i in range(n_starts - 1):
        start_points.append((f'random_{i + 1}', random_start_point()))
    all_results = []
    for name, start in start_points:
        print(f'\n--- START: {name} ---')
        t0 = time.time()
        best_params, best_score, history = gradient_descent(ticker, timeframe, htf, is_data, start, importance, max_iter=max_iter, base_lr=1.0, patience=5)
        elapsed = time.time() - t0
        print(f'  Итог: IS score={best_score:.4f} за {elapsed:.1f}s')
        all_results.append({'start_name': name, 'best_params': best_params, 'best_score_IS': best_score, 'elapsed': elapsed})
    best_overall = max(all_results, key=lambda r: r['best_score_IS'])
    print(f"\nBest overall: start={best_overall['start_name']}, score={best_overall['best_score_IS']:.4f}")
    return best_overall

def apply_on_slice(ticker: str, timeframe: str, htf: str, params: dict, data_slice: pd.DataFrame, label: str) -> dict:
    full_params = dict(params)
    full_params.setdefault('macd_fast', 12)
    full_params.setdefault('macd_slow', 26)
    full_params.setdefault('macd_signal', 9)
    if 'entry_percentage' not in full_params:
        full_params['entry_percentage'] = round(100.0 / full_params.get('pyramiding', 2), 2)
    full_params.setdefault('is_using_stops', False)
    full_params.setdefault('is_using_take_profits', False)
    full_params.setdefault('is_using_trailing_stop', False)
    full_params.setdefault('is_price_step', False)
    full_params.setdefault('is_using_trend_analysis', True)
    full_params.setdefault('check_info', 'BUY & SELL')
    full_params.setdefault('is_rsi_changer', False)
    full_params.setdefault('rsi_changer', 0)
    if full_params.get('L_long_period', 0) <= full_params.get('L_short_period', 0):
        full_params['L_long_period'] = full_params.get('L_short_period', 100) + 50
    if full_params.get('H_long_period', 0) <= full_params.get('H_short_period', 0):
        full_params['H_long_period'] = full_params.get('H_short_period', 100) + 50
    stats, _ = run_strategy_on_slice(ticker, timeframe, full_params, data_slice, htf)
    if stats is None:
        print(f'  {label}: FAILED')
        return None
    print(f"  {label}: trades={stats.get('Количество сделок', 0)}, win={stats.get('Процент прибыльных сделок', 0)}%, profit={stats.get('Процент роста', 0)}%, maxdd={stats.get('Макс. просадка (%)', 0)}%, sharpe={stats.get('Sharpe', '—')}")
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--split', default='2025-01-01')
    parser.add_argument('--trials', type=int, default=500)
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--skip-gradient', action='store_true')
    parser.add_argument('--output', default='results/vf_strategy')
    args = parser.parse_args()
    htf = '1W' if args.timeframe == '1D' else '1D'
    print(f"\n{'#' * 70}")
    print(f'#  HONEST OOS VALIDATION — {args.ticker} {args.timeframe}')
    print(f'#  Split: {args.split}')
    print(f'#  Optimize ONLY on IS, apply on OOS')
    print(f"{'#' * 70}")
    full_data, is_data, oos_data = load_and_split(args.ticker, args.timeframe, args.split)
    print(f'\nIS: {len(is_data)} bars ({is_data.index[0].date()}..{is_data.index[-1].date()})')
    print(f'OOS: {len(oos_data)} bars ({oos_data.index[0].date()}..{oos_data.index[-1].date()})')
    results = []
    bayes_result = run_bayesian_is(args.ticker, args.timeframe, htf, is_data, args.trials, args.output)
    print(f'\n--- Apply Bayesian best on IS, OOS, Full ---')
    bayes_is_stats = apply_on_slice(args.ticker, args.timeframe, htf, bayes_result['best_params'], is_data, 'IS    ')
    bayes_oos_stats = apply_on_slice(args.ticker, args.timeframe, htf, bayes_result['best_params'], oos_data, 'OOS   ')
    bayes_full_stats = apply_on_slice(args.ticker, args.timeframe, htf, bayes_result['best_params'], full_data, 'Full  ')
    results.append({'method': 'Bayesian_OOS', 'best_params': bayes_result['best_params'], 'IS_stats': bayes_is_stats, 'OOS_stats': bayes_oos_stats, 'Full_stats': bayes_full_stats, 'elapsed': bayes_result['elapsed']})
    if not args.skip_gradient:
        grad_result = run_gradient_is(args.ticker, args.timeframe, htf, is_data, args.iters, args.starts, args.output)
        print(f'\n--- Apply Gradient best on IS, OOS, Full ---')
        grad_is_stats = apply_on_slice(args.ticker, args.timeframe, htf, grad_result['best_params'], is_data, 'IS    ')
        grad_oos_stats = apply_on_slice(args.ticker, args.timeframe, htf, grad_result['best_params'], oos_data, 'OOS   ')
        grad_full_stats = apply_on_slice(args.ticker, args.timeframe, htf, grad_result['best_params'], full_data, 'Full  ')
        results.append({'method': 'Gradient_OOS', 'best_params': grad_result['best_params'], 'IS_stats': grad_is_stats, 'OOS_stats': grad_oos_stats, 'Full_stats': grad_full_stats, 'elapsed': grad_result['elapsed'], 'start_name': grad_result.get('start_name')})
    out = {'ticker': args.ticker, 'timeframe': args.timeframe, 'split_date': args.split, 'IS_bars': len(is_data), 'OOS_bars': len(oos_data), 'results': results}
    json_path = os.path.join(args.output, f'{args.ticker}_honest_oos_{args.timeframe}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {json_path}')
    rows = []
    for r in results:
        for period in ['IS', 'OOS', 'Full']:
            st = r.get(f'{period}_stats')
            if st is None:
                continue
            rows.append({'Method': r['method'], 'Period': period, 'Trades': st.get('Количество сделок', 0), 'Win%': st.get('Процент прибыльных сделок', 0), 'Profit%': st.get('Процент роста', 0), 'MaxDD%': st.get('Макс. просадка (%)', 0), 'Sharpe': st.get('Sharpe', None), 'Sortino': st.get('Sortino', None)})
    csv_df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output, f'{args.ticker}_honest_oos_{args.timeframe}.csv')
    csv_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Saved: {csv_path}')
    print(f"\n{'=' * 100}\nHONEST OOS VALIDATION — СВОДКА\n{'=' * 100}")
    print(f"{'Method':<18} {'Period':<6} | {'Trades':>6} {'Win%':>6} {'Profit%':>9} {'MaxDD%':>8} {'Sharpe':>7}")
    print('-' * 80)
    for row in rows:
        print(f"{row['Method']:<18} {row['Period']:<6} | {row['Trades']:>6} {row['Win%']:>6.1f} {row['Profit%']:>9.2f} {row['MaxDD%']:>8.2f} {str(row['Sharpe']):>7}")
if __name__ == '__main__':
    main()
