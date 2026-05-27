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
from .vf_optimizer import run_strategy_on_slice, composite_score
PARAM_BOUNDS = {'rsi_period': (7, 30, 'int'), 'rsi_low': (15, 40, 'int'), 'rsi_high': (60, 85, 'int'), 'pivot_period': (2, 10, 'int'), 'L_short_period': (20, 150, 'int'), 'L_long_period': (40, 350, 'int'), 'H_short_period': (20, 150, 'int'), 'H_long_period': (40, 350, 'int'), 'smart_stop': (1.0, 10.0, 'float'), 'smart_stop_activation': (5.0, 25.0, 'float'), 'sell_count': (5, 30, 'int'), 'pyramiding': (1, 4, 'int'), 'is_using_macd': (False, True, 'bool'), 'is_smart_stop_activated': (False, True, 'bool'), 'allowshort': (False, True, 'bool')}
STEP_SIZES = {'rsi_period': 2, 'rsi_low': 2, 'rsi_high': 2, 'pivot_period': 1, 'L_short_period': 10, 'L_long_period': 10, 'H_short_period': 10, 'H_long_period': 10, 'smart_stop': 0.5, 'smart_stop_activation': 1.0, 'sell_count': 2, 'pyramiding': 1}

def get_default_importance():
    return {k: 1.0 for k in PARAM_BOUNDS.keys()}

def load_importance(ticker: str, timeframe: str, output_dir: str) -> dict:
    imp_file = f'{output_dir}/{ticker}_importance_{timeframe}.csv'
    if not os.path.exists(imp_file):
        return get_default_importance()
    df = pd.read_csv(imp_file)
    importance = {row['param']: row['importance'] for _, row in df.iterrows()}
    max_imp = max(importance.values()) if importance else 1.0
    weights = {}
    for k in PARAM_BOUNDS.keys():
        imp = importance.get(k, max_imp * 0.01)
        weights[k] = max(0.1, imp / max_imp)
    return weights

def clip_param(name: str, value):
    lo, hi, typ = PARAM_BOUNDS[name]
    if typ == 'bool':
        return bool(value)
    v = max(lo, min(hi, value))
    if typ == 'int':
        return int(round(v))
    return float(v)

def evaluate(ticker: str, timeframe: str, htf: str, data: pd.DataFrame, params: dict) -> float:
    full = dict(params)
    full.setdefault('macd_fast', 12)
    full.setdefault('macd_slow', 26)
    full.setdefault('macd_signal', 9)
    full.setdefault('entry_percentage', round(100.0 / full.get('pyramiding', 2), 2))
    full.setdefault('is_using_stops', False)
    full.setdefault('is_using_take_profits', False)
    full.setdefault('is_using_trailing_stop', False)
    full.setdefault('is_price_step', False)
    full.setdefault('is_using_trend_analysis', True)
    full.setdefault('check_info', 'BUY & SELL')
    full.setdefault('is_rsi_changer', False)
    full.setdefault('rsi_changer', 0)
    if full.get('L_long_period', 0) <= full.get('L_short_period', 0):
        full['L_long_period'] = full.get('L_short_period', 100) + 50
    if full.get('H_long_period', 0) <= full.get('H_short_period', 0):
        full['H_long_period'] = full.get('H_short_period', 100) + 50
    stats, _ = run_strategy_on_slice(ticker, timeframe, full, data, htf)
    if stats is None:
        return -1000000000.0
    return composite_score(stats)

def compute_gradient(ticker: str, timeframe: str, htf: str, data: pd.DataFrame, params: dict, current_score: float) -> dict:
    grad = {}
    for name in PARAM_BOUNDS.keys():
        lo, hi, typ = PARAM_BOUNDS[name]
        step = STEP_SIZES.get(name, 1)
        if typ == 'bool':
            flipped = dict(params)
            flipped[name] = not params[name]
            score_flip = evaluate(ticker, timeframe, htf, data, flipped)
            grad[name] = score_flip - current_score
        else:
            p_plus = dict(params)
            p_plus[name] = clip_param(name, params[name] + step)
            p_minus = dict(params)
            p_minus[name] = clip_param(name, params[name] - step)
            if p_plus[name] == p_minus[name]:
                if params[name] + step <= hi:
                    s_plus = evaluate(ticker, timeframe, htf, data, p_plus)
                    grad[name] = (s_plus - current_score) / step
                elif params[name] - step >= lo:
                    s_minus = evaluate(ticker, timeframe, htf, data, p_minus)
                    grad[name] = (current_score - s_minus) / step
                else:
                    grad[name] = 0.0
            else:
                s_plus = evaluate(ticker, timeframe, htf, data, p_plus)
                s_minus = evaluate(ticker, timeframe, htf, data, p_minus)
                grad[name] = (s_plus - s_minus) / (2 * step)
    return grad

def random_start_point() -> dict:
    params = {}
    for name, (lo, hi, typ) in PARAM_BOUNDS.items():
        if typ == 'bool':
            params[name] = random.choice([True, False])
        elif typ == 'int':
            params[name] = random.randint(lo, hi)
        else:
            params[name] = round(random.uniform(lo, hi), 2)
    if params['L_long_period'] <= params['L_short_period']:
        params['L_long_period'] = min(350, params['L_short_period'] + 50)
    if params['H_long_period'] <= params['H_short_period']:
        params['H_long_period'] = min(350, params['H_short_period'] + 50)
    return params

def load_warm_start(ticker: str, timeframe: str, output_dir: str) -> dict:
    pareto_csv = f'{output_dir}/{ticker}_v2_pareto_{timeframe}.csv'
    if not os.path.exists(pareto_csv):
        return random_start_point()
    df = pd.read_csv(pareto_csv)
    sort_col = 'avg_sharpe' if 'avg_sharpe' in df.columns else 'sharpe'
    df = df.sort_values(sort_col, ascending=False)
    if df.empty:
        return random_start_point()
    row = df.iloc[0]
    params = {}
    for name in ['rsi_period', 'rsi_low', 'rsi_high', 'pivot_period', 'L_short_period', 'L_long_period', 'H_short_period', 'H_long_period', 'sell_count', 'pyramiding', 'smart_stop', 'smart_stop_activation']:
        if name in row.index and (not pd.isna(row[name])):
            params[name] = clip_param(name, row[name])
    params.setdefault('is_using_macd', False)
    params.setdefault('is_smart_stop_activated', True)
    params.setdefault('allowshort', False)
    params.setdefault('smart_stop', 6.0)
    params.setdefault('smart_stop_activation', 17.0)
    for name, (lo, hi, typ) in PARAM_BOUNDS.items():
        if name not in params:
            if typ == 'bool':
                params[name] = False
            elif typ == 'int':
                params[name] = (lo + hi) // 2
            else:
                params[name] = (lo + hi) / 2
    return params

def gradient_descent(ticker: str, timeframe: str, htf: str, data: pd.DataFrame, start_params: dict, importance: dict, max_iter: int=30, base_lr: float=1.0, patience: int=5) -> tuple:
    params = dict(start_params)
    current_score = evaluate(ticker, timeframe, htf, data, params)
    history = [{'iter': 0, 'score': current_score, **params}]
    best_score = current_score
    best_params = dict(params)
    no_improve_count = 0
    for it in range(1, max_iter + 1):
        t0 = time.time()
        grad = compute_gradient(ticker, timeframe, htf, data, params, current_score)
        new_params = dict(params)
        for name in PARAM_BOUNDS.keys():
            lo, hi, typ = PARAM_BOUNDS[name]
            step = STEP_SIZES.get(name, 1)
            w = importance.get(name, 0.5)
            if typ == 'bool':
                if grad[name] > 0:
                    new_params[name] = not params[name]
            else:
                delta = base_lr * w * grad[name] * step
                new_params[name] = clip_param(name, params[name] + delta)
        new_score = evaluate(ticker, timeframe, htf, data, new_params)
        elapsed = time.time() - t0
        actual_lr = base_lr
        if new_score <= current_score:
            for shrink in [0.5, 0.25, 0.1]:
                backup_params = dict(params)
                for name in PARAM_BOUNDS.keys():
                    lo, hi, typ = PARAM_BOUNDS[name]
                    step = STEP_SIZES.get(name, 1)
                    w = importance.get(name, 0.5)
                    if typ == 'bool':
                        continue
                    delta = base_lr * shrink * w * grad[name] * step
                    backup_params[name] = clip_param(name, params[name] + delta)
                alt_score = evaluate(ticker, timeframe, htf, data, backup_params)
                if alt_score > current_score:
                    new_params = backup_params
                    new_score = alt_score
                    actual_lr = base_lr * shrink
                    break
        improvement = new_score - current_score
        print(f'  iter {it:>2}: score {current_score:.4f} → {new_score:.4f} (Δ={improvement:+.4f}, lr={actual_lr:.2f}, {elapsed:.1f}s)')
        params = new_params
        current_score = new_score
        history.append({'iter': it, 'score': current_score, **params})
        if current_score > best_score:
            best_score = current_score
            best_params = dict(params)
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            print(f'  Early stop: нет улучшения {patience} итераций')
            break
    return (best_params, best_score, history)

def run_gradient_multi_start(ticker: str, timeframe: str, n_starts: int=3, max_iter: int=30, output_dir: str='results/vf_strategy'):
    htf = '1W' if timeframe == '1D' else '1D'
    if timeframe == '1D':
        data_file = f'data/{ticker}_data_new.csv'
    else:
        data_file = f'data/{ticker}_hourly_data_new.csv'
    df = pd.read_csv(data_file)
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df.sort_index(inplace=True)
    importance = load_importance(ticker, timeframe, output_dir)
    print(f"\n{'#' * 70}")
    print(f'#  GRADIENT DESCENT OPTIMIZATION — FULL SEARCH SPACE')
    print(f'#  {ticker} {timeframe}, htf={htf}, starts={n_starts}, max_iter={max_iter}')
    print(f'#  All 14 parameters; importance-weighted learning rates')
    print(f"{'#' * 70}")
    random.seed(42)
    start_points = []
    start_points.append(('warm_from_V2', load_warm_start(ticker, timeframe, output_dir)))
    for i in range(n_starts - 1):
        start_points.append((f'random_{i + 1}', random_start_point()))
    all_results = []
    total_t0 = time.time()
    for name, start in start_points:
        print(f'\n--- START: {name} ---')
        t0 = time.time()
        best_params, best_score, history = gradient_descent(ticker, timeframe, htf, df, start, importance, max_iter=max_iter, base_lr=1.0, patience=5)
        elapsed = time.time() - t0
        print(f'  Итог: score={best_score:.4f} за {elapsed:.1f}s')
        all_results.append({'start_name': name, 'best_params': best_params, 'best_score': best_score, 'history': history, 'elapsed_sec': elapsed})
    total_elapsed = time.time() - total_t0
    print(f'\nВсе starts завершены за {total_elapsed:.1f}s')
    best_overall = max(all_results, key=lambda r: r['best_score'])
    print(f"\n{'=' * 70}")
    print(f"BEST OVERALL (start={best_overall['start_name']}): score={best_overall['best_score']:.4f}")
    print(f"{'=' * 70}")
    for k, v in best_overall['best_params'].items():
        print(f'  {k} = {v}')
    os.makedirs(output_dir, exist_ok=True)
    history_rows = []
    for r in all_results:
        for h in r['history']:
            row = {'start': r['start_name'], **h}
            history_rows.append(row)
    hist_df = pd.DataFrame(history_rows)
    hist_csv = os.path.join(output_dir, f'{ticker}_gradient_full_{timeframe}.csv')
    hist_df.to_csv(hist_csv, index=False, encoding='utf-8-sig')
    print(f'\nSaved: {hist_csv}')
    out = {'ticker': ticker, 'timeframe': timeframe, 'method': 'Gradient descent (finite diff) with multi-start + importance-weighted LR', 'n_starts': n_starts, 'max_iter': max_iter, 'elapsed_sec': total_elapsed, 'best_score': best_overall['best_score'], 'best_start': best_overall['start_name'], 'best_params': best_overall['best_params'], 'all_starts': [{'name': r['start_name'], 'best_score': r['best_score'], 'best_params': r['best_params']} for r in all_results]}
    json_path = os.path.join(output_dir, f'{ticker}_gradient_full_{timeframe}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'Saved: {json_path}')
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--iters', type=int, default=30)
    parser.add_argument('--starts', type=int, default=3)
    parser.add_argument('--output', default='results/vf_strategy')
    args = parser.parse_args()
    run_gradient_multi_start(args.ticker, args.timeframe, n_starts=args.starts, max_iter=args.iters, output_dir=args.output)
if __name__ == '__main__':
    main()
