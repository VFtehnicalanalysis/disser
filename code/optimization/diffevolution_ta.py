import argparse
import os
import time
import json
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from .bayesian_ta import compute_metrics, composite_score, run_strategy_on_period, IS_START, IS_END, OOS_START, OOS_END, FULL_START, FULL_END, TICKERS, STRATEGY_TYPES, _period_stats_flat
from .utils import list_bounds, vec_to_params, params_to_constraint_ok, save_topn
OUT_DIR = 'results/optimization_de'

def _de_bounds(stype):
    bounds = list_bounds(stype)
    return [(b[2], b[3]) for b in bounds]

def _objective_factory(stype, ticker, timeframe='1D'):
    history = []

    def objective(vec):
        params = vec_to_params(stype, vec)
        if not params_to_constraint_ok(stype, params):
            return 1000000.0
        res = run_strategy_on_period(stype, ticker, params, IS_START, IS_END, timeframe=timeframe)
        if res is None:
            return 1000000.0
        score = composite_score(res['sharpe'], res['return_pct'], res['max_dd_pct'])
        history.append((vec.copy(), params, score, res))
        return -score
    return (objective, history)

def _top_n_unique(history, n=5):
    history_sorted = sorted(history, key=lambda t: t[2], reverse=True)
    seen = set()
    top = []
    for vec, params, score, metrics in history_sorted:
        key = json.dumps(params, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        top.append((vec, params, score, metrics))
        if len(top) >= n:
            break
    return top

def optimize_ticker_type_de(stype, ticker, maxiter=50, popsize=20, seed=42, top_n=5, verbose=True, timeframe='1D'):
    t0 = time.time()
    if verbose:
        print(f'  {ticker}/{stype}: DE optimization (maxiter={maxiter}, popsize={popsize})...')
    bounds = _de_bounds(stype)
    objective, history = _objective_factory(stype, ticker)
    try:
        differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, tol=0.0001, seed=seed, polish=False, workers=1, updating='deferred', init='sobol', disp=False)
    except Exception as e:
        if verbose:
            print(f'    Ошибка DE: {e}')
        return []
    optim_params_json = json.dumps({'algorithm': 'differential_evolution', 'scipy.version': 'scipy.optimize.differential_evolution', 'maxiter': maxiter, 'popsize': popsize, 'seed': seed, 'init': 'sobol', 'polish': False, 'tol': 0.0001, 'objective': 'composite = Sharpe * (Return% / MaxDD%)'}, ensure_ascii=False, separators=(',', ':'))
    top = _top_n_unique(history, n=top_n)
    rows = []
    for rank, (vec, params, score, _is_metrics_short) in enumerate(top, 1):
        is_res = run_strategy_on_period(stype, ticker, params, IS_START, IS_END, timeframe=timeframe, full_stats=True)
        oos_res = run_strategy_on_period(stype, ticker, params, OOS_START, OOS_END, timeframe=timeframe, full_stats=True)
        full_res = run_strategy_on_period(stype, ticker, params, FULL_START, FULL_END, timeframe=timeframe, full_stats=True)
        if is_res is None:
            continue
        entry = {'ticker': ticker, 'strategy_type': stype, 'timeframe': timeframe, 'allowshort': bool(params.get('allowshort', False)), 'method': 'DiffEvolution', 'optim_params': optim_params_json, 'rank': rank, 'composite_score': round(score, 3), 'params': json.dumps(params, separators=(',', ':'), default=str), 'IS_start': IS_START, 'IS_end': IS_END, 'OOS_start': OOS_START, 'OOS_end': OOS_END, 'FULL_start': FULL_START, 'FULL_end': FULL_END}
        entry.update(_period_stats_flat(is_res, 'IS_'))
        entry.update(_period_stats_flat(oos_res, 'OOS_'))
        entry.update(_period_stats_flat(full_res, 'FULL_'))
        rows.append(entry)
    elapsed = time.time() - t0
    if verbose and rows:
        best = rows[0]
        print(f"    Best: score={best['composite_score']}, IS_ret%={best.get('IS_Процент роста')}, OOS_ret%={best.get('OOS_Процент роста')}, FULL_ret%={best.get('FULL_Процент роста')}, evals={len(history)}, {elapsed:.0f}s")
    return rows

def _history_to_trials_df(history, stype, ticker, timeframe):
    rows = []
    for i, (vec, params, score, res) in enumerate(history):
        row = {'ticker': ticker, 'strategy_type': stype, 'timeframe': timeframe, 'method': 'DiffEvolution', 'trial_number': i, 'composite_score': round(score, 3)}
        for pname, pval in params.items():
            row[f'param_{pname}'] = pval
        row['metric_return_%'] = res['return_pct']
        row['metric_sharpe'] = res['sharpe']
        row['metric_sortino'] = res['sortino']
        row['metric_maxdd_%'] = res['max_dd_pct']
        row['metric_n_trades'] = res['n_trades']
        row['metric_win_rate_%'] = res['win_rate']
        rows.append(row)
    return pd.DataFrame(rows)

def run_for_type(stype, tickers, maxiter=50, popsize=20, top_n=5, timeframe='1D'):
    from optimization_utils import merge_append
    all_rows = []
    all_trials_frames = []
    optim_params_json = json.dumps({'algorithm': 'differential_evolution', 'maxiter': maxiter, 'popsize': popsize, 'seed': 42, 'init': 'sobol', 'polish': False, 'tol': 0.0001, 'objective': 'composite = Sharpe * (Return% / MaxDD%)'}, ensure_ascii=False, separators=(',', ':'))
    for ticker in tickers:
        t0 = time.time()
        print(f'  {ticker}/{stype}/{timeframe}: DE optimization (maxiter={maxiter}, popsize={popsize})...')
        bounds = _de_bounds(stype)
        objective, history = _objective_factory(stype, ticker, timeframe=timeframe)
        try:
            differential_evolution(objective, bounds, maxiter=maxiter, popsize=popsize, tol=0.0001, seed=42, polish=False, workers=1, updating='deferred', init='sobol', disp=False)
        except Exception as e:
            print(f'    Ошибка DE: {e}')
            continue
        trials_full_rows = []
        for i, (vec, params, score, is_res) in enumerate(history):
            if not params_to_constraint_ok(stype, params):
                continue
            if is_res is None:
                continue
            row = {'ticker': ticker, 'strategy_type': stype, 'timeframe': timeframe, 'method': 'DiffEvolution', 'trial_number': i, 'allowshort': bool(params.get('allowshort', False)), 'optim_params': optim_params_json, 'composite_score': round(score, 3), 'params': json.dumps(params, separators=(',', ':'), default=str), 'IS_start': IS_START, 'IS_end': IS_END, 'OOS_start': OOS_START, 'OOS_end': OOS_END, 'FULL_start': FULL_START, 'FULL_end': FULL_END}
            for pname, pval in params.items():
                row[f'param_{pname}'] = pval
            row['metric_return_%'] = is_res['return_pct']
            row['metric_sharpe'] = is_res['sharpe']
            row['metric_sortino'] = is_res['sortino']
            row['metric_maxdd_%'] = is_res['max_dd_pct']
            row['metric_n_trades'] = is_res['n_trades']
            row['metric_win_rate_%'] = is_res['win_rate']
            trials_full_rows.append(row)
        trials_df = pd.DataFrame(trials_full_rows)
        if not trials_df.empty:
            all_trials_frames.append(trials_df)
        top = _top_n_unique(history, n=top_n)
        for rank, (vec, params, score, _is_quick) in enumerate(top, 1):
            is_res = run_strategy_on_period(stype, ticker, params, IS_START, IS_END, timeframe=timeframe, full_stats=True)
            oos_res = run_strategy_on_period(stype, ticker, params, OOS_START, OOS_END, timeframe=timeframe, full_stats=True)
            full_res = run_strategy_on_period(stype, ticker, params, FULL_START, FULL_END, timeframe=timeframe, full_stats=True)
            if is_res is None:
                continue
            entry = {'ticker': ticker, 'strategy_type': stype, 'timeframe': timeframe, 'allowshort': bool(params.get('allowshort', False)), 'method': 'DiffEvolution', 'optim_params': optim_params_json, 'rank': rank, 'composite_score': round(score, 3), 'params': json.dumps(params, separators=(',', ':'), default=str), 'IS_start': IS_START, 'IS_end': IS_END, 'OOS_start': OOS_START, 'OOS_end': OOS_END, 'FULL_start': FULL_START, 'FULL_end': FULL_END}
            entry.update(_period_stats_flat(is_res, 'IS_'))
            entry.update(_period_stats_flat(oos_res, 'OOS_'))
            entry.update(_period_stats_flat(full_res, 'FULL_'))
            all_rows.append(entry)
        print(f'    evals={len(history)}, {time.time() - t0:.0f}s')
    os.makedirs(OUT_DIR, exist_ok=True)
    out_file = os.path.join(OUT_DIR, f'{stype}_de_top{top_n}.csv')
    save_topn(all_rows, out_file)
    if all_trials_frames:
        df_all = pd.concat(all_trials_frames, ignore_index=True)
        trials_file = os.path.join(OUT_DIR, f'{stype}_de_all_trials.csv')
        df_out = merge_append(trials_file, df_all)
        df_out.to_csv(trials_file, index=False)
        print(f'  Сохранено: {trials_file} (+{len(df_all)}, всего {len(df_out)} trials)')
    return pd.DataFrame(all_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='all', help=f"Тип стратегии или all. Варианты: {','.join(STRATEGY_TYPES)}")
    parser.add_argument('--ticker', type=str, default=None, help='Один тикер (иначе все)')
    parser.add_argument('--maxiter', type=int, default=50, help='DE max iterations')
    parser.add_argument('--popsize', type=int, default=20, help='DE population size multiplier')
    parser.add_argument('--top', type=int, default=5, help='Сколько лучших точек сохранять')
    parser.add_argument('--timeframe', type=str, default='1D', help="TF для прогона оптимизации: '1D' или '1h'")
    args = parser.parse_args()
    tickers = [args.ticker] if args.ticker else TICKERS
    types = STRATEGY_TYPES if args.type == 'all' else [args.type]
    if args.type != 'all' and args.type not in STRATEGY_TYPES:
        print(f'Неверный тип: {args.type}. Допустимо: {STRATEGY_TYPES}')
        return
    all_dfs = []
    for stype in types:
        print(f'\n=== DE оптимизация {stype} [{args.timeframe}] ===')
        df = run_for_type(stype, tickers, maxiter=args.maxiter, popsize=args.popsize, top_n=args.top, timeframe=args.timeframe)
        all_dfs.append(df)
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_file = os.path.join(OUT_DIR, f'all_de_top{args.top}.csv')
        save_topn(combined.to_dict('records'), combined_file)
if __name__ == '__main__':
    main()
