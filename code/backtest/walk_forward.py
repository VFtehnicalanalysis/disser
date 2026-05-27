import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
from optimization.bayesian_ta import build_strategy, sample_params, run_strategy_on_period, compute_metrics, composite_score, TICKERS, STRATEGY_TYPES
WINDOWS = [('2014-01-01', '2016-12-31', '2017-01-01', '2017-12-31'), ('2015-01-01', '2017-12-31', '2018-01-01', '2018-12-31'), ('2016-01-01', '2018-12-31', '2019-01-01', '2019-12-31'), ('2017-01-01', '2019-12-31', '2020-01-01', '2020-12-31'), ('2018-01-01', '2020-12-31', '2021-01-01', '2021-12-31'), ('2019-01-01', '2021-12-31', '2022-01-01', '2022-12-31'), ('2020-01-01', '2022-12-31', '2023-01-01', '2023-12-31'), ('2021-01-01', '2023-12-31', '2024-01-01', '2024-12-31'), ('2022-01-01', '2024-12-31', '2025-01-01', '2025-12-31'), ('2023-01-01', '2025-12-31', '2026-01-01', '2026-04-20')]

def optimize_window(stype, ticker, train_start, train_end, n_trials):

    def objective(trial):
        try:
            params = sample_params(trial, stype)
        except optuna.TrialPruned:
            return -1000000.0
        res = run_strategy_on_period(stype, ticker, params, train_start, train_end)
        if res is None:
            return -1000000.0
        return composite_score(res['sharpe'], res['return_pct'], res['max_dd_pct'])
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except Exception as e:
        return (None, -1000000.0)
    return (study.best_params, study.best_value)

def run_strategy_on_window_get_equity(stype, ticker, params, period_start, period_end):
    try:
        strat = build_strategy(stype, ticker, '1D', params)
        strat.generate_signals()
        t0 = pd.Timestamp(period_start)
        t1 = pd.Timestamp(period_end)
        mask_data = (strat.data.index >= t0) & (strat.data.index <= t1)
        strat.data = strat.data.loc[mask_data]
        if strat.signals_df is not None and (not strat.signals_df.empty):
            mask_sig = (strat.signals_df.index >= t0) & (strat.signals_df.index <= t1)
            strat.signals_df = strat.signals_df.loc[mask_sig]
        if strat.data.empty or len(strat.data) < 10:
            return None
        strat.execute_trades()
        eq_df = strat.get_equity_curve()
        if eq_df.empty:
            return None
        return eq_df['equity']
    except Exception:
        return None

def walk_forward(stype, ticker, n_trials=20, verbose=True):
    results = []
    test_equities = []
    t_start = time.time()
    for i, (train_start, train_end, test_start, test_end) in enumerate(WINDOWS):
        if verbose:
            print(f'    Окно {i + 1}/{len(WINDOWS)}: train {train_start[:7]}...{train_end[:7]}, test {test_start[:7]}')
        best_params, best_score = optimize_window(stype, ticker, train_start, train_end, n_trials)
        if best_params is None:
            continue
        test_eq = run_strategy_on_window_get_equity(stype, ticker, best_params, test_start, test_end)
        if test_eq is None or test_eq.empty:
            continue
        ret_pct, sharpe, max_dd = compute_metrics(test_eq)
        entry = {'window': i + 1, 'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end, 'params': json.dumps(best_params, separators=(',', ':')), 'train_score': round(best_score, 3), 'test_return_%': round(ret_pct, 2), 'test_sharpe': round(sharpe, 3), 'test_maxdd_%': round(max_dd, 2)}
        results.append(entry)
        normalized_eq = test_eq / test_eq.iloc[0]
        test_equities.append(normalized_eq)
    stitched = []
    running_capital = 1000000
    for eq in test_equities:
        scaled = eq * running_capital
        stitched.append(scaled)
        running_capital = scaled.iloc[-1]
    full_eq = pd.concat(stitched).drop_duplicates().sort_index() if stitched else pd.Series(dtype=float)
    os.makedirs('results/walkforward', exist_ok=True)
    params_df = pd.DataFrame(results)
    params_df.to_csv(f'results/walkforward/{ticker}_{stype}_wf_params.csv', index=False)
    full_eq.name = 'equity'
    full_eq.to_csv(f'results/walkforward/{ticker}_{stype}_wf_equity.csv')
    if verbose and (not params_df.empty):
        total_ret, total_sh, total_dd = compute_metrics(full_eq)
        elapsed = time.time() - t_start
        print(f'    {ticker}/{stype} ИТОГ: OOS ret={total_ret:.1f}%, Sharpe={total_sh:.2f}, DD={total_dd:.1f}%, время {elapsed:.0f}с')
    return (params_df, full_eq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='ema', help=f"Тип стратегии или all. Варианты: {','.join(STRATEGY_TYPES)}")
    parser.add_argument('--ticker', type=str, default=None, help='Один тикер (иначе все)')
    parser.add_argument('--trials', type=int, default=20, help='Trials на окно')
    args = parser.parse_args()
    tickers = [args.ticker] if args.ticker else TICKERS
    types = STRATEGY_TYPES if args.type == 'all' else [args.type]
    for stype in types:
        print(f'\n=== Walk-forward для типа {stype} ===')
        for ticker in tickers:
            print(f'  {ticker}:')
            try:
                walk_forward(stype, ticker, n_trials=args.trials)
            except Exception as e:
                import traceback
                traceback.print_exc()
if __name__ == '__main__':
    main()
