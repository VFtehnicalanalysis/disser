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
from strategies.technical import EMACrossoverStrategy, RSIStrategy, MACDHistogramStrategy, BreakoutPivotStrategy, BreakoutOLSPivotStrategy, RetestPivotStrategy, RetestOLSPivotStrategy, OBVSignalStrategy, VWAPSignalStrategy, EngulfingStrategy, HammerDojiStrategy, HaramiStrategy, CandleEnsembleStrategy
TICKERS = ['LKOH', 'SBER', 'GAZP', 'ROSN', 'VKCO', 'AFKS', 'VTBR', 'SNGSP', 'GMKN', 'PLZL']
STRATEGY_TYPES = ['ema', 'rsi', 'macd', 'breakout_pivot', 'breakout_ols', 'retest_pivot', 'retest_ols', 'obv', 'vwap', 'engulfing', 'hammerdoji', 'harami', 'candle_ensemble']
IS_START = '2014-01-01'
IS_END = '2024-12-31'
OOS_START = '2025-01-01'
OOS_END = '2026-04-20'
FULL_START = '2014-01-01'
FULL_END = '2026-04-20'
OUT_DIR = 'results/optimization'

def compute_metrics(equity):
    if equity.empty or len(equity) < 2:
        return (0.0, 0.0, 0.0, 0.0)
    returns = equity.pct_change().dropna()
    initial, final = (equity.iloc[0], equity.iloc[-1])
    ret_pct = (final / initial - 1) * 100 if initial != 0 else 0.0
    running_max = equity.cummax()
    max_dd = (running_max - equity) / running_max * 100
    max_dd_pct = max_dd.max() if not max_dd.empty else 0.0
    sharpe = 0.0
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    downside = returns[returns < 0]
    sortino = 0.0
    if len(downside) > 1 and downside.std() > 0:
        sortino = returns.mean() / downside.std() * np.sqrt(252)
    return (ret_pct, sharpe, sortino, max_dd_pct)

def composite_score(sharpe, ret_pct, max_dd_pct):
    if max_dd_pct < 0.01:
        max_dd_pct = 0.01
    score = sharpe * (ret_pct / max_dd_pct)
    if np.isnan(score) or np.isinf(score):
        return -1000000.0
    return score

def build_strategy(stype, ticker, timeframe, params):
    ia = 1000000
    ep = 100
    if stype == 'ema':
        return EMACrossoverStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, short_ema=params['short_ema'], long_ema=params['long_ema'], pyramiding=1, allowshort=params['allowshort'], debug=False)
    elif stype == 'rsi':
        return RSIStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, rsi_lower=params['rsi_lower'], rsi_upper=params['rsi_upper'], rsi_window=params['rsi_window'], pyramiding=1, allowshort=params['allowshort'])
    elif stype == 'macd':
        return MACDHistogramStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, macd_fast=params['macd_fast'], macd_slow=params['macd_slow'], macd_signal=params['macd_signal'], pyramiding=1, allowshort=params['allowshort'])
    elif stype == 'breakout_pivot':
        return BreakoutPivotStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, pyramiding=1, allowshort=params['allowshort'], lookback=params['lookback'], pivot_window=params['pivot_window'], min_touches=params['min_touches'], tolerance_atr_mult=params['tolerance_atr_mult'], cluster_atr_mult=params['cluster_atr_mult'], confirmation_bars=params['confirmation_bars'], cooldown_bars=params['cooldown_bars'])
    elif stype == 'breakout_ols':
        return BreakoutOLSPivotStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, pyramiding=1, allowshort=params['allowshort'], lookback=params['lookback'], pivot_window=params['pivot_window'], min_touches=params['min_touches'], tolerance_atr_mult=params['tolerance_atr_mult'], cluster_atr_mult=params['cluster_atr_mult'], confirmation_bars=params['confirmation_bars'], cooldown_bars=params['cooldown_bars'])
    elif stype == 'retest_pivot':
        return RetestPivotStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, pyramiding=1, allowshort=params['allowshort'], lookback=params['lookback'], pivot_window=params['pivot_window'], min_touches=params['min_touches'], tolerance_atr_mult=params['tolerance_atr_mult'], cluster_atr_mult=params['cluster_atr_mult'], confirmation_bars=params['confirmation_bars'], cooldown_bars=params['cooldown_bars'], retest_window=params['retest_window'], cancel_atr_mult=params['cancel_atr_mult'])
    elif stype == 'retest_ols':
        return RetestOLSPivotStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, pyramiding=1, allowshort=params['allowshort'], lookback=params['lookback'], pivot_window=params['pivot_window'], min_touches=params['min_touches'], tolerance_atr_mult=params['tolerance_atr_mult'], cluster_atr_mult=params['cluster_atr_mult'], confirmation_bars=params['confirmation_bars'], cooldown_bars=params['cooldown_bars'], retest_window=params['retest_window'], cancel_atr_mult=params['cancel_atr_mult'])
    elif stype == 'obv':
        return OBVSignalStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, obv_ema_window=params['obv_ema_window'], confirmation_bars=params['confirmation_bars'], pyramiding=1, allowshort=params['allowshort'])
    elif stype == 'vwap':
        return VWAPSignalStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, vwap_window=params['vwap_window'], vwap_tol=params['vwap_tol'], vwap_mode=params['vwap_mode'], pyramiding=1, allowshort=params['allowshort'])
    elif stype in ('engulfing', 'hammerdoji', 'harami'):
        cls_map = {'engulfing': EngulfingStrategy, 'hammerdoji': HammerDojiStrategy, 'harami': HaramiStrategy}
        return cls_map[stype](ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, doji_tol=params['doji_tol'], wick_body_ratio=params['wick_body_ratio'], pyramiding=1, allowshort=params['allowshort'])
    elif stype == 'candle_ensemble':
        return CandleEnsembleStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=ia, entry_percentage=ep, ensemble_window=params['ensemble_window'], min_votes=params['min_votes'], doji_tol=params['doji_tol'], wick_body_ratio=params['wick_body_ratio'], pyramiding=1, allowshort=params['allowshort'])
    raise ValueError(f'Unknown strategy type: {stype}')

def sample_params(trial, stype):
    if stype == 'ema':
        short_ema = trial.suggest_int('short_ema', 5, 100)
        long_ema = trial.suggest_int('long_ema', 50, 200)
        if short_ema >= long_ema:
            raise optuna.TrialPruned()
        return {'short_ema': short_ema, 'long_ema': long_ema, 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    elif stype == 'rsi':
        rsi_lower = trial.suggest_int('rsi_lower', 10, 45)
        rsi_upper = trial.suggest_int('rsi_upper', 55, 90)
        return {'rsi_lower': rsi_lower, 'rsi_upper': rsi_upper, 'rsi_window': trial.suggest_int('rsi_window', 7, 30), 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    elif stype == 'macd':
        macd_fast = trial.suggest_int('macd_fast', 5, 20)
        macd_slow = trial.suggest_int('macd_slow', 20, 50)
        if macd_fast >= macd_slow:
            raise optuna.TrialPruned()
        return {'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_signal': trial.suggest_int('macd_signal', 5, 15), 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    elif stype in ('breakout_pivot', 'breakout_ols'):
        return {'lookback': trial.suggest_int('lookback', 100, 300), 'pivot_window': trial.suggest_int('pivot_window', 3, 12), 'min_touches': trial.suggest_int('min_touches', 2, 5), 'tolerance_atr_mult': trial.suggest_float('tolerance_atr_mult', 0.3, 1.2, step=0.05), 'cluster_atr_mult': trial.suggest_float('cluster_atr_mult', 0.3, 1.2, step=0.05), 'confirmation_bars': trial.suggest_int('confirmation_bars', 1, 3), 'cooldown_bars': trial.suggest_int('cooldown_bars', 3, 20), 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    elif stype in ('retest_pivot', 'retest_ols'):
        return {'lookback': trial.suggest_int('lookback', 100, 300), 'pivot_window': trial.suggest_int('pivot_window', 3, 12), 'min_touches': trial.suggest_int('min_touches', 2, 5), 'tolerance_atr_mult': trial.suggest_float('tolerance_atr_mult', 0.3, 1.2, step=0.05), 'cluster_atr_mult': trial.suggest_float('cluster_atr_mult', 0.3, 1.2, step=0.05), 'confirmation_bars': trial.suggest_int('confirmation_bars', 1, 3), 'cooldown_bars': trial.suggest_int('cooldown_bars', 3, 20), 'retest_window': trial.suggest_int('retest_window', 10, 40), 'cancel_atr_mult': trial.suggest_float('cancel_atr_mult', 0.5, 2.5, step=0.1), 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    elif stype == 'obv':
        return {'obv_ema_window': trial.suggest_int('obv_ema_window', 5, 30), 'confirmation_bars': trial.suggest_int('confirmation_bars', 1, 5), 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    elif stype == 'vwap':
        return {'vwap_window': trial.suggest_int('vwap_window', 20, 200), 'vwap_tol': trial.suggest_float('vwap_tol', 0.001, 0.02, step=0.001), 'vwap_mode': trial.suggest_categorical('vwap_mode', ['rolling', 'yearly']), 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    elif stype in ('engulfing', 'hammerdoji', 'harami'):
        return {'doji_tol': trial.suggest_float('doji_tol', 0.03, 0.2, step=0.01), 'wick_body_ratio': trial.suggest_float('wick_body_ratio', 1.5, 3.5, step=0.1), 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    elif stype == 'candle_ensemble':
        return {'ensemble_window': trial.suggest_int('ensemble_window', 3, 10), 'min_votes': trial.suggest_int('min_votes', 2, 4), 'doji_tol': trial.suggest_float('doji_tol', 0.03, 0.2, step=0.01), 'wick_body_ratio': trial.suggest_float('wick_body_ratio', 1.5, 3.5, step=0.1), 'allowshort': trial.suggest_categorical('allowshort', [True, False])}
    raise ValueError(f'Unknown type {stype}')

def run_strategy_on_period(stype, ticker, params, period_start, period_end, timeframe='1D', full_stats=False):
    try:
        strat = build_strategy(stype, ticker, timeframe, params)
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
        strat.equity_df = eq_df
        eq = eq_df['equity']
        ret_pct, sharpe, sortino, max_dd = compute_metrics(eq)
        closed = [t for t in strat.trades if t.is_closed]
        n_trades = len(closed)
        profitable = sum((1 for t in closed if t.result > 0))
        win_rate = round(profitable / n_trades * 100, 2) if n_trades > 0 else 0.0
        out = {'return_pct': round(ret_pct, 2), 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3), 'max_dd_pct': round(max_dd, 2), 'n_trades': n_trades, 'win_rate': win_rate}
        if full_stats:
            out['stats'] = strat.get_stats()
        return out
    except Exception as e:
        return None
_STATS_KEYS_FOR_PERIOD = ['Количество сигналов', 'Количество сделок', 'Прибыльных сделок', 'Убыточных сделок', 'Процент прибыльных сделок', 'Средняя прибыль', 'Средний убыток', 'Среднее время удержания (дней)', 'Среднее время удержания (часов)', 'Макс. просадка (%)', 'Макс. прибыль по сделке', 'Макс. убыток по сделке', 'Процент роста', 'Sharpe', 'Sortino', 'Std. return', 'Reward/Risk']

def _period_stats_flat(res, prefix):
    if res is None or 'stats' not in res:
        return {f'{prefix}{k}': None for k in _STATS_KEYS_FOR_PERIOD}
    stats = res['stats'] or {}
    return {f'{prefix}{k}': stats.get(k) for k in _STATS_KEYS_FOR_PERIOD}

def _metrics_tuple(res, prefix):
    if res is None:
        return {f'{prefix}return_%': None, f'{prefix}sharpe': None, f'{prefix}sortino': None, f'{prefix}maxdd_%': None, f'{prefix}win_rate_%': None, f'{prefix}n_trades': None}
    return {f'{prefix}return_%': res['return_pct'], f'{prefix}sharpe': res['sharpe'], f'{prefix}sortino': res['sortino'], f'{prefix}maxdd_%': res['max_dd_pct'], f'{prefix}win_rate_%': res['win_rate'], f'{prefix}n_trades': res['n_trades']}

def optimize_ticker_type(stype, ticker, n_trials=50, seed=42, verbose=True, timeframe='1D'):
    t0 = time.time()
    if verbose:
        print(f'  {ticker}/{stype}: IS optimization (Bayesian TPE, trials={n_trials})...')

    def objective(trial):
        try:
            params = sample_params(trial, stype)
        except optuna.TrialPruned:
            return -1000000.0
        res = run_strategy_on_period(stype, ticker, params, IS_START, IS_END, timeframe=timeframe)
        if res is None:
            return -1000000.0
        return composite_score(res['sharpe'], res['return_pct'], res['max_dd_pct'])
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except Exception as e:
        if verbose:
            print(f'    Ошибка Optuna: {e}')
        return []
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    trials.sort(key=lambda t: t.value, reverse=True)
    top_trials = trials[:5]
    optim_params_json = json.dumps({'sampler': 'TPESampler', 'seed': seed, 'n_trials': n_trials, 'direction': 'maximize', 'objective': 'composite = Sharpe * (Return% / MaxDD%)'}, ensure_ascii=False, separators=(',', ':'))
    top5 = []
    for rank, trial in enumerate(top_trials, 1):
        params = trial.params.copy()
        is_res = run_strategy_on_period(stype, ticker, params, IS_START, IS_END, timeframe=timeframe, full_stats=True)
        oos_res = run_strategy_on_period(stype, ticker, params, OOS_START, OOS_END, timeframe=timeframe, full_stats=True)
        full_res = run_strategy_on_period(stype, ticker, params, FULL_START, FULL_END, timeframe=timeframe, full_stats=True)
        if is_res is None:
            continue
        entry = {'ticker': ticker, 'strategy_type': stype, 'timeframe': timeframe, 'allowshort': bool(params.get('allowshort', False)), 'method': 'Bayesian_TPE', 'optim_params': optim_params_json, 'rank': rank, 'composite_score': round(trial.value, 3), 'params': json.dumps(params, separators=(',', ':'), default=str), 'IS_start': IS_START, 'IS_end': IS_END, 'OOS_start': OOS_START, 'OOS_end': OOS_END, 'FULL_start': FULL_START, 'FULL_end': FULL_END}
        entry.update(_period_stats_flat(is_res, 'IS_'))
        entry.update(_period_stats_flat(oos_res, 'OOS_'))
        entry.update(_period_stats_flat(full_res, 'FULL_'))
        top5.append(entry)
    elapsed = time.time() - t0
    if verbose and top5:
        best = top5[0]
        print(f"    Best: score={best['composite_score']}, IS_ret%={best.get('IS_Процент роста')}, OOS_ret%={best.get('OOS_Процент роста')}, FULL_ret%={best.get('FULL_Процент роста')}, {elapsed:.0f}s")
    return top5

def optimize_ticker_type_with_trials(stype, ticker, n_trials=50, seed=42, verbose=True, timeframe='1D'):
    t0 = time.time()
    if verbose:
        print(f'  {ticker}/{stype}/{timeframe}: IS optimization (Bayesian TPE, trials={n_trials})...')

    def objective(trial):
        try:
            params = sample_params(trial, stype)
        except optuna.TrialPruned:
            return -1000000.0
        res = run_strategy_on_period(stype, ticker, params, IS_START, IS_END, timeframe=timeframe)
        if res is None:
            return -1000000.0
        return composite_score(res['sharpe'], res['return_pct'], res['max_dd_pct'])
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    except Exception as e:
        if verbose:
            print(f'    Ошибка Optuna: {e}')
        return ([], pd.DataFrame())
    optim_params_json = json.dumps({'sampler': 'TPESampler', 'seed': seed, 'n_trials': n_trials, 'direction': 'maximize', 'objective': 'composite = Sharpe * (Return% / MaxDD%)'}, ensure_ascii=False, separators=(',', ':'))
    all_trial_rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None or t.value <= -100000.0:
            continue
        params = t.params.copy()
        is_res = run_strategy_on_period(stype, ticker, params, IS_START, IS_END, timeframe=timeframe, full_stats=True)
        oos_res = run_strategy_on_period(stype, ticker, params, OOS_START, OOS_END, timeframe=timeframe, full_stats=True)
        full_res = run_strategy_on_period(stype, ticker, params, FULL_START, FULL_END, timeframe=timeframe, full_stats=True)
        if is_res is None:
            continue
        row = {'ticker': ticker, 'strategy_type': stype, 'timeframe': timeframe, 'method': 'Bayesian_TPE', 'trial_number': t.number, 'allowshort': bool(params.get('allowshort', False)), 'optim_params': optim_params_json, 'composite_score': round(t.value, 3), 'params': json.dumps(params, separators=(',', ':'), default=str), 'IS_start': IS_START, 'IS_end': IS_END, 'OOS_start': OOS_START, 'OOS_end': OOS_END, 'FULL_start': FULL_START, 'FULL_end': FULL_END}
        for pname, pval in params.items():
            row[f'param_{pname}'] = pval
        row['metric_return_%'] = is_res['return_pct']
        row['metric_sharpe'] = is_res['sharpe']
        row['metric_sortino'] = is_res['sortino']
        row['metric_maxdd_%'] = is_res['max_dd_pct']
        row['metric_n_trades'] = is_res['n_trades']
        row['metric_win_rate_%'] = is_res['win_rate']
        row.update(_period_stats_flat(is_res, 'IS_'))
        row.update(_period_stats_flat(oos_res, 'OOS_'))
        row.update(_period_stats_flat(full_res, 'FULL_'))
        all_trial_rows.append(row)
    all_trials_df = pd.DataFrame(all_trial_rows)
    top5 = []
    if not all_trials_df.empty:
        sorted_df = all_trials_df.sort_values('composite_score', ascending=False).head(5)
        for rank, (_, r) in enumerate(sorted_df.iterrows(), 1):
            entry = r.to_dict()
            entry['rank'] = rank
            entry.pop('trial_number', None)
            for k in list(entry.keys()):
                if k.startswith('param_') or k.startswith('metric_'):
                    entry.pop(k, None)
            top5.append(entry)
    elapsed = time.time() - t0
    if verbose and top5:
        best = top5[0]
        print(f"    Best: score={best['composite_score']}, IS_ret%={best.get('IS_Процент роста')}, OOS_ret%={best.get('OOS_Процент роста')}, FULL_ret%={best.get('FULL_Процент роста')}, {elapsed:.0f}s")
    return (top5, all_trials_df)

def run_for_type(stype, tickers, n_trials=50, timeframe='1D'):
    from optimization_utils import save_topn, merge_append
    all_top = []
    all_trials_frames = []
    for ticker in tickers:
        top5, trials_df = optimize_ticker_type_with_trials(stype, ticker, n_trials=n_trials, timeframe=timeframe)
        all_top.extend(top5)
        if not trials_df.empty:
            all_trials_frames.append(trials_df)
    os.makedirs(OUT_DIR, exist_ok=True)
    df_top = pd.DataFrame(all_top)
    top_file = os.path.join(OUT_DIR, f'{stype}_top5.csv')
    save_topn(all_top, top_file)
    if all_trials_frames:
        df_all = pd.concat(all_trials_frames, ignore_index=True)
        trials_file = os.path.join(OUT_DIR, f'{stype}_all_trials.csv')
        df_out = merge_append(trials_file, df_all)
        df_out.to_csv(trials_file, index=False)
        print(f'  Сохранено: {trials_file} (+{len(df_all)}, всего {len(df_out)} trials)')
    return df_top

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='all', help=f"Тип стратегии или all. Варианты: {','.join(STRATEGY_TYPES)}")
    parser.add_argument('--ticker', type=str, default=None, help='Один тикер (иначе все)')
    parser.add_argument('--trials', type=int, default=50, help='Optuna trials per (ticker,type)')
    parser.add_argument('--timeframe', type=str, default='1D', help="TF для прогона оптимизации: '1D' или '1h'")
    args = parser.parse_args()
    tickers = [args.ticker] if args.ticker else TICKERS
    types = STRATEGY_TYPES if args.type == 'all' else [args.type]
    if args.type != 'all' and args.type not in STRATEGY_TYPES:
        print(f'Неверный тип: {args.type}. Допустимо: {STRATEGY_TYPES}')
        return
    all_dfs = []
    for stype in types:
        print(f'\n=== Оптимизация {stype} [{args.timeframe}] ===')
        df = run_for_type(stype, tickers, n_trials=args.trials, timeframe=args.timeframe)
        all_dfs.append(df)
    if len(all_dfs) > 1:
        from optimization_utils import save_topn
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_file = os.path.join(OUT_DIR, 'all_top5.csv')
        save_topn(combined.to_dict('records'), combined_file)
if __name__ == '__main__':
    main()
