import os
import sys
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
from strategies.vf import MyStrategy
from .vf_optimizer_v2 import DEFAULT_FIXED_PARAMS
from .vf_pareto import apply_params

def select_top_n_pareto(pareto_df: pd.DataFrame, top_n: int=5) -> pd.DataFrame:
    df = pareto_df.copy()
    df = df[df['avg_maxdd'] <= 50.0]
    df = df[df['std_sharpe'] < 1.0]
    df = df[df['avg_sharpe'] > 0]
    df = df.sort_values('avg_sharpe', ascending=False).head(top_n)
    return df

def run_pareto_strategies(ticker: str, timeframe: str, pareto_subset: pd.DataFrame) -> list:
    strategies = []
    for i, (_, point) in enumerate(pareto_subset.iterrows(), 1):
        params = {k: v for k, v in point.items() if k not in ['trial', 'is_pareto', 'avg_sharpe', 'avg_maxdd', 'std_sharpe', 'sharpe', 'maxdd', 'sh_loss', 'dd_loss', 'st_loss', 'total_loss'] and (not pd.isna(v))}
        try:
            strat = apply_params(ticker, timeframe, params, f"VF_pareto#{int(point['trial'])}_top{i}")
            strat._opt_avg_sharpe = float(point['avg_sharpe'])
            strat._opt_avg_maxdd = float(point['avg_maxdd'])
            strategies.append(strat)
        except Exception as e:
            print(f"  Skipped trial #{int(point['trial'])}: {e}")
    return strategies

def build_ensemble_equity(strategies: list, weights: np.ndarray, initial_capital: float=1000000) -> pd.DataFrame:
    eq_dfs = []
    for s in strategies:
        if not hasattr(s, 'equity_df') or s.equity_df is None or s.equity_df.empty:
            continue
        eq = s.equity_df.copy()
        eq_dfs.append(eq.rename(columns={'equity': s.name}))
    if not eq_dfs:
        return pd.DataFrame()
    combined = eq_dfs[0]
    for df in eq_dfs[1:]:
        combined = pd.merge(combined, df, left_index=True, right_index=True, how='outer')
    combined = combined.sort_index()
    combined = combined.ffill()
    combined = combined.bfill()
    norm = combined / initial_capital
    if len(weights) != norm.shape[1]:
        raise ValueError(f'Weights ({len(weights)}) != strategies ({norm.shape[1]})')
    weighted = (norm.values * weights).sum(axis=1)
    ensemble_equity = weighted * initial_capital
    return pd.DataFrame({'equity': ensemble_equity}, index=combined.index)

def compute_ensemble_stats(equity_df: pd.DataFrame, timeframe: str, initial_capital: float=1000000) -> dict:
    if equity_df.empty:
        return {}
    eq = equity_df['equity']
    final = eq.iloc[-1]
    profit_pct = (final - initial_capital) / initial_capital * 100
    running_max = eq.cummax()
    drawdowns = (running_max - eq) / running_max * 100
    max_dd = float(drawdowns.max())
    returns = eq.pct_change().dropna()
    if timeframe == '1D':
        scale = np.sqrt(252)
    elif timeframe == '1h':
        scale = np.sqrt(252 * 6.5)
    else:
        scale = 1.0
    sharpe = returns.mean() / returns.std() * scale if returns.std() > 0 else float('nan')
    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * scale if downside.std() > 0 else float('nan')
    return {'Final equity': round(float(final), 2), 'Profit %': round(float(profit_pct), 2), 'MaxDD %': round(max_dd, 2), 'Sharpe': round(float(sharpe), 3) if not pd.isna(sharpe) else None, 'Sortino': round(float(sortino), 3) if not pd.isna(sortino) else None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--output', default='results/vf_strategy')
    args = parser.parse_args()
    pareto_path = os.path.join(args.output, f'{args.ticker}_v2_pareto_{args.timeframe}.csv')
    if not os.path.exists(pareto_path):
        print(f'Файл не найден: {pareto_path}')
        return
    pareto_df = pd.read_csv(pareto_path)
    print(f'Loaded {len(pareto_df)} Pareto-точек из {pareto_path}')
    top_n = select_top_n_pareto(pareto_df, args.top)
    print(f'\nВыбрано top-{len(top_n)} Pareto-точек (фильтр: maxdd≤50%, std<1.0, sharpe>0):')
    for _, p in top_n.iterrows():
        print(f"  Trial #{int(p['trial']):>4}  avgSh={p['avg_sharpe']:.3f}  avgDD={p['avg_maxdd']:>5.2f}%  stdSh={p['std_sharpe']:.3f}")
    print(f"\n{'=' * 70}\nЗАПУСК ИНДИВИДУАЛЬНЫХ СТРАТЕГИЙ\n{'=' * 70}")
    strategies = run_pareto_strategies(args.ticker, args.timeframe, top_n)
    if not strategies:
        print('Нет валидных стратегий')
        return
    print(f"\n{'=' * 70}\nПОСТРОЕНИЕ АНСАМБЛЕЙ\n{'=' * 70}")
    n = len(strategies)
    eq_weights = np.array([1.0 / n] * n)
    eq_ensemble_equity = build_ensemble_equity(strategies, eq_weights)
    eq_stats = compute_ensemble_stats(eq_ensemble_equity, args.timeframe)
    print(f'\nEqual-weighted ({n} стратегий, веса по 1/{n}):')
    for k, v in eq_stats.items():
        print(f'  {k}: {v}')
    sharpes = np.array([s._opt_avg_sharpe for s in strategies])
    sharpes = np.clip(sharpes, 0.01, None)
    sw_weights = sharpes / sharpes.sum()
    sw_ensemble_equity = build_ensemble_equity(strategies, sw_weights)
    sw_stats = compute_ensemble_stats(sw_ensemble_equity, args.timeframe)
    print(f'\nSharpe-weighted (веса = avg_sharpe / sum):')
    for s, w in zip(strategies, sw_weights):
        print(f'  {s.name}: weight={w:.3f} (avgSh={s._opt_avg_sharpe:.3f})')
    for k, v in sw_stats.items():
        print(f'  {k}: {v}')
    out_csv = os.path.join(args.output, f'{args.ticker}_ensemble_equity_{args.timeframe}.csv')
    combined = pd.merge(eq_ensemble_equity.rename(columns={'equity': 'EqualWeighted'}), sw_ensemble_equity.rename(columns={'equity': 'SharpeWeighted'}), left_index=True, right_index=True, how='outer')
    for s in strategies:
        if s.equity_df is not None and (not s.equity_df.empty):
            combined = pd.merge(combined, s.equity_df[['equity']].rename(columns={'equity': s.name}), left_index=True, right_index=True, how='outer')
    combined.to_csv(out_csv, encoding='utf-8-sig')
    print(f'\nSaved ensemble equity: {out_csv}')
    summary_rows = []
    for s in strategies:
        st = s.get_stats()
        summary_rows.append({'Strategy': s.name, 'Trades': st.get('Количество сделок', 0), 'Win%': st.get('Процент прибыльных сделок', 0), 'Profit%': st.get('Процент роста', 0), 'MaxDD%': st.get('Макс. просадка (%)', 0), 'Sharpe': st.get('Sharpe', None), 'Type': 'Individual'})
    summary_rows.append({'Strategy': f'Ensemble_Equal_{n}', 'Trades': '-', 'Win%': '-', **{f'{k}': v for k, v in [('Profit%', eq_stats.get('Profit %')), ('MaxDD%', eq_stats.get('MaxDD %')), ('Sharpe', eq_stats.get('Sharpe'))]}, 'Type': 'Ensemble'})
    summary_rows.append({'Strategy': f'Ensemble_Sharpe_{n}', 'Trades': '-', 'Win%': '-', **{f'{k}': v for k, v in [('Profit%', sw_stats.get('Profit %')), ('MaxDD%', sw_stats.get('MaxDD %')), ('Sharpe', sw_stats.get('Sharpe'))]}, 'Type': 'Ensemble'})
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(args.output, f'{args.ticker}_ensemble_summary_{args.timeframe}.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f'Saved summary: {summary_csv}')
    print(f"\n{'=' * 70}\nИТОГОВОЕ СРАВНЕНИЕ\n{'=' * 70}")
    for _, row in summary_df.iterrows():
        print(f"  {row['Strategy']:<28} Profit={row['Profit%']:>7}, MaxDD={row['MaxDD%']:>6}, Sharpe={row['Sharpe']}")
if __name__ == '__main__':
    main()
