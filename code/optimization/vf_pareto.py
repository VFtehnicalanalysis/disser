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
from strategies.ema_longterm import LongTermHold
from reports.vf_report import generate_vf_report, export_vf_csv
from .vf_optimizer_v2 import DEFAULT_FIXED_PARAMS

def select_pareto_point(pareto_df: pd.DataFrame, strategy: str='balanced') -> pd.Series:
    if pareto_df.empty:
        raise ValueError('Pareto frontier пуст')
    if strategy == 'best_sharpe':
        return pareto_df.sort_values('avg_sharpe', ascending=False).iloc[0]
    elif strategy == 'best_dd':
        return pareto_df.sort_values('avg_maxdd').iloc[0]
    elif strategy == 'best_stable':
        return pareto_df.sort_values('std_sharpe').iloc[0]
    elif strategy == 'balanced':
        df = pareto_df.copy()
        sh_max = df['avg_sharpe'].max()
        sh_min = df['avg_sharpe'].min()
        df['sh_loss'] = (sh_max - df['avg_sharpe']) / (sh_max - sh_min + 1e-09)
        dd_max = df['avg_maxdd'].max()
        dd_min = df['avg_maxdd'].min()
        df['dd_loss'] = (df['avg_maxdd'] - dd_min) / (dd_max - dd_min + 1e-09)
        st_max = df['std_sharpe'].max()
        st_min = df['std_sharpe'].min()
        df['st_loss'] = (df['std_sharpe'] - st_min) / (st_max - st_min + 1e-09)
        df['total_loss'] = df['sh_loss'] + df['dd_loss'] + df['st_loss']
        return df.sort_values('total_loss').iloc[0]
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

def apply_params(ticker: str, timeframe: str, params: dict, name: str) -> MyStrategy:
    htf = '1W' if timeframe == '1D' else '1D'
    full = dict(DEFAULT_FIXED_PARAMS)
    full.update(params)
    full['check_info'] = 'BUY & SELL'
    full['is_using_trend_analysis'] = True
    full['is_rsi_changer'] = False
    full['rsi_changer'] = 0
    full['is_using_stops'] = False
    full['is_using_take_profits'] = False
    full['is_using_trailing_stop'] = False
    full['is_price_step'] = False
    full['stop_l_percent'] = 3.0
    full['stop_s_percent'] = 1.0
    full['TP_l_percent'] = 6.0
    full['TP_s_percent'] = 3.0
    full['TRS_percent'] = 3.0
    full['price_step'] = 0.1
    if 'entry_percentage' not in full or pd.isna(full.get('entry_percentage', np.nan)):
        full['entry_percentage'] = round(100.0 / full.get('pyramiding', 2), 2)
    full.pop('initial_account_size', None)
    for k in ['trial', 'is_pareto', 'avg_sharpe', 'avg_maxdd', 'std_sharpe', 'sharpe', 'maxdd', 'sh_loss', 'dd_loss', 'st_loss', 'total_loss']:
        full.pop(k, None)
    int_params = ['rsi_period', 'rsi_low', 'rsi_high', 'pivot_period', 'L_short_period', 'L_long_period', 'H_short_period', 'H_long_period', 'sell_count', 'pyramiding', 'macd_fast', 'macd_slow', 'macd_signal', 'rsi_changer']
    for k in int_params:
        if k in full and (not pd.isna(full[k])):
            full[k] = int(full[k])
    bool_params = ['is_using_macd', 'is_smart_stop_activated', 'allowshort', 'is_using_trend_analysis', 'is_using_stops', 'is_using_take_profits', 'is_using_trailing_stop', 'is_price_step', 'is_rsi_changer']
    for k in bool_params:
        if k in full and (not pd.isna(full[k])):
            full[k] = bool(full[k])
    print(f'\n>> {name}: timeframe={timeframe}, htf={htf}')
    print(f"   Params: rsi_p={full['rsi_period']}, rsi[{full['rsi_low']},{full['rsi_high']}], piv={full['pivot_period']}, L[{full['L_short_period']}/{full['L_long_period']}], H[{full['H_short_period']}/{full['H_long_period']}], sell={full['sell_count']}, pyr={full['pyramiding']}×{full['entry_percentage']}%, MACD={full['is_using_macd']}, SS={full['is_smart_stop_activated']}, short={full['allowshort']}")
    strat = MyStrategy(name=name, ticker=ticker, timeframe=timeframe, htf=htf, initial_account_size=1000000, **full)
    strat.load_data()
    strat.generate_signals()
    strat.execute_trades()
    strat.finalize()
    s = strat.get_stats()
    print(f"   → Trades: {s.get('Количество сделок', 0)}, Win%: {s.get('Процент прибыльных сделок', 0)}, Profit: {s.get('Процент роста', 0)}%, MaxDD: {s.get('Макс. просадка (%)', 0)}%, Sharpe: {s.get('Sharpe', '—')}")
    return strat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--output', default='results/vf_strategy')
    parser.add_argument('--strategies', nargs='+', default=['best_sharpe', 'best_dd', 'best_stable', 'balanced'], help='Какие стратегии выбора Pareto-точки применить')
    args = parser.parse_args()
    pareto_path = os.path.join(args.output, f'{args.ticker}_v2_pareto_{args.timeframe}.csv')
    if not os.path.exists(pareto_path):
        print(f'Файл не найден: {pareto_path}')
        print(f'Запустите сначала: python vf_optimizer_v2.py {args.ticker} {args.timeframe}')
        return
    pareto_df = pd.read_csv(pareto_path)
    print(f'Loaded {len(pareto_df)} Pareto-оптимальных решений из {pareto_path}\n')
    v1_path = os.path.join(args.output, f'{args.ticker}_optimization_{args.timeframe}.json')
    strategies = []
    if os.path.exists(v1_path):
        with open(v1_path, 'r', encoding='utf-8') as f:
            v1 = json.load(f)
        if v1['folds']:
            v1_params = v1['folds'][-1]['best_params']
            print(f'=== V1 BASELINE (single-objective TPE из vf_optimizer.py) ===')
            v1_strat = apply_params(args.ticker, args.timeframe, v1_params, 'V1_optimized_TPE')
            strategies.append(v1_strat)
    print(f'\n=== V2 PARETO-OPTIMAL POINTS (NSGA-II) ===')
    selected_points = {}
    for sel_strategy in args.strategies:
        try:
            point = select_pareto_point(pareto_df, sel_strategy)
            selected_points[sel_strategy] = point
        except Exception as e:
            print(f"Не удалось выбрать '{sel_strategy}': {e}")
    for sel_strategy, point in selected_points.items():
        print(f"\n--- Pareto strategy: '{sel_strategy}' ---")
        print(f"Trial #{int(point['trial'])}: avg_Sharpe={point['avg_sharpe']:.3f}, avg_MaxDD={point['avg_maxdd']:.2f}%, std_Sharpe={point['std_sharpe']:.3f}")
        params = {k: v for k, v in point.items() if k not in ['trial', 'is_pareto', 'avg_sharpe', 'avg_maxdd', 'std_sharpe', 'sh_loss', 'dd_loss', 'st_loss', 'total_loss'] and (not pd.isna(v))}
        strat = apply_params(args.ticker, args.timeframe, params, f'V2_pareto_{sel_strategy}')
        strategies.append(strat)
    print(f'\n=== BENCHMARK ===')
    bh = LongTermHold(args.ticker, '1D', 1000000, 100)
    bh.generate_signals()
    bh.execute_trades()
    bh.equity_df = bh.get_equity_curve()
    s = bh.get_stats()
    print(f">> LongTermHold: Profit={s.get('Процент роста', 0)}%, MaxDD={s.get('Макс. просадка (%)', 0)}%, Sharpe={s.get('Sharpe', '—')}")
    strategies.append(bh)
    print(f'\n=== ЭКСПОРТ ===')
    export_vf_csv(args.ticker, strategies, output_dir=args.output)
    generate_vf_report(args.ticker, strategies, output_dir=args.output, report_suffix=f'vf_v2_pareto_{args.timeframe}', price_data_tf='1h')
    print(f"\n{'=' * 90}")
    print(f'ИТОГОВОЕ СРАВНЕНИЕ ({args.ticker} {args.timeframe})')
    print(f"{'=' * 90}")
    print(f"{'Стратегия':<28} {'TF':<5} {'Trades':>7} {'Win%':>6} {'Profit%':>9} {'MaxDD%':>8} {'Sharpe':>7}")
    print('-' * 80)
    for s in strategies:
        st = s.get_stats()
        print(f"{s.name:<28} {s.timeframe:<5} {st.get('Количество сделок', 0):>7} {st.get('Процент прибыльных сделок', 0):>6.1f} {st.get('Процент роста', 0):>9.1f} {st.get('Макс. просадка (%)', 0):>8.1f} {str(st.get('Sharpe', '—')):>7}")
if __name__ == '__main__':
    main()
