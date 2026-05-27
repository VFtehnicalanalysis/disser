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
from strategies.vf import MyStrategy
from strategies.ema_longterm import LongTermHold
from reports.vf_report import generate_vf_report, export_vf_csv
BASELINE_1D = dict(timeframe='1D', pyramiding=2, allowshort=True, entry_percentage=50.0, htf='1W', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=True, macd_fast=12, macd_slow=26, macd_signal=9, is_using_trend_analysis=True, check_info='BUY & SELL', is_smart_stop_activated=True, smart_stop=5.0, smart_stop_activation=10.0, is_rsi_changer=False, rsi_changer=0, sell_count=5, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, is_price_step=False)
BASELINE_1H = dict(timeframe='1h', pyramiding=2, allowshort=True, entry_percentage=50.0, htf='1D', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=True, macd_fast=12, macd_slow=26, macd_signal=9, is_using_trend_analysis=True, check_info='BUY & SELL', is_smart_stop_activated=True, smart_stop=5.0, smart_stop_activation=10.0, is_rsi_changer=False, rsi_changer=0, sell_count=5, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, is_price_step=False)

def build_full_params(best_from_optuna: dict, timeframe: str) -> dict:
    base = BASELINE_1D if timeframe == '1D' else BASELINE_1H
    htf_value = '1W' if timeframe == '1D' else '1D'
    full = dict(base)
    full.update(best_from_optuna)
    full['timeframe'] = timeframe
    full['htf'] = htf_value
    full['initial_account_size'] = 1000000
    full['entry_percentage'] = 50.0
    full['check_info'] = 'BUY & SELL'
    full['is_using_trend_analysis'] = True
    full['is_using_stops'] = False
    full['is_using_take_profits'] = False
    full['is_using_trailing_stop'] = False
    full['is_price_step'] = False
    full['is_rsi_changer'] = False
    full['rsi_changer'] = 0
    return full

def run_strategy(name: str, ticker: str, params: dict) -> MyStrategy:
    print(f"\n>> {name}: timeframe={params.get('timeframe')}, htf={params.get('htf')}")
    p = dict(params)
    p.pop('initial_account_size', None)
    strat = MyStrategy(name=name, ticker=ticker, initial_account_size=1000000, **p)
    strat.load_data()
    strat.generate_signals()
    strat.execute_trades()
    strat.finalize()
    s = strat.get_stats()
    print(f"   Trades: {s.get('Количество сделок', 0)}, Win%: {s.get('Процент прибыльных сделок', 0)}, Profit: {s.get('Процент роста', 0)}%, MaxDD: {s.get('Макс. просадка (%)', 0)}%, Sharpe: {s.get('Sharpe', '—')}")
    return strat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('--output', default='results/vf_strategy')
    args = parser.parse_args()
    output_dir = args.output
    strategies = []
    json_1d = os.path.join(output_dir, f'{args.ticker}_optimization_1D.json')
    if os.path.exists(json_1d):
        with open(json_1d, 'r', encoding='utf-8') as f:
            opt_1d = json.load(f)
        if opt_1d['folds']:
            best_1d = opt_1d['folds'][-1]['best_params']
            print(f'\n=== 1D BASELINE vs OPTIMIZED ===')
            baseline_1d_strat = run_strategy('VF_D_baseline', args.ticker, BASELINE_1D)
            strategies.append(baseline_1d_strat)
            optimized_1d_strat = run_strategy('VF_D_optimized', args.ticker, build_full_params(best_1d, '1D'))
            strategies.append(optimized_1d_strat)
        else:
            print(f'⚠ 1D оптимизация дала 0 fold-результатов — пропускаем')
    else:
        print(f'⚠ Не найден файл оптимизации 1D: {json_1d}')
    json_1h = os.path.join(output_dir, f'{args.ticker}_optimization_1h.json')
    if os.path.exists(json_1h):
        with open(json_1h, 'r', encoding='utf-8') as f:
            opt_1h = json.load(f)
        if opt_1h['folds']:
            best_1h = opt_1h['folds'][-1]['best_params']
            print(f'\n=== 1h BASELINE vs OPTIMIZED ===')
            baseline_1h_strat = run_strategy('VF_H_baseline', args.ticker, BASELINE_1H)
            strategies.append(baseline_1h_strat)
            optimized_1h_strat = run_strategy('VF_H_optimized', args.ticker, build_full_params(best_1h, '1h'))
            strategies.append(optimized_1h_strat)
        else:
            print(f'⚠ 1h оптимизация дала 0 fold-результатов — пропускаем')
    else:
        print(f'⚠ Не найден файл оптимизации 1h: {json_1h}')
    print(f'\n=== BENCHMARK ===')
    bh = LongTermHold(args.ticker, '1D', 1000000, 100)
    bh.generate_signals()
    bh.execute_trades()
    bh.equity_df = bh.get_equity_curve()
    s = bh.get_stats()
    print(f">> LongTermHold: Profit: {s.get('Процент роста', 0)}%, MaxDD: {s.get('Макс. просадка (%)', 0)}%, Sharpe: {s.get('Sharpe', '—')}")
    strategies.append(bh)
    print(f'\n=== ЭКСПОРТ ===')
    export_vf_csv(args.ticker, strategies, output_dir=output_dir)
    generate_vf_report(args.ticker, strategies, output_dir=output_dir, report_suffix='vf_optimized_comparison', price_data_tf='1h')
    print(f"\n{'=' * 80}")
    print(f'ИТОГОВОЕ СРАВНЕНИЕ (baseline vs optimized)')
    print(f"{'=' * 80}")
    print(f"{'Стратегия':<22} {'TF':<5} {'Trades':>7} {'Win%':>6} {'Profit%':>9} {'MaxDD%':>8} {'Sharpe':>7}")
    print('-' * 75)
    for s in strategies:
        st = s.get_stats()
        print(f"{s.name:<22} {s.timeframe:<5} {st.get('Количество сделок', 0):>7} {st.get('Процент прибыльных сделок', 0):>6.1f} {st.get('Процент роста', 0):>9.1f} {st.get('Макс. просадка (%)', 0):>8.1f} {str(st.get('Sharpe', '—')):>7}")
if __name__ == '__main__':
    main()
