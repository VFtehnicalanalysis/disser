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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from strategies.vf import MyStrategy
from optimization.vf_optimizer import run_strategy_on_slice
from optimization.vf_optimizer_v2 import DEFAULT_FIXED_PARAMS
from optimization.vf_pareto import apply_params as _apply_full
SPLIT_DATE = '2025-01-01'

def load_best_params(ticker: str, timeframe: str, output_dir: str) -> dict:
    result = {}
    pareto = f'{output_dir}/{ticker}_v2_pareto_{timeframe}.csv'
    if os.path.exists(pareto):
        df = pd.read_csv(pareto)
        sort_col = 'avg_sharpe' if 'avg_sharpe' in df.columns else 'sharpe'
        df = df.sort_values(sort_col, ascending=False)
        if not df.empty:
            row = df.iloc[0]
            p = {k: v for k, v in row.items() if k not in ['trial', 'is_pareto', 'avg_sharpe', 'avg_maxdd', 'std_sharpe', 'sharpe', 'maxdd'] and (not pd.isna(v))}
            result['NSGA-II_V2'] = p
    bayes = f'{output_dir}/{ticker}_bayesian_full_{timeframe}.json'
    if os.path.exists(bayes):
        with open(bayes, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result['Bayesian'] = data.get('best_params')
    grad = f'{output_dir}/{ticker}_gradient_full_{timeframe}.json'
    if os.path.exists(grad):
        with open(grad, 'r', encoding='utf-8') as f:
            data = json.load(f)
        result['Gradient'] = data.get('best_params')
    return result

def load_ticker_data(ticker: str, timeframe: str) -> pd.DataFrame:
    if timeframe == '1D':
        file = f'data/{ticker}_data_new.csv'
    else:
        file = f'data/{ticker}_hourly_data_new.csv'
    df = pd.read_csv(file)
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df.sort_index(inplace=True)
    return df

def prepare_params_for_run(params: dict) -> dict:
    p = dict(params)
    for k in ['trial', 'is_pareto', 'avg_sharpe', 'avg_maxdd', 'std_sharpe', 'sharpe', 'maxdd', 'L_long_offset', 'H_long_offset']:
        p.pop(k, None)
    p.setdefault('macd_fast', 12)
    p.setdefault('macd_slow', 26)
    p.setdefault('macd_signal', 9)
    if 'entry_percentage' not in p:
        p['entry_percentage'] = round(100.0 / p.get('pyramiding', 2), 2)
    p.setdefault('is_using_stops', False)
    p.setdefault('is_using_take_profits', False)
    p.setdefault('is_using_trailing_stop', False)
    p.setdefault('is_price_step', False)
    p.setdefault('is_using_trend_analysis', True)
    p.setdefault('check_info', 'BUY & SELL')
    p.setdefault('is_rsi_changer', False)
    p.setdefault('rsi_changer', 0)
    for k in ['rsi_period', 'rsi_low', 'rsi_high', 'pivot_period', 'L_short_period', 'L_long_period', 'H_short_period', 'H_long_period', 'sell_count', 'pyramiding', 'macd_fast', 'macd_slow', 'macd_signal', 'rsi_changer']:
        if k in p and (not pd.isna(p[k])):
            p[k] = int(p[k])
    for k in ['is_using_macd', 'is_smart_stop_activated', 'allowshort', 'is_using_trend_analysis', 'is_using_stops', 'is_using_take_profits', 'is_using_trailing_stop', 'is_price_step', 'is_rsi_changer']:
        if k in p and (not pd.isna(p[k])):
            p[k] = bool(p[k])
    if p.get('L_long_period', 0) <= p.get('L_short_period', 0):
        p['L_long_period'] = p.get('L_short_period', 100) + 50
    if p.get('H_long_period', 0) <= p.get('H_short_period', 0):
        p['H_long_period'] = p.get('H_short_period', 100) + 50
    return p

def run_on_slice(ticker: str, timeframe: str, params: dict, data_slice: pd.DataFrame, htf: str, label: str) -> dict:
    p = prepare_params_for_run(params)
    stats, strat = run_strategy_on_slice(ticker, timeframe, p, data_slice, htf)
    if stats is None:
        return None
    print(f"    {label}: trades={stats.get('Количество сделок', 0)}, win={stats.get('Процент прибыльных сделок', 0)}%, profit={stats.get('Процент роста', 0)}%, maxdd={stats.get('Макс. просадка (%)', 0)}%, sharpe={stats.get('Sharpe', '—')}")
    return {'stats': stats, 'strat': strat}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--split', default=SPLIT_DATE)
    parser.add_argument('--output', default='results/vf_strategy')
    args = parser.parse_args()
    htf = '1W' if args.timeframe == '1D' else '1D'
    split_ts = pd.Timestamp(args.split)
    print(f"\n{'#' * 70}")
    print(f'#  QUICK OOS CHECK — {args.ticker} {args.timeframe}')
    print(f'#  Split: {args.split}')
    print(f'#  IS: до {args.split}, OOS: с {args.split}')
    print(f"{'#' * 70}")
    df = load_ticker_data(args.ticker, args.timeframe)
    df_is = df[df.index < split_ts].copy()
    df_oos = df[df.index >= split_ts].copy()
    df_full = df.copy()
    print(f'\nIS: {len(df_is)} bars ({df_is.index[0].date()}..{df_is.index[-1].date()})')
    print(f'OOS: {len(df_oos)} bars ({df_oos.index[0].date()}..{df_oos.index[-1].date()})')
    best = load_best_params(args.ticker, args.timeframe, args.output)
    print(f'\nFound best params для методов: {list(best.keys())}')
    results = []
    for method, params in best.items():
        if params is None:
            continue
        print(f'\n>> {method}:')
        full = run_on_slice(args.ticker, args.timeframe, params, df_full, htf, 'FULL   ')
        is_ = run_on_slice(args.ticker, args.timeframe, params, df_is, htf, 'IS     ')
        oos = run_on_slice(args.ticker, args.timeframe, params, df_oos, htf, 'OOS    ')
        row = {'Method': method}
        for period, r in [('Full', full), ('IS', is_), ('OOS', oos)]:
            if r:
                s = r['stats']
                row[f'{period}_Trades'] = s.get('Количество сделок', 0)
                row[f'{period}_Win%'] = s.get('Процент прибыльных сделок', 0)
                row[f'{period}_Profit%'] = s.get('Процент роста', 0)
                row[f'{period}_MaxDD%'] = s.get('Макс. просадка (%)', 0)
                row[f'{period}_Sharpe'] = s.get('Sharpe', None)
            else:
                for m in ['Trades', 'Win%', 'Profit%', 'MaxDD%', 'Sharpe']:
                    row[f'{period}_{m}'] = None
        results.append(row)
    res_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output, f'{args.ticker}_oos_check_{args.timeframe}.csv')
    res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'\nSaved: {csv_path}')
    print(f"\n{'=' * 100}")
    print(f'QUICK OOS CHECK — СВОДКА (текущие params применены на IS/OOS раздельно)')
    print(f"{'=' * 100}")
    print(f"{'Method':<15} | {'Full Sh':>8} {'IS Sh':>7} {'OOS Sh':>7} | {'Full P%':>8} {'IS P%':>7} {'OOS P%':>7} | {'OOS Trades':>10}")
    print('-' * 100)
    for r in results:
        print(f"{r['Method']:<15} | {str(r.get('Full_Sharpe', '—')):>8} {str(r.get('IS_Sharpe', '—')):>7} {str(r.get('OOS_Sharpe', '—')):>7} | {r.get('Full_Profit%', 0):>8.1f} {r.get('IS_Profit%', 0):>7.1f} {r.get('OOS_Profit%', 0):>7.1f} | {r.get('OOS_Trades', 0):>10}")
    fig = make_subplots(rows=1, cols=3, subplot_titles=['Sharpe', 'Profit %', 'MaxDD %'])
    methods = [r['Method'] for r in results]
    for col_idx, (metric_name, metric_key) in enumerate([('Sharpe', 'Sharpe'), ('Profit %', 'Profit%'), ('MaxDD %', 'MaxDD%')], 1):
        for period in ['Full', 'IS', 'OOS']:
            values = [r.get(f'{period}_{metric_key}', 0) or 0 for r in results]
            fig.add_trace(go.Bar(x=methods, y=values, name=period, showlegend=col_idx == 1), row=1, col=col_idx)
    fig.update_layout(title=f'{args.ticker} {args.timeframe} — Quick OOS Check: Full vs IS vs OOS', height=450, barmode='group')
    res_html = res_df.round(2).to_html(index=False, classes='table table-striped', border=0)
    html = f"""\n<html><head><meta charset="utf-8">\n<title>{args.ticker} {args.timeframe} — Quick OOS Check</title>\n<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">\n<style>body {{ margin: 24px; }}\n.warn {{ background: #fff3cd; padding: 16px; border-left: 4px solid #ffc107;\n       border-radius: 4px; margin: 16px 0; }}</style>\n</head><body>\n<h1>{args.ticker} {args.timeframe} — Quick OOS Check</h1>\n<div class="warn">\n⚠ <strong>Это НЕ честная OOS валидация.</strong> Best params были оптимизированы\nна ВСЕХ данных (включая 2025-2026). Этот тест показывает как параметры "ретроспективно"\nработают на IS/OOS split — если OOS сильно хуже IS, это признак переобучения.\nДля честной OOS используй <code>vf_oos_reoptimize.py</code>.\n</div>\n<h2>Разделение данных</h2>\n<ul>\n  <li><strong>IS (In-Sample):</strong> до {args.split} ({len(df_is)} bars)</li>\n  <li><strong>OOS (Out-of-Sample):</strong> с {args.split} ({len(df_oos)} bars)</li>\n</ul>\n<h2>Сравнительная таблица</h2>\n{res_html}\n<h2>Визуализация</h2>\n{plot(fig, include_plotlyjs=False, output_type='div')}\n</body></html>\n"""
    html_path = os.path.join(args.output, f'{args.ticker}_oos_check_{args.timeframe}.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {html_path}')
if __name__ == '__main__':
    main()
