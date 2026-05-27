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
from strategies.ema_longterm import LongTermHold
from .vf_pareto import apply_params

def get_nsga2_best(ticker: str, timeframe: str, output_dir: str) -> dict:
    pareto_csv = f'{output_dir}/{ticker}_v2_pareto_{timeframe}.csv'
    if not os.path.exists(pareto_csv):
        return None
    df = pd.read_csv(pareto_csv)
    sort_col = 'avg_sharpe' if 'avg_sharpe' in df.columns else 'sharpe'
    df = df.sort_values(sort_col, ascending=False)
    if df.empty:
        return None
    row = df.iloc[0]
    params = {k: v for k, v in row.items() if k not in ['trial', 'is_pareto', 'avg_sharpe', 'avg_maxdd', 'std_sharpe', 'sharpe', 'maxdd'] and (not pd.isna(v))}
    return params

def get_bayesian_best(ticker: str, timeframe: str, output_dir: str) -> dict:
    json_path = f'{output_dir}/{ticker}_bayesian_full_{timeframe}.json'
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('best_params')

def get_gradient_best(ticker: str, timeframe: str, output_dir: str) -> dict:
    json_path = f'{output_dir}/{ticker}_gradient_full_{timeframe}.json'
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('best_params')

def run_method(ticker: str, timeframe: str, params: dict, name: str) -> MyStrategy:
    if params is None:
        print(f'\n>> {name}: параметры не найдены, пропускаем')
        return None
    p = dict(params)
    for k in ['L_long_offset', 'H_long_offset']:
        p.pop(k, None)
    try:
        return apply_params(ticker, timeframe, p, name)
    except Exception as e:
        print(f'\n>> {name}: ошибка {e}')
        return None

def make_convergence_plot(ticker: str, timeframe: str, output_dir: str) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Bayesian TPE — best_so_far', 'Gradient Descent — score по итерациям'])
    bayes_csv = f'{output_dir}/{ticker}_bayesian_full_{timeframe}.csv'
    if os.path.exists(bayes_csv):
        df = pd.read_csv(bayes_csv)
        df = df[df['best_so_far'] > -100000000.0]
        if not df.empty:
            fig.add_trace(go.Scatter(x=df['trial'], y=df['best_so_far'], mode='lines', line=dict(color='blue', width=2), name='TPE best_so_far', showlegend=True), row=1, col=1)
            fig.add_trace(go.Scatter(x=df['trial'], y=df['value'], mode='markers', marker=dict(color='lightblue', size=4, opacity=0.5), name='TPE trial score', showlegend=True), row=1, col=1)
        fig.update_xaxes(title_text='Trial #', row=1, col=1)
        fig.update_yaxes(title_text='Composite score', row=1, col=1)
    grad_csv = f'{output_dir}/{ticker}_gradient_full_{timeframe}.csv'
    if os.path.exists(grad_csv):
        df = pd.read_csv(grad_csv)
        df = df[df['score'] > -100000000.0]
        colors = ['red', 'orange', 'purple', 'green']
        for i, (start, grp) in enumerate(df.groupby('start')):
            fig.add_trace(go.Scatter(x=grp['iter'], y=grp['score'], mode='lines+markers', line=dict(color=colors[i % len(colors)]), name=f'Grad {start}'), row=1, col=2)
        fig.update_xaxes(title_text='Iteration', row=1, col=2)
        fig.update_yaxes(title_text='Composite score', row=1, col=2)
    fig.update_layout(title=f'{ticker} {timeframe} — Convergence Bayesian vs Gradient', height=450)
    return fig

def make_params_comparison_table(methods_params: dict) -> str:
    all_keys = set()
    for p in methods_params.values():
        if p:
            all_keys.update(p.keys())
    skip = {'trial', 'is_pareto', 'avg_sharpe', 'avg_maxdd', 'std_sharpe', 'sharpe', 'maxdd', 'L_long_offset', 'H_long_offset', 'macd_fast', 'macd_slow', 'macd_signal', 'entry_percentage', 'initial_account_size'}
    all_keys = sorted((k for k in all_keys if k not in skip))
    rows = []
    for k in all_keys:
        row = {'Параметр': k}
        for method, p in methods_params.items():
            if p and k in p:
                v = p[k]
                if isinstance(v, float):
                    v = round(v, 3)
                row[method] = v
            else:
                row[method] = '—'
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.to_html(index=False, classes='table table-striped table-sm', border=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--output', default='results/vf_strategy')
    args = parser.parse_args()
    output_dir = args.output
    ticker = args.ticker
    timeframe = args.timeframe
    methods_params = {'NSGA-II (V2)': get_nsga2_best(ticker, timeframe, output_dir), 'Bayesian TPE': get_bayesian_best(ticker, timeframe, output_dir), 'Gradient': get_gradient_best(ticker, timeframe, output_dir)}
    print(f"\n{'=' * 70}")
    print(f'СРАВНЕНИЕ МЕТОДОВ — {ticker} {timeframe}')
    print(f"{'=' * 70}")
    strategies = []
    for name, params in methods_params.items():
        strat = run_method(ticker, timeframe, params, name)
        if strat is not None:
            strategies.append(strat)
    print(f'\n>> Buy & Hold')
    bh = LongTermHold(ticker, '1D', 1000000, 100)
    bh.generate_signals()
    bh.execute_trades()
    bh.equity_df = bh.get_equity_curve()
    strategies.append(bh)
    print(f"\n{'=' * 85}")
    print(f"{'Метод':<22} {'Trades':>7} {'Win%':>6} {'Profit%':>9} {'MaxDD%':>8} {'Sharpe':>7} {'Sortino':>8}")
    print('-' * 85)
    summary_rows = []
    for s in strategies:
        st = s.get_stats()
        row = {'Метод': s.name, 'TF': s.timeframe, 'Trades': st.get('Количество сделок', 0), 'Win%': st.get('Процент прибыльных сделок', 0), 'Profit%': st.get('Процент роста', 0), 'MaxDD%': st.get('Макс. просадка (%)', 0), 'Sharpe': st.get('Sharpe', None), 'Sortino': st.get('Sortino', None)}
        summary_rows.append(row)
        print(f"{s.name:<22} {st.get('Количество сделок', 0):>7} {st.get('Процент прибыльных сделок', 0):>6.1f} {st.get('Процент роста', 0):>9.1f} {st.get('Макс. просадка (%)', 0):>8.1f} {str(st.get('Sharpe', '—')):>7} {str(st.get('Sortino', '—')):>8}")
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, f'{ticker}_methods_comparison_{timeframe}.csv')
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
    print(f'\nSaved: {summary_csv}')
    fig_conv = make_convergence_plot(ticker, timeframe, output_dir)
    fig_eq = go.Figure()
    initial = 1000000
    for s in strategies:
        if not hasattr(s, 'equity_df') or s.equity_df is None:
            continue
        eq = s.equity_df['equity']
        fig_eq.add_trace(go.Scatter(x=eq.index, y=eq.values, mode='lines', name=s.name, line=dict(width=2)))
    fig_eq.update_layout(title=f'{ticker} {timeframe} — Equity curves: comparison of methods', xaxis_title='Дата', yaxis_title='Equity (₽)', height=500, hovermode='x unified')
    params_table_html = make_params_comparison_table(methods_params)
    summary_table_html = summary_df.to_html(index=False, classes='table table-striped', border=0)
    html = f"""\n<html>\n<head>\n    <meta charset="utf-8">\n    <title>{ticker} {timeframe} — Comparison of Optimization Methods</title>\n    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">\n    <style>\n      body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}\n      h1 {{ color: #2c3e50; }}\n      h2 {{ margin-top: 32px; padding-bottom: 8px; border-bottom: 2px solid #ecf0f1; }}\n      .table td, .table th {{ font-size: 0.9rem; }}\n      .info {{ background-color: #f8f9fa; padding: 16px; border-radius: 8px;\n              margin: 16px 0; border-left: 4px solid #3498db; }}\n    </style>\n</head>\n<body>\n    <h1>Сравнение методов оптимизации — {ticker} {timeframe}</h1>\n\n    <div class="info">\n        <strong>Методы:</strong><br>\n        • <strong>NSGA-II (V2)</strong> — Multi-objective на reduced search space (8 параметров) с walk-forward (1h)<br>\n        • <strong>Bayesian TPE</strong> — Single-objective (composite Sharpe×Profit/MaxDD) на ВСЕХ 14 параметрах, warm-start из V2<br>\n        • <strong>Gradient Descent</strong> — Finite differences с importance-weighted LR, 3 стартовые точки<br>\n        • <strong>Buy & Hold</strong> — бенчмарк\n    </div>\n\n    <h2>1. Сводная таблица результатов (применение best params на полных данных)</h2>\n    {summary_table_html}\n\n    <h2>2. Convergence (сходимость оптимизации)</h2>\n    {plot(fig_conv, include_plotlyjs=False, output_type='div')}\n\n    <h2>3. Equity curves (реальный backtest)</h2>\n    {plot(fig_eq, include_plotlyjs=False, output_type='div')}\n\n    <h2>4. Какие параметры выбрал каждый метод</h2>\n    {params_table_html}\n</body>\n</html>\n"""
    out_path = os.path.join(output_dir, f'{ticker}_methods_comparison_{timeframe}.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {out_path}')
if __name__ == '__main__':
    main()
