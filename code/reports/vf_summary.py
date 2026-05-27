import os
import sys
import json
import argparse
import glob
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import plot

def load_ticker_results(ticker: str, timeframe: str, output_dir: str) -> dict:
    res = {'ticker': ticker, 'timeframe': timeframe}
    imp = os.path.join(output_dir, f'{ticker}_importance_{timeframe}.csv')
    if os.path.exists(imp):
        res['importance'] = pd.read_csv(imp)
    par = os.path.join(output_dir, f'{ticker}_v2_pareto_{timeframe}.csv')
    if os.path.exists(par):
        res['pareto'] = pd.read_csv(par)
    all_t = os.path.join(output_dir, f'{ticker}_v2_all_trials_{timeframe}.csv')
    if os.path.exists(all_t):
        res['all_trials'] = pd.read_csv(all_t)
    ens = os.path.join(output_dir, f'{ticker}_ensemble_summary_{timeframe}.csv')
    if os.path.exists(ens):
        res['ensemble'] = pd.read_csv(ens)
    return res

def build_metrics_matrix(all_results: list) -> pd.DataFrame:
    rows = []
    for r in all_results:
        if 'ensemble' not in r:
            continue
        for _, e in r['ensemble'].iterrows():
            rows.append({'Ticker': r['ticker'], 'Timeframe': r['timeframe'], 'Strategy': e.get('Strategy', '?'), 'Type': e.get('Type', '?'), 'Profit %': e.get('Profit%'), 'MaxDD %': e.get('MaxDD%'), 'Sharpe': e.get('Sharpe'), 'Trades': e.get('Trades'), 'Win %': e.get('Win%')})
    return pd.DataFrame(rows)

def make_heatmap_metric(df: pd.DataFrame, metric: str, timeframe: str) -> go.Figure:
    sub = df[df['Timeframe'] == timeframe].copy()
    if sub.empty:
        return go.Figure()
    sub_ens = sub[sub['Type'] == 'Ensemble']
    if sub_ens.empty:
        sub_ens = sub
    pv = sub_ens.pivot_table(index='Ticker', columns='Strategy', values=metric, aggfunc='first')
    pv = pv.round(2)
    if metric in ['MaxDD %', 'std_sharpe']:
        colorscale = 'RdYlGn_r'
    else:
        colorscale = 'RdYlGn'
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale=colorscale, text=pv.values, texttemplate='%{text}', colorbar=dict(title=metric)))
    fig.update_layout(title=f'{metric} ({timeframe}): тикеры × стратегии', height=max(300, 30 * len(pv.index) + 100))
    return fig

def make_importance_comparison(all_results: list, timeframe: str) -> go.Figure:
    rows = []
    for r in all_results:
        if 'importance' not in r or r['timeframe'] != timeframe:
            continue
        for _, imp in r['importance'].iterrows():
            rows.append({'Ticker': r['ticker'], 'Param': imp['param'], 'Importance': imp['importance']})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    pv = df.pivot_table(index='Ticker', columns='Param', values='Importance', aggfunc='first')
    pv = pv.round(3)
    col_order = pv.mean().sort_values(ascending=False).index
    pv = pv[col_order]
    fig = go.Figure(go.Heatmap(z=pv.values, x=pv.columns, y=pv.index, colorscale='Viridis', text=pv.values, texttemplate='%{text:.2f}', colorbar=dict(title='Importance')))
    fig.update_layout(title=f'Importance параметров ({timeframe}): тикеры × параметры', height=max(300, 30 * len(pv.index) + 100))
    return fig

def make_pareto_size_chart(all_results: list) -> go.Figure:
    rows = []
    for r in all_results:
        n_pareto = len(r.get('pareto', []))
        n_total = len(r.get('all_trials', []))
        rows.append({'Ticker': r['ticker'], 'Timeframe': r['timeframe'], 'Pareto size': n_pareto, 'Total trials': n_total})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for tf in df['Timeframe'].unique():
        sub = df[df['Timeframe'] == tf]
        fig.add_trace(go.Bar(x=sub['Ticker'], y=sub['Pareto size'], name=f'Pareto size ({tf})', text=sub['Pareto size'], textposition='outside'))
    fig.update_layout(title='Размер Pareto frontier по тикерам (больше = сложнее задача)', yaxis_title='Кол-во Pareto-точек', barmode='group', height=400)
    return fig

def make_ensemble_vs_individual(all_results: list, timeframe: str) -> go.Figure:
    rows = []
    for r in all_results:
        if 'ensemble' not in r or r['timeframe'] != timeframe:
            continue
        df = r['ensemble']
        ind = df[df['Type'] == 'Individual']
        ens = df[df['Type'] == 'Ensemble']
        if ind.empty or ens.empty:
            continue
        ind_sorted = ind.copy()
        ind_sorted['Sharpe_num'] = pd.to_numeric(ind_sorted['Sharpe'], errors='coerce')
        best_ind = ind_sorted.sort_values('Sharpe_num', ascending=False).iloc[0]
        for _, row in pd.concat([pd.DataFrame([{'Ticker': r['ticker'], 'Variant': 'Best Individual', 'Sharpe': pd.to_numeric(best_ind['Sharpe'], errors='coerce'), 'Profit %': pd.to_numeric(best_ind['Profit%'], errors='coerce'), 'MaxDD %': pd.to_numeric(best_ind['MaxDD%'], errors='coerce')}]), ens.assign(Ticker=r['ticker'], Variant=ens['Strategy'])[['Ticker', 'Variant', 'Sharpe', 'Profit%', 'MaxDD%']].rename(columns={'Profit%': 'Profit %', 'MaxDD%': 'MaxDD %'})]).iterrows():
            rows.append(row.to_dict())
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows)
    df['Sharpe'] = pd.to_numeric(df['Sharpe'], errors='coerce')
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Sharpe', 'Profit %'])
    for variant in df['Variant'].unique():
        sub = df[df['Variant'] == variant]
        fig.add_trace(go.Bar(x=sub['Ticker'], y=sub['Sharpe'], name=variant, showlegend=True), row=1, col=1)
        fig.add_trace(go.Bar(x=sub['Ticker'], y=pd.to_numeric(sub['Profit %'], errors='coerce'), name=variant, showlegend=False), row=1, col=2)
    fig.update_layout(title=f'Best Individual vs Ensemble по тикерам ({timeframe})', height=400, barmode='group')
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/vf_strategy')
    parser.add_argument('--tickers', nargs='+', default=None, help='Если задан — только эти тикеры')
    parser.add_argument('--timeframes', nargs='+', default=['1h', '1D'])
    args = parser.parse_args()
    if args.tickers is None:
        files = glob.glob(os.path.join(args.output, '*_v2_pareto_*.csv'))
        tickers = set()
        for f in files:
            base = os.path.basename(f)
            ticker = base.split('_v2_pareto_')[0]
            tickers.add(ticker)
        tickers = sorted(tickers)
    else:
        tickers = args.tickers
    print(f'Тикеры: {tickers}')
    print(f'Таймфреймы: {args.timeframes}')
    all_results = []
    for ticker in tickers:
        for tf in args.timeframes:
            r = load_ticker_results(ticker, tf, args.output)
            if 'pareto' in r or 'ensemble' in r:
                all_results.append(r)
                print(f"  Loaded: {ticker} {tf} (Pareto={len(r.get('pareto', []))}, Ensemble rows={len(r.get('ensemble', []))})")
    if not all_results:
        print('Нет данных для сводного отчёта')
        return
    metrics_df = build_metrics_matrix(all_results)
    metrics_csv = os.path.join(args.output, 'ALL_TICKERS_summary.csv')
    metrics_df.to_csv(metrics_csv, index=False, encoding='utf-8-sig')
    print(f'\nSaved metrics matrix: {metrics_csv}')
    figures_html = []
    for tf in args.timeframes:
        for metric in ['Sharpe', 'Profit %', 'MaxDD %']:
            fig = make_heatmap_metric(metrics_df, metric, tf)
            figures_html.append((f'Heatmap: {metric} ({tf})', plot(fig, include_plotlyjs=False, output_type='div')))
        fig = make_importance_comparison(all_results, tf)
        figures_html.append((f'Importance параметров ({tf})', plot(fig, include_plotlyjs=False, output_type='div')))
        fig = make_ensemble_vs_individual(all_results, tf)
        figures_html.append((f'Best Individual vs Ensemble ({tf})', plot(fig, include_plotlyjs=False, output_type='div')))
    fig = make_pareto_size_chart(all_results)
    figures_html.append(('Размер Pareto frontier по тикерам', plot(fig, include_plotlyjs=False, output_type='div')))
    metrics_html = metrics_df.round(2).to_html(index=False, classes='table table-striped table-sm', border=0)
    figures_str = '\n'.join([f'<h2>{title}</h2>\n{div}' for title, div in figures_html])
    ticker_links = []
    for r in all_results:
        link = f"{r['ticker']}_optimization_visual_{r['timeframe']}.html"
        ticker_links.append(f'''<li><a href="{link}">{r['ticker']} {r['timeframe']}</a></li>''')
    html = f"""\n<html>\n<head>\n    <meta charset="utf-8">\n    <title>VF Strategy — Multi-Ticker Summary Report</title>\n    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">\n    <style>\n      body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}\n      h1 {{ color: #2c3e50; }}\n      h2 {{ margin-top: 32px; padding-bottom: 8px; border-bottom: 2px solid #ecf0f1; }}\n      .table-sm td, .table-sm th {{ font-size: 0.85rem; padding: 0.4rem; }}\n      .info {{ background-color: #f8f9fa; padding: 16px; border-radius: 8px;\n              margin: 16px 0; border-left: 4px solid #3498db; }}\n    </style>\n</head>\n<body>\n    <h1>VF Strategy — Сводный отчёт по {len(tickers)} тикерам</h1>\n\n    <div class="info">\n        <strong>Метод:</strong> Sensitivity analysis + NSGA-II Multi-objective + Ensembles<br>\n        <strong>Тикеры:</strong> {', '.join(tickers)}<br>\n        <strong>Таймфреймы:</strong> {', '.join(args.timeframes)}\n    </div>\n\n    <h2>Индивидуальные отчёты</h2>\n    <ul>{''.join(ticker_links)}</ul>\n\n    {figures_str}\n\n    <h2>Полная сводная таблица</h2>\n    {metrics_html}\n</body>\n</html>\n"""
    out_path = os.path.join(args.output, 'ALL_TICKERS_visual_report.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Summary report saved: {out_path}')
if __name__ == '__main__':
    main()
