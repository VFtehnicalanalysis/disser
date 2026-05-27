import os
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from strategies.technical import EMACrossoverStrategy, RSIStrategy, MACDHistogramStrategy
from strategies.fundamental import DDMGordonStrategy, DividendYieldStrategy, DCFTwoStageStrategy, DCFMultiStageStrategy, MultipleRegressionStrategy, FUND_TICKERS
from strategies.statistical import ARIMASignalStrategy
warnings.filterwarnings('ignore')
INITIAL_CAPITAL = 1000000
ENTRY_PCT = 1.0
RESULTS_DIR = 'results'
TRADING_DAYS_PER_YEAR = 252

def _safe_metrics(strat):
    try:
        eq = strat.get_equity_curve()
    except Exception as e:
        print(f'    [{strat.name}/{strat.ticker}] equity curve failed: {e}')
        return None
    if eq is None or eq.empty:
        return None
    eq_series = eq['equity']
    final = float(eq_series.iloc[-1])
    total_return = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    daily_ret = eq_series.pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    else:
        sharpe = 0.0
    cum_max = eq_series.cummax()
    dd = (eq_series - cum_max) / cum_max
    max_dd = float(dd.min()) if not dd.empty else 0.0
    closed = [t for t in strat.trades if t.is_closed]
    n_trades = len(closed)
    n_wins = sum((1 for t in closed if t.result > 0))
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    return {'Стратегия': strat.name, 'Тикер': strat.ticker, 'Total Return': total_return, 'Sharpe (annualized)': sharpe, 'Max Drawdown': max_dd, 'Кол-во сделок': n_trades, 'Win Rate': win_rate, 'Final Equity': final, '_equity': eq_series}

def _run_one(constructor, kwargs, label):
    try:
        strat = constructor(**kwargs)
        strat.generate_signals()
        strat.execute_trades()
        return strat
    except Exception as e:
        print(f'    [{label}] failed: {type(e).__name__}: {e}')
        return None

def run_all_strategies(ticker):
    out = []
    has_ta_file = os.path.exists(f'data/{ticker}_data_new.csv')
    if has_ta_file:
        for cls, extra in [(EMACrossoverStrategy, {'short_ema': 50, 'long_ema': 100}), (MACDHistogramStrategy, {}), (RSIStrategy, {'rsi_lower': 30, 'rsi_upper': 70, 'rsi_window': 14})]:
            kwargs = dict(ticker=ticker, timeframe='1D', initial_account_size=INITIAL_CAPITAL, entry_percentage=ENTRY_PCT, pyramiding=1, allowshort=True, **extra)
            strat = _run_one(cls, kwargs, f'TA/{cls.__name__}')
            if strat is not None:
                out.append(('Technical', strat))
    else:
        print(f'  Skipping TA strategies for {ticker}: data/{ticker}_data_new.csv not found')
    fund_kwargs = dict(ticker=ticker, initial_account_size=INITIAL_CAPITAL, entry_percentage=ENTRY_PCT, pyramiding=1, allowshort=True)
    for cls in [DDMGordonStrategy, DividendYieldStrategy, DCFTwoStageStrategy, DCFMultiStageStrategy, MultipleRegressionStrategy]:
        strat = _run_one(cls, fund_kwargs, f'F/{cls.__name__}')
        if strat is not None:
            out.append(('Fundamental', strat))
    strat = _run_one(ARIMASignalStrategy, fund_kwargs, 'S/ARIMASignalStrategy')
    if strat is not None:
        out.append(('Statistical', strat))
    return out

def _format_metrics_table(records):
    df = pd.DataFrame(records)
    if df.empty:
        return '<p>Нет данных.</p>'
    display = df[['Тип', 'Тикер', 'Стратегия', 'Total Return', 'Sharpe (annualized)', 'Max Drawdown', 'Кол-во сделок', 'Win Rate']].copy()
    display['Total Return'] = display['Total Return'].apply(lambda x: f'{x:.2%}')
    display['Max Drawdown'] = display['Max Drawdown'].apply(lambda x: f'{x:.2%}')
    display['Win Rate'] = display['Win Rate'].apply(lambda x: f'{x:.2%}')
    display['Sharpe (annualized)'] = display['Sharpe (annualized)'].apply(lambda x: f'{x:.3f}')
    return display.to_html(index=False, classes='table table-striped table-hover', border=0)

def _equity_chart_for_ticker(ticker, ticker_records):
    fig = go.Figure()
    color_map = {'Technical': 'blue', 'Fundamental': 'green', 'Statistical': 'orange'}
    for r in ticker_records:
        eq = r['_equity']
        kind = r['Тип']
        fig.add_trace(go.Scatter(x=eq.index, y=eq.values, mode='lines', name=f"{r['Стратегия']} [{kind}]", line=dict(color=color_map.get(kind, 'gray'), width=1.2), opacity=0.85))
    fig.add_hline(y=INITIAL_CAPITAL, line=dict(dash='dot', color='black'), annotation_text='Initial Capital')
    fig.update_layout(title=f'Equity curves — {ticker}', height=460, xaxis_title='Дата', yaxis_title='Equity (RUB)', legend=dict(font=dict(size=10)))
    return plot(fig, include_plotlyjs=False, output_type='div')

def _aggregate_by_kind_chart(all_records):
    df = pd.DataFrame(all_records)
    if df.empty:
        return ''
    agg = df.groupby('Тип').agg(mean_sharpe=('Sharpe (annualized)', 'mean'), mean_return=('Total Return', 'mean'), count=('Стратегия', 'size')).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Avg Sharpe (annualized)', x=agg['Тип'], y=agg['mean_sharpe'], marker_color='steelblue'))
    fig.add_trace(go.Bar(name='Avg Total Return', x=agg['Тип'], y=agg['mean_return'], marker_color='seagreen', yaxis='y2'))
    fig.update_layout(title='Среднее Sharpe и Total Return по типу стратегии (по всем тикерам)', barmode='group', height=400, yaxis=dict(title='Sharpe'), yaxis2=dict(title='Total Return', overlaying='y', side='right', tickformat='.0%'))
    return plot(fig, include_plotlyjs=False, output_type='div')

def _summary_block(records):
    if not records:
        return ''
    df = pd.DataFrame(records)
    by_kind = df.groupby('Тип').size().to_dict()
    top_sharpe = df.nlargest(5, 'Sharpe (annualized)')[['Тип', 'Тикер', 'Стратегия', 'Sharpe (annualized)', 'Total Return']]
    top_return = df.nlargest(5, 'Total Return')[['Тип', 'Тикер', 'Стратегия', 'Total Return', 'Sharpe (annualized)']]
    top_sharpe_html = top_sharpe.to_html(index=False, classes='table table-sm table-bordered', border=0, float_format=lambda x: f'{x:.3f}')
    top_return_html = top_return.to_html(index=False, classes='table table-sm table-bordered', border=0, float_format=lambda x: f'{x:.3f}')
    by_kind_html = ''.join((f'<li><b>{k}</b>: {v}</li>' for k, v in by_kind.items()))
    return f'\n    <h2>Сводка</h2>\n    <p>Всего обработано стратегий: <b>{len(df)}</b>.</p>\n    <ul>{by_kind_html}</ul>\n    <h3>Топ-5 по Sharpe</h3>\n    {top_sharpe_html}\n    <h3>Топ-5 по Total Return</h3>\n    {top_return_html}\n    '

def generate_comparative_report(tickers=None):
    if tickers is None:
        tickers = FUND_TICKERS
    all_records = []
    by_ticker = {}
    for ticker in tickers:
        print(f'\n=== {ticker} ===')
        items = run_all_strategies(ticker)
        ticker_records = []
        for kind, strat in items:
            metrics = _safe_metrics(strat)
            if metrics is None:
                continue
            metrics['Тип'] = kind
            all_records.append(metrics)
            ticker_records.append(metrics)
            print(f"  ✓ {kind}/{strat.name}: TR={metrics['Total Return']:.1%}, Sharpe={metrics['Sharpe (annualized)']:.2f}, n={metrics['Кол-во сделок']}")
        by_ticker[ticker] = ticker_records
    if not all_records:
        print('Нет данных для сравнительного отчёта.')
        return
    summary_html = _summary_block(all_records)
    metrics_table = _format_metrics_table(all_records)
    aggregate_chart = _aggregate_by_kind_chart(all_records)
    chart_blocks = []
    for ticker in tickers:
        if not by_ticker.get(ticker):
            continue
        chart_blocks.append(f'<h3>{ticker}</h3>')
        chart_blocks.append(_equity_chart_for_ticker(ticker, by_ticker[ticker]))
    html = f"""<!DOCTYPE html>\n<html lang="ru">\n<head>\n    <meta charset="utf-8">\n    <title>Сравнительный отчёт: TA vs Fundamental vs Statistical</title>\n    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">\n    <style>\n      body {{ margin: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}\n      h1, h2, h3 {{ margin-top: 24px; }}\n      h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 6px; }}\n      table {{ margin-top: 12px; font-size: 0.92em; }}\n    </style>\n</head>\n<body>\n    <h1>Сравнительный отчёт: TA vs Fundamental vs Statistical</h1>\n    <p><i>Бэктест: 2015-01-01 — 2024-01-01 (фундаментальные/статистические),\n    TA — на всём периоде данных в data/{{ticker}}_data_new.csv.</i></p>\n\n    {summary_html}\n\n    <h2>Агрегированное сравнение по типам</h2>\n    {aggregate_chart}\n\n    <h2>Полная таблица метрик</h2>\n    {metrics_table}\n\n    <h2>Equity curves по тикерам</h2>\n    {''.join(chart_blocks)}\n</body>\n</html>\n"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, 'comparative_report.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n✓ Сравнительный отчёт сохранён: {out}')
if __name__ == '__main__':
    generate_comparative_report()
