import os
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot

def get_marker_symbol(signal):
    s = str(signal).lower()
    if 'close long' in s or ('short' in s and 'close short' not in s):
        return 'triangle-down'
    elif 'close short' in s or ('long' in s and 'close long' not in s):
        return 'triangle-up'
    return 'circle'

def generate_vf_report(ticker, strategies, output_dir='results/vf_strategy', report_suffix='report', price_data_tf='1h'):
    if price_data_tf == '1h':
        candle_file = f'data/{ticker}_hourly_data_new.csv'
    else:
        candle_file = f'data/{ticker}_data_new.csv'
    if not os.path.exists(candle_file):
        raise FileNotFoundError(f'Файл с данными для свечей не найден: {candle_file}')
    df_candles = pd.read_csv(candle_file)
    df_candles['begin'] = pd.to_datetime(df_candles['begin'])
    df_candles.sort_values('begin', inplace=True)
    candle_fig = go.Figure(data=[go.Candlestick(x=df_candles['begin'], open=df_candles['open'], high=df_candles['high'], low=df_candles['low'], close=df_candles['close'], name=f'Свечи {price_data_tf}')])
    trades_list = []
    for strat in strategies:
        df_t = strat.get_trades_df().copy()
        if df_t.empty:
            continue
        df_t['Strategy'] = f'{strat.name} ({strat.timeframe})'
        trades_list.append(df_t)
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    if not trades_df.empty:
        time_col = 'raw_date' if 'raw_date' in trades_df.columns else 'Дата'
        trades_df[time_col] = pd.to_datetime(trades_df[time_col])
        price_col = 'Цена'
        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000', '#ffd700']
        for i, (strat_name, group) in enumerate(trades_df.groupby('Strategy')):
            color = palette[i % len(palette)]
            symbols = group['Сигнал'].apply(get_marker_symbol).tolist()
            hover = group.apply(lambda r: f"Стратегия: {strat_name}<br>Тип: {str(r.get('Сигнал', '')).lower()}<br>Цена: {r.get(price_col, '')}<br>Кол-во: {r.get('Количество', '')}<br>PnL: {r.get('Прибыль/Убыток', '')}<br>Комм.: {r.get('Комментарий', '')}", axis=1).tolist()
            candle_fig.add_trace(go.Scatter(x=group[time_col].tolist(), y=group[price_col].tolist(), mode='markers', marker=dict(color=color, symbol=symbols, size=10), hoverinfo='text', hovertext=hover, name=strat_name, showlegend=True))
    candle_fig.update_layout(title=f'{ticker} — Свечной график со сделками VF Strategy', xaxis_title='Время', yaxis_title='Цена', height=600)
    equity_dfs = []
    for strat in strategies:
        if not hasattr(strat, 'equity_df') or strat.equity_df is None or strat.equity_df.empty:
            continue
        eq = strat.equity_df.copy()
        eq.rename(columns={'equity': f'Equity_{strat.name} ({strat.timeframe})'}, inplace=True)
        equity_dfs.append(eq)
    pnl_div = ''
    if equity_dfs:
        combined_eq = equity_dfs[0]
        for df in equity_dfs[1:]:
            combined_eq = pd.merge(combined_eq, df, left_index=True, right_index=True, how='outer')
        combined_eq.sort_index(inplace=True)
        repo_file = 'data/MOEXREPO_data.csv'
        if os.path.exists(repo_file):
            repo_df = pd.read_csv(repo_file, parse_dates=['begin']).set_index('begin').sort_index()
            base = strategies[0].initial_account_size
            riskfree_values = []
            for year, group in combined_eq.groupby(combined_eq.index.year):
                start_year = pd.Timestamp(year, 1, 1)
                end_year = pd.Timestamp(year, 12, 31)
                D = (end_year - start_year).days + 1
                year_repo_rate = repo_df['close'].loc[repo_df.index.year == year].mean()
                if pd.isna(year_repo_rate):
                    year_repo_rate = 0.0
                for current_date in group.index:
                    d = (current_date - start_year).days
                    rf_value = base * (1 + year_repo_rate / 100 * (d / D))
                    riskfree_values.append((current_date, rf_value))
                base = base * (1 + year_repo_rate / 100)
            combined_eq['Riskfree'] = pd.Series(dict(riskfree_values)).reindex(combined_eq.index).ffill()
        pnl_df = combined_eq - strategies[0].initial_account_size
        pnl_fig = go.Figure()
        for col in pnl_df.columns:
            sub = pnl_df[[col]].dropna().reset_index()
            sub.columns = ['time', col]
            pnl_fig.add_trace(go.Scatter(x=sub['time'], y=sub[col], mode='lines', name=col))
        pnl_fig.update_layout(title='Кумулятивный PnL по стратегиям VF', xaxis_title='Время', yaxis_title='PnL (₽)', height=500)
        pnl_div = plot(pnl_fig, include_plotlyjs=False, output_type='div')
    stats_list = [s.get_stats() for s in strategies]
    stats_df = pd.DataFrame(stats_list)
    stats_html = stats_df.to_html(index=False, classes='table table-striped table-sm', border=0)
    trades_html = trades_df.to_html(index=False, classes='table table-bordered table-sm', border=0) if not trades_df.empty else '<p>Нет сделок.</p>'
    candle_div = plot(candle_fig, include_plotlyjs=False, output_type='div')
    html = f'\n    <html>\n      <head>\n        <meta charset="utf-8">\n        <title>{ticker} — VF Strategy Report</title>\n        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">\n        <style>\n          body {{ margin: 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}\n          h1, h2 {{ margin-top: 24px; margin-bottom: 16px; }}\n          .table-sm td, .table-sm th {{ font-size: 0.85rem; padding: 0.3rem; }}\n        </style>\n      </head>\n      <body>\n        <h1>VF Strategy Report — {ticker}</h1>\n        <p><em>Детальное сравнение нескольких конфигураций VF Strategy на исторических данных.</em></p>\n\n        <h2>1. Свечной график со сделками</h2>\n        {candle_div}\n\n        <h2>2. Кумулятивный PnL по стратегиям</h2>\n        {pnl_div}\n\n        <h2>3. Сводная статистика</h2>\n        {stats_html}\n\n        <h2>4. Таблица всех сделок</h2>\n        {trades_html}\n      </body>\n    </html>\n    '
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, f'{ticker}_{report_suffix}.html')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'VF Report сохранён: {report_file}')
    return report_file

def export_vf_csv(ticker, strategies, output_dir='results/vf_strategy'):
    os.makedirs(output_dir, exist_ok=True)
    stats_df = pd.DataFrame([s.get_stats() for s in strategies])
    stats_file = os.path.join(output_dir, f'{ticker}_vf_stats.csv')
    stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f'Stats: {stats_file}')
    equity_dfs = []
    for strat in strategies:
        if hasattr(strat, 'equity_df') and strat.equity_df is not None and (not strat.equity_df.empty):
            eq = strat.equity_df.copy()
            eq.rename(columns={'equity': f'{strat.name} ({strat.timeframe})'}, inplace=True)
            equity_dfs.append(eq)
    if equity_dfs:
        combined_eq = equity_dfs[0]
        for df in equity_dfs[1:]:
            combined_eq = pd.merge(combined_eq, df, left_index=True, right_index=True, how='outer')
        combined_eq.sort_index(inplace=True)
        equity_file = os.path.join(output_dir, f'{ticker}_vf_equity.csv')
        combined_eq.to_csv(equity_file, encoding='utf-8-sig')
        print(f'Equity: {equity_file}')
    trades_list = []
    for strat in strategies:
        df_t = strat.get_trades_df().copy()
        if df_t.empty:
            continue
        df_t['Strategy'] = f'{strat.name} ({strat.timeframe})'
        trades_list.append(df_t)
    if trades_list:
        all_trades = pd.concat(trades_list, ignore_index=True)
        trades_file = os.path.join(output_dir, f'{ticker}_vf_trades.csv')
        all_trades.to_csv(trades_file, index=False, encoding='utf-8-sig')
        print(f'Trades: {trades_file}')
