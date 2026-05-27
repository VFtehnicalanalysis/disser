import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
OOS_START = pd.Timestamp('2025-01-01')
INITIAL_CAPITAL = 1000000

def get_marker_symbol(signal):
    s = str(signal).lower()
    if 'close long' in s or ('short' in s and 'close short' not in s):
        return 'triangle-down'
    elif 'close short' in s or ('long' in s and 'close long' not in s):
        return 'triangle-up'
    else:
        return 'circle'

def compute_drawdown(equity_series):
    if equity_series.empty:
        return pd.Series(dtype=float)
    running_max = equity_series.cummax()
    dd = (equity_series - running_max) / running_max * 100
    return dd

def compute_monthly_returns(equity_series):
    if equity_series.empty:
        return pd.DataFrame()
    eq = equity_series.copy()
    eq.index = pd.to_datetime(eq.index)
    try:
        monthly_last = eq.resample('ME').last()
    except ValueError:
        monthly_last = eq.resample('M').last()
    monthly_ret = monthly_last.pct_change() * 100
    monthly_ret = monthly_ret.dropna()
    if monthly_ret.empty:
        return pd.DataFrame()
    monthly_ret.index = monthly_ret.index.to_period('M')
    df = monthly_ret.to_frame(name='ret')
    df['year'] = df.index.year
    df['month'] = df.index.month
    pivot = df.pivot_table(index='year', columns='month', values='ret')
    return pivot

def compute_rolling_sharpe(equity_series, window=63, periods_per_year=252):
    if equity_series.empty:
        return pd.Series(dtype=float)
    returns = equity_series.pct_change().dropna()
    if len(returns) < window:
        return pd.Series(dtype=float)
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()
    sharpe = rolling_mean / rolling_std * np.sqrt(periods_per_year)
    return sharpe

def compute_period_stats(equity_series, period_mask, periods_per_year=252):
    eq = equity_series[period_mask]
    if eq.empty or len(eq) < 2:
        return {'Доходность, %': 0.0, 'Макс. просадка, %': 0.0, 'Sharpe': None, 'Sortino': None, 'Волатильность, %': 0.0, 'Дней': 0}
    returns = eq.pct_change().dropna()
    initial = eq.iloc[0]
    final = eq.iloc[-1]
    total_return = (final / initial - 1) * 100 if initial != 0 else 0.0
    max_dd = compute_drawdown(eq).min()
    annual_vol = returns.std() * np.sqrt(periods_per_year) * 100
    sharpe = None
    sortino = None
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(periods_per_year)
    downside = returns[returns < 0]
    if not downside.empty and downside.std() > 0:
        sortino = returns.mean() / downside.std() * np.sqrt(periods_per_year)
    return {'Доходность, %': round(total_return, 2), 'Макс. просадка, %': round(abs(max_dd), 2), 'Sharpe': round(sharpe, 2) if sharpe is not None and (not np.isnan(sharpe)) else None, 'Sortino': round(sortino, 2) if sortino is not None and (not np.isnan(sortino)) else None, 'Волатильность, %': round(annual_vol, 2), 'Дней': len(eq)}

def build_benchmark_equity(ticker, ref_index):
    index_file = f'data/{ref_index}_data.csv'
    if not os.path.exists(index_file):
        return None
    df = pd.read_csv(index_file, parse_dates=['begin']).set_index('begin').sort_index()
    if df.empty:
        return None
    first_price = df['close'].iloc[0]
    bench_eq = df['close'] / first_price * INITIAL_CAPITAL
    bench_eq.name = f'BuyHold_{ref_index}'
    return bench_eq

def build_riskfree_curve(combined_eq_index, initial=INITIAL_CAPITAL):
    repo_file = 'data/RISKFREE_data.csv'
    if not os.path.exists(repo_file):
        return None
    repo_df = pd.read_csv(repo_file, parse_dates=['begin']).set_index('begin').sort_index()
    rate_series = repo_df['close'].reindex(combined_eq_index, method='ffill')
    values = []
    base = initial
    for year, group in pd.Series(1, index=combined_eq_index).groupby(combined_eq_index.year):
        start_year = pd.Timestamp(year, 1, 1)
        end_year = pd.Timestamp(year, 12, 31)
        D = (end_year - start_year).days + 1
        year_rate = rate_series[rate_series.index.year == year].mean()
        if pd.isna(year_rate):
            year_rate = rate_series.mean()
        for current_date in group.index:
            d = (current_date - start_year).days
            values.append((current_date, base * (1 + year_rate / 100 * (d / D))))
        base = base * (1 + year_rate / 100)
    return pd.Series(dict(values)).sort_index()

def build_optimization_section(ticker):
    strategy_types = ['ema', 'rsi', 'macd', 'breakout_pivot', 'breakout_ols', 'retest_pivot', 'retest_ols']
    type_titles = {'ema': 'EMA Crossover', 'rsi': 'RSI Strategy', 'macd': 'MACD Histogram', 'breakout_pivot': 'Breakout (pivot-pair lines)', 'breakout_ols': 'Breakout (OLS через пивоты)', 'retest_pivot': 'Retest (pivot-pair lines)', 'retest_ols': 'Retest (OLS через пивоты)'}
    html_parts = []
    any_data = False
    for stype in strategy_types:
        path = f'results/optimization/{stype}_top5.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df_t = df[df['ticker'] == ticker].copy()
        if df_t.empty:
            continue
        any_data = True

        def color_oos(row):
            v = row.get('OOS_return_%')
            if v is None or pd.isna(v):
                return ''
            try:
                v = float(v)
            except (ValueError, TypeError):
                return ''
            if v > 5:
                return 'background-color: #d4edda;'
            if v < -5:
                return 'background-color: #f8d7da;'
            return ''
        display_cols = ['rank', 'composite_score', 'params', 'IS_return_%', 'IS_sharpe', 'IS_maxdd_%', 'IS_n_trades', 'OOS_return_%', 'OOS_sharpe', 'OOS_maxdd_%', 'OOS_n_trades']
        df_show = df_t[[c for c in display_cols if c in df_t.columns]].copy()
        if 'params' in df_show.columns:

            def shorten(p):
                try:
                    obj = json.loads(p)
                    return ', '.join((f'{k}={v}' for k, v in obj.items()))
                except Exception:
                    return p[:100]
            df_show['params'] = df_show['params'].apply(shorten)
        try:
            styler = df_show.style.apply(lambda row: [color_oos(row) if c in ('OOS_return_%', 'OOS_sharpe') else '' for c in df_show.columns], axis=1)
            styler.hide(axis='index')
            html_table = styler.to_html()
        except Exception:
            html_table = df_show.to_html(index=False, classes='table table-sm table-striped', border=0, na_rep='—')
        html_parts.append(f'<h4>{type_titles.get(stype, stype)}</h4>')
        html_parts.append(html_table)
    if not any_data:
        return '<p><i>Результаты оптимизации ещё не готовы (запустите <code>python optimize_strategies.py --type all</code>).</i></p>'
    summary_rows = []
    for stype in strategy_types:
        path = f'results/optimization/{stype}_top5.csv'
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df_t = df[(df['ticker'] == ticker) & (df['rank'] == 1)]
        if df_t.empty:
            continue
        r = df_t.iloc[0]
        summary_rows.append({'Тип': type_titles.get(stype, stype), 'Composite': r.get('composite_score'), 'IS return, %': r.get('IS_return_%'), 'IS Sharpe': r.get('IS_sharpe'), 'OOS return, %': r.get('OOS_return_%'), 'OOS Sharpe': r.get('OOS_sharpe'), 'OOS MaxDD, %': r.get('OOS_maxdd_%'), 'OOS сделок': r.get('OOS_n_trades')})
    summary_html = ''
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        if 'OOS Sharpe' in summary_df.columns:
            summary_df = summary_df.sort_values('OOS Sharpe', ascending=False, na_position='last')
        summary_html = '<h3>Сводка: best-of-top-5 (rank=1) по типам</h3>' + summary_df.to_html(index=False, classes='table table-sm table-striped', border=0, na_rep='—') + '<hr>'
    intro = '<p>Для каждого типа стратегии выполнена оптимизация через Optuna TPE на in-sample периоде (2014-2024). Отображаются Top-5 параметрических наборов с наибольшим composite score = Sharpe × (Return / MaxDD). Каждый набор независимо протестирован на out-of-sample 2025-2026. Зелёная подсветка — OOS return > +5%, красная — OOS return < -5%.</p>'
    return intro + summary_html + '\n'.join(html_parts)

def build_narrative(ticker, strategies, period_stats_full, period_stats_is, period_stats_oos):
    lines = []
    lines.append(f'<h3>Ключевые наблюдения по {ticker}</h3><ul>')
    full_df = pd.DataFrame(period_stats_full).T
    if 'Sharpe' in full_df.columns:
        full_df_sharpe = full_df.dropna(subset=['Sharpe'])
        if not full_df_sharpe.empty:
            best_sharpe_name = full_df_sharpe['Sharpe'].astype(float).idxmax()
            best_sharpe_val = full_df_sharpe.loc[best_sharpe_name, 'Sharpe']
            lines.append(f'<li><b>Лучший Sharpe за 2014-2026:</b> {best_sharpe_name} ({best_sharpe_val:.2f})</li>')
    if 'Доходность, %' in full_df.columns:
        best_ret_name = full_df['Доходность, %'].astype(float).idxmax()
        best_ret_val = full_df.loc[best_ret_name, 'Доходность, %']
        lines.append(f'<li><b>Максимальная доходность:</b> {best_ret_name} ({best_ret_val:.1f}%)</li>')
        worst_ret_name = full_df['Доходность, %'].astype(float).idxmin()
        worst_ret_val = full_df.loc[worst_ret_name, 'Доходность, %']
        if worst_ret_val < 0:
            lines.append(f'<li><b>Максимальный убыток:</b> {worst_ret_name} ({worst_ret_val:.1f}%)</li>')
    if 'Макс. просадка, %' in full_df.columns:
        max_dd_name = full_df['Макс. просадка, %'].astype(float).idxmax()
        max_dd_val = full_df.loc[max_dd_name, 'Макс. просадка, %']
        lines.append(f'<li><b>Максимальная просадка:</b> {max_dd_name} (−{max_dd_val:.1f}%)</li>')
    is_df = pd.DataFrame(period_stats_is).T
    oos_df = pd.DataFrame(period_stats_oos).T
    if not is_df.empty and (not oos_df.empty) and ('Sharpe' in is_df.columns):
        common = is_df.index.intersection(oos_df.index)
        degraded = []
        improved = []
        for s in common:
            is_sh = is_df.loc[s, 'Sharpe']
            oos_sh = oos_df.loc[s, 'Sharpe']
            if is_sh is not None and oos_sh is not None and (not pd.isna(is_sh)) and (not pd.isna(oos_sh)):
                try:
                    is_sh_f, oos_sh_f = (float(is_sh), float(oos_sh))
                    if is_sh_f > 0.5 and oos_sh_f < 0:
                        degraded.append(s)
                    elif is_sh_f < oos_sh_f and oos_sh_f > 0.5:
                        improved.append(s)
                except (ValueError, TypeError):
                    pass
        if degraded:
            lines.append(f"<li><b>Деградация на OOS (признаки overfitting):</b> {', '.join(degraded[:3])}</li>")
        if improved:
            lines.append(f"<li><b>Улучшение на OOS:</b> {', '.join(improved[:3])}</li>")
    warnings = []
    for s in full_df.index:
        sh = full_df.loc[s, 'Sharpe'] if 'Sharpe' in full_df.columns else None
        try:
            if sh is not None and (not pd.isna(sh)) and (float(sh) > 3):
                warnings.append(f'Подозрительно высокий Sharpe у {s} ({float(sh):.2f}) — проверить на переобучение')
        except (ValueError, TypeError):
            pass
    if warnings:
        lines.append('<li><b>⚠️ Предупреждения:</b><ul>')
        for w in warnings:
            lines.append(f'<li>{w}</li>')
        lines.append('</ul></li>')
    no_trades = []
    for strat in strategies:
        closed = [t for t in strat.trades if t.is_closed]
        if len(closed) == 0:
            no_trades.append(f'{strat.name} ({strat.timeframe})')
    if no_trades:
        lines.append(f"<li><b>⚠️ Стратегии без сделок:</b> {', '.join(no_trades)}</li>")
    lines.append('</ul>')
    return '\n'.join(lines)

def generate_report(ticker, strategies):
    daily_file = f'data/{ticker}_data_new.csv'
    if not os.path.exists(daily_file):
        raise FileNotFoundError(f'Нет данных для {ticker}: {daily_file}')
    df_price = pd.read_csv(daily_file)
    price_label = 'Дневные свечи'
    df_price['begin'] = pd.to_datetime(df_price['begin'])
    df_price.sort_values('begin', inplace=True)
    candle = go.Candlestick(x=df_price['begin'], open=df_price['open'], high=df_price['high'], low=df_price['low'], close=df_price['close'], name=price_label)
    candlestick_fig = go.Figure(data=[candle])
    trades_list = []
    for strat in strategies:
        df_trades = strat.get_trades_df().copy()
        if df_trades.empty:
            continue
        if 'Strategy' not in df_trades.columns:
            df_trades['Strategy'] = f'{strat.name} ({strat.timeframe})'
        trades_list.append(df_trades)
    trades_df = pd.concat(trades_list, ignore_index=True) if trades_list else pd.DataFrame()
    time_col = None
    price_col = 'Цена' if 'Цена' in trades_df.columns else None
    if not trades_df.empty:
        if 'raw_date' in trades_df.columns:
            trades_df['raw_date'] = pd.to_datetime(trades_df['raw_date'])
            time_col = 'raw_date'
        elif 'Дата' in trades_df.columns:
            trades_df['Дата'] = pd.to_datetime(trades_df['Дата'])
            time_col = 'Дата'
    if time_col and price_col and (not trades_df.empty):
        for strategy_name, group in trades_df.groupby('Strategy'):
            if len(group) > 100:
                continue
            times = group[time_col].tolist()
            prices = group[price_col].tolist()
            hover_texts = group.apply(lambda row: f"<b>{strategy_name}</b><br>Тип: {row.get('Сигнал', '')}<br>P&L: {row.get('Прибыль/Убыток', '')}", axis=1).tolist()
            symbols = group['Сигнал'].apply(get_marker_symbol).tolist()
            candlestick_fig.add_trace(go.Scatter(x=times, y=prices, mode='markers', marker=dict(symbol=symbols, size=8), hoverinfo='text', hovertext=hover_texts, name=strategy_name, showlegend=True))
    candlestick_fig.add_vline(x=OOS_START.timestamp() * 1000, line_dash='dash', line_color='red', annotation_text='OOS старт', annotation_position='top')
    candlestick_fig.update_layout(title=f'{ticker}: {price_label} и торговые сигналы', xaxis_title='Время', yaxis_title='Цена', height=600)
    equity_dfs = []
    equity_dfs_full = []
    for strat in strategies:
        if not hasattr(strat, 'equity_df') or strat.equity_df is None or strat.equity_df.empty:
            continue
        eq_df_full = strat.equity_df.copy()
        eq_df_full.rename(columns={'equity': f'Equity_{strat.name} ({strat.timeframe})'}, inplace=True)
        equity_dfs_full.append(eq_df_full)
        if strat.timeframe == '1h':
            eq_df = eq_df_full.resample('D').last().dropna()
        else:
            eq_df = eq_df_full
        equity_dfs.append(eq_df)
    if not equity_dfs:
        print(f'  Нет equity для {ticker} — базовый отчёт')
        html_content = f'<html><body><h1>{ticker}</h1><p>Нет данных equity — стратегии не торговали.</p></body></html>'
        os.makedirs('results', exist_ok=True)
        report_file = os.path.join('results', f'{ticker}_report.html')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return
    combined_eq = equity_dfs[0]
    for df in equity_dfs[1:]:
        combined_eq = pd.merge(combined_eq, df, left_index=True, right_index=True, how='outer')
    combined_eq.sort_index(inplace=True)
    combined_eq.index = pd.to_datetime(combined_eq.index)
    bench_eq = build_benchmark_equity(ticker, 'IMOEX')
    if bench_eq is not None:
        bench_eq = bench_eq.reindex(combined_eq.index, method='ffill')
        combined_eq['BuyHold_IMOEX'] = bench_eq
    rf_eq = build_riskfree_curve(combined_eq.index)
    if rf_eq is not None:
        combined_eq['Riskfree'] = rf_eq
    equity_fig = go.Figure()
    for col in combined_eq.columns:
        sub = combined_eq[[col]].dropna()
        if sub.empty:
            continue
        equity_fig.add_trace(go.Scatter(x=sub.index, y=sub[col], mode='lines', name=col))
    equity_fig.add_vline(x=OOS_START.timestamp() * 1000, line_dash='dash', line_color='red', annotation_text='OOS', annotation_position='top')
    equity_fig.update_layout(title=f'{ticker}: Equity стратегий vs бенчмарк IMOEX vs Risk-free', xaxis_title='Время', yaxis_title='Equity, руб', height=500, hovermode='x unified')
    dd_fig = go.Figure()
    for col in combined_eq.columns:
        if col in ('Riskfree',):
            continue
        series = combined_eq[col].dropna()
        if series.empty:
            continue
        dd = compute_drawdown(series)
        dd_fig.add_trace(go.Scatter(x=dd.index, y=dd, mode='lines', name=col, fill='tozeroy', opacity=0.5))
    dd_fig.add_vline(x=OOS_START.timestamp() * 1000, line_dash='dash', line_color='red')
    dd_fig.update_layout(title=f'{ticker}: Просадка (Drawdown %) по стратегиям', xaxis_title='Время', yaxis_title='Drawdown, %', height=400, hovermode='x unified')
    heatmap_fig = None
    full_stats = {}
    for col in combined_eq.columns:
        if col in ('Riskfree', 'BuyHold_IMOEX'):
            continue
        series = combined_eq[col].dropna()
        if series.empty:
            continue
        stats = compute_period_stats(series, series.index == series.index)
        full_stats[col] = stats
    if full_stats:
        best_col = None
        best_sh = -99
        for col, s in full_stats.items():
            sh = s.get('Sharpe')
            if sh is not None and (not pd.isna(sh)) and (float(sh) > best_sh):
                best_sh = float(sh)
                best_col = col
        if best_col:
            pivot = compute_monthly_returns(combined_eq[best_col].dropna())
            if not pivot.empty:
                heatmap_fig = go.Figure(data=go.Heatmap(z=pivot.values, x=[f'Мес {m}' for m in pivot.columns], y=pivot.index.astype(str), colorscale='RdYlGn', zmid=0, text=pivot.round(1).values, texttemplate='%{text}', hovertemplate='Год: %{y}<br>Месяц: %{x}<br>Return: %{z:.2f}%<extra></extra>'))
                heatmap_fig.update_layout(title=f'{ticker}: Месячные доходности, % — лучшая стратегия ({best_col})', xaxis_title='Месяц', yaxis_title='Год', height=400)
    rs_fig = go.Figure()
    for col in combined_eq.columns:
        if col in ('Riskfree',):
            continue
        series = combined_eq[col].dropna()
        if series.empty:
            continue
        rs = compute_rolling_sharpe(series, window=63)
        if rs.empty:
            continue
        rs_fig.add_trace(go.Scatter(x=rs.index, y=rs, mode='lines', name=col))
    rs_fig.add_hline(y=0, line_dash='dot', line_color='gray')
    rs_fig.add_vline(x=OOS_START.timestamp() * 1000, line_dash='dash', line_color='red')
    rs_fig.update_layout(title=f'{ticker}: Rolling Sharpe (окно 63 дня)', xaxis_title='Время', yaxis_title='Sharpe', height=400, hovermode='x unified')
    period_stats_full = {}
    period_stats_is = {}
    period_stats_oos = {}
    is_mask_base = combined_eq.index < OOS_START
    oos_mask_base = combined_eq.index >= OOS_START
    for col in combined_eq.columns:
        if col == 'Riskfree':
            continue
        series = combined_eq[col].dropna()
        if series.empty:
            continue
        period_stats_full[col] = compute_period_stats(series, series.index == series.index)
        period_stats_is[col] = compute_period_stats(series, series.index < OOS_START)
        period_stats_oos[col] = compute_period_stats(series, series.index >= OOS_START)

    def stats_to_html(stats_dict, label):
        if not stats_dict:
            return f'<p>Нет данных для {label}.</p>'
        df = pd.DataFrame(stats_dict).T
        df.index.name = 'Стратегия'
        df = df.reset_index()
        if 'Sharpe' in df.columns:
            df = df.sort_values('Sharpe', ascending=False, na_position='last')
        return df.to_html(index=False, classes='table table-striped table-sm', border=0, na_rep='—')
    stats_full_html = stats_to_html(period_stats_full, 'Полный период (2014-2026)')
    stats_is_html = stats_to_html(period_stats_is, 'In-Sample (2014-2024)')
    stats_oos_html = stats_to_html(period_stats_oos, 'Out-of-Sample (2025-2026)')
    full_df = pd.DataFrame(period_stats_full).T.reset_index().rename(columns={'index': 'Стратегия'})
    ranking_html = ''
    if not full_df.empty:
        for metric, ascending in [('Sharpe', False), ('Sortino', False), ('Доходность, %', False), ('Макс. просадка, %', True)]:
            if metric not in full_df.columns:
                continue
            top = full_df.sort_values(metric, ascending=ascending, na_position='last').head(5)
            ranking_html += f'<h4>Топ-5 по «{metric}»</h4>'
            ranking_html += top[['Стратегия', metric]].to_html(index=False, classes='table table-sm', border=0, na_rep='—')
    narrative_html = build_narrative(ticker, strategies, period_stats_full, period_stats_is, period_stats_oos)
    optimization_html = build_optimization_section(ticker)
    if not trades_df.empty:
        trades_display = trades_df.copy()
        if 'raw_date' in trades_display.columns:
            trades_display = trades_display.drop(columns=['raw_date'], errors='ignore')
        if len(trades_display) > 200:
            trades_display = trades_display.tail(200)
            note = f'<p><i>Показаны последние 200 сделок из {len(trades_df)} всего. Полный журнал: <code>results/{ticker}_strategies_trades.csv</code>.</i></p>'
        else:
            note = ''
        trades_html = note + trades_display.to_html(index=False, classes='table table-bordered table-sm', border=0)
    else:
        trades_html = '<p>Сделок нет.</p>'
    os.makedirs('results', exist_ok=True)
    strat_key_map = {f'Equity_{s.name} ({s.timeframe})': s for s in strategies}

    def _enrich_df(raw_df, period_start_str, period_end_str):
        df = raw_df.copy()
        if 'Стратегия' not in df.columns:
            df = df.reset_index().rename(columns={'index': 'Стратегия'})

        def _meta(col_name, attr, default=None):
            s = strat_key_map.get(col_name)
            return getattr(s, attr, default) if s is not None else default
        df['Тикер'] = df['Стратегия'].map(lambda n: _meta(n, 'ticker', ticker))
        df['TF'] = df['Стратегия'].map(lambda n: _meta(n, 'timeframe'))
        df['AllowShort'] = df['Стратегия'].map(lambda n: bool(_meta(n, 'allowshort', False)) if strat_key_map.get(n) else None)
        df['Период начало'] = period_start_str
        df['Период окончание'] = period_end_str
        meta_cols = ['Тикер', 'Стратегия', 'TF', 'AllowShort', 'Период начало', 'Период окончание']
        meta_cols = [c for c in meta_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in meta_cols]
        return df[meta_cols + other_cols]
    full_df_enr = _enrich_df(full_df, str(combined_eq.index[0]), str(combined_eq.index[-1]))
    is_df = pd.DataFrame(period_stats_is).T.reset_index().rename(columns={'index': 'Стратегия'})
    is_start = str(combined_eq.index[0])
    is_end = str(combined_eq.index[combined_eq.index < OOS_START][-1]) if (combined_eq.index < OOS_START).any() else ''
    is_df_enr = _enrich_df(is_df, is_start, is_end)
    oos_df = pd.DataFrame(period_stats_oos).T.reset_index().rename(columns={'index': 'Стратегия'})
    oos_start = str(combined_eq.index[combined_eq.index >= OOS_START][0]) if (combined_eq.index >= OOS_START).any() else ''
    oos_end = str(combined_eq.index[-1])
    oos_df_enr = _enrich_df(oos_df, oos_start, oos_end)
    full_df_enr.to_csv(f'results/{ticker}_period_stats_full.csv', index=False)
    is_df_enr.to_csv(f'results/{ticker}_period_stats_IS.csv', index=False)
    oos_df_enr.to_csv(f'results/{ticker}_period_stats_OOS.csv', index=False)
    candlestick_div = plot(candlestick_fig, include_plotlyjs=False, output_type='div')
    equity_div = plot(equity_fig, include_plotlyjs=False, output_type='div')
    dd_div = plot(dd_fig, include_plotlyjs=False, output_type='div')
    rs_div = plot(rs_fig, include_plotlyjs=False, output_type='div')
    heatmap_div = plot(heatmap_fig, include_plotlyjs=False, output_type='div') if heatmap_fig else '<p>Heatmap недоступен.</p>'
    html_content = f'<!DOCTYPE html>\n<html lang="ru">\n<head>\n  <meta charset="utf-8">\n  <title>{ticker} — отчёт по бэктестированию</title>\n  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n  <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">\n  <style>\n    body {{ margin: 30px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}\n    h1 {{ color: #1a1a1a; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}\n    h2 {{ color: #0066cc; margin-top: 40px; }}\n    h3 {{ color: #333; }}\n    .summary-box {{ background: #f5f5f5; padding: 15px; border-left: 4px solid #0066cc; margin: 20px 0; }}\n    .table {{ margin-top: 15px; font-size: 0.9em; }}\n    .nav-tabs {{ margin-top: 20px; }}\n    .tab-content {{ padding-top: 15px; }}\n  </style>\n</head>\n<body>\n  <h1>{ticker} — отчёт по бэктестированию</h1>\n  <p><b>Период:</b> 2014-2026 (IS: 2014-2024, OOS: 2025-2026 — граница отмечена красной пунктирной линией на графиках)</p>\n  <p><b>Стратегий протестировано:</b> {len(strategies)}</p>\n\n  <div class="summary-box">\n    {narrative_html}\n  </div>\n\n  <h2>1. Торговые сигналы на графике цены</h2>\n  {candlestick_div}\n\n  <h2>2. Equity-кривые стратегий vs бенчмарки</h2>\n  <p><b>BuyHold_IMOEX</b> — эталонное удержание индекса MOEX. <b>Riskfree</b> — капитал, размещённый по безрисковой ставке (MOEXREPO/RUSFAR).</p>\n  {equity_div}\n\n  <h2>3. Просадка (Drawdown) по стратегиям</h2>\n  {dd_div}\n\n  <h2>4. Rolling Sharpe (скользящая оценка качества)</h2>\n  <p>Sharpe на скользящем окне 63 дня — видно, как меняется качество стратегии во времени.</p>\n  {rs_div}\n\n  <h2>5. Тепловая карта месячных доходностей (лучшая стратегия)</h2>\n  {heatmap_div}\n\n  <h2>6. Метрики по периодам</h2>\n  <h3>6.1 Полный период (2014-2026)</h3>\n  {stats_full_html}\n  <h3>6.2 In-Sample (2014-2024)</h3>\n  {stats_is_html}\n  <h3>6.3 Out-of-Sample (2025-2026)</h3>\n  {stats_oos_html}\n\n  <h2>7. Ранжирование стратегий (по базовым параметрам)</h2>\n  {ranking_html}\n\n  <h2>8. Top-5 оптимизированных стратегий по типам (IS → OOS)</h2>\n  {optimization_html}\n\n  <h2>9. Журнал сделок</h2>\n  {trades_html}\n\n  <hr>\n  <p style="color: #888; font-size: 0.85em;">Отчёт сгенерирован автоматически. Risk-free комбинация: MOEXREPO до 2018, RUSFAR с 2018.</p>\n</body>\n</html>'
    report_file = os.path.join('results', f'{ticker}_report.html')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Отчёт сохранён в файл: {report_file}')
if __name__ == '__main__':
    print('Модуль report.py. Используйте через generate_report(ticker, strategies).')
