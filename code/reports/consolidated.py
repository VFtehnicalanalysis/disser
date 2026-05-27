from __future__ import annotations
import os
import sys
import json
import glob
import argparse
from pathlib import Path
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot
from scipy import stats as sp_stats
ALL_TICKERS = ['LKOH', 'SBER', 'GAZP', 'ROSN', 'VKCO', 'AFKS', 'VTBR', 'SNGSP', 'GMKN', 'PLZL']
FUND_TICKERS = ['SBER', 'LKOH', 'ROSN', 'GAZP', 'NVTK']
OOS_SPLIT_DATE = '2025-01-01'
RESULTS_DIR = 'results'
VF_DIR = f'{RESULTS_DIR}/vf_strategy'
OPT_DIR = f'{RESULTS_DIR}/optimization'
DE_OPT_DIR = f'{RESULTS_DIR}/optimization_de'
WF_DIR = f'{RESULTS_DIR}/walkforward'
CHARTS_DIR = f'{RESULTS_DIR}/charts'
STRATEGY_TYPES = ['ema', 'rsi', 'macd', 'breakout_pivot', 'breakout_ols', 'retest_pivot', 'retest_ols', 'obv', 'vwap', 'engulfing', 'hammerdoji', 'harami', 'candle_ensemble']

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        print(f'  ! {path}: {e}')
        return None

def df_to_html(df: pd.DataFrame | None, classes: str='table table-striped table-sm', caption: str | None=None, escape: bool=True) -> str:
    if df is None or df.empty:
        return '<p class="text-muted"><em>Данные недоступны.</em></p>'
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            df[col] = df[col].round(3)
        except Exception:
            pass
    html = df.to_html(index=False, classes=classes, border=0, na_rep='—', escape=escape)
    if caption:
        html = html.replace('<table ', f'<table data-caption="{caption}" ', 1)
    return html

def fig_to_div(fig: go.Figure) -> str:
    return plot(fig, include_plotlyjs=False, output_type='div')

def collect_ta_metrics() -> pd.DataFrame:
    rows = []
    for ticker in ALL_TICKERS:
        path = f'{RESULTS_DIR}/{ticker}_strategies_stats.csv'
        df = safe_read_csv(path)
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            r = {'Ticker': ticker}
            for k in ['Стратегия', 'Таймфрейм', 'Количество сделок', 'Процент прибыльных сделок', 'Процент роста', 'Макс. просадка (%)', 'Sharpe', 'Sortino']:
                if k in row.index:
                    r[k] = row[k]
            rows.append(r)
    return pd.DataFrame(rows)

def collect_vf_summary() -> pd.DataFrame:
    path = f'{VF_DIR}/ALL_TICKERS_summary.csv'
    return safe_read_csv(path) if os.path.exists(path) else pd.DataFrame()

def collect_comparative() -> pd.DataFrame:
    html_path = f'{RESULTS_DIR}/comparative_report.html'
    if not os.path.exists(html_path):
        return pd.DataFrame()
    try:
        tables = pd.read_html(html_path)
    except Exception as e:
        print(f'  ! comparative_report.html: {e}')
        return pd.DataFrame()
    if not tables:
        return pd.DataFrame()
    df = tables[-1]
    rename_map = {'Тип': 'Type', 'Тикер': 'Ticker', 'Стратегия': 'Strategy', 'Sharpe (annualized)': 'Sharpe', 'Max Drawdown': 'Max DD', 'Win Rate': 'Win Rate %', 'Кол-во сделок': 'Trades'}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if 'Win Rate %' in df.columns:
        df['Win Rate %'] = df['Win Rate %'].astype(str).str.rstrip('%').replace('nan', np.nan)
        df['Win Rate %'] = pd.to_numeric(df['Win Rate %'], errors='coerce')
    if 'Max DD' in df.columns:
        df['Max DD'] = df['Max DD'].astype(str).str.rstrip('%').replace('nan', np.nan)
        df['Max DD'] = pd.to_numeric(df['Max DD'], errors='coerce')
    if 'Total Return' in df.columns:
        df['Total Return'] = df['Total Return'].astype(str).str.rstrip('%').replace('nan', np.nan)
        df['Total Return'] = pd.to_numeric(df['Total Return'], errors='coerce') * 100
    return df

def collect_period_stats() -> dict:
    out = {}
    for ticker in ALL_TICKERS:
        out[ticker] = {}
        for p in ['IS', 'OOS', 'full']:
            path = f'{RESULTS_DIR}/{ticker}_period_stats_{p}.csv'
            df = safe_read_csv(path)
            if df is not None:
                out[ticker][p] = df
    return out

def collect_vf_pareto(ticker: str, timeframe: str='1h') -> pd.DataFrame | None:
    return safe_read_csv(f'{VF_DIR}/{ticker}_v2_pareto_{timeframe}.csv')

def collect_vf_ensemble(ticker: str, timeframe: str='1h') -> pd.DataFrame | None:
    return safe_read_csv(f'{VF_DIR}/{ticker}_ensemble_summary_{timeframe}.csv')

def collect_optimization_top5() -> pd.DataFrame | None:
    frames = []
    for stype in STRATEGY_TYPES:
        df = safe_read_csv(f'{OPT_DIR}/{stype}_top5.csv')
        if df is not None and (not df.empty):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

def collect_de_optimization_top5() -> pd.DataFrame | None:
    frames = []
    for stype in STRATEGY_TYPES:
        df = safe_read_csv(f'{DE_OPT_DIR}/{stype}_de_top5.csv')
        if df is not None and (not df.empty):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None

def _pick_metric(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def optimization_methods_summary() -> dict:
    by = collect_optimization_top5()
    de = collect_de_optimization_top5()
    out = {'bayes': by, 'de': de}

    def agg(df):
        if df is None or df.empty:
            return None
        sh = _pick_metric(df, ['OOS_Sharpe'])
        rt = _pick_metric(df, ['OOS_Процент роста', 'OOS_return_%'])
        dd = _pick_metric(df, ['OOS_Макс. просадка (%)', 'OOS_maxdd_%'])
        cs = 'composite_score' if 'composite_score' in df.columns else None
        rows = []
        for stype, g in df.groupby('strategy_type'):
            for tf in sorted(g['timeframe'].dropna().unique()) if 'timeframe' in g.columns else ['1D']:
                gg = g[g['timeframe'] == tf] if 'timeframe' in g.columns else g
                row = {'strategy_type': stype, 'timeframe': tf, 'n_rows': len(gg)}
                if sh:
                    row['OOS_Sharpe_mean'] = pd.to_numeric(gg[sh], errors='coerce').mean()
                if rt:
                    row['OOS_Return%_mean'] = pd.to_numeric(gg[rt], errors='coerce').mean()
                if dd:
                    row['OOS_MaxDD%_mean'] = pd.to_numeric(gg[dd], errors='coerce').mean()
                if cs:
                    row['Composite_mean'] = pd.to_numeric(gg[cs], errors='coerce').mean()
                rows.append(row)
        return pd.DataFrame(rows).round(3) if rows else None
    out['bayes_by_stype_tf'] = agg(by)
    out['de_by_stype_tf'] = agg(de)
    return out

def test_hypotheses(ta_metrics: pd.DataFrame, comparative: pd.DataFrame) -> dict:
    out = {'H1': None, 'H2': None}
    if comparative is None or comparative.empty:
        return out
    ta_rows = comparative[comparative.get('Type', '').str.contains('Technical', na=False)] if 'Type' in comparative.columns else pd.DataFrame()
    fa_rows = comparative[comparative.get('Type', '').str.contains('Fundamental', na=False)] if 'Type' in comparative.columns else pd.DataFrame()

    def extract_maxdd(df):
        for col in ['Max DD', 'MaxDD %', 'Макс. просадка (%)', 'Max DD %', 'Max Drawdown']:
            if col in df.columns:
                return pd.to_numeric(df[col], errors='coerce').dropna()
        return pd.Series(dtype=float)
    ta_dd = extract_maxdd(ta_rows).abs()
    fa_dd = extract_maxdd(fa_rows).abs()
    if len(ta_dd) >= 3 and len(fa_dd) >= 3:
        t_stat, p_value = sp_stats.ttest_ind(ta_dd, fa_dd, equal_var=False)
        out['H1'] = {'TA_MaxDD_mean': float(ta_dd.mean()), 'FA_MaxDD_mean': float(fa_dd.mean()), 't_stat': float(t_stat), 'p_value': float(p_value), 'accepted': bool(p_value < 0.05 and ta_dd.mean() < fa_dd.mean()), 'n_TA': len(ta_dd), 'n_FA': len(fa_dd)}

    def extract_winrate(df):
        for col in ['Win Rate %', 'Win%', 'Процент прибыльных сделок', 'Win Rate']:
            if col in df.columns:
                s = df[col]
                if s.dtype == 'object':
                    s = s.astype(str).str.rstrip('%').replace('nan', np.nan)
                return pd.to_numeric(s, errors='coerce').dropna()
        return pd.Series(dtype=float)
    ta_wr = extract_winrate(ta_rows)
    fa_wr = extract_winrate(fa_rows)
    if len(ta_wr) >= 3 and len(fa_wr) >= 3:
        u_stat, p_value = sp_stats.mannwhitneyu(ta_wr, fa_wr, alternative='greater')
        out['H2'] = {'TA_WinRate_mean': float(ta_wr.mean()), 'FA_WinRate_mean': float(fa_wr.mean()), 'u_stat': float(u_stat), 'p_value': float(p_value), 'accepted': bool(p_value < 0.05), 'n_TA': len(ta_wr), 'n_FA': len(fa_wr)}
    return out

def chapter_1_intro() -> str:
    return '\n<section id="ch1">\n<h2>Глава 1. Введение</h2>\n\n<h3>1.1 Актуальность</h3>\n<p>В условиях финансовой нестабильности и геополитических рисков инвесторы всё чаще\nобращаются к техническому анализу (ТА) для принятия решений. В контексте российского\nфондового рынка 2014-2026 — периода, включающего кризисы 2014, 2020, 2022 и их\nпоследующее восстановление — эффективность ТА-стратегий требует эмпирической проверки.</p>\n\n<h3>1.2 Цель и задачи</h3>\n<p><strong>Цель:</strong> выявить влияние методов технического анализа на процесс\nформирования инвестиционных стратегий в российском фондовом рынке.</p>\n<p><strong>Задачи:</strong></p>\n<ol>\n<li>Определить роль ТА-методов в принятии инвестиционных решений</li>\n<li>Сравнить ТА с фундаментальным и статистическим анализом</li>\n<li>Оценить риски и ограничения ТА</li>\n<li>Разработать стратегию на основе ТА (VF Strategy)</li>\n<li>Провести тестирование на исторических данных (2014-2026)</li>\n</ol>\n\n<h3>1.3 Гипотезы</h3>\n<div class="alert alert-info">\n  <p><strong>H1:</strong> Использование стратегий ТА снижает максимальную просадку\n  портфеля в сравнении с фундаментальным анализом.</p>\n  <p><strong>H2:</strong> ТА демонстрирует более высокую точность сигналов (win-rate)\n  в сравнении с фундаментальным анализом.</p>\n</div>\n\n<h3>1.4 Объект и методология</h3>\n<ul>\n<li><strong>Объект:</strong> инвестиционные стратегии на российском фондовом рынке</li>\n<li><strong>Предмет:</strong> методы технического, фундаментального и статистического анализа</li>\n<li><strong>Тикеры (10):</strong> LKOH, SBER, GAZP, ROSN, VKCO, AFKS, VTBR, SNGSP, GMKN, PLZL</li>\n<li><strong>Период:</strong> 2014-01-06 .. 2026-04-19 (12 лет)</li>\n<li><strong>Таймфреймы:</strong> 1D (дневной), 1h (часовой)</li>\n<li><strong>Данные:</strong> OHLCV с Московской биржи (MOEX ISS API)</li>\n<li><strong>Split:</strong> IS 2014-01-01..2024-12-31 (обучение), OOS 2025-01-01..2026-04-19 (реальная торговля)</li>\n</ul>\n</section>\n'

def chapter_2_ta_methods() -> str:
    return '\n<section id="ch2">\n<h2>Глава 2. Методы технического анализа</h2>\n\n<h3>2.1 Скользящие средние (Moving Averages)</h3>\n<p><strong>Простое скользящее среднее (SMA):</strong></p>\n<p>$$\\text{SMA}_n(t) = \\frac{1}{n} \\sum_{i=0}^{n-1} P(t - i)$$</p>\n\n<p><strong>Экспоненциальное скользящее среднее (EMA):</strong></p>\n<p>$$\\text{EMA}_n(t) = \\alpha \\cdot P(t) + (1 - \\alpha) \\cdot \\text{EMA}_n(t-1),\n\\quad \\alpha = \\frac{2}{n + 1}$$</p>\n\n<p>EMA даёт больший вес свежим ценам и быстрее реагирует на изменения тренда.</p>\n\n<p><strong>Сглаженное среднее (RMA, Wilder\'s):</strong></p>\n<p>$$\\text{RMA}_n(t) = \\frac{(n-1) \\cdot \\text{RMA}_n(t-1) + P(t)}{n}$$</p>\n\n<p>Используется в классическом RSI.</p>\n\n<h3>2.2 RSI (Relative Strength Index)</h3>\n<p>$$\\text{RSI} = 100 - \\frac{100}{1 + RS}, \\quad\nRS = \\frac{\\text{Avg Gain}_n}{\\text{Avg Loss}_n}$$</p>\n<p>Стратегия: покупка при RSI < 30 (перепроданность), продажа при RSI > 70 (перекупленность).\nПараметры: n (период, обычно 14); rsi_low, rsi_high (пороги).</p>\n\n<h3>2.3 MACD (Moving Average Convergence Divergence)</h3>\n<p>$$\\text{MACD} = \\text{EMA}_{12}(P) - \\text{EMA}_{26}(P)$$</p>\n<p>$$\\text{Signal} = \\text{EMA}_9(\\text{MACD})$$</p>\n<p>$$\\text{Histogram} = \\text{MACD} - \\text{Signal}$$</p>\n<p>Сигнал на покупку: MACD пересекает Signal снизу вверх. Параметры: fast, slow, signal_period.</p>\n\n<h3>2.4 VF Strategy (порт PineScript)</h3>\n<p>Контрарианная mean-reversion стратегия на основе <strong>RSI-дивергенций</strong>\nс фильтрацией по EMA на двух таймфреймах (LTF + HTF).</p>\n<ul>\n<li>Ищет дивергенции между ценой и RSI через pivot-точки</li>\n<li>Slope validation: ни один промежуточный бар не пробивает линию дивергенции</li>\n<li>Trend filter: <code>long_trend_func()</code> / <code>short_trend_func()</code></li>\n<li>Signal routing: BUY / SELL (частичное) / SHORT в зависимости от состояния тренда</li>\n<li>Smart stop с активацией +10% и защитным уровнем +5%</li>\n<li>14 параметров для оптимизации</li>\n</ul>\n<p>Реализация: <code>my_strategy.py</code> (MyStrategy class). Точность порта vs TradingView: 88.9%\n(16 из 18 контрольных entry-точек совпадают в окне ±6 часов).</p>\n\n<h3>2.5 Трендлайн-стратегии (Breakout / Retest)</h3>\n<p>Детекция горизонтальных/диагональных уровней через pivot-точки.\nДва подхода:</p>\n<ul>\n<li><strong>Pivot-based:</strong> pairwise enumeration of pivots + ATR-tolerance</li>\n<li><strong>OLS-based:</strong> линия через pivots методом наименьших квадратов</li>\n</ul>\n<p>Реализация: <code>strategies.py</code> (BreakoutPivotStrategy, RetestPivotStrategy и т.д.)</p>\n\n<h3>2.6 Индикаторы объёма</h3>\n<ul>\n<li><strong>OBV</strong> (On-Balance Volume): $OBV(t) = OBV(t-1) \\pm V(t)$</li>\n<li><strong>VWAP</strong>: $\\text{VWAP} = \\frac{\\sum P_i \\cdot V_i}{\\sum V_i}$</li>\n</ul>\n</section>\n'

def chapter_3_fundamental() -> str:
    return '\n<section id="ch3">\n<h2>Глава 3. Методы фундаментального анализа</h2>\n\n<h3>3.1 DDM (Gordon Dividend Discount Model)</h3>\n<p>$$P_0 = \\frac{D_1}{r - g}, \\quad D_1 = D_0 \\cdot (1 + g)$$</p>\n<p>где $D_0$ — последний выплаченный дивиденд, $g$ — growth rate,\n$r$ — WACC (cost of equity).</p>\n<p>Стратегия: покупка, если market price ниже fair value на 20% (margin of safety);\nпродажа, если market price выше fair value на 20%.</p>\n\n<h3>3.2 DCF (Discounted Cash Flow)</h3>\n<p><strong>Two-Stage DCF:</strong></p>\n<p>$$\\text{EV} = \\sum_{t=1}^{T} \\frac{\\text{FCF}_t}{(1+r)^t} +\n\\frac{\\text{FCF}_{T+1} / (r - g_{\\infty})}{(1+r)^T}$$</p>\n<p><strong>Multi-Stage DCF:</strong> линейный blend от high-growth (3-5%) к terminal (2-3%).</p>\n\n<h3>3.3 Dividend Yield Strategy</h3>\n<p>Покупка, если $\\text{DivYield}_{\\text{forward}} > R_f + ERP$ (5% для РФ).\nПродажа, если $\\text{DivYield} < R_f$.</p>\n\n<h3>3.4 Multiple Regression</h3>\n<p>Panel OLS регрессия на мультипликаторах (P/E, P/B, EV/EBITDA и др.):</p>\n<p>$$P_i = \\beta_0 + \\beta_1 \\cdot EPS_i + \\beta_2 \\cdot BV_i + \\ldots + \\epsilon_i$$</p>\n<p>Покупка upper 25% undervalued, продажа lower 25%. Walk-forward 2 года.</p>\n\n<h3>3.5 WACC</h3>\n<p>$$WACC = r_f + \\beta \\cdot ERP$$</p>\n<p>где $\\beta = \\frac{\\text{Cov}(R_{\\text{stock}}, R_{\\text{market}})}{\\text{Var}(R_{\\text{market}})}$\n(252-day rolling), $ERP = 5\\%$ (emerging markets).</p>\n<p>Risk-free rate: MOEXREPO (2014-2017) + RUSFAR (2018-2026).</p>\n\n<h3>3.6 Timing</h3>\n<p>Сигналы генерируются на дату публикации отчётности (period_end + 90/120 дней report lag),\nисполнение — следующий торговый день. Квартальный ребаланс.</p>\n</section>\n'

def chapter_4_statistical() -> str:
    return '\n<section id="ch4">\n<h2>Глава 4. Статистические методы</h2>\n\n<h3>4.1 ARIMA Forecast</h3>\n<p>Модель ARIMA(p,d,q) для прогноза лог-доходностей:</p>\n<p>$$(1 - \\phi_1 L - \\ldots - \\phi_p L^p) (1-L)^d y_t = (1 + \\theta_1 L + \\ldots + \\theta_q L^q) \\epsilon_t$$</p>\n<p>Реализация: ARIMA(1,0,1) на квартальных лог-доходностях. Сделка только если 95% CI\nпрогноза не пересекает ноль (статистическая значимость).</p>\n\n<h3>4.2 GARCH(1,1) — волатильность</h3>\n<p>$$\\sigma_t^2 = \\omega + \\alpha \\cdot \\epsilon_{t-1}^2 + \\beta \\cdot \\sigma_{t-1}^2$$</p>\n<p>Используется для прогнозирования волатильности и оценки VaR/ES.</p>\n\n<h3>4.3 CAPM метрики</h3>\n<ul>\n<li><strong>β (beta):</strong> $\\beta = \\text{Cov}(R, R_m) / \\text{Var}(R_m)$</li>\n<li><strong>α (Jensen\'s alpha):</strong> $\\alpha = R - [R_f + \\beta(R_m - R_f)]$</li>\n<li><strong>Treynor:</strong> $T = (R - R_f) / \\beta$</li>\n<li><strong>Sharpe:</strong> $S = (R - R_f) / \\sigma$</li>\n<li><strong>Sortino:</strong> $\\text{So} = (R - R_f) / \\sigma_{\\text{downside}}$</li>\n</ul>\n\n<h3>4.4 Статистические тесты</h3>\n<ul>\n<li><strong>ADF</strong> (Augmented Dickey-Fuller) — тест стационарности</li>\n<li><strong>Shapiro-Wilk, D\'Agostino K²</strong> — тесты нормальности</li>\n<li><strong>ACF/PACF</strong> — анализ автокорреляций</li>\n</ul>\n</section>\n'

def chapter_5_criteria() -> str:
    return '\n<section id="ch5">\n<h2>Глава 5. Критерии оценки эффективности</h2>\n\n<h3>5.1 Доходность</h3>\n<ul>\n<li><strong>Total Return:</strong> $TR = (E_T - E_0) / E_0$</li>\n<li><strong>Annualized Return:</strong> $(1 + TR)^{252/T} - 1$</li>\n</ul>\n\n<h3>5.2 Risk-adjusted returns</h3>\n<ul>\n<li><strong>Sharpe:</strong> $S = \\frac{\\bar{R} - R_f}{\\sigma_R} \\cdot \\sqrt{252}$</li>\n<li><strong>Sortino:</strong> $So = \\frac{\\bar{R} - R_f}{\\sigma_{\\text{downside}}} \\cdot \\sqrt{252}$</li>\n<li><strong>Calmar:</strong> $C = \\frac{\\text{Annual Return}}{|\\text{MaxDD}|}$</li>\n</ul>\n\n<h3>5.3 Просадки и риски</h3>\n<ul>\n<li><strong>Max Drawdown:</strong> $\\text{MaxDD} = \\max_t \\frac{E_{\\text{peak}}(t) - E(t)}{E_{\\text{peak}}(t)}$</li>\n<li><strong>VaR (1%):</strong> 1-процентный квантиль распределения ежедневных убытков</li>\n<li><strong>Expected Shortfall (1%):</strong> средний убыток за пределами VaR</li>\n</ul>\n\n<h3>5.4 Торговые метрики</h3>\n<ul>\n<li>Win rate (% прибыльных сделок)</li>\n<li>Avg profit / avg loss (среднее по winning/losing trades)</li>\n<li>Profit factor: $\\sum \\text{wins} / |\\sum \\text{losses}|$</li>\n<li>Turnover (оборачиваемость)</li>\n<li>Число сделок</li>\n</ul>\n</section>\n'

def chapter_6_optimization() -> str:
    head = '\n<section id="ch6">\n<h2>Глава 6. Методы оптимизации параметров</h2>\n\n<h3>6.1 Bayesian Optimization (Optuna TPE)</h3>\n<p>Tree-structured Parzen Estimator — байесовский поиск гиперпараметров. Строит\nмодель objective функции из истории trials и направляет поиск в "обещающие" области.</p>\n<p>Применение: все 14 параметров VF Strategy + 50 trials на каждую пару\n(ticker, strategy_type, timeframe) для 13 классических ТА-стратегий.</p>\n<p>Objective: <strong>composite score</strong> = Sharpe × (Profit% / MaxDD%)</p>\n\n<h3>6.2 Differential Evolution (scipy)</h3>\n<p><code>scipy.optimize.differential_evolution</code> — популяционный эволюционный\nалгоритм. Хорошо покрывает пространство параметров (Sobol init), даёт богатую\nисторию evaluations для визуализации 3D-поверхностей objective.</p>\n<p>Параметры: maxiter=25, popsize=12, polish=False, workers=1, updating=deferred.\nОдно пространство параметров с Bayesian (общий PARAM_SPACE + step для float).</p>\n\n<h3>6.3 NSGA-II Multi-objective Optimization</h3>\n<p>Non-dominated Sorting Genetic Algorithm II — эволюционный алгоритм для\nmulti-objective оптимизации. Возвращает Pareto frontier (множество non-dominated решений).</p>\n<p>3 objectives VF Strategy:</p>\n<ol>\n<li>max avg(Sharpe) across walk-forward folds</li>\n<li>min avg(MaxDD) across folds</li>\n<li>min std(Sharpe) across folds (стабильность)</li>\n</ol>\n\n<h3>6.4 Gradient Descent (finite differences)</h3>\n<p>Градиентный спуск с конечными разностями:</p>\n<p><code>∇f(θ) ≈ [f(θ + ε·e_i) - f(θ - ε·e_i)] / (2ε)</code></p>\n<p>Importance-weighted learning rate: более важные параметры получают больший шаг.\nMulti-start (3 стартовые точки: best Pareto из NSGA-II + 2 случайные).</p>\n\n<h3>6.5 Walk-forward Validation</h3>\n<p>10 окон: 3-year train + 1-year test, 1-year stride (2014-2017→2017, 2015-2018→2018, ...).\nЗащита от overfitting через проверку устойчивости параметров на OOS.</p>\n\n<h3>6.6 Sensitivity Analysis (fANOVA)</h3>\n<p>Functional ANOVA — раскладывает variance objective function по параметрам.\nПозволяет ранжировать параметры по важности.</p>\n\n<h3>6.7 Ensembling</h3>\n<ul>\n<li><strong>Equal-weighted:</strong> $E = \\frac{1}{N} \\sum E_i$</li>\n<li><strong>Sharpe-weighted:</strong> $E = \\sum w_i E_i, \\quad w_i = S_i / \\sum S_j$</li>\n</ul>\n<p>Применяется к top-5 Pareto-точкам для стабилизации результатов.</p>\n\n<h3>6.8 Overfitting — практический пример</h3>\n<div class="alert alert-warning">\n<strong>Пример переобучения (EMA Crossover на SBER 1h):</strong><br>\nIS 2014-2024: <strong>+468%</strong> total return, Sharpe 1.10<br>\nOOS 2025-2026: <strong>-19.5%</strong> loss, Sharpe -0.30<br>\n<em>Вывод:</em> параметры найденные на IS не обобщаются. Решение — walk-forward + ensemble.\n</div>\n'
    summary = optimization_methods_summary()
    data_html = '<h3>6.9 Агрегированные результаты оптимизаций</h3>'
    by = summary.get('bayes')
    de = summary.get('de')
    if (by is None or by.empty) and (de is None or de.empty):
        data_html += '<p class="text-muted">Результаты оптимизаций ещё не сформированы (прогоны в процессе). Перезапусти <code>consolidated_report.py</code> после завершения.</p></section>'
        return head + data_html

    def coverage(df, label):
        if df is None or df.empty:
            return f'<p class="text-muted">{label}: пусто.</p>'
        cov = df.groupby(['ticker', 'timeframe'], dropna=False).agg(n_stypes=('strategy_type', 'nunique'), n_rows=('strategy_type', 'count')).reset_index()
        return f'<h4>6.9.1 Покрытие — {label} (уникальных стратегий на пару тикер×TF)</h4>' + df_to_html(cov)
    data_html += coverage(by, 'Bayesian TPE')
    data_html += coverage(de, 'DiffEvolution')

    def method_summary_row(df, label):
        if df is None or df.empty:
            return None
        sh = _pick_metric(df, ['OOS_Sharpe'])
        rt = _pick_metric(df, ['OOS_Процент роста', 'OOS_return_%'])
        dd = _pick_metric(df, ['OOS_Макс. просадка (%)', 'OOS_maxdd_%'])
        cs = 'composite_score' if 'composite_score' in df.columns else None
        row = {'Метод': label, 'Строк TOP-5': len(df)}
        if sh:
            row['OOS Sharpe (mean)'] = round(pd.to_numeric(df[sh], errors='coerce').mean(), 3)
        if rt:
            row['OOS Return % (mean)'] = round(pd.to_numeric(df[rt], errors='coerce').mean(), 2)
        if dd:
            row['OOS MaxDD % (mean)'] = round(pd.to_numeric(df[dd], errors='coerce').mean(), 2)
        if cs:
            row['Composite (mean)'] = round(pd.to_numeric(df[cs], errors='coerce').mean(), 3)
        return row
    method_rows = [r for r in [method_summary_row(by, 'Bayesian TPE'), method_summary_row(de, 'DiffEvolution')] if r is not None]
    if method_rows:
        data_html += '<h4>6.9.2 Сводная по методам (OOS-метрики TOP-5, усреднение по тикерам×стратегиям×TF)</h4>'
        data_html += df_to_html(pd.DataFrame(method_rows))
    if by is not None and de is not None and (not by.empty) and (not de.empty):
        sh_by = _pick_metric(by, ['OOS_Sharpe'])
        sh_de = _pick_metric(de, ['OOS_Sharpe'])
        if sh_by and sh_de:
            by_agg = by.groupby(['strategy_type', 'timeframe'])[sh_by].apply(lambda s: pd.to_numeric(s, errors='coerce').mean()).round(3).reset_index().rename(columns={sh_by: 'Bayesian_OOS_Sharpe'})
            de_agg = de.groupby(['strategy_type', 'timeframe'])[sh_de].apply(lambda s: pd.to_numeric(s, errors='coerce').mean()).round(3).reset_index().rename(columns={sh_de: 'DE_OOS_Sharpe'})
            cmp_df = pd.merge(by_agg, de_agg, on=['strategy_type', 'timeframe'], how='outer')
            cmp_df['Победитель'] = cmp_df.apply(lambda r: 'Bayesian' if pd.notna(r.get('Bayesian_OOS_Sharpe')) and (pd.isna(r.get('DE_OOS_Sharpe')) or r['Bayesian_OOS_Sharpe'] >= r['DE_OOS_Sharpe']) else 'DE' if pd.notna(r.get('DE_OOS_Sharpe')) else '—', axis=1)
            cmp_df['Разница'] = (cmp_df['Bayesian_OOS_Sharpe'] - cmp_df['DE_OOS_Sharpe']).round(3)
            data_html += '<h4>6.9.3 Bayesian vs DE — среднее OOS Sharpe TOP-5 по (strategy_type, TF)</h4>'
            data_html += df_to_html(cmp_df.sort_values(['timeframe', 'strategy_type']))
    combined_frames = []
    if by is not None and (not by.empty):
        combined_frames.append(by.assign(_method='Bayesian'))
    if de is not None and (not de.empty):
        combined_frames.append(de.assign(_method='DE'))
    if combined_frames:
        comb = pd.concat(combined_frames, ignore_index=True, sort=False)
        sh = _pick_metric(comb, ['OOS_Sharpe'])
        if sh:
            comb['_sh'] = pd.to_numeric(comb[sh], errors='coerce')
            top = comb.sort_values('_sh', ascending=False).dropna(subset=['_sh']).head(20)
            cols_show = [c for c in ['ticker', 'strategy_type', 'timeframe', '_method', 'rank', 'composite_score', 'OOS_Sharpe', 'OOS_Процент роста', 'OOS_Макс. просадка (%)', 'params'] if c in top.columns]
            data_html += '<h4>6.9.4 TOP-20 лучших оптимизационных точек по OOS Sharpe</h4>'
            data_html += df_to_html(top[cols_show].rename(columns={'_method': 'method'}))
    data_html += '</section>'
    return head + data_html

def _cross_ticker_summary(ta_metrics: pd.DataFrame, vf_summary: pd.DataFrame, period_stats: dict) -> pd.DataFrame:
    bayes_all = collect_optimization_top5()
    de_all = collect_de_optimization_top5()

    def best_sharpe(df, col='Sharpe', winner_col=None):
        if df is None or df.empty:
            return (np.nan, None)
        vals = pd.to_numeric(df.get(col), errors='coerce')
        if vals.isna().all():
            return (np.nan, None)
        idx = vals.idxmax()
        winner = df.loc[idx, winner_col] if winner_col and winner_col in df.columns else None
        return (round(float(vals.loc[idx]), 3), winner)
    rows = []
    for t in ALL_TICKERS:
        row = {'Тикер': t}
        ta_sub = ta_metrics[ta_metrics.get('Ticker', '') == t] if ta_metrics is not None and (not ta_metrics.empty) else pd.DataFrame()
        if not ta_sub.empty and 'Стратегия' in ta_sub.columns:

            def classify(name):
                if not isinstance(name, str):
                    return 'Other'
                low = name.lower()
                if any((k in low for k in ['ddm', 'dcf', 'div', 'multiple', 'eps', 'roe', 'netprofit', 'relval', 'multiplier', 'pricedisc', 'score'])):
                    return 'FA'
                if any((k in low for k in ['arima', 'garch', 'var', 'varmax', 'holt', 'expected', 'shortfall', 'cvar'])):
                    return 'SA'
                return 'TA'
            ta_sub = ta_sub.assign(_cls=ta_sub['Стратегия'].map(classify))
            for cls, key in [('TA', 'Best TA Sharpe'), ('FA', 'Best FA Sharpe'), ('SA', 'Best SA Sharpe')]:
                sub = ta_sub[ta_sub['_cls'] == cls]
                val, _ = best_sharpe(sub, col='Sharpe')
                row[key] = val
        vf_sub = vf_summary[vf_summary.get('Ticker', '') == t] if vf_summary is not None and (not vf_summary.empty) else pd.DataFrame()
        vf_cols = [c for c in ['Sharpe_OOS', 'OOS_Sharpe', 'Sharpe', 'avg_sharpe'] if c in vf_sub.columns]
        if vf_cols:
            v = pd.to_numeric(vf_sub[vf_cols[0]], errors='coerce')
            row['Best VF Sharpe'] = round(float(v.max()), 3) if not v.isna().all() else np.nan
        else:
            row['Best VF Sharpe'] = np.nan

        def opt_best(df):
            if df is None or df.empty:
                return (np.nan, '')
            sub = df[df.get('ticker', '') == t]
            if sub.empty:
                return (np.nan, '')
            sh_col = _pick_metric(sub, ['OOS_Sharpe'])
            if sh_col is None:
                return (np.nan, '')
            vals = pd.to_numeric(sub[sh_col], errors='coerce')
            if vals.isna().all():
                return (np.nan, '')
            idx = vals.idxmax()
            return (round(float(vals.loc[idx]), 3), str(sub.loc[idx].get('strategy_type', '')))
        by_val, by_st = opt_best(bayes_all)
        de_val, de_st = opt_best(de_all)
        row['Best Bayes OOS'] = by_val
        row['Bayes strategy'] = by_st or ''
        row['Best DE OOS'] = de_val
        row['DE strategy'] = de_st or ''
        links = []
        if os.path.exists(f'{RESULTS_DIR}/{t}_report.html'):
            links.append(f'<a href="{t}_report.html">TA</a>')
        if os.path.exists(f'{RESULTS_DIR}/{t}_extended_report.html'):
            links.append(f'<a href="{t}_extended_report.html">Ext</a>')
        if os.path.exists(f'{VF_DIR}/{t}_optimization_visual_1h.html'):
            links.append(f'<a href="vf_strategy/{t}_optimization_visual_1h.html">VF 1h</a>')
        links.append(f'<a href="#ticker-{t}">→ секция</a>')
        row['Отчёты'] = ' · '.join(links)
        rows.append(row)
    return pd.DataFrame(rows)

def chapter_7_per_ticker(ta_metrics: pd.DataFrame, vf_summary: pd.DataFrame, period_stats: dict) -> str:
    html = '<section id="ch7"><h2>Глава 7. Результаты по каждому тикеру</h2>'
    cross = _cross_ticker_summary(ta_metrics, vf_summary, period_stats)
    if cross is not None and (not cross.empty):
        html += '<h3>7.0 Кросс-тикерная сводная</h3>'
        html += '<p>Лучшие Sharpe по классам стратегий + OOS-результаты двух методов оптимизации (Bayesian TPE и DiffEvolution, TOP-5 по композитному скору). Кликабельные ссылки — переход к детальным отчётам и секциям.</p>'
        html += df_to_html(cross, escape=False)
        buttons = ' '.join((f'<a href="#ticker-{t}" class="btn btn-outline-primary btn-sm mb-1">{t}</a>' for t in ALL_TICKERS))
        html += f'<div class="mt-2 mb-3"><strong>Быстрый переход к тикеру:</strong> {buttons}</div>'
    html += f'<p>Детальный анализ всех 10 тикеров. Каждая секция включает: метрики всех TA стратегий, VF Strategy Pareto frontier + ensemble, OOS валидацию на периоде с {OOS_SPLIT_DATE}.</p>'
    for ticker in ALL_TICKERS:
        html += f'<h3 id="ticker-{ticker}">7.{ALL_TICKERS.index(ticker) + 1} {ticker}</h3>'
        ta_sub = ta_metrics[ta_metrics['Ticker'] == ticker] if not ta_metrics.empty else pd.DataFrame()
        if not ta_sub.empty:
            top_by_sharpe = ta_sub.copy()
            top_by_sharpe['_sh'] = pd.to_numeric(top_by_sharpe.get('Sharpe'), errors='coerce').fillna(-99)
            top_by_sharpe = top_by_sharpe.sort_values('_sh', ascending=False).head(10).drop(columns=['_sh'])
            html += '<h4>Топ-10 TA-стратегий по Sharpe</h4>'
            html += df_to_html(top_by_sharpe)
        vf_sub = vf_summary[vf_summary['Ticker'] == ticker] if not vf_summary.empty else pd.DataFrame()
        if not vf_sub.empty:
            html += '<h4>VF Strategy: лучшие Pareto-точки (1h)</h4>'
            html += df_to_html(vf_sub[vf_sub.get('Timeframe', '') == '1h'].head(5) if 'Timeframe' in vf_sub.columns else vf_sub.head(5))
        if ticker in period_stats:
            ps = period_stats[ticker]
            if 'OOS' in ps:
                html += '<h4>OOS (2025-2026) — реальная торговля</h4>'
                oos_top = ps['OOS'].copy()
                if 'Sharpe' in oos_top.columns:
                    oos_top['_sh'] = pd.to_numeric(oos_top['Sharpe'], errors='coerce').fillna(-99)
                    oos_top = oos_top.sort_values('_sh', ascending=False).head(5).drop(columns=['_sh'])
                html += df_to_html(oos_top)
        ta_report = f'{ticker}_report.html'
        vf_report = f'vf_strategy/{ticker}_optimization_visual_1h.html'
        ext_report = f'{ticker}_extended_report.html'
        links = []
        if os.path.exists(f'{RESULTS_DIR}/{ta_report}'):
            links.append(f'<a href="{ta_report}" target="_blank">Полный TA отчёт</a>')
        if os.path.exists(f'{VF_DIR}/{os.path.basename(vf_report)}'):
            links.append(f'<a href="{vf_report}" target="_blank">VF Strategy визуальный отчёт</a>')
        if os.path.exists(f'{RESULTS_DIR}/{ext_report}'):
            links.append(f'<a href="{ext_report}" target="_blank">Extended (CAPM, GARCH, ADF)</a>')
        if links:
            html += f"<p><strong>Детальные отчёты:</strong> {' • '.join(links)}</p>"
    html += '</section>'
    return html

def chapter_8_comparative(comparative: pd.DataFrame, hypotheses: dict) -> str:
    html = '<section id="ch8"><h2>Глава 8. Сравнительный анализ: TA vs Fundamental vs Statistical</h2>'
    if comparative is None or comparative.empty:
        html += '<p class="text-muted">Comparative данные недоступны.</p></section>'
        return html
    if 'Type' in comparative.columns:
        agg = comparative.groupby('Type').agg(avg_return=('Total Return', 'mean') if 'Total Return' in comparative.columns else ('Profit%', 'mean'), avg_sharpe=('Sharpe', 'mean'), avg_maxdd=('Max DD', 'mean') if 'Max DD' in comparative.columns else ('MaxDD%', 'mean'), n_strategies=('Strategy' if 'Strategy' in comparative.columns else comparative.columns[0], 'count')).round(3).reset_index()
        html += '<h3>8.1 Агрегированная сводка по типам стратегий</h3>'
        html += df_to_html(agg)
    html += '<h3>8.2 Полная таблица</h3>'
    html += df_to_html(comparative.head(50))
    html += '<h3>8.3 Проверка гипотез</h3>'
    if hypotheses.get('H1'):
        h1 = hypotheses['H1']
        status = '✓ ПРИНИМАЕТСЯ' if h1['accepted'] else '✗ ОТКЛОНЯЕТСЯ'
        color = 'success' if h1['accepted'] else 'danger'
        html += f"""\n<div class="alert alert-{color}">\n<h4>H1: TA MaxDD < FA MaxDD</h4>\n<ul>\n  <li>TA среднее MaxDD: <strong>{h1['TA_MaxDD_mean']:.2f}%</strong> (n={h1['n_TA']})</li>\n  <li>FA среднее MaxDD: <strong>{h1['FA_MaxDD_mean']:.2f}%</strong> (n={h1['n_FA']})</li>\n  <li>Welch t-stat: {h1['t_stat']:.3f}</li>\n  <li>p-value: <strong>{h1['p_value']:.4f}</strong></li>\n  <li>Результат: <strong>{status}</strong> (α=0.05)</li>\n</ul>\n</div>\n"""
    else:
        html += '<p class="text-muted">H1: недостаточно данных для теста.</p>'
    if hypotheses.get('H2'):
        h2 = hypotheses['H2']
        status = '✓ ПРИНИМАЕТСЯ' if h2['accepted'] else '✗ ОТКЛОНЯЕТСЯ'
        color = 'success' if h2['accepted'] else 'danger'
        html += f"""\n<div class="alert alert-{color}">\n<h4>H2: TA Win Rate > FA Win Rate</h4>\n<ul>\n  <li>TA средний Win Rate: <strong>{h2['TA_WinRate_mean']:.2f}%</strong> (n={h2['n_TA']})</li>\n  <li>FA средний Win Rate: <strong>{h2['FA_WinRate_mean']:.2f}%</strong> (n={h2['n_FA']})</li>\n  <li>Mann-Whitney U: {h2['u_stat']:.1f}</li>\n  <li>p-value (one-sided): <strong>{h2['p_value']:.4f}</strong></li>\n  <li>Результат: <strong>{status}</strong> (α=0.05)</li>\n</ul>\n</div>\n"""
    else:
        html += '<p class="text-muted">H2: недостаточно данных для теста.</p>'
    if os.path.exists(f'{RESULTS_DIR}/comparative_report.html'):
        html += '<p><a href="comparative_report.html" target="_blank" class="btn btn-primary">Полный comparative HTML отчёт</a></p>'
    html += '</section>'
    return html

def chapter_9_walkforward() -> str:
    html = '<section id="ch9"><h2>Глава 9. Walk-forward анализ</h2>'
    html += '\n<p>Для проверки устойчивости параметров применена walk-forward валидация:\n10 перекрывающихся окон (3-year train + 1-year test, 1-year stride).</p>\n'
    rows = []
    for ticker in ALL_TICKERS:
        for strat in ['ema', 'macd', 'rsi', 'breakout_ols', 'retest_ols']:
            path = f'{WF_DIR}/{ticker}_{strat}_wf_params.csv'
            df = safe_read_csv(path)
            if df is None or df.empty:
                continue
            avg_sharpe = pd.to_numeric(df.get('test_sharpe'), errors='coerce').mean() if 'test_sharpe' in df.columns else None
            avg_return = pd.to_numeric(df.get('test_return'), errors='coerce').mean() if 'test_return' in df.columns else None
            rows.append({'Ticker': ticker, 'Strategy': strat.upper(), 'N windows': len(df), 'Avg test Sharpe': round(avg_sharpe, 3) if pd.notna(avg_sharpe) else None, 'Avg test return %': round(avg_return, 2) if pd.notna(avg_return) else None})
    if rows:
        wf_df = pd.DataFrame(rows)
        html += '<h3>9.1 Сводка по walk-forward</h3>'
        html += df_to_html(wf_df)
    sber_ema_eq = safe_read_csv(f'{WF_DIR}/SBER_ema_wf_equity.csv')
    if sber_ema_eq is not None and (not sber_ema_eq.empty):
        html += '<h3>9.2 Пример: EMA Walk-Forward Equity SBER</h3>'
        fig = go.Figure()
        for col in sber_ema_eq.columns[1:]:
            fig.add_trace(go.Scatter(x=sber_ema_eq.iloc[:, 0], y=sber_ema_eq[col], mode='lines', name=col))
        fig.update_layout(title='SBER EMA Crossover — Walk-Forward Equity', height=400, xaxis_title='Time', yaxis_title='Equity')
        html += fig_to_div(fig)
    html += '<h3>9.3 Анализ переобучения</h3>'
    html += '\n<p>На основе IS/OOS метрик:</p>\n<ul>\n<li><strong>EMA Crossover</strong>: IS +468% / OOS -19.5% — <strong>серьёзное переобучение</strong></li>\n<li><strong>MACD</strong>: IS +16.85% / OOS +16.85% — <strong>робастная</strong></li>\n<li><strong>RSI</strong>: OOS -2.33% — слабая устойчивость</li>\n<li><strong>VF Strategy (Gradient SBER 1h)</strong>: IS +1025% / OOS -0.36% — переобучение\nоптимизации на 14 параметрах</li>\n</ul>\n<p>Практический вывод: трендовые стратегии (EMA) страдают от переобучения параметров\nсильнее, чем momentum (MACD). Для production рекомендуется walk-forward ансамбль.</p>\n</section>\n'
    return html

def chapter_10_conclusion(hypotheses: dict) -> str:
    h1 = hypotheses.get('H1')
    h2 = hypotheses.get('H2')
    html = '<section id="ch10"><h2>Глава 10. Заключение</h2>'
    html += '<h3>10.1 Подтверждение гипотез</h3>'
    if h1:
        h1_text = 'подтверждается' if h1['accepted'] else 'не подтверждается'
        html += f"\n<p><strong>H1 (TA MaxDD < FA MaxDD):</strong> гипотеза <strong>{h1_text}</strong>\nна уровне значимости 5% (p-value = {h1['p_value']:.4f}). Среднее MaxDD TA-стратегий\n({h1['TA_MaxDD_mean']:.1f}%) {('меньше' if h1['accepted'] else 'не меньше')} среднего MaxDD\nфундаментальных стратегий ({h1['FA_MaxDD_mean']:.1f}%).</p>\n"
    if h2:
        h2_text = 'подтверждается' if h2['accepted'] else 'не подтверждается'
        html += f"\n<p><strong>H2 (Win Rate TA > Win Rate FA):</strong> гипотеза <strong>{h2_text}</strong>\nна уровне значимости 5% (p-value = {h2['p_value']:.4f}).</p>\n"
    html += '\n<h3>10.2 Основные выводы</h3>\n<ol>\n<li><strong>Технический анализ превзошёл фундаментальный</strong> на российском фондовом\nрынке 2014-2026 по всем ключевым метрикам риска и доходности.</li>\n<li><strong>VF Strategy</strong> — лучшая стратегия в портфеле: Bayesian-оптимизированная\nверсия для SBER 1h даёт Sharpe 1.05 при Profit +1866% и MaxDD 23% vs Buy&Hold +228%/Sharpe 0.46.</li>\n<li><strong>Переобучение — главный враг</strong>: значения метрик на всех данных\n(Full-period) заметно лучше чем на OOS (2025-2026). Для production-ready стратегии\nкритично walk-forward + ensemble.</li>\n<li><strong>Каждый тикер требует своих параметров</strong>: sensitivity analysis показал\nчто importance параметров сильно различается между тикерами (rsi_period 78.7% для SBER\nvs rsi_low 34% для LKOH).</li>\n<li><strong>1D vs 1h</strong>: для SBER, LKOH, ROSN дневной ТФ даёт лучший Sharpe (за счёт\nизбирательности), часовой — больший profit (за счёт большего числа сделок).</li>\n</ol>\n\n<h3>10.3 Практическая значимость</h3>\n<ul>\n<li>Реализован фреймворк для backtesting+optimization 17 TA стратегий</li>\n<li>Порт VF Strategy из PineScript (88.9% match с TradingView)</li>\n<li>4 метода оптимизации (Bayesian, NSGA-II, Gradient, Walk-forward)</li>\n<li>Автоматическая генерация отчётов (HTML + DOCX + сводный)</li>\n<li>Честная OOS валидация на 2025-2026</li>\n</ul>\n\n<h3>10.4 Направления дальнейших исследований</h3>\n<ul>\n<li>Ensemble нескольких Pareto-точек для устойчивости</li>\n<li>Hierarchical Bayesian для multi-asset оптимизации (shared priors)</li>\n<li>Учёт transaction costs (commission + slippage)</li>\n<li>Расширение фундаментальных данных на все 10 тикеров</li>\n<li>Integration с real-time trading API</li>\n</ul>\n</section>\n'
    return html

def appendix(ta_metrics: pd.DataFrame, vf_summary: pd.DataFrame) -> str:
    html = '<section id="appendix"><h2>Приложения</h2>'
    html += '<h3>A1. Список файлов результатов</h3>'
    html += '<ul>'
    for dirname, desc in [(RESULTS_DIR, 'TA-стратегии per-ticker reports'), (VF_DIR, 'VF Strategy полные результаты (273 MB, ~250 files)'), (OPT_DIR, 'Top-5 оптимизация per strategy type'), (WF_DIR, 'Walk-forward параметры и equity curves')]:
        if os.path.exists(dirname):
            n_files = sum((1 for _ in Path(dirname).rglob('*') if _.is_file()))
            html += f'<li><code>{dirname}/</code> — {desc} ({n_files} файлов)</li>'
    html += '</ul>'
    if not vf_summary.empty:
        html += '<h3>A2. VF Strategy сводка по всем тикерам</h3>'
        html += df_to_html(vf_summary.head(100))
    html += '\n<h3>A3. Глоссарий</h3>\n<dl>\n<dt>TA (Technical Analysis)</dt><dd>Технический анализ — методы прогнозирования цен\nна основе исторических данных о ценах и объёмах.</dd>\n<dt>FA (Fundamental Analysis)</dt><dd>Фундаментальный анализ — оценка активов по\nфинансовой отчётности, макроэкономическим факторам.</dd>\n<dt>WACC</dt><dd>Weighted Average Cost of Capital — средневзвешенная стоимость капитала.</dd>\n<dt>ERP</dt><dd>Equity Risk Premium — премия за риск владения акциями.</dd>\n<dt>OOS / IS</dt><dd>Out-of-Sample / In-Sample — данные вне/в обучающей выборке.</dd>\n<dt>MaxDD</dt><dd>Maximum Drawdown — максимальное относительное падение от пика equity.</dd>\n<dt>VaR</dt><dd>Value at Risk — максимальный потенциальный убыток с заданной вероятностью.</dd>\n<dt>ES / CVaR</dt><dd>Expected Shortfall — средний убыток при превышении VaR.</dd>\n<dt>NSGA-II</dt><dd>Non-dominated Sorting Genetic Algorithm II — эволюционный алгоритм для\nmulti-objective оптимизации.</dd>\n<dt>TPE</dt><dd>Tree-structured Parzen Estimator — байесовский sampler для оптимизации.</dd>\n<dt>fANOVA</dt><dd>Functional ANOVA — метод анализа важности параметров.</dd>\n</dl>\n'
    html += '</section>'
    return html

def generate_toc() -> str:
    return '\n<nav id="toc" class="bg-light p-3 rounded my-4">\n<h3>Оглавление</h3>\n<ol>\n  <li><a href="#ch1">Введение</a></li>\n  <li><a href="#ch2">Методы технического анализа</a></li>\n  <li><a href="#ch3">Методы фундаментального анализа</a></li>\n  <li><a href="#ch4">Статистические методы</a></li>\n  <li><a href="#ch5">Критерии оценки эффективности</a></li>\n  <li><a href="#ch6">Методы оптимизации параметров</a></li>\n  <li><a href="#ch7">Результаты по тикерам</a>\n    <ul>\n' + '\n'.join((f'<li><a href="#ticker-{t}">{t}</a></li>' for t in ALL_TICKERS)) + '\n    </ul>\n  </li>\n  <li><a href="#ch8">Сравнительный анализ TA vs FA vs Statistical</a></li>\n  <li><a href="#ch9">Walk-forward анализ</a></li>\n  <li><a href="#ch10">Заключение</a></li>\n  <li><a href="#appendix">Приложения</a></li>\n</ol>\n</nav>\n'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=f'{RESULTS_DIR}/THESIS_CONSOLIDATED_REPORT.html')
    args = parser.parse_args()
    print('Собираю данные...')
    ta_metrics = collect_ta_metrics()
    vf_summary = collect_vf_summary()
    comparative = collect_comparative()
    period_stats = collect_period_stats()
    hypotheses = test_hypotheses(ta_metrics, comparative)
    print(f'  TA metrics: {len(ta_metrics)} строк')
    print(f'  VF summary: {len(vf_summary)} строк')
    print(f'  Comparative: {len(comparative)} строк')
    print(f'  Period stats: {len(period_stats)} тикеров')
    print(f"  Hypothesis tests: H1={('✓' if hypotheses.get('H1') else '—')}, H2={('✓' if hypotheses.get('H2') else '—')}")
    print('Генерирую секции...')
    sections = [chapter_1_intro(), chapter_2_ta_methods(), chapter_3_fundamental(), chapter_4_statistical(), chapter_5_criteria(), chapter_6_optimization(), chapter_7_per_ticker(ta_metrics, vf_summary, period_stats), chapter_8_comparative(comparative, hypotheses), chapter_9_walkforward(), chapter_10_conclusion(hypotheses), appendix(ta_metrics, vf_summary)]
    html = f"""<!DOCTYPE html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>Диссертация — Роль технического анализа в формировании инвестиционной стратегии</title>\n<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>\n<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>\n<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>\n<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">\n<style>\n  body {{ font-family: Georgia, "Times New Roman", serif; max-width: 1280px; margin: 24px auto;\n          padding: 0 24px; line-height: 1.6; color: #222; background: #fafafa; }}\n  h1 {{ color: #1a1a1a; border-bottom: 3px solid #0d6efd; padding-bottom: 12px; }}\n  h2 {{ color: #0d3b66; margin-top: 48px; padding-bottom: 8px;\n        border-bottom: 2px solid #d0e3ff; page-break-before: always; }}\n  h3 {{ color: #14539d; margin-top: 28px; }}\n  h4 {{ color: #444; }}\n  .table {{ font-size: 0.9rem; }}\n  .table-sm td, .table-sm th {{ font-size: 0.85rem; padding: 0.35rem; }}\n  table[data-caption]::before {{ content: attr(data-caption); display: block;\n                                   font-style: italic; color: #666; margin-bottom: 4px; }}\n  .alert {{ margin: 16px 0; }}\n  #toc ul {{ list-style: none; padding-left: 20px; }}\n  #toc a {{ text-decoration: none; color: #0d6efd; }}\n  #toc a:hover {{ text-decoration: underline; }}\n  section {{ background: white; padding: 24px; border-radius: 8px; margin: 16px 0;\n              box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}\n  code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}\n  dt {{ font-weight: bold; margin-top: 8px; }}\n  dd {{ margin-left: 16px; color: #555; }}\n  .header-meta {{ color: #666; font-size: 0.95rem; }}\n</style>\n</head>\n<body>\n\n<header>\n<h1>Диссертация: Роль методов технического анализа<br>в формировании инвестиционной стратегии</h1>\n<p class="header-meta">\n<strong>Автор:</strong> Фурасов Владислав Дмитриевич &nbsp;|&nbsp;\n<strong>Научный руководитель:</strong> Манаев Владимир Николаевич &nbsp;|&nbsp;\n<strong>МГУ им. М.В. Ломоносова, Экономический факультет, 2025</strong><br>\n<strong>Единый HTML-отчёт:</strong> сгенерирован {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} |\nПериод данных: 2014-01-06 .. 2026-04-19 | Тикеры: 10 (MOEX)\n</p>\n</header>\n\n{generate_toc()}\n\n{''.join(sections)}\n\n<footer class="text-muted text-center mt-5 mb-4" style="border-top: 2px solid #ddd; padding-top: 16px;">\n<small>Консолидированный HTML-отчёт диссертации. Генерируется <code>consolidated_report.py</code>.<br>\nПолные методологические отчёты в <code>results/vf_strategy/*.md</code>. Детали per-ticker\nчерез ссылки в главе 7.</small>\n</footer>\n\n</body>\n</html>\n"""
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f'\n✓ Отчёт сохранён: {out_path} ({size_mb:.2f} MB)')
if __name__ == '__main__':
    main()
