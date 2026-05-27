import os
import pandas as pd
import numpy as np
from strategies.technical import EMACrossoverStrategy, LongTermHold, RSIStrategy, MACDHistogramStrategy, TrendlineBreakoutStrategy, TrendlineRetestStrategy, BreakoutPivotStrategy, BreakoutOLSPivotStrategy, RetestPivotStrategy, RetestOLSPivotStrategy, OBVSignalStrategy, VWAPSignalStrategy, EngulfingStrategy, HammerDojiStrategy, HaramiStrategy, CandleEnsembleStrategy
from strategies.fundamental import DDMGordonStrategy, DividendYieldStrategy, DCFTwoStageStrategy, DCFMultiStageStrategy, MultipleRegressionStrategy, FUND_TICKERS, EPSGrowthStrategy, ROEStrategy, NetProfitStrategy, RelativeValuationStrategy, MultiplierTrendStrategy, ComplexScoringStrategy, PriceDisconnectStrategy
from strategies.statistical import ARIMASignalStrategy, GARCHVolTargetStrategy, TGARCHVolTargetStrategy, VARSignalStrategy, VARMAXSignalStrategy, ImprovedVARStrategy, ExponentialSmoothingStrategy, ExpectedShortfallStrategy
from reports.report import generate_report
from optimization.bayesian_ema import optimize_ema_parameters, optimize_ema_parameters_1h
from optimization.gradient_ema import gradient_descent_optimization

def run_optimized_strategy(ticker):
    print('Запуск оптимизированной стратегии (1D) с найденными параметрами...')
    best_params, best_composite = optimize_ema_parameters(n_trials=200)
    print('Оптимальные параметры (1D):', best_params)
    print('Лучший составной показатель (1D):', best_composite)
    strat_opt = EMACrossoverStrategy(ticker=ticker, timeframe='1D', initial_account_size=1000000, entry_percentage=100, short_ema=best_params['short_ema'], long_ema=best_params['long_ema'], pyramiding=1, allowshort=best_params['allowshort'], debug=False)
    strat_opt.generate_signals()
    strat_opt.execute_trades()
    strat_opt.print_results()
    strat_opt.equity_df = strat_opt.get_equity_curve()
    return strat_opt

def run_optimized_strategy_1h(ticker):
    print('Запуск оптимизированной стратегии (1h) с найденными параметрами...')
    best_params, best_composite = optimize_ema_parameters_1h(n_trials=200)
    print('Оптимальные параметры (1h):', best_params)
    print('Лучший составной показатель (1h):', best_composite)
    strat_opt_1h = EMACrossoverStrategy(ticker=ticker, timeframe='1h', initial_account_size=1000000, entry_percentage=100, short_ema=best_params['short_ema'], long_ema=best_params['long_ema'], pyramiding=1, allowshort=best_params['allowshort'], debug=False)
    strat_opt_1h.generate_signals()
    strat_opt_1h.execute_trades()
    strat_opt_1h.print_results()
    strat_opt_1h.equity_df = strat_opt_1h.get_equity_curve()
    return strat_opt_1h

def run_gradient_descent_strategy(ticker, timeframe='1D', allowshort=False):
    print(f'Запуск оптимизированной стратегии (градиентный спуск) для {timeframe} данных, allowshort={allowshort}...')
    best_params, best_score = gradient_descent_optimization(allowshort, timeframe, ticker, initial_params=[50, 100], lr=1.0, max_iter=100, tol=0.001)
    print('Оптимальные параметры (градиентный спуск, {}):'.format(timeframe), best_params)
    print('Лучший composite score (градиентный спуск, {}):'.format(timeframe), best_score)
    strat_grad = EMACrossoverStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=1000000, entry_percentage=100, short_ema=int(round(best_params[0])), long_ema=int(round(best_params[1])), pyramiding=1, allowshort=allowshort, debug=False)
    strat_grad.generate_signals()
    strat_grad.execute_trades()
    strat_grad.print_results()
    strat_grad.equity_df = strat_grad.get_equity_curve()
    return strat_grad

def run_gradient_descent_strategy_1h(ticker):
    return run_gradient_descent_strategy(ticker, timeframe='1h', allowshort=True)

def run_strategy_safe(strat):
    try:
        strat.generate_signals()
        strat.execute_trades()
        strat.print_results()
        return True
    except Exception as e:
        print(f'  Ошибка в стратегии {strat.name}: {e}')
        return False

def run_strategies(ticker):
    initial_account_size = 1000000
    entry_percentage = 100
    strategies_list = []
    for short_ema, long_ema in [(50, 100), (50, 200)]:
        strat = EMACrossoverStrategy(ticker, '1D', initial_account_size, entry_percentage, pyramiding=1, allowshort=False, short_ema=short_ema, long_ema=long_ema)
        if run_strategy_safe(strat):
            strategies_list.append(strat)
    for short_ema, long_ema in [(50, 100), (50, 200)]:
        strat = EMACrossoverStrategy(ticker, '1h', initial_account_size, entry_percentage, pyramiding=1, allowshort=False, short_ema=short_ema, long_ema=long_ema)
        if run_strategy_safe(strat):
            strategies_list.append(strat)
    for short_ema, long_ema in [(50, 100), (50, 200)]:
        strat = EMACrossoverStrategy(ticker, '1D', initial_account_size, entry_percentage, pyramiding=1, allowshort=True, short_ema=short_ema, long_ema=long_ema)
        if run_strategy_safe(strat):
            strategies_list.append(strat)
    for short_ema, long_ema in [(50, 100), (50, 200)]:
        strat = EMACrossoverStrategy(ticker, '1h', initial_account_size, entry_percentage, pyramiding=1, allowshort=True, short_ema=short_ema, long_ema=long_ema)
        if run_strategy_safe(strat):
            strategies_list.append(strat)
    strat_bh = LongTermHold(ticker, '1D', initial_account_size, entry_percentage)
    if run_strategy_safe(strat_bh):
        strategies_list.append(strat_bh)
    for tf in ['1D', '1h']:
        for rsi_lower, rsi_upper in [(30, 70), (20, 80)]:
            for allowshort in (True, False):
                strat = RSIStrategy(ticker, tf, initial_account_size, entry_percentage, rsi_lower=rsi_lower, rsi_upper=rsi_upper, rsi_window=14, pyramiding=1, allowshort=allowshort)
                if run_strategy_safe(strat):
                    strategies_list.append(strat)
    for tf in ['1D', '1h']:
        for allowshort in (True, False):
            strat = MACDHistogramStrategy(ticker, tf, initial_account_size, entry_percentage, pyramiding=1, allowshort=allowshort)
            if run_strategy_safe(strat):
                strategies_list.append(strat)
    for tf in ['1D', '1h']:
        for allowshort in (True, False):
            strat_obv = OBVSignalStrategy(ticker, tf, initial_account_size, entry_percentage, obv_ema_window=14, pyramiding=1, allowshort=allowshort)
            if run_strategy_safe(strat_obv):
                strategies_list.append(strat_obv)
    for tf in ['1D', '1h']:
        mode = 'yearly' if tf == '1D' else 'rolling'
        for allowshort in (True, False):
            strat_vwap = VWAPSignalStrategy(ticker, tf, initial_account_size, entry_percentage, vwap_window=100, vwap_tol=0.005, vwap_mode=mode, pyramiding=1, allowshort=allowshort)
            if run_strategy_safe(strat_vwap):
                strategies_list.append(strat_vwap)
    for tf in ['1D', '1h']:
        for allowshort in (True, False):
            for cls in (EngulfingStrategy, HammerDojiStrategy, HaramiStrategy):
                strat_c = cls(ticker, tf, initial_account_size, entry_percentage, pyramiding=1, allowshort=allowshort)
                if run_strategy_safe(strat_c):
                    strategies_list.append(strat_c)
            strat_ens = CandleEnsembleStrategy(ticker, tf, initial_account_size, entry_percentage, ensemble_window=5, min_votes=2, pyramiding=1, allowshort=allowshort)
            if run_strategy_safe(strat_ens):
                strategies_list.append(strat_ens)
    strat_tbr = TrendlineBreakoutStrategy(ticker, '1D', initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
    if run_strategy_safe(strat_tbr):
        strategies_list.append(strat_tbr)
    strat_trt = TrendlineRetestStrategy(ticker, '1D', initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
    if run_strategy_safe(strat_trt):
        strategies_list.append(strat_trt)
    strat_bp = BreakoutPivotStrategy(ticker, '1D', initial_account_size, entry_percentage, pyramiding=1, allowshort=True, pivot_window=5, min_touches=3, lookback=200)
    if run_strategy_safe(strat_bp):
        strategies_list.append(strat_bp)
    strat_bols = BreakoutOLSPivotStrategy(ticker, '1D', initial_account_size, entry_percentage, pyramiding=1, allowshort=True, pivot_window=5, min_touches=3, lookback=200)
    if run_strategy_safe(strat_bols):
        strategies_list.append(strat_bols)
    strat_rp = RetestPivotStrategy(ticker, '1D', initial_account_size, entry_percentage, pyramiding=1, allowshort=True, pivot_window=5, min_touches=3, lookback=200)
    if run_strategy_safe(strat_rp):
        strategies_list.append(strat_rp)
    strat_rols = RetestOLSPivotStrategy(ticker, '1D', initial_account_size, entry_percentage, pyramiding=1, allowshort=True, pivot_window=5, min_touches=3, lookback=200)
    if run_strategy_safe(strat_rols):
        strategies_list.append(strat_rols)
    strat_arima = ARIMASignalStrategy(ticker, timeframe='1D', initial_account_size=initial_account_size, entry_percentage=entry_percentage, pyramiding=1, allowshort=True)
    if run_strategy_safe(strat_arima):
        strategies_list.append(strat_arima)
    strat_garch = GARCHVolTargetStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
    if run_strategy_safe(strat_garch):
        strategies_list.append(strat_garch)
    strat_tgarch = TGARCHVolTargetStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
    if run_strategy_safe(strat_tgarch):
        strategies_list.append(strat_tgarch)
    sa_extra = [VARSignalStrategy(ticker, timeframe='1D', initial_account_size=initial_account_size, entry_percentage=entry_percentage, pyramiding=1, allowshort=True), VARMAXSignalStrategy(ticker, timeframe='1D', initial_account_size=initial_account_size, entry_percentage=entry_percentage, pyramiding=1, allowshort=True), ImprovedVARStrategy(ticker, timeframe='1D', initial_account_size=initial_account_size, entry_percentage=entry_percentage, pyramiding=1, allowshort=True), ExponentialSmoothingStrategy(ticker, timeframe='1D', initial_account_size=initial_account_size, entry_percentage=entry_percentage, pyramiding=1, allowshort=True), ExpectedShortfallStrategy(ticker, timeframe='1D', initial_account_size=initial_account_size, entry_percentage=entry_percentage, pyramiding=1, allowshort=True)]
    for strat in sa_extra:
        if run_strategy_safe(strat):
            strategies_list.append(strat)
    if ticker in FUND_TICKERS:
        strat_ddm = DDMGordonStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
        if run_strategy_safe(strat_ddm):
            strategies_list.append(strat_ddm)
        strat_dy = DividendYieldStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
        if run_strategy_safe(strat_dy):
            strategies_list.append(strat_dy)
        strat_dcf2 = DCFTwoStageStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
        if run_strategy_safe(strat_dcf2):
            strategies_list.append(strat_dcf2)
        strat_dcfm = DCFMultiStageStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
        if run_strategy_safe(strat_dcfm):
            strategies_list.append(strat_dcfm)
        strat_mr = MultipleRegressionStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True)
        if run_strategy_safe(strat_mr):
            strategies_list.append(strat_mr)
        fa_extra = [EPSGrowthStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True), ROEStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True), NetProfitStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True), RelativeValuationStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, multiplier='P/E'), RelativeValuationStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, multiplier='P/B'), MultiplierTrendStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, multiplier='P/E'), ComplexScoringStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, multipliers=('P/E', 'P/B'), weights=(0.6, 0.4)), PriceDisconnectStrategy(ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True)]
        for strat in fa_extra:
            if run_strategy_safe(strat):
                strategies_list.append(strat)
    for strat in strategies_list:
        if not hasattr(strat, 'equity_df') or strat.equity_df is None:
            strat.equity_df = strat.get_equity_curve()
    export_combined_trades(ticker, strategies_list)
    export_combined_stats(ticker, strategies_list)
    export_combined_equity(ticker, strategies_list)
    generate_report(ticker, strategies_list)

def export_combined_trades(ticker, strategies):
    dfs = []
    for strat in strategies:
        df_closed = strat.get_trades_df().copy(deep=True)
        open_records = []
        for trade in strat.trades:
            if not trade.is_closed:
                fmt = '%Y-%m-%d' if strat.timeframe == '1D' else '%Y-%m-%d %H:%M'
                open_records.append({'№ сделки': trade.trade_id, 'Дата': trade.entry_time.strftime(fmt), 'Цена': trade.entry_price, 'Сигнал': trade.trade_type if trade.trade_type is not None else 'open ' + ('long' if trade.direction == 'long' else 'short'), 'Количество': trade.quantity, 'Прибыль/Убыток': trade.result, 'Время удержания': '', 'Комментарий': trade.comment, 'raw_date': trade.entry_time})
        df_open = pd.DataFrame(open_records).copy(deep=True)
        if not df_closed.empty and (not df_open.empty):
            df_all = pd.concat([df_closed, df_open], ignore_index=True).copy(deep=True)
        elif not df_closed.empty:
            df_all = df_closed.copy(deep=True)
        elif not df_open.empty:
            df_all = df_open.copy(deep=True)
        else:
            continue
        df_all = df_all.copy(deep=True).set_index('raw_date')
        prefix = f'{strat.name} ({strat.timeframe})'
        df_all = df_all.rename(columns={'Дата': f'Дата_{prefix}', 'Цена': f'Цена_{prefix}', 'Сигнал': f'Сигнал_{prefix}', 'Количество': f'Кол-во_{prefix}', 'Прибыль/Убыток': f'Прибыль/Убыток_{prefix}', 'Время удержания': f'Время удержания_{prefix}', 'Комментарий': f'Комментарий_{prefix}'}).copy(deep=True)
        dfs.append(df_all)
    if not dfs:
        print('Нет данных по сделкам для экспорта.')
        return
    combined_df = pd.concat(dfs, axis=1, join='outer').copy(deep=True)
    combined_df.index = pd.to_datetime(combined_df.index)
    combined_df = combined_df.sort_index().copy(deep=True)
    combined_df['Дата+час'] = combined_df.index.strftime('%Y-%m-%d %H:%M')
    cols = ['Дата+час'] + [col for col in combined_df.columns if col != 'Дата+час']
    combined_df = combined_df[cols].copy(deep=True)
    os.makedirs('results', exist_ok=True)
    filename = os.path.join('results', f'{ticker}_strategies_trades.csv')
    combined_df.to_csv(filename, index=False)
    print(f'Таблица сделок сохранена в файл: {filename}')
_STRATEGY_SERVICE_ATTRS = {'data', 'signals_df', 'trades', 'equity_df', 'ticker', 'timeframe', 'name', 'current_account_size', 'current_profit', 'initial_account_size', 'entry_percentage', 'allowshort', 'pyramiding', 'trade_counter', 'partial_entry_pct', 'partial_close_pct', 'unrealized_profit', 'debug', 'income', 'balance', 'cashflow', 'multipliers', 'income_q', 'balance_q', 'cashflow_q', 'multipliers_q', 'rf', 'market_close', 'trading_days'}

def _extract_strategy_params(strat):
    import json
    params = {}
    for k, v in vars(strat).items():
        if k in _STRATEGY_SERVICE_ATTRS or k.startswith('_'):
            continue
        if isinstance(v, (pd.DataFrame, pd.Series)):
            continue
        if isinstance(v, (bool, int, float, str)):
            params[k] = v
        elif isinstance(v, (tuple, list)):
            try:
                json.dumps(v, default=str)
                params[k] = list(v)
            except (TypeError, ValueError):
                continue
    return params

def export_combined_stats(ticker, strategies):
    import json
    stats_list = []
    for strat in strategies:
        stats = strat.get_stats()
        start_date = end_date = None
        if getattr(strat, 'equity_df', None) is not None and (not strat.equity_df.empty):
            start_date = str(strat.equity_df.index[0])
            end_date = str(strat.equity_df.index[-1])
        elif getattr(strat, 'data', None) is not None and (not strat.data.empty):
            start_date = str(strat.data.index[0])
            end_date = str(strat.data.index[-1])
        params_dict = _extract_strategy_params(strat)
        enriched = {'Тикер': strat.ticker, 'Стратегия': stats.pop('Стратегия', strat.name), 'Таймфрейм': stats.pop('Таймфрейм', strat.timeframe), 'AllowShort': bool(getattr(strat, 'allowshort', False)), 'Pyramiding': int(getattr(strat, 'pyramiding', 1)), 'Initial Account Size': float(getattr(strat, 'initial_account_size', 0.0)), 'Entry %': float(getattr(strat, 'entry_percentage', 0.0)), 'Дата начала': start_date, 'Дата окончания': end_date, 'Параметры стратегии': json.dumps(params_dict, ensure_ascii=False, separators=(',', ':'), default=str)}
        enriched.update(stats)
        stats_list.append(enriched)
    if not stats_list:
        print('Нет статистики для экспорта.')
        return
    stats_df = pd.DataFrame(stats_list).copy(deep=True)
    os.makedirs('results', exist_ok=True)
    filename = os.path.join('results', f'{ticker}_strategies_stats.csv')
    stats_df.to_csv(filename, index=False)
    print(f'Статистика стратегий сохранена в файл: {filename}')

def export_combined_equity(ticker, strategies):
    equity_dfs = []
    for strat in strategies:
        eq_df = strat.equity_df.copy(deep=True)
        eq_df.rename(columns={'equity': f'Equity_{strat.name} ({strat.timeframe})'}, inplace=True)
        equity_dfs.append(eq_df)
    if not equity_dfs:
        print('Нет данных по equity для экспорта.')
        return
    combined_eq = None
    for df in equity_dfs:
        if combined_eq is None:
            combined_eq = df.copy(deep=True)
        else:
            combined_eq = pd.merge(combined_eq, df, left_index=True, right_index=True, how='outer').copy(deep=True)
    combined_eq.sort_index(inplace=True)
    repo_file = 'data/RISKFREE_data.csv'
    if not os.path.exists(repo_file):
        print('Файл data/RISKFREE_data.csv не найден (запустите scripts/build_riskfree.py). Риск‑фри кривая не рассчитана.')
        return
    repo_df = pd.read_csv(repo_file)
    repo_df['begin'] = pd.to_datetime(repo_df['begin'])
    repo_df = repo_df.set_index('begin').sort_index().copy(deep=True)
    repo_rate_series = repo_df['close']
    repo_rate_series = repo_rate_series.reindex(combined_eq.index, method='ffill')
    riskfree_values = []
    base = strategies[0].initial_account_size
    for year, group in combined_eq.groupby(combined_eq.index.year):
        start_year = pd.Timestamp(year, 1, 1)
        end_year = pd.Timestamp(year, 12, 31)
        D = (end_year - start_year).days + 1
        year_repo_rate = repo_rate_series[repo_rate_series.index.year == year].mean()
        for current_date in group.index:
            d = (current_date - pd.Timestamp(year, 1, 1)).days
            riskfree_value = base * (1 + year_repo_rate / 100 * (d / D))
            riskfree_values.append((current_date, riskfree_value))
        base = base * (1 + year_repo_rate / 100)
    riskfree_series = pd.Series(dict(riskfree_values)).sort_index()
    combined_eq.index = pd.to_datetime(combined_eq.index)
    combined_eq['Riskfree'] = riskfree_series
    os.makedirs('results', exist_ok=True)
    filename = os.path.join('results', f'{ticker}_strategies_equity.csv')
    combined_eq.to_csv(filename)
    print(f'Equity curve data сохранены в файл: {filename}')
if __name__ == '__main__':
    ticker = 'SBER'
    run_strategies(ticker)
