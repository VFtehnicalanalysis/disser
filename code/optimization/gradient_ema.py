import numpy as np
import pandas as pd
from copy import deepcopy
from strategies.ema_longterm import EMACrossoverStrategy

def evaluate_strategy(short_ema, long_ema, allowshort, timeframe, ticker):
    short_ema = int(round(short_ema))
    long_ema = int(round(long_ema))
    if short_ema >= long_ema:
        return -1000000.0
    strat = EMACrossoverStrategy(ticker=ticker, timeframe=timeframe, initial_account_size=1000000, entry_percentage=100, short_ema=short_ema, long_ema=long_ema, pyramiding=1, allowshort=allowshort, debug=False)
    try:
        strat.load_data()
    except Exception as e:
        print('Ошибка загрузки данных:', e)
        return -1000000.0
    data = strat.data
    data.index = pd.to_datetime(data.index)
    mask = (data.index >= pd.Timestamp('2014-01-01')) & (data.index < pd.Timestamp('2023-01-01'))
    strat.data = data.loc[mask]
    if strat.data.empty:
        return -1000000.0
    strat.generate_signals()
    strat.execute_trades()
    equity_df = strat.get_equity_curve()
    if equity_df.empty:
        return -1000000.0
    valid_mask = (equity_df.index >= pd.Timestamp('2022-01-01')) & (equity_df.index < pd.Timestamp('2023-01-01'))
    equity_valid = equity_df.loc[valid_mask]
    if equity_valid.empty:
        return -1000000.0
    initial_equity = equity_valid.iloc[0]['equity']
    final_equity = equity_valid.iloc[-1]['equity']
    profit_growth = (final_equity / initial_equity - 1) * 100
    running_max = equity_valid['equity'].cummax()
    drawdowns = (running_max - equity_valid['equity']) / running_max * 100
    max_drawdown = drawdowns.max()
    returns = equity_valid['equity'].pct_change().dropna()
    if returns.std() == 0:
        sharpe = np.nan
    elif timeframe == '1D':
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    elif timeframe == '1h':
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6.5)
    else:
        sharpe = returns.mean() / returns.std()
    if sharpe is None or np.isnan(sharpe) or profit_growth is None or np.isnan(profit_growth):
        return -1000000.0
    if max_drawdown is None or np.isnan(max_drawdown) or max_drawdown < 1e-05:
        max_drawdown = 1e-05
    composite_score = sharpe * (profit_growth / max_drawdown)
    return composite_score

def objective_value(params, allowshort, timeframe, ticker):
    short_ema, long_ema = params
    if short_ema >= long_ema:
        return 1000000.0
    score = evaluate_strategy(short_ema, long_ema, allowshort, timeframe, ticker)
    return -score

def compute_gradient(params, allowshort, timeframe, ticker, delta=1.0):
    grad = np.zeros_like(params, dtype=float)
    f0 = objective_value(params, allowshort, timeframe, ticker)
    for i in range(len(params)):
        params_delta = params.copy()
        params_delta[i] += delta
        f_delta = objective_value(params_delta, allowshort, timeframe, ticker)
        grad[i] = (f_delta - f0) / delta
    return grad

def gradient_descent_optimization(allowshort, timeframe, ticker, initial_params=[50, 100], lr=1.0, max_iter=10, tol=0.001):
    params = np.array(initial_params, dtype=float)
    best_val = objective_value(params, allowshort, timeframe, ticker)
    best_params = params.copy()
    for iteration in range(max_iter):
        grad = compute_gradient(params, allowshort, timeframe, ticker)
        new_params = params - lr * grad
        new_params[0] = np.clip(new_params[0], 5, 100)
        new_params[1] = np.clip(new_params[1], 50, 200)
        if new_params[0] >= new_params[1]:
            new_params[0] = new_params[1] - 1
        new_val = objective_value(new_params, allowshort, timeframe, ticker)
        print(f'Iteration {iteration}: params = {new_params}, objective = {new_val}')
        if abs(new_val - best_val) < tol:
            best_val = new_val
            best_params = new_params.copy()
            break
        if new_val < best_val:
            best_val = new_val
            best_params = new_params.copy()
        params = new_params
    return (best_params, -best_val)
if __name__ == '__main__':
    ticker = 'SBER'
    allowshort = False
    timeframe = '1D'
    best_params, best_score = gradient_descent_optimization(allowshort, timeframe, ticker, initial_params=[50, 100], lr=1.0, max_iter=10)
    print('Оптимальные параметры:', best_params)
    print('Лучший composite score:', best_score)
