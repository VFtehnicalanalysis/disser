import optuna
import numpy as np
import pandas as pd
from strategies.ema_longterm import EMACrossoverStrategy

def objective(trial):
    short_ema = trial.suggest_int('short_ema', 5, 100)
    long_ema = trial.suggest_int('long_ema', 50, 200)
    allowshort = trial.suggest_categorical('allowshort', [True, False])
    if short_ema >= long_ema:
        return np.inf
    strat = EMACrossoverStrategy(ticker='SBER', timeframe='1D', initial_account_size=1000000, entry_percentage=100, short_ema=short_ema, long_ema=long_ema, pyramiding=1, allowshort=allowshort, debug=False)
    try:
        strat.load_data()
    except Exception as e:
        print('Ошибка загрузки данных:', e)
        return np.inf
    data = strat.data
    data.index = pd.to_datetime(data.index)
    train_val_mask = (data.index >= pd.Timestamp('2014-01-01')) & (data.index < pd.Timestamp('2023-01-01'))
    strat.data = data.loc[train_val_mask]
    if strat.data.empty:
        return np.inf
    strat.generate_signals()
    strat.execute_trades()
    equity_df = strat.get_equity_curve()
    if equity_df.empty:
        return np.inf
    equity_valid = equity_df[(equity_df.index >= pd.Timestamp('2022-01-01')) & (equity_df.index < pd.Timestamp('2023-01-01'))]
    if equity_valid.empty:
        return np.inf
    initial_equity_valid = equity_valid.iloc[0]['equity']
    final_equity_valid = equity_valid.iloc[-1]['equity']
    profit_growth = (final_equity_valid / initial_equity_valid - 1) * 100
    running_max = equity_valid['equity'].cummax()
    drawdowns = (running_max - equity_valid['equity']) / running_max * 100
    max_drawdown = drawdowns.max()
    returns = equity_valid['equity'].pct_change().dropna()
    if returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
    if sharpe is None or np.isnan(sharpe):
        return np.inf
    if profit_growth is None or np.isnan(profit_growth):
        return np.inf
    if max_drawdown is None or np.isnan(max_drawdown) or max_drawdown < 1e-05:
        max_drawdown = 1e-05
    composite_score = sharpe * (profit_growth / max_drawdown)
    trial.set_user_attr('sharpe', sharpe)
    trial.set_user_attr('profit_growth', profit_growth)
    trial.set_user_attr('max_drawdown', max_drawdown)
    trial.set_user_attr('composite_score', composite_score)
    return -composite_score

def optimize_ema_parameters(n_trials=200):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_composite = -study.best_value
    print('Лучшие параметры для дневных данных:', best_params)
    print('Лучший составной показатель (composite score) для дневных данных:', best_composite)
    return (best_params, best_composite)

def objective_1h(trial):
    short_ema = trial.suggest_int('short_ema', 5, 100)
    long_ema = trial.suggest_int('long_ema', 50, 200)
    allowshort = trial.suggest_categorical('allowshort', [True, False])
    if short_ema >= long_ema:
        return np.inf
    strat = EMACrossoverStrategy(ticker='SBER', timeframe='1h', initial_account_size=1000000, entry_percentage=100, short_ema=short_ema, long_ema=long_ema, pyramiding=1, allowshort=allowshort, debug=False)
    try:
        strat.load_data()
    except Exception as e:
        print('Ошибка загрузки данных:', e)
        return np.inf
    data = strat.data
    data.index = pd.to_datetime(data.index)
    train_val_mask = (data.index >= pd.Timestamp('2014-01-01')) & (data.index < pd.Timestamp('2023-01-01'))
    strat.data = data.loc[train_val_mask]
    if strat.data.empty:
        return np.inf
    strat.generate_signals()
    strat.execute_trades()
    equity_df = strat.get_equity_curve()
    if equity_df.empty:
        return np.inf
    equity_valid = equity_df[(equity_df.index >= pd.Timestamp('2022-01-01')) & (equity_df.index < pd.Timestamp('2023-01-01'))]
    if equity_valid.empty:
        return np.inf
    initial_equity_valid = equity_valid.iloc[0]['equity']
    final_equity_valid = equity_valid.iloc[-1]['equity']
    profit_growth = (final_equity_valid / initial_equity_valid - 1) * 100
    running_max = equity_valid['equity'].cummax()
    drawdowns = (running_max - equity_valid['equity']) / running_max * 100
    max_drawdown = drawdowns.max()
    returns = equity_valid['equity'].pct_change().dropna()
    if returns.std() == 0:
        sharpe = np.nan
    else:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6.5)
    if sharpe is None or np.isnan(sharpe):
        return np.inf
    if profit_growth is None or np.isnan(profit_growth):
        return np.inf
    if max_drawdown is None or np.isnan(max_drawdown) or max_drawdown < 1e-05:
        max_drawdown = 1e-05
    composite_score = sharpe * (profit_growth / max_drawdown)
    trial.set_user_attr('sharpe', sharpe)
    trial.set_user_attr('profit_growth', profit_growth)
    trial.set_user_attr('max_drawdown', max_drawdown)
    trial.set_user_attr('composite_score', composite_score)
    return -composite_score

def optimize_ema_parameters_1h(n_trials=200):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_1h, n_trials=n_trials)
    best_params = study.best_params
    best_composite = -study.best_value
    print('Лучшие параметры для часовых данных:', best_params)
    print('Лучший составной показатель (composite score) для часовых данных:', best_composite)
    return (best_params, best_composite)
if __name__ == '__main__':
    print('Оптимизация для дневных данных:')
    optimize_ema_parameters()
    print('Оптимизация для часовых данных:')
    optimize_ema_parameters_1h()
