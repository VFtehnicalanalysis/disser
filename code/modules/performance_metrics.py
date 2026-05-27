import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def get_statistics(weights, returns):
    weights = np.array(weights)
    portfolio_return = np.dot(weights, returns.mean()) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    return np.array([portfolio_return, portfolio_volatility])

def add_cal_to_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data['CAL_LKOH'] = np.nan
    data['CAL_SBER'] = np.nan
    tickers = ['LKOH', 'SBER']
    for i in range(1, len(data)):
        sub_data = data.iloc[:i]
        for ticker in tickers:
            returns = sub_data[f'log_return_{ticker}']
            returns = returns.dropna().to_frame()
            if returns.empty:
                continue
            num_assets = returns.shape[1]
            bounds = tuple(((0, 1) for _ in range(num_assets)))
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            risk_free_rate = sub_data['risk_free_rate'].iloc[-1] / 100
            result = minimize(lambda weights: get_statistics(weights, returns)[1], num_assets * [1.0 / num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
            weights = result.x
            opt_return, opt_volatility = get_statistics(weights, returns)
            max_sharpe_ratio = (opt_return - risk_free_rate) / opt_volatility
            cal_value = risk_free_rate + max_sharpe_ratio * opt_volatility
            data.at[data.index[i], f'CAL_{ticker}'] = cal_value
    data.to_csv(file_path)
    print("CAL values have been added to 'market_data.csv'.")

def add_cml_to_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data['CML_LKOH'] = np.nan
    data['CML_SBER'] = np.nan
    tickers = ['LKOH', 'SBER']
    for i in range(1, len(data)):
        sub_data = data.iloc[:i]
        market_returns = sub_data['log_return_IMOEX']
        market_return = np.sum(market_returns) * 252 / len(market_returns)
        market_volatility = market_returns.std() * np.sqrt(252)
        risk_free_rate = sub_data['risk_free_rate'].iloc[-1] / 100
        for ticker in tickers:
            returns = sub_data[f'log_return_{ticker}']
            ticker_volatility = returns.std() * np.sqrt(252)
            cml_value = risk_free_rate + (market_return - risk_free_rate) / market_volatility * ticker_volatility
            data.at[data.index[i], f'CML_{ticker}'] = cml_value
    data.to_csv(file_path)
    print("CML values have been added to 'market_data.csv'.")

def add_sml_to_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data['SML_LKOH'] = np.nan
    data['SML_SBER'] = np.nan
    tickers = ['LKOH', 'SBER']
    for i in range(1, len(data)):
        sub_data = data.iloc[:i]
        risk_free_rate = sub_data['risk_free_rate'].iloc[-1] / 100
        market_returns = sub_data['log_return_IMOEX']
        market_return = np.sum(market_returns) * 252 / len(market_returns)
        for ticker in tickers:
            returns = sub_data[f'log_return_{ticker}']
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance != 0 else 0
            sml_value = risk_free_rate + beta * (market_return - risk_free_rate)
            data.at[data.index[i], f'SML_{ticker}'] = sml_value
    data.to_csv(file_path)
    print("SML values have been added to 'market_data.csv'.")

def add_treynor_ratio_to_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data['Treynor_LKOH'] = np.nan
    data['Treynor_SBER'] = np.nan
    tickers = ['LKOH', 'SBER']
    for i in range(1, len(data)):
        sub_data = data.iloc[:i]
        market_returns = sub_data['log_return_IMOEX']
        market_return = np.sum(market_returns) * 252 / len(market_returns)
        risk_free_rate = sub_data['risk_free_rate'].iloc[-1] / 100
        for ticker in tickers:
            returns = sub_data[f'log_return_{ticker}']
            portfolio_return = np.sum(returns) * 252 / len(returns)
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance != 0 else 0
            treynor_ratio = (portfolio_return - risk_free_rate) / beta if beta != 0 else np.nan
            data.at[data.index[i], f'Treynor_{ticker}'] = treynor_ratio
    data.to_csv(file_path)
    print("Treynor Ratio values have been added to 'market_data.csv'.")

def add_sortino_ratio_to_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data['Sortino_LKOH'] = np.nan
    data['Sortino_SBER'] = np.nan
    tickers = ['LKOH', 'SBER']
    for i in range(1, len(data)):
        sub_data = data.iloc[:i]
        risk_free_rate = sub_data['risk_free_rate'].iloc[-1] / 100
        for ticker in tickers:
            returns = sub_data[f'log_return_{ticker}']
            portfolio_return = np.sum(returns) * 252 / len(returns)
            negative_returns = returns[returns < 0]
            downside_risk = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else np.nan
            sortino_ratio = (portfolio_return - risk_free_rate) / downside_risk if downside_risk != 0 else np.nan
            data.at[data.index[i], f'Sortino_{ticker}'] = sortino_ratio
    data.to_csv(file_path)
    print("Sortino Ratio values have been added to 'market_data.csv'.")

def add_jensens_alpha_to_csv(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data['Jensens_Alpha_LKOH'] = np.nan
    data['Jensens_Alpha_SBER'] = np.nan
    tickers = ['LKOH', 'SBER']
    for i in range(1, len(data)):
        sub_data = data.iloc[:i]
        market_returns = sub_data['log_return_IMOEX']
        market_return = np.sum(market_returns) * 252 / len(market_returns)
        risk_free_rate = sub_data['risk_free_rate'].iloc[-1] / 100
        for ticker in tickers:
            returns = sub_data[f'log_return_{ticker}']
            portfolio_return = np.sum(returns) * 252 / len(returns)
            covariance = np.cov(returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance != 0 else 0
            jensens_alpha = portfolio_return - (risk_free_rate + beta * (market_return - risk_free_rate))
            data.at[data.index[i], f'Jensens_Alpha_{ticker}'] = jensens_alpha
    data.to_csv(file_path)
    print("Jensen's Alpha values have been added to 'market_data.csv'.")

def static_plots(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    risk_free_rate = data['risk_free_rate'].mean() / 100
    market_return = data['log_return_IMOEX'].mean() * 252
    market_volatility = data['log_return_IMOEX'].std() * np.sqrt(252)
    tickers = ['LKOH', 'SBER']
    fig, ax = plt.subplots(3, 1, figsize=(10, 18))
    for ticker in tickers:
        ticker_returns = data[f'log_return_{ticker}']
        ticker_volatility = ticker_returns.std() * np.sqrt(252)
        cal_values = data[f'CAL_{ticker}']
        ax[0].plot(ticker_volatility, cal_values.mean(), 'o', label=f'{ticker} Optimal Portfolio')
    x = np.linspace(0, max(data[[f'CAL_{ticker}' for ticker in tickers]].max()), 100)
    y = risk_free_rate + (market_return - risk_free_rate) / market_volatility * x
    ax[0].plot(x, y, label='Capital Allocation Line', color='orange')
    ax[0].set_xlabel('Portfolio Standard Deviation')
    ax[0].set_ylabel('Portfolio Expected Return')
    ax[0].legend()
    ax[0].set_title('Capital Allocation Line (CAL) and Optimal Portfolio')
    for ticker in tickers:
        ticker_returns = data[f'log_return_{ticker}']
        beta = np.cov(ticker_returns, data['log_return_IMOEX'])[0, 1] / np.var(data['log_return_IMOEX'])
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        ax[1].plot(beta, expected_return, 'o', label=f'{ticker}')
    x = np.linspace(-1, 2, 100)
    y = risk_free_rate + x * (market_return - risk_free_rate)
    ax[1].plot(x, y, label='Security Market Line', color='green')
    ax[1].set_xlabel('Beta (β)')
    ax[1].set_ylabel('Expected Return')
    ax[1].legend()
    ax[1].set_title('Security Market Line (SML)')
    for ticker in tickers:
        ticker_volatility = data[f'log_return_{ticker}'].std() * np.sqrt(252)
        cml_values = data[f'CML_{ticker}']
        ax[2].plot(ticker_volatility, cml_values.mean(), 'o', label=f'{ticker}')
    x = np.linspace(0, max(data[[f'CML_{ticker}' for ticker in tickers]].max()), 100)
    y = risk_free_rate + (market_return - risk_free_rate) / market_volatility * x
    ax[2].plot(x, y, label='Capital Market Line', color='blue')
    ax[2].set_xlabel('Portfolio Standard Deviation')
    ax[2].set_ylabel('Portfolio Expected Return')
    ax[2].legend()
    ax[2].set_title('Capital Market Line (CML)')
    plt.tight_layout()
    plt.show()
    plt.savefig('CML_CAL_SML.png')

def plot_performance_ratios(file_path):
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date', skiprows=range(2, 252))
    tickers = ['LKOH', 'SBER']
    metrics = ['Treynor', 'Sortino', 'Jensens_Alpha']
    fig, ax = plt.subplots(3, 1, figsize=(10, 18))
    for i, metric in enumerate(metrics):
        for ticker in tickers:
            ax[i].plot(data.index, data[f'{metric}_{ticker}'], label=f'{ticker}')
        ax[i].set_xlabel('Date')
        ax[i].set_ylabel(metric)
        ax[i].legend()
        ax[i].set_title(f'{metric} over Time')
    plt.tight_layout()
    plt.show()
    plt.savefig('TreynorSortinoJensensAlpha.png')
