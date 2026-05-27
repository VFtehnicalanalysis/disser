import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy.stats import normaltest
import seaborn as sns
import pandas as pd
import mpld3
from infographics.infographics import add_to_html_file

def plot_and_test_distribution(tickers, field_name):
    for ticker in tickers:
        df = pd.read_csv(f'{ticker}_new_data.csv')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.histplot(df[field_name].dropna(), kde=True, color='blue', bins=30)
        plt.title(f'Histogram for {ticker} - {field_name}')
        plt.xlabel(field_name)
        plt.ylabel('Frequency')
        fig_html = mpld3.fig_to_html(plt.gcf())
        plt.close()
        add_to_html_file(f'<h2>Histogram for {ticker} - {field_name}</h2>{fig_html}')
        stat_sw, p_value_sw = shapiro(df[field_name].dropna())
        add_to_html_file(f'Shapiro-Wilk Test for {ticker} - {field_name}:')
        add_to_html_file(f'Statistic: {stat_sw}, p-value: {p_value_sw}\n')
        stat_ks, p_value_ks = normaltest(df[field_name].dropna())
        add_to_html_file(f'Normality Test (Kolmogorov-Smirnov) for {ticker} - {field_name}:')
        add_to_html_file(f'Statistic: {stat_ks}, p-value: {p_value_ks}\n')
        alpha_sw = 0.05
        if p_value_sw > alpha_sw:
            add_to_html_file(f'The distribution of {ticker} - {field_name} appears to be normal (p-value: {p_value_sw})')
        else:
            add_to_html_file(f'The distribution of {ticker} - {field_name} does not appear to be normal (p-value: {p_value_sw})')
        alpha_ks = 0.05
        if p_value_ks > alpha_ks:
            add_to_html_file(f'The distribution of {ticker} - {field_name} appears to be normal (p-value: {p_value_ks})')
        else:
            add_to_html_file(f'The distribution of {ticker} - {field_name} does not appear to be normal (p-value: {p_value_ks})')
from statsmodels.tsa.stattools import adfuller

def make_stationary(ticker, field_name):
    df = pd.read_csv(f'{ticker}_new_data.csv')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if field_name not in df.columns:
        print(f"Error: '{field_name}' column not found in {ticker} data. Exiting make_stationary function.")
        return
    diff_field_name = f'{field_name}_Diff'
    df[diff_field_name] = df[field_name].diff()
    df.to_csv(f'{ticker}_new_data.csv', index=False)
    print(f'Data for {ticker} processed and saved with added {diff_field_name} column.')

def check_stationarity(tickers, field_name):
    for ticker in tickers:
        df = pd.read_csv(f'{ticker}_new_data.csv')
        adf_result = adfuller(df[field_name].dropna(), autolag='AIC')
        add_to_html_file(f'ADF Test for {ticker} - {field_name}:')
        add_to_html_file(f'Test Statistic: {adf_result[0]}')
        add_to_html_file(f'p-value: {adf_result[1]}')
        add_to_html_file(f'Critical Values: {adf_result[4]}')
        add_to_html_file('Is Stationary: ' + str(adf_result[1] < 0.05))
        if adf_result[1] >= 0.05:
            add_to_html_file(f'Differencing needed for {ticker} - {field_name} to achieve stationarity.')
            make_stationary(ticker, field_name)
        add_to_html_file('')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
from itertools import combinations

def plot_autocorrelation(tickers, field_name):
    add_to_html_file(f'<h2>Autocorrelation Plots for {field_name}</h2>')
    for ticker1 in tickers:
        add_to_html_file(f'<h3>Correlation {field_name} for {ticker1}</h3>')
        for ticker2 in tickers:
            if ticker1 != ticker2:
                df_ticker1 = pd.read_csv(f'{ticker1}_new_data.csv')
                df_ticker2 = pd.read_csv(f'{ticker2}_new_data.csv')
                "fig, ax = plt.subplots(figsize=(8, 5))\n\n                # Plot autocorrelation for each pair\n                ax.acorr(df_ticker1[field_name].dropna(), maxlags=20, label=f'{ticker1}-{field_name}')\n                ax.acorr(df_ticker2[field_name].dropna(), maxlags=20, label=f'{ticker2}-{field_name}')\n                \n                \n                # Add labels and title to the plot\n                ax.set_title(f'Autocorrelation for {ticker1}-{ticker2}-{field_name}')\n                ax.set_xlabel('Lag')\n                ax.set_ylabel('Autocorrelation')\n                ax.legend()\n                "
                fig, ax = plt.subplots(figsize=(10, 6))
                autocorr_series = df_ticker1[field_name].rolling(window=20).corr(df_ticker2[field_name])
                ax.plot(autocorr_series, label=f'{ticker1}-{field_name}', color='blue')
                ax.set_title(f'Autocorrelation over time for {ticker1}-{ticker2} ({field_name})')
                ax.set_xlabel('Time')
                ax.set_ylabel('Autocorrelation')
                ax.legend()
                ax.grid(True)
                fig_html = mpld3.fig_to_html(fig)
                plt.close()
                add_to_html_file(f'<h4>Correlation {field_name} {ticker1}-{ticker2}</h4>')
                add_to_html_file(f'<div>{fig_html}</div>')
    add_to_html_file('')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
from itertools import combinations

def cross_correlation(series1, series2):
    result = np.correlate(series1, series2, mode='full')
    return result / np.max(result)

def plot_cross_correlation(tickers, field_name):
    add_to_html_file(f'<h2>Cross-Correlation Plots for {field_name}</h2>')
    for ticker1, ticker2 in combinations(tickers, 2):
        df_ticker1 = pd.read_csv(f'{ticker1}_new_data.csv')
        df_ticker2 = pd.read_csv(f'{ticker2}_new_data.csv')
        series1 = df_ticker1[field_name].dropna()
        series2 = df_ticker2[field_name].dropna()
        cross_corr = cross_correlation(series1, series2)
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(-len(series1) + 1, len(series2)), cross_corr, label=f'{ticker1}-{ticker2}-{field_name}')
        plt.title(f'Cross-Correlation for {ticker1}-{ticker2}-{field_name}')
        plt.xlabel('Lag')
        plt.ylabel('Cross-Correlation')
        plt.legend()
        fig_html = mpld3.fig_to_html(plt.gcf())
        plt.close()
        add_to_html_file(f'<h3>Cross-Correlation {field_name} for {ticker1}-{ticker2}</h3>')
        add_to_html_file(f'<div>{fig_html}</div>')
    add_to_html_file('')
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from itertools import combinations

def plot_cointegration(tickers, field_name):
    data = {}
    for ticker in tickers:
        filename = f'./{ticker}_new_data.csv'
        data[ticker] = pd.read_csv(filename)[:200]
    add_to_html_file(f'<h2>Cointegration Plots for {field_name}</h2>')
    for ticker1, ticker2 in combinations(tickers, 2):
        df_ticker1 = pd.read_csv(f'{ticker1}_new_data.csv')[:200]
        df_ticker2 = pd.read_csv(f'{ticker2}_new_data.csv')[:200]
        series1 = df_ticker1[field_name].dropna()
        series2 = df_ticker2[field_name].dropna()
        coint_result = sm.tsa.coint(series1, series2)
        plt.figure(figsize=(10, 5))
        plt.plot(series1, label=f'{ticker1}-{field_name}')
        plt.plot(series2, label=f'{ticker2}-{field_name}')
        plt.title(f'Cointegration for {ticker1} and {ticker2} - {field_name}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        fig_html = mpld3.fig_to_html(plt.gcf())
        plt.close()
        add_to_html_file(f'<h3>Cointegration {field_name} for {ticker1}-{ticker2}</h3>')
        add_to_html_file(f'<div>{fig_html}</div>')
import statsmodels.api as sm
import matplotlib.pyplot as plt
from itertools import combinations

def plot_acf_pacf(tickers, field_name):
    add_to_html_file(f'<h2>ACF and PACF Plots for {field_name}</h2>')
    for ticker1, ticker2 in combinations(tickers, 2):
        df_ticker1 = pd.read_csv(f'{ticker1}_new_data.csv')[:200]
        df_ticker2 = pd.read_csv(f'{ticker2}_new_data.csv')[:200]
        series1 = df_ticker1[field_name].dropna()
        series2 = df_ticker2[field_name].dropna()
        otg1diff1 = series1.diff().dropna()
        otg1diff2 = series2.diff().dropna()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        sm.graphics.tsa.plot_acf(otg1diff1, lags=25, ax=ax1)
        ax1.set_title(f'ACF for {ticker1}-{field_name}')
        sm.graphics.tsa.plot_pacf(otg1diff1, lags=25, ax=ax2)
        ax2.set_title(f'PACF for {ticker1}-{field_name}')
        fig_html = mpld3.fig_to_html(plt.gcf())
        plt.close()
        add_to_html_file(f'<h3>ACF and PACF {field_name} for {ticker1}</h3>')
        add_to_html_file(f'<div>{fig_html}</div>')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        sm.graphics.tsa.plot_acf(otg1diff2, lags=25, ax=ax1)
        ax1.set_title(f'ACF for {ticker2}-{field_name}')
        sm.graphics.tsa.plot_pacf(otg1diff2, lags=25, ax=ax2)
        ax2.set_title(f'PACF for {ticker2}-{field_name}')
        fig_html = mpld3.fig_to_html(plt.gcf())
        plt.close()
        add_to_html_file(f'<h3>ACF and PACF {field_name} for {ticker2}</h3>')
        add_to_html_file(f'<div>{fig_html}</div>')
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mpld3
import statsmodels.api as sm
import matplotlib.pyplot as plt
import mpld3
from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA

def build_arima_model(tickers, field_name):
    for ticker in tickers:
        try:
            df_ticker = pd.read_csv(f'{ticker}_new_data.csv')[:200]
            series = df_ticker[field_name].dropna()
            series = series.squeeze()
            series.index = pd.to_datetime(df_ticker['TRADEDATE'][:len(series)], errors='coerce')
            series = pd.to_numeric(series, errors='coerce')
            auto_arima_fit = auto_arima(series, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
            arima_order = auto_arima_fit.order
            model = sm.tsa.arima.model.ARIMA(series, order=(1, 1, 1)).fit()
            results = model.fit()
            add_to_html_file(f'<h3>ARIMA Model Results for {ticker}-{field_name}</h3>')
            add_to_html_file(f'<p>{results.summary()}</p>')
        except Exception as e:
            add_to_html_file(f'<p>Error processing {ticker}: {str(e)}</p>')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result = adfuller(series, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values: {result[4]}')
    return result

def fit_arima(series, order):
    model = ARIMA(series, order=order)
    results = model.fit()
    print(results.summary())
    return results

def arima_model_for_ticker(ticker, field_name):
    df = pd.read_csv(f'{ticker}_new_data.csv', index_col='TRADEDATE', parse_dates=True)
    series = df[field_name].dropna()
    adf_result = adf_test(series)
    if adf_result[1] > 0.05:
        series_diff = series.diff().dropna()
        adf_test(series_diff)
    else:
        series_diff = series
    auto_arima_fit = auto_arima(series, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    arima_order = auto_arima_fit.order
    arima_model = fit_arima(series_diff, arima_order)
    forecast_index = pd.date_range(start=series.index[-1], periods=6, freq=series.index.freq)
    arima_forecast = arima_model.forecast(steps=5)
    arima_forecast.index = forecast_index[:-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series_diff.index, series_diff, label=f'{ticker} - {field_name}')
    ax.plot(arima_model.fittedvalues.index, arima_model.fittedvalues, color='red', label='ARIMA Fitted Values')
    ax.plot(arima_forecast.index, arima_forecast, color='blue', linestyle='dashed', label='ARIMA Forecast')
    ax.set_title(f'ARIMA Forecast for {ticker} - {field_name}')
    ax.legend()
    fig_html = mpld3.fig_to_html(fig)
    fig_html = mpld3.fig_to_html(plt.gcf())
    add_to_html_file(f'<h3>ARIMA {field_name} for {ticker}</h3>')
    add_to_html_file(f'<div>{fig_html}</div>')
    add_to_html_file(f'<h3>ARIMA summary {field_name} for {ticker}</h3>')
    add_to_html_file(f'<div>{arima_model.summary().as_html()}</div>')
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def arfima_model_for_ticker(ticker, field_name):
    df = pd.read_csv(f'{ticker}_new_data.csv', index_col='TRADEDATE', parse_dates=True)
    series = df[field_name].dropna()
    adf_result = adfuller(series)
    if adf_result[1] > 0.05:
        series_diff = series.diff().dropna()
        adf_test(series_diff)
    else:
        series_diff = series
    decomposition = seasonal_decompose(series_diff, model='additive', period=1)
    d = 1
    p = 1
    q = 1
    P = 1
    D = 1
    Q = 1
    s = 24
    model = SARIMAX(series_diff, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results = model.fit(disp=False)
    add_to_html_file(f'<h3>Модель AFRIMA для {ticker}</h3>')
    add_to_html_file(results.summary().tables[1].as_html())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(series_diff.index, series_diff, label='Дифференцированные данные')
    ax.plot(series_diff.index, results.fittedvalues, label='Предсказания', color='red')
    forecast_index = pd.date_range(start=series_diff.index[-1], periods=6, freq=series_diff.index.freq)
    forecast = results.get_forecast(steps=5)
    forecast_index = pd.date_range(start=series.index[-1], periods=6, freq=series.index.freq)
    ax.plot(forecast_index[:-1], forecast.predicted_mean, label='Прогноз', color='green')
    ax.set_title('Модель AFRIMA для {}'.format(ticker))
    ax.legend()
    fig_html = mpld3.fig_to_html(fig)
    add_to_html_file(fig_html)

def arch_garch_models_for_ticker(ticker, field_name):
    df = pd.read_csv(f'{ticker}_new_data.csv', index_col='TRADEDATE', parse_dates=True)
    if field_name == 'CLOSE':
        returns = df[field_name].pct_change().dropna()
    else:
        returns = df[field_name].dropna()
    arch_model_spec = arch_model(returns, vol='Arch', p=1)
    arch_results = arch_model_spec.fit(disp='off')
    garch_model_spec = arch_model(returns, vol='Garch', p=1, q=1)
    garch_results = garch_model_spec.fit(disp='off')
    add_to_html_file(f'<h3>Модели ARCH и GARCH для {ticker}</h3>')
    add_to_html_file('<h4>Модель ARCH(1)</h4>')
    add_to_html_file(arch_results.summary().as_html())
    add_to_html_file('<h4>Модель GARCH(1,1)</h4>')
    add_to_html_file(garch_results.summary().as_html())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(returns.index, returns, label='Данные')
    ax.plot(returns.index, arch_results.conditional_volatility, label='Прогноз волатильности (ARCH)', color='blue')
    ax.plot(returns.index, garch_results.conditional_volatility, label='Прогноз волатильности (GARCH)', color='green')
    ax.set_title(f'Модели ARCH и GARCH для {ticker}')
    ax.legend()
    fig_html = mpld3.fig_to_html(fig)
    add_to_html_file(fig_html)

def arima_adl_model_for_ticker(ticker, returns_column, exogenous_column):
    df = pd.read_csv(f'{ticker}_new_data.csv', index_col='TRADEDATE', parse_dates=True)
    returns = df[returns_column].pct_change().dropna()
    exogenous_variable = df[exogenous_column].shift(1).dropna()
    X = sm.add_constant(exogenous_variable)
    model = sm.tsa.statespace.SARIMAX(returns, order=(1, 0, 1), exog=X)
    results = model.fit(disp=False)
    add_to_html_file(f'<h3>Модель ARIMA с экзогенными переменными (ADL) для {ticker}</h3>')
    add_to_html_file(results.summary().tables[1].as_html())
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(returns.index, returns, label='Данные')
    ax.plot(returns.index, results.fittedvalues, label='Предсказания', color='red')
    ax.set_title(f'Модель ARIMA с экзогенными переменными (ADL) для {ticker}')
    ax.legend()
    fig_html = mpld3.fig_to_html(fig)
    add_to_html_file(fig_html)
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, VECM
import mpld3

def var_vecm_model_for_tickers(ticker1, ticker2, returns_column1, returns_column2):
    df = pd.read_csv(f'{ticker1}_new_data.csv', index_col='TRADEDATE', parse_dates=True)
    df2 = pd.read_csv(f'{ticker2}_new_data.csv', index_col='TRADEDATE', parse_dates=True)
    returns1 = df[returns_column1].pct_change().dropna()
    returns2 = df2[returns_column2].pct_change().dropna()
    returns_df = pd.DataFrame({'returns1': returns1, 'returns2': returns2})
    model_var = VAR(returns_df)
    results_var = model_var.fit()
    fig = results_var.plot_forecast(steps=10)
    ax = fig.gca()
    ax.set_title(f'VAR модель ддля {ticker1} и {ticker2}')
    fig_html = mpld3.fig_to_html(fig)
    add_to_html_file(fig_html)
    model_vecm = VECM(returns_df, k_ar_diff=1, coint_rank=1)
    results_vecm = model_vecm.fit()
    fig, ax = plt.subplots(figsize=(12, 6))
    sm.graphics.tsa.plot_acf(results_vecm.resid[:, :1], lags=30, ax=ax)
    ax.set_title(f'Автокорреляция остатков VECM для {ticker1} и {ticker2}')
    fig_html = mpld3.fig_to_html(fig)
    add_to_html_file(fig_html)

def sharp_for_ticker(ticker, field_name='CLOSE'):
    df = pd.read_csv(f'{ticker}_new_data.csv', index_col='TRADEDATE', parse_dates=True)
    df['Normed_Return'] = df[field_name] / df.iloc[0][field_name]
    sharp_ratio = (df['Normed_Return'].mean() - 0) / df['Normed_Return'].std()
    return (df, sharp_ratio)

def negative_sharpe_ratio(weights, returns):
    portfolio_return = np.dot(returns.mean(), weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    sharpe_ratio = -portfolio_return / portfolio_volatility
    return sharpe_ratio
import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(tickers, field_name):
    data = {ticker: sharp_for_ticker(ticker, field_name)[0] for ticker in tickers}
    combined_data = pd.concat([data[ticker]['Normed_Return'] for ticker in tickers], axis=1)
    combined_data.columns = tickers
    initial_weights = np.ones(len(tickers)) / len(tickers)
    result = minimize(negative_sharpe_ratio, initial_weights, args=(combined_data,), method='SLSQP', bounds=[(0, 1) for _ in range(len(tickers))], constraints={'type': 'eq', 'fun': lambda weights: 1 - np.sum(weights)})
    optimal_weights = result.x
    optimal_portfolio = {tickers[i]: optimal_weights[i] for i in range(len(tickers))}
    add_to_html_file('<h2>Optimal Portfolio</h2>')
    for ticker, weight in optimal_portfolio.items():
        add_to_html_file(f'{ticker}: {round(weight * 100, 2)}%')
    fig, ax = plt.subplots()
    ax.pie(optimal_weights, labels=tickers, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    fig_html = mpld3.fig_to_html(fig)
    add_to_html_file(fig_html)
    return optimal_portfolio

def show_portfolio(tickers, optimal_portfolio, field_name, initial_capital, output_html='infographics.html'):
    all_pos_vals = []
    for ticker, weight in optimal_portfolio.items():
        df, _ = sharp_for_ticker(ticker, field_name)
        df['Position Value'] = weight * initial_capital * (1 + df[field_name].pct_change(fill_method=None)).cumprod()
        all_pos_vals.append(df['Position Value'])
    portfolio_val = pd.concat(all_pos_vals, axis=1)
    portfolio_val.columns = tickers
    portfolio_val['Total'] = portfolio_val.sum(axis=1)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8))
    axes[0].plot(portfolio_val['Total'])
    axes[0].set_title('Изменение стоимости портфеля')
    axes[0].set_xlabel('Дата')
    axes[0].set_ylabel('Стоимость портфеля')
    portfolio_val.drop('Total', axis=1).plot(ax=axes[1])
    axes[1].set_title('Изменение стоимости каждой позиции')
    axes[1].set_xlabel('Дата')
    axes[1].set_ylabel('Стоимость позиции')
    plt.tight_layout()
    fig_html = mpld3.fig_to_html(fig)
    add_to_html_file(fig_html)
    return portfolio_val
import matplotlib.pyplot as plt

def show_statistics(portfolio_val, output_html='infographics.html'):
    portfolio_val['Daily Return'] = portfolio_val['Total'].pct_change(1)
    portfolio_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    avg_daily_return = portfolio_val['Daily Return'].mean()
    std_daily_return = portfolio_val['Daily Return'].std()
    if std_daily_return != 0:
        sharpe_ratio = avg_daily_return / std_daily_return
    else:
        sharpe_ratio = np.nan
    print(f'Средняя ежедневная доходность: {avg_daily_return}')
    print(f'Стандартное отклонение ежедневной доходности: {std_daily_return}')
    print(f'Коэффициент Шарпа: {sharpe_ratio}')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(portfolio_val['Daily Return'].dropna(), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title('Distribution of Daily Returns')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequency')
    ax.grid(True)
    fig_html = mpld3.fig_to_html(fig)
    add_to_html_file(fig_html)
