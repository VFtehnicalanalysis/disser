from data.data_loader import load_moex_data, load_moex_hourly_data, load_alpha_vantage_data, load_yahoo_finance_data, load_moex_finance_data
from modules.performance_metrics import add_cal_to_csv, add_cml_to_csv, add_sml_to_csv
from modules.performance_metrics import add_treynor_ratio_to_csv, add_sortino_ratio_to_csv, add_jensens_alpha_to_csv
from modules.performance_metrics import static_plots, plot_performance_ratios
from TA.ta_loader import change_data, change_hourly_data, process_hourly_ta_data, process_ta_data
from backtest.portfolio import run_strategies
from reports.report import generate_report

def add_me_suffix(tickers):
    return [f'{ticker}.ME' for ticker in tickers]

def main():
    tickers = ['LKOH', 'SBER', 'GAZP', 'ROSN', 'VKCO', 'AFKS', 'VTBR', 'SNGSP', 'GMKN', 'PLZL']
    index = 'IMOEX'
    risk_free_rate_ticker = 'MOEXREPO'
    start_date = '2014-01-01'
    end_date = '2026-04-20'
    load_moex_data(tickers, start_date, end_date)
    load_moex_data([index, risk_free_rate_ticker], start_date, end_date, market='index')
    load_moex_hourly_data(tickers, start_date, end_date)
    load_moex_hourly_data([index, risk_free_rate_ticker], start_date, end_date, market='index')
    change_data(tickers, start_date, end_date)
    change_data([index, risk_free_rate_ticker], start_date, end_date)
    change_hourly_data(tickers)
    change_hourly_data([index, risk_free_rate_ticker])
    for ticker in tickers:
        print(f'\nОбрабатывается тикер: {ticker}')
        run_strategies(ticker)
        print(f'Обработка тикера {ticker} завершена.')
if __name__ == '__main__':
    main()
