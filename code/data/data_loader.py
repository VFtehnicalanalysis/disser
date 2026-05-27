import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import apimoex

def load_moex_data(tickers, start_date, end_date, market='shares'):
    base_url = f'https://iss.moex.com/iss/engines/stock/markets/{market}/securities'
    os.makedirs('data', exist_ok=True)
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    with requests.Session() as session:
        for ticker in tickers:
            print(f'Fetching data for {ticker}...')
            all_data = []
            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=500), end_date)
                url = f'{base_url}/{ticker}/candles.json'
                params = {'from': current_start.strftime('%Y-%m-%d'), 'till': current_end.strftime('%Y-%m-%d'), 'interval': 24}
                try:
                    response = session.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    if 'candles' in data and 'data' in data['candles']:
                        columns = data['candles']['columns']
                        records = data['candles']['data']
                        if records:
                            df = pd.DataFrame(records, columns=columns)
                            all_data.append(df)
                            print(f'Fetched {len(records)} rows for {ticker} from {current_start} to {current_end}')
                        else:
                            print(f'No data for {ticker} from {current_start} to {current_end}')
                    else:
                        print(f'No data returned for {ticker}. Skipping this interval...')
                except Exception as e:
                    print(f'Error occurred for {ticker} from {current_start} to {current_end}: {e}')
                current_start = current_end + timedelta(days=1)
            if all_data:
                full_data = pd.concat(all_data, ignore_index=True)
                if 'begin' in full_data.columns:
                    full_data['begin'] = pd.to_datetime(full_data['begin'])
                    full_data.set_index('begin', inplace=True)
                file_name = f'data/{ticker}_data.csv'
                full_data.to_csv(file_name)
                print(f'Data for {ticker} saved to {file_name}')
            else:
                print(f'No data available for {ticker} over the entire period.')

def load_yahoo_finance_data(tickers, start_date, end_date):
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        if not data.empty:
            data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
            file_name = f'data/{ticker}_yahoo.csv'
            os.makedirs('data', exist_ok=True)
            data.dropna(inplace=True)
            data.to_csv(file_name)
            print(f'Yahoo Finance data for {ticker} saved to {file_name}')
        else:
            print(f'No data found for {ticker} on Yahoo Finance.')

def load_alpha_vantage_data(api_key, tickers):
    base_url = 'https://www.alphavantage.co/query'
    for ticker in tickers:
        params = {'function': 'OVERVIEW', 'symbol': ticker, 'apikey': 'AR56AXJ764Y5C49G'}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame([data])
                file_name = f'data/{ticker}_fundamental.csv'
                os.makedirs('data', exist_ok=True)
                df.to_csv(file_name, index=False)
                print(f'Alpha Vantage data for {ticker} saved to {file_name}')
            else:
                print(f'No data found for {ticker} on Alpha Vantage.')
        else:
            print(f'Failed to fetch data for {ticker} from Alpha Vantage. Status code: {response.status_code}')

def load_moex_hourly_data(tickers, start_date, end_date, market='shares'):
    base_url = f'https://iss.moex.com/iss/engines/stock/markets/{market}/securities'
    os.makedirs('data', exist_ok=True)
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    with requests.Session() as session:
        for ticker in tickers:
            print(f'Fetching hourly data for {ticker}...')
            all_data = []
            current_start = start_date
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=30), end_date)
                url = f'{base_url}/{ticker}/candles.json'
                params = {'from': current_start.strftime('%Y-%m-%d'), 'till': current_end.strftime('%Y-%m-%d'), 'interval': 60}
                try:
                    response = session.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    if 'candles' in data and 'data' in data['candles']:
                        columns = data['candles']['columns']
                        records = data['candles']['data']
                        if records:
                            df = pd.DataFrame(records, columns=columns)
                            all_data.append(df)
                            print(f'Fetched {len(records)} hourly rows for {ticker} from {current_start} to {current_end}')
                        else:
                            print(f'No hourly data for {ticker} from {current_start} to {current_end}')
                    else:
                        print(f'No data returned for {ticker}. Skipping this interval...')
                except Exception as e:
                    print(f'Error occurred for {ticker} from {current_start} to {current_end}: {e}')
                current_start = current_end + timedelta(days=1)
            if all_data:
                full_data = pd.concat(all_data, ignore_index=True)
                if 'begin' in full_data.columns:
                    full_data['begin'] = pd.to_datetime(full_data['begin'])
                    full_data.set_index('begin', inplace=True)
                file_name = f'data/{ticker}_hourly_data.csv'
                full_data.to_csv(file_name)
                print(f'Hourly data for {ticker} saved to {file_name}')
            else:
                print(f'No hourly data available for {ticker} over the entire period.')

def change_data(tickers, start_date, end_date):
    for ticker in tickers:
        input_file = f'data/{ticker}_data.csv'
        output_file = f'data/{ticker}_new_data.csv'
        if not os.path.exists(input_file):
            print(f'Error: File {input_file} not found for {ticker}. Skipping...')
            continue
        df = pd.read_csv(input_file, parse_dates=['begin'])
        df.set_index('begin', inplace=True)
        df.ffill()
        df['Daily_Return'] = df['close'].pct_change(fill_method=None)
        df['Log_Return'] = df['close'].pct_change(fill_method=None).apply(lambda x: 0 if pd.isnull(x) or np.isinf(x) else np.log(1 + x))
        df['Daily_Volume_Change'] = df['volume'].diff()
        df.to_csv(output_file)
        print(f'Data for {ticker} processed and saved to {output_file}')
import yfinance as yf
import pandas as pd
import os

def load_moex_finance_data(tickers):
    os.makedirs('finance_data', exist_ok=True)
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            quarterly_financials = stock.quarterly_financials.T
            annual_financials = stock.financials.T
            quarterly_file = os.path.join('finance_data', f'{ticker}_quarterly_financials.csv')
            quarterly_financials.to_csv(quarterly_file)
            annual_file = os.path.join('finance_data', f'{ticker}_annual_financials.csv')
            annual_financials.to_csv(annual_file)
            print(f'Data for {ticker} saved to {quarterly_file} and {annual_file}')
        except Exception as e:
            print(f'Error occurred for {ticker}: {e}')
