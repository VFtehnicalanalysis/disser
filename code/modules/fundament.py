import os
import csv
import requests

def fetch_alpha_vantage_data(api_key, tickers):
    base_url = 'https://www.alphavantage.co/query'
    for ticker in tickers:
        params = {'function': 'OVERVIEW', 'symbol': ticker, 'apikey': api_key}
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'Symbol' in data:
                csv_file = f'{ticker}_multipliers.csv'
                with open(csv_file, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    headers = data.keys()
                    writer.writerow(headers)
                    values = data.values()
                    writer.writerow(values)
                print(f'Data for {ticker} saved to {csv_file}')
            else:
                print(f'No data found for ticker {ticker}')
        else:
            print(f'Failed to fetch data for ticker {ticker}. HTTP Status code: {response.status_code}')
if __name__ == '__main__':
    api_key = 'AR56AXJ764Y5C49G'
    tickers = ['AAPL', 'SBER.ME', 'GAZP-RM']
    fetch_alpha_vantage_data(api_key, tickers)
