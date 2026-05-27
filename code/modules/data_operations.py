import os
import pandas as pd
import numpy as np
from infographics.infographics import add_to_html_file

def change_data(tickers, start_date, end_date):
    for ticker in tickers:
        input_file = f'{ticker}_data.csv'
        output_file = f'{ticker}_new_data.csv'
        if not os.path.exists(input_file):
            print(f'Error: File {input_file} not found for {ticker}. Skipping...')
            continue
        df = pd.read_csv(input_file, parse_dates=['TRADEDATE'])
        df.set_index('TRADEDATE', inplace=True)
        df.ffill()
        df['Daily_Return'] = df['CLOSE'].pct_change(fill_method=None)
        df['Log_Return'] = df['CLOSE'].pct_change(fill_method=None).apply(lambda x: 0 if pd.isnull(x) or np.isinf(x) else np.log(1 + x))
        df['Daily_Volume_Change'] = df['VOLUME'].diff()
        df.to_csv(output_file)
        print(f'Data for {ticker} processed and saved to {output_file}')
        add_to_html_file(f'<p>Data for {ticker} processed and saved to {output_file}</p>')
