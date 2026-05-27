import os
import pandas as pd
from TA.advanced_methods import add_signals_to_data
from TA.base_methods import calculate_ma, calculate_ema, calculate_rsi, calculate_macd, calculate_obv, calculate_vwap

def calculate_technical_indicators(df):
    df = df.sort_index()
    df['SMA_50'] = calculate_ma(df['close'], window=50)
    df['SMA_100'] = calculate_ma(df['close'], window=200)
    df['EMA_21'] = calculate_ema(df['close'], window=21)
    df['EMA_50'] = calculate_ema(df['close'], window=50)
    df['EMA_100'] = calculate_ema(df['close'], window=100)
    df['EMA_200'] = calculate_ema(df['close'], window=200)
    df['RSI_14'] = calculate_rsi(df['close'], window=14)
    macd, signal = calculate_macd(df['close'])
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['OBV'] = calculate_obv(df['close'], df['volume'])
    df['VWAP'] = calculate_vwap(df['close'], df['volume'])
    return df

def change_data(tickers, start_date, end_date):
    for ticker in tickers:
        input_file = f'data/{ticker}_data.csv'
        output_file = f'data/{ticker}_data_new.csv'
        if not os.path.exists(input_file):
            print(f'Error: File {input_file} not found for {ticker}. Skipping...')
            continue
        df = pd.read_csv(input_file, parse_dates=['begin'])
        df.set_index('begin', inplace=True)
        df.ffill(inplace=True)
        df = calculate_technical_indicators(df)
        df.to_csv(output_file)
        print(f'Data for {ticker} processed and saved to {output_file}')

def change_hourly_data(tickers):
    for ticker in tickers:
        input_file = f'data/{ticker}_hourly_data.csv'
        output_file = f'data/{ticker}_hourly_data_new.csv'
        if not os.path.exists(input_file):
            print(f'Error: File {input_file} not found for {ticker}. Skipping...')
            continue
        df = pd.read_csv(input_file, parse_dates=['begin'])
        df.set_index('begin', inplace=True)
        df.ffill(inplace=True)
        df = calculate_technical_indicators(df)
        df.to_csv(output_file)
        print(f'Hourly data for {ticker} processed and saved to {output_file}')

def process_ta_data(tickers):
    for ticker in tickers:
        input_file = f'data/{ticker}_data_new.csv'
        output_file = f'data/{ticker}_data_with_signals.csv'
        if not os.path.exists(input_file):
            print(f'Error: File {input_file} not found for {ticker}. Skipping...')
            continue
        df = pd.read_csv(input_file, parse_dates=['begin'])
        df.set_index('begin', inplace=True)
        df = add_signals_to_data(df)
        df.to_csv(output_file)
        print(f'Data with signals for {ticker} saved to {output_file}')

def process_hourly_ta_data(tickers):
    for ticker in tickers:
        input_file = f'data/{ticker}_hourly_data_new.csv'
        output_file = f'data/{ticker}_hourly_data_with_signals.csv'
        if not os.path.exists(input_file):
            print(f'Error: File {input_file} not found for {ticker}. Skipping...')
            continue
        df = pd.read_csv(input_file, parse_dates=['begin'])
        df.set_index('begin', inplace=True)
        df = add_signals_to_data(df)
        df.to_csv(output_file)
        print(f'Hourly data with signals for {ticker} saved to {output_file}')
