import os
import pandas as pd
FINANCEMARKER_DIR = 'finance_data/financemarker'
SUPPORTED_TICKERS = ['SBER', 'LKOH', 'ROSN', 'GAZP', 'NVTK']
META_COLS = {'code', 'year', 'month', 'period', 'type', 'amount', 'curr'}

def _period_end_date(year, month):
    return pd.to_datetime(f'{int(year):04d}-{int(month):02d}-01') + pd.offsets.MonthEnd(0)

def _filter_and_dedupe(df):
    df = df[df['curr'] == 'RUB'].copy()
    df = df[df['period'].isin(['Y', 'Q'])].copy()
    df['_prio'] = df['type'].map({'МСФО': 0, 'РСБУ': 1}).fillna(2)
    df = df.sort_values('_prio').drop_duplicates(subset=['year', 'month', 'period'], keep='first').drop(columns='_prio')
    df['date'] = df.apply(lambda r: _period_end_date(r['year'], r['month']), axis=1)
    return df.sort_values('date').reset_index(drop=True)

def _apply_units(df):
    numeric_cols = [c for c in df.columns if c not in META_COLS and c != 'date']
    amount = pd.to_numeric(df['amount'], errors='coerce')
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce') * amount
    return df

def _read(ticker, statement):
    path = os.path.join(FINANCEMARKER_DIR, f'{ticker}_{statement}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Не найден файл {path}')
    return pd.read_csv(path)

def load_income(ticker):
    df = _filter_and_dedupe(_read(ticker, 'income'))
    return _apply_units(df)

def load_balance(ticker):
    df = _filter_and_dedupe(_read(ticker, 'balance'))
    return _apply_units(df)

def load_cashflow(ticker):
    df = _filter_and_dedupe(_read(ticker, 'cashflow'))
    return _apply_units(df)

def load_multipliers(ticker):
    df = _filter_and_dedupe(_read(ticker, 'multipliers'))
    numeric_cols = [c for c in df.columns if c not in META_COLS and c != 'date']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def load_all(ticker):
    return {'income': load_income(ticker), 'balance': load_balance(ticker), 'cashflow': load_cashflow(ticker), 'multipliers': load_multipliers(ticker)}

def load_panel(statement, tickers=None):
    if tickers is None:
        tickers = SUPPORTED_TICKERS
    loader = {'income': load_income, 'balance': load_balance, 'cashflow': load_cashflow, 'multipliers': load_multipliers}[statement]
    parts = []
    for t in tickers:
        df = loader(t)
        df = df.copy()
        df['code'] = t
        parts.append(df)
    return pd.concat(parts, ignore_index=True)
