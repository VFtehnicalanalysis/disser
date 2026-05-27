import os
import numpy as np
import pandas as pd
from .base import Strategy
from data.financemarker_loader import load_income, load_balance, load_cashflow, load_multipliers, SUPPORTED_TICKERS
FUND_TICKERS = SUPPORTED_TICKERS
BANK_TICKERS = {'SBER'}
ERP = 0.05
MARGIN_OF_SAFETY = 0.2
REPORT_LAG_QUARTERLY_DAYS = 90
REPORT_LAG_ANNUAL_DAYS = 120
BETA_WINDOW = 252
GROWTH_LOOKBACK_QUARTERS = 12
GROWTH_CAP = 0.25
DCF_TWOSTAGE_PERIOD = 5
DCF_MULTISTAGE_PERIOD = 10
G_TERMINAL = 0.03
DDM_GROWTH_CAP = 0.1
BACKTEST_START = pd.Timestamp('2015-01-01')
BACKTEST_END = pd.Timestamp('2024-01-01')
REG_TRAIN_QUARTERS = 8
REG_TOP_QUANTILE = 0.75
REG_BOT_QUANTILE = 0.25
DIVYIELD_BUY_PREMIUM = ERP
RISKFREE_FILE = 'data/RISKFREE_data.csv'
MARKET_FILE = 'data/IMOEX_data.csv'
_CACHE = {}

def _load_riskfree():
    if 'rf' in _CACHE:
        return _CACHE['rf']
    df = pd.read_csv(RISKFREE_FILE, parse_dates=['begin'])
    df = df.set_index('begin').sort_index()
    _CACHE['rf'] = df['close']
    return _CACHE['rf']

def _load_market_close():
    if 'market' in _CACHE:
        return _CACHE['market']
    df = pd.read_csv(MARKET_FILE, parse_dates=['begin'])
    df = df.set_index('begin').sort_index()
    _CACHE['market'] = df['close']
    return _CACHE['market']

def _load_dividends(ticker):
    path = f'finance_data/dividends/{ticker}_dividends.csv'
    df = pd.read_csv(path, parse_dates=['date'])
    return df.set_index('date')['amount'].sort_index()

def _signal_date(period_end, period_type, trading_days):
    lag = REPORT_LAG_ANNUAL_DAYS if period_type == 'Y' else REPORT_LAG_QUARTERLY_DAYS
    pub_date = period_end + pd.Timedelta(days=lag)
    after = trading_days[trading_days > pub_date]
    return after[0] if len(after) > 0 else None

def _ttm(quarterly_series, end_date):
    s = quarterly_series.loc[:end_date].dropna()
    if len(s) < 4:
        return None
    return float(s.iloc[-4:].sum())

def _avg_yoy_growth(quarterly_series, end_date, n_periods=GROWTH_LOOKBACK_QUARTERS, cap=GROWTH_CAP):
    s = quarterly_series.loc[:end_date].dropna()
    if len(s) < n_periods + 4:
        return None
    g = s.pct_change(periods=4).iloc[-n_periods:]
    g = g.replace([np.inf, -np.inf], np.nan).dropna()
    if g.empty:
        return None
    avg = float(g.mean())
    return min(max(avg, -cap), cap)

def _compute_beta(stock_close, market_close, end_date, window=BETA_WINDOW):
    sr = np.log(stock_close.loc[:end_date]).diff().dropna().iloc[-window:]
    mr = np.log(market_close.loc[:end_date]).diff().dropna().iloc[-window:]
    df = pd.concat([sr, mr], axis=1, join='inner').dropna()
    if len(df) < window // 2:
        return None
    cov = df.iloc[:, 0].cov(df.iloc[:, 1])
    var = df.iloc[:, 1].var()
    if var <= 0:
        return None
    return float(cov / var)

def _rf_at(date, rf_series):
    rf_pct = rf_series.asof(date)
    if pd.isna(rf_pct):
        return None
    return float(rf_pct) / 100.0

def _wacc(rf_decimal, beta, erp=ERP):
    if rf_decimal is None or beta is None:
        return None
    return rf_decimal + beta * erp

def _two_stage_dcf(fcf_now, g_high, g_terminal, wacc, period_high=DCF_TWOSTAGE_PERIOD):
    if wacc is None or wacc <= g_terminal or fcf_now is None or (fcf_now <= 0):
        return None
    pv = 0.0
    fcf = fcf_now
    for t in range(1, period_high + 1):
        fcf = fcf * (1 + g_high)
        pv += fcf / (1 + wacc) ** t
    fcf_terminal = fcf * (1 + g_terminal)
    tv = fcf_terminal / (wacc - g_terminal)
    pv += tv / (1 + wacc) ** period_high
    return pv

def _multi_stage_dcf(fcf_now, g_high, g_terminal, wacc, n_years=DCF_MULTISTAGE_PERIOD):
    if wacc is None or wacc <= g_terminal or fcf_now is None or (fcf_now <= 0):
        return None
    pv = 0.0
    fcf = fcf_now
    for t in range(1, n_years + 1):
        frac = (t - 1) / max(n_years - 1, 1)
        g_t = g_high + (g_terminal - g_high) * frac
        fcf = fcf * (1 + g_t)
        pv += fcf / (1 + wacc) ** t
    fcf_terminal = fcf * (1 + g_terminal)
    tv = fcf_terminal / (wacc - g_terminal)
    pv += tv / (1 + wacc) ** n_years
    return pv

def _gordon_ddm(d_next, r, g):
    if r is None or r <= g or d_next is None or (d_next <= 0):
        return None
    return d_next / (r - g)

def _equity_per_share_from_ev(ev, net_debt, total_shares):
    if ev is None or total_shares is None or total_shares <= 0:
        return None
    eq = ev - (net_debt if net_debt is not None and (not pd.isna(net_debt)) else 0.0)
    return eq / total_shares

class _FundamentalStrategy(Strategy):

    def __init__(self, name, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True):
        super().__init__(name, ticker, '1D', initial_account_size, entry_percentage, pyramiding, allowshort)

    def load_data(self):
        file_path = f'data/{self.ticker}_data.csv'
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Файл данных не найден: {file_path}')
        df = pd.read_csv(file_path)
        df['begin'] = pd.to_datetime(df['begin'])
        df = df.set_index('begin').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        self.data = df

    def _prepare(self):
        self.load_data()
        self.income = load_income(self.ticker)
        self.balance = load_balance(self.ticker)
        self.cashflow = load_cashflow(self.ticker)
        self.multipliers = load_multipliers(self.ticker)
        self.income_q = self.income[self.income['period'] == 'Q'].set_index('date').sort_index()
        self.balance_q = self.balance[self.balance['period'] == 'Q'].set_index('date').sort_index()
        self.cashflow_q = self.cashflow[self.cashflow['period'] == 'Q'].set_index('date').sort_index()
        self.multipliers_q = self.multipliers[self.multipliers['period'] == 'Q'].set_index('date').sort_index()
        self.rf = _load_riskfree()
        self.market_close = _load_market_close()
        self.trading_days = self.data.index

    def _emit_in_range(self, signals, date, sig_type, price):
        if date is None or pd.isna(price):
            return
        if date < BACKTEST_START or date > BACKTEST_END:
            return
        signals.append((date, sig_type, float(price)))

    def _signals_to_df(self, signals):
        if not signals:
            return pd.DataFrame(columns=['signal', 'close'])
        df = pd.DataFrame(signals, columns=['time', 'signal', 'close'])
        df = df.set_index('time').sort_index()
        return df

def _decide_signal(fair_price, market_price, margin_of_safety=MARGIN_OF_SAFETY):
    if fair_price is None or market_price is None or market_price <= 0:
        return None
    margin = (fair_price - market_price) / market_price
    if margin > margin_of_safety:
        return 'buy'
    if margin < -margin_of_safety:
        return 'sell'
    return None

class DDMGordonStrategy(_FundamentalStrategy):

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True):
        super().__init__('DDM_Gordon', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)

    def generate_signals(self):
        self._prepare()
        divs = _load_dividends(self.ticker)
        signals = []
        for period_end in self.income_q.index:
            sig_date = _signal_date(period_end, 'Q', self.trading_days)
            if sig_date is None or sig_date < BACKTEST_START:
                continue
            year_ago = period_end - pd.DateOffset(years=1)
            ttm_div = divs[(divs.index > year_ago) & (divs.index <= period_end)].sum()
            if ttm_div <= 0:
                continue
            yearly = divs[divs.index <= period_end].resample('Y').sum()
            yearly = yearly[yearly > 0]
            yearly_growth = yearly.pct_change().dropna().iloc[-5:]
            if len(yearly_growth) > 0:
                g_div = float(yearly_growth.mean())
                g_div = min(max(g_div, -DDM_GROWTH_CAP), DDM_GROWTH_CAP)
            else:
                g_div = 0.0
            beta = _compute_beta(self.data['close'], self.market_close, sig_date)
            rf_dec = _rf_at(sig_date, self.rf)
            r = _wacc(rf_dec, beta, ERP)
            if r is None:
                continue
            d_next = ttm_div * (1 + g_div)
            fv = _gordon_ddm(d_next, r, g_div)
            if fv is None:
                continue
            market_price = self.data['close'].asof(sig_date)
            signal = _decide_signal(fv, market_price)
            if signal:
                self._emit_in_range(signals, sig_date, signal, market_price)
        self.signals_df = self._signals_to_df(signals)

class DividendYieldStrategy(_FundamentalStrategy):

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True):
        super().__init__('DividendYield', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)

    def generate_signals(self):
        self._prepare()
        divs = _load_dividends(self.ticker)
        signals = []
        for period_end in self.income_q.index:
            sig_date = _signal_date(period_end, 'Q', self.trading_days)
            if sig_date is None or sig_date < BACKTEST_START:
                continue
            year_ago = period_end - pd.DateOffset(years=1)
            ttm_div = divs[(divs.index > year_ago) & (divs.index <= period_end)].sum()
            yearly = divs[divs.index <= period_end].resample('Y').sum()
            yearly = yearly[yearly > 0]
            yearly_growth = yearly.pct_change().dropna().iloc[-5:]
            if len(yearly_growth) > 0:
                g_div = float(yearly_growth.mean())
                g_div = min(max(g_div, -DDM_GROWTH_CAP), DDM_GROWTH_CAP)
            else:
                g_div = 0.0
            forward_div = ttm_div * (1 + g_div)
            market_price = self.data['close'].asof(sig_date)
            if pd.isna(market_price) or market_price <= 0:
                continue
            forward_yield = forward_div / market_price
            rf_dec = _rf_at(sig_date, self.rf)
            if rf_dec is None:
                continue
            if forward_yield > rf_dec + DIVYIELD_BUY_PREMIUM:
                self._emit_in_range(signals, sig_date, 'buy', market_price)
            elif forward_yield < rf_dec:
                self._emit_in_range(signals, sig_date, 'sell', market_price)
        self.signals_df = self._signals_to_df(signals)

class _DCFStrategyBase(_FundamentalStrategy):
    DCF_KIND = None

    def _compute_fair_price(self, period_end, sig_date):
        if self.ticker in BANK_TICKERS:
            return None
        fcf_now = _ttm(self.cashflow_q['fcf'], period_end)
        if fcf_now is None or fcf_now <= 0:
            return None
        g_high = _avg_yoy_growth(self.cashflow_q['fcf'], period_end)
        if g_high is None:
            return None
        beta = _compute_beta(self.data['close'], self.market_close, sig_date)
        rf_dec = _rf_at(sig_date, self.rf)
        wacc = _wacc(rf_dec, beta, ERP)
        if wacc is None:
            return None
        if self.DCF_KIND == 'twostage':
            ev = _two_stage_dcf(fcf_now, g_high, G_TERMINAL, wacc)
        elif self.DCF_KIND == 'multistage':
            ev = _multi_stage_dcf(fcf_now, g_high, G_TERMINAL, wacc)
        else:
            return None
        if ev is None:
            return None
        net_debt = self.balance_q['net_debt'].asof(period_end) if 'net_debt' in self.balance_q.columns else 0.0
        mult_row = self.multipliers_q.asof(period_end)
        if mult_row is None or pd.isna(mult_row.get('num1', np.nan)):
            return None
        n1 = mult_row.get('num1', 0) or 0
        n2 = mult_row.get('num2', 0) or 0
        total_shares = (n1 if not pd.isna(n1) else 0) + (n2 if not pd.isna(n2) else 0)
        return _equity_per_share_from_ev(ev, net_debt, total_shares)

    def generate_signals(self):
        self._prepare()
        signals = []
        for period_end in self.income_q.index:
            sig_date = _signal_date(period_end, 'Q', self.trading_days)
            if sig_date is None or sig_date < BACKTEST_START:
                continue
            fv = self._compute_fair_price(period_end, sig_date)
            market_price = self.data['close'].asof(sig_date)
            signal = _decide_signal(fv, market_price)
            if signal:
                self._emit_in_range(signals, sig_date, signal, market_price)
        self.signals_df = self._signals_to_df(signals)

class DCFTwoStageStrategy(_DCFStrategyBase):
    DCF_KIND = 'twostage'

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True):
        super().__init__('DCF_TwoStage', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)

class DCFMultiStageStrategy(_DCFStrategyBase):
    DCF_KIND = 'multistage'

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True):
        super().__init__('DCF_MultiStage', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)

class MultipleRegressionStrategy(_FundamentalStrategy):
    _panel_cache = None
    _predictions_cache = None

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True):
        super().__init__('MultipleRegression', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)

    @classmethod
    def _build_panel(cls):
        if cls._panel_cache is not None:
            return cls._panel_cache
        rows = []
        market_close = _load_market_close()
        for ticker in FUND_TICKERS:
            try:
                income = load_income(ticker)
                balance = load_balance(ticker)
                cashflow = load_cashflow(ticker)
                multipliers = load_multipliers(ticker)
            except FileNotFoundError:
                continue
            income_q = income[income['period'] == 'Q'].set_index('date').sort_index()
            balance_q = balance[balance['period'] == 'Q'].set_index('date').sort_index()
            cashflow_q = cashflow[cashflow['period'] == 'Q'].set_index('date').sort_index()
            mult_q = multipliers[multipliers['period'] == 'Q'].set_index('date').sort_index()
            price_path = f'data/{ticker}_data.csv'
            if not os.path.exists(price_path):
                continue
            prices = pd.read_csv(price_path, parse_dates=['begin']).set_index('begin').sort_index()
            prices = prices[~prices.index.duplicated(keep='first')]
            divs = _load_dividends(ticker)
            for period_end in income_q.index:
                sig_date = _signal_date(period_end, 'Q', prices.index)
                next_period_end = period_end + pd.DateOffset(months=3)
                next_sig_date = _signal_date(next_period_end, 'Q', prices.index)
                if sig_date is None or next_sig_date is None:
                    continue
                p0 = prices['close'].asof(sig_date)
                p1 = prices['close'].asof(next_sig_date)
                if pd.isna(p0) or pd.isna(p1) or p0 <= 0 or (p1 <= 0):
                    continue
                fwd_return = float(np.log(p1 / p0))
                earnings_ttm = _ttm(income_q['earnings'], period_end)
                mult_row = mult_q.asof(period_end)
                if mult_row is None:
                    continue
                n1 = mult_row.get('num1', 0) or 0
                n2 = mult_row.get('num2', 0) or 0
                total_shares = (n1 if not pd.isna(n1) else 0) + (n2 if not pd.isna(n2) else 0)
                if total_shares <= 0 or earnings_ttm is None or earnings_ttm <= 0:
                    pe = np.nan
                else:
                    eps_ttm = earnings_ttm / total_shares
                    pe = p0 / eps_ttm
                equity = balance_q['equity'].asof(period_end) if 'equity' in balance_q.columns else np.nan
                if total_shares > 0 and equity > 0:
                    book_per_share = equity / total_shares
                    pb = p0 / book_per_share
                else:
                    pb = np.nan
                roe = earnings_ttm / equity if earnings_ttm is not None and equity and (equity > 0) else np.nan
                total_debt = balance_q['total_debt'].asof(period_end) if 'total_debt' in balance_q.columns else np.nan
                de = total_debt / equity if equity and equity > 0 and (not pd.isna(total_debt)) else np.nan
                op_inc_ttm = _ttm(income_q['operating_income'], period_end) if 'operating_income' in income_q.columns else None
                rev_ttm = _ttm(income_q['revenue'], period_end) if 'revenue' in income_q.columns else None
                op_margin = op_inc_ttm / rev_ttm if op_inc_ttm is not None and rev_ttm and (rev_ttm > 0) else np.nan
                rev_growth = _avg_yoy_growth(income_q['revenue'], period_end, n_periods=4) if 'revenue' in income_q.columns else np.nan
                three_mo_ago = sig_date - pd.DateOffset(months=3)
                p_3m = prices['close'].asof(three_mo_ago)
                momentum_3m = float(np.log(p0 / p_3m)) if not pd.isna(p_3m) and p_3m > 0 else np.nan
                window_returns = np.log(prices['close'].loc[:sig_date]).diff().iloc[-63:]
                vol_3m = float(window_returns.std()) if len(window_returns) > 5 else np.nan
                year_ago = period_end - pd.DateOffset(years=1)
                ttm_div = divs[(divs.index > year_ago) & (divs.index <= period_end)].sum()
                div_yield = float(ttm_div / p0) if p0 > 0 else np.nan
                rows.append({'ticker': ticker, 'period_end': period_end, 'sig_date': sig_date, 'fwd_return': fwd_return, 'pe': pe, 'pb': pb, 'roe': roe, 'de': de, 'op_margin': op_margin, 'rev_growth': rev_growth, 'momentum_3m': momentum_3m, 'vol_3m': vol_3m, 'div_yield': div_yield})
        cls._panel_cache = pd.DataFrame(rows)
        return cls._panel_cache

    @classmethod
    def _build_predictions(cls):
        if cls._predictions_cache is not None:
            return cls._predictions_cache
        panel = cls._build_panel()
        if panel.empty:
            cls._predictions_cache = {}
            return cls._predictions_cache
        feature_cols = ['pe', 'pb', 'roe', 'de', 'op_margin', 'rev_growth', 'momentum_3m', 'vol_3m', 'div_yield']
        quarters = sorted(panel['period_end'].unique())
        predictions = {}
        for i, q in enumerate(quarters):
            if i < REG_TRAIN_QUARTERS:
                continue
            train_qs = quarters[i - REG_TRAIN_QUARTERS:i]
            train = panel[panel['period_end'].isin(train_qs)].dropna(subset=feature_cols + ['fwd_return'])
            test = panel[panel['period_end'] == q].dropna(subset=feature_cols)
            if len(train) < len(feature_cols) + 1 or test.empty:
                continue
            X_tr = train[feature_cols].values
            tr_dum = pd.get_dummies(train['ticker'], prefix='t', drop_first=True).values.astype(float)
            X_tr_full = np.column_stack([np.ones(len(X_tr)), X_tr, tr_dum])
            y_tr = train['fwd_return'].values
            try:
                beta_hat, *_ = np.linalg.lstsq(X_tr_full, y_tr, rcond=None)
            except np.linalg.LinAlgError:
                continue
            tr_ticker_cols = pd.get_dummies(train['ticker'], prefix='t', drop_first=True).columns
            test_dum = pd.get_dummies(test['ticker'], prefix='t').reindex(columns=tr_ticker_cols, fill_value=0).values.astype(float)
            X_te_full = np.column_stack([np.ones(len(test)), test[feature_cols].values, test_dum])
            y_pred = X_te_full @ beta_hat
            for (ticker, sig_date), pred in zip(zip(test['ticker'], test['sig_date']), y_pred):
                predictions[ticker, pd.Timestamp(sig_date)] = float(pred)
        cls._predictions_cache = predictions
        return cls._predictions_cache

    def generate_signals(self):
        self._prepare()
        predictions = self._build_predictions()
        all_preds = pd.DataFrame([{'ticker': t, 'date': d, 'pred': p} for (t, d), p in predictions.items()])
        signals = []
        if all_preds.empty:
            self.signals_df = self._signals_to_df(signals)
            return
        for sig_date, group in all_preds.groupby('date'):
            if sig_date < BACKTEST_START or sig_date > BACKTEST_END:
                continue
            top = group['pred'].quantile(REG_TOP_QUANTILE)
            bot = group['pred'].quantile(REG_BOT_QUANTILE)
            row = group[group['ticker'] == self.ticker]
            if row.empty:
                continue
            pred = float(row['pred'].iloc[0])
            market_price = self.data['close'].asof(sig_date)
            if pred >= top:
                self._emit_in_range(signals, sig_date, 'buy', market_price)
            elif pred <= bot:
                self._emit_in_range(signals, sig_date, 'sell', market_price)
        self.signals_df = self._signals_to_df(signals)

def _total_shares_at(multipliers_q: pd.DataFrame, period_end) -> float | None:
    if multipliers_q is None or multipliers_q.empty:
        return None
    subset = multipliers_q.loc[multipliers_q.index <= period_end]
    if subset.empty:
        return None
    last = subset.iloc[-1]
    n1 = last.get('num1')
    n2 = last.get('num2')
    n1 = 0.0 if n1 is None or pd.isna(n1) else float(n1)
    n2 = 0.0 if n2 is None or pd.isna(n2) else float(n2)
    total = n1 + n2
    return total if total > 0 else None

def _series_ttm(quarterly_series: pd.Series) -> pd.Series:
    if quarterly_series is None or quarterly_series.empty:
        return pd.Series(dtype=float)
    s = pd.to_numeric(quarterly_series, errors='coerce').dropna()
    if s.empty:
        return s
    return s.rolling(4, min_periods=4).sum()

def _pe_series(income_q: pd.DataFrame, multipliers_q: pd.DataFrame, prices: pd.Series) -> pd.Series:
    if 'earnings' not in income_q.columns:
        return pd.Series(dtype=float)
    earn_ttm = _series_ttm(income_q['earnings'])
    out = {}
    for period_end, e_ttm in earn_ttm.items():
        if pd.isna(e_ttm) or e_ttm <= 0:
            continue
        total_shares = _total_shares_at(multipliers_q, period_end)
        if total_shares is None:
            continue
        eps_ttm = e_ttm / total_shares
        if eps_ttm <= 0:
            continue
        p = prices.asof(period_end)
        if pd.isna(p) or p <= 0:
            continue
        out[period_end] = float(p) / float(eps_ttm)
    return pd.Series(out).sort_index()

def _pb_series(balance_q: pd.DataFrame, multipliers_q: pd.DataFrame, prices: pd.Series) -> pd.Series:
    if 'equity' not in balance_q.columns:
        return pd.Series(dtype=float)
    out = {}
    for period_end, equity in balance_q['equity'].items():
        equity = pd.to_numeric(equity, errors='coerce')
        if pd.isna(equity) or equity <= 0:
            continue
        total_shares = _total_shares_at(multipliers_q, period_end)
        if total_shares is None:
            continue
        bvps = equity / total_shares
        if bvps <= 0:
            continue
        p = prices.asof(period_end)
        if pd.isna(p) or p <= 0:
            continue
        out[period_end] = float(p) / float(bvps)
    return pd.Series(out).sort_index()

def _ev_ebitda_series(income_q: pd.DataFrame, balance_q: pd.DataFrame, multipliers_q: pd.DataFrame, prices: pd.Series) -> pd.Series:
    ebitda_col = next((c for c in ('ebitda', 'EBITDA') if c in income_q.columns), None)
    if ebitda_col is None and 'operating_income' in income_q.columns:
        ebitda_col = 'operating_income'
    if ebitda_col is None:
        return pd.Series(dtype=float)
    ebitda_ttm = _series_ttm(income_q[ebitda_col])
    out = {}
    for period_end, ebitda in ebitda_ttm.items():
        if pd.isna(ebitda) or ebitda <= 0:
            continue
        total_shares = _total_shares_at(multipliers_q, period_end)
        if total_shares is None:
            continue
        p = prices.asof(period_end)
        if pd.isna(p) or p <= 0:
            continue
        market_cap = p * total_shares
        net_debt = 0.0
        if balance_q is not None and 'net_debt' in balance_q.columns:
            nd = balance_q['net_debt'].asof(period_end)
            net_debt = 0.0 if pd.isna(nd) else float(nd)
        ev = market_cap + net_debt
        out[period_end] = ev / float(ebitda)
    return pd.Series(out).sort_index()

def _change_signal_from_series(period_series: pd.Series, trading_days, min_change_pct: float=5.0) -> list:
    changes = period_series.pct_change().dropna() * 100.0
    signals = []
    for period_end, change in changes.items():
        sig_date = _signal_date(period_end, 'Q', trading_days)
        if sig_date is None:
            continue
        if change > min_change_pct:
            signals.append((sig_date, 'buy', change))
        elif change < -min_change_pct:
            signals.append((sig_date, 'sell', change))
    return signals

class EPSGrowthStrategy(_FundamentalStrategy):

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, min_change_pct=5.0):
        super().__init__('EPSGrowth', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.min_change_pct = float(min_change_pct)

    def generate_signals(self):
        self._prepare()
        if 'earnings' not in self.income_q.columns:
            self.signals_df = self._signals_to_df([])
            return
        earn_ttm = _series_ttm(self.income_q['earnings'])
        eps_series = {}
        for period_end, e in earn_ttm.items():
            if pd.isna(e) or e <= 0:
                continue
            total_shares = _total_shares_at(self.multipliers_q, period_end)
            if total_shares is None:
                continue
            eps_series[period_end] = e / total_shares
        eps_s = pd.Series(eps_series).sort_index()
        raw = _change_signal_from_series(eps_s, self.trading_days, self.min_change_pct)
        signals = []
        for sig_date, sig, _ in raw:
            price = self.data['close'].asof(sig_date)
            self._emit_in_range(signals, sig_date, sig, price)
        self.signals_df = self._signals_to_df(signals)

class ROEStrategy(_FundamentalStrategy):

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, min_change_pct=5.0):
        super().__init__('ROE', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.min_change_pct = float(min_change_pct)

    def generate_signals(self):
        self._prepare()
        if 'earnings' not in self.income_q.columns or 'equity' not in self.balance_q.columns:
            self.signals_df = self._signals_to_df([])
            return
        earn_ttm = _series_ttm(self.income_q['earnings'])
        roe = {}
        for period_end, e in earn_ttm.items():
            if pd.isna(e):
                continue
            equity = self.balance_q['equity'].asof(period_end)
            equity = pd.to_numeric(equity, errors='coerce')
            if pd.isna(equity) or equity <= 0:
                continue
            roe[period_end] = float(e) / float(equity)
        roe_s = pd.Series(roe).sort_index()
        raw = _change_signal_from_series(roe_s, self.trading_days, self.min_change_pct)
        signals = []
        for sig_date, sig, _ in raw:
            price = self.data['close'].asof(sig_date)
            self._emit_in_range(signals, sig_date, sig, price)
        self.signals_df = self._signals_to_df(signals)

class NetProfitStrategy(_FundamentalStrategy):

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, min_change_pct=5.0):
        super().__init__('NetProfit', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.min_change_pct = float(min_change_pct)

    def generate_signals(self):
        self._prepare()
        if 'earnings' not in self.income_q.columns:
            self.signals_df = self._signals_to_df([])
            return
        earn_ttm = _series_ttm(self.income_q['earnings']).dropna()
        raw = _change_signal_from_series(earn_ttm, self.trading_days, self.min_change_pct)
        signals = []
        for sig_date, sig, _ in raw:
            price = self.data['close'].asof(sig_date)
            self._emit_in_range(signals, sig_date, sig, price)
        self.signals_df = self._signals_to_df(signals)

class RelativeValuationStrategy(_FundamentalStrategy):
    MULTIPLIER_BUILDERS = {'P/E': lambda inc, bal, mul, pr: _pe_series(inc, mul, pr), 'P/B': lambda inc, bal, mul, pr: _pb_series(bal, mul, pr), 'EV/EBITDA': lambda inc, bal, mul, pr: _ev_ebitda_series(inc, bal, mul, pr)}

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, multiplier='P/E', history_window=8, threshold=0.2):
        name = f"RelValuation_{multiplier.replace('/', '')}"
        super().__init__(name, ticker, initial_account_size, entry_percentage, pyramiding, allowshort)
        if multiplier not in self.MULTIPLIER_BUILDERS:
            raise ValueError(f'Unknown multiplier {multiplier}')
        self.multiplier = multiplier
        self.history_window = int(history_window)
        self.threshold = float(threshold)

    def generate_signals(self):
        self._prepare()
        mult_s = self.MULTIPLIER_BUILDERS[self.multiplier](self.income_q, self.balance_q, self.multipliers_q, self.data['close'])
        if mult_s.empty or 'earnings' not in self.income_q.columns:
            self.signals_df = self._signals_to_df([])
            return
        earn_ttm = _series_ttm(self.income_q['earnings']).dropna()
        earn_change = earn_ttm.pct_change()
        signals = []
        for i in range(self.history_window, len(mult_s)):
            period_end = mult_s.index[i]
            current = mult_s.iloc[i]
            hist_avg = mult_s.iloc[i - self.history_window:i].mean()
            if pd.isna(hist_avg) or hist_avg == 0:
                continue
            deviation = (current - hist_avg) / hist_avg
            fund_growth = earn_change.asof(period_end) if not earn_change.empty else 0.0
            if pd.isna(fund_growth):
                fund_growth = 0.0
            sig_date = _signal_date(period_end, 'Q', self.trading_days)
            if sig_date is None:
                continue
            price = self.data['close'].asof(sig_date)
            if deviation < -self.threshold and fund_growth > 0:
                self._emit_in_range(signals, sig_date, 'buy', price)
            elif deviation > self.threshold and fund_growth < 0:
                self._emit_in_range(signals, sig_date, 'sell', price)
        self.signals_df = self._signals_to_df(signals)

class MultiplierTrendStrategy(_FundamentalStrategy):

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, multiplier='P/E', trend_periods=3, threshold=0.05):
        name = f"MultTrend_{multiplier.replace('/', '')}"
        super().__init__(name, ticker, initial_account_size, entry_percentage, pyramiding, allowshort)
        if multiplier not in RelativeValuationStrategy.MULTIPLIER_BUILDERS:
            raise ValueError(f'Unknown multiplier {multiplier}')
        self.multiplier = multiplier
        self.trend_periods = int(trend_periods)
        self.threshold = float(threshold)

    def generate_signals(self):
        self._prepare()
        mult_s = RelativeValuationStrategy.MULTIPLIER_BUILDERS[self.multiplier](self.income_q, self.balance_q, self.multipliers_q, self.data['close'])
        if mult_s.empty or 'earnings' not in self.income_q.columns:
            self.signals_df = self._signals_to_df([])
            return
        mult_change = mult_s.pct_change()
        earn_ttm = _series_ttm(self.income_q['earnings']).dropna()
        earn_change = earn_ttm.pct_change()
        signals = []
        for i in range(self.trend_periods, len(mult_s)):
            period_end = mult_s.index[i]
            window = mult_change.iloc[i - self.trend_periods + 1:i + 1].dropna()
            if len(window) < self.trend_periods:
                continue
            all_down = all((c < -self.threshold for c in window))
            all_up = all((c > self.threshold for c in window))
            fund_growth = earn_change.asof(period_end) if not earn_change.empty else 0.0
            if pd.isna(fund_growth):
                fund_growth = 0.0
            sig_date = _signal_date(period_end, 'Q', self.trading_days)
            if sig_date is None:
                continue
            price = self.data['close'].asof(sig_date)
            if all_down and fund_growth >= 0:
                self._emit_in_range(signals, sig_date, 'buy', price)
            elif all_up and fund_growth < 0:
                self._emit_in_range(signals, sig_date, 'sell', price)
        self.signals_df = self._signals_to_df(signals)

class ComplexScoringStrategy(_FundamentalStrategy):

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, multipliers=('P/E', 'P/B', 'EV/EBITDA'), weights=(0.4, 0.3, 0.3), history_window=8, lambda_coef=0.5, threshold=0.1):
        super().__init__('ComplexScoring', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)
        if len(multipliers) != len(weights):
            raise ValueError('multipliers and weights length mismatch')
        self.multipliers = tuple(multipliers)
        self.weights = tuple((float(w) for w in weights))
        self.history_window = int(history_window)
        self.lambda_coef = float(lambda_coef)
        self.threshold = float(threshold)

    def generate_signals(self):
        self._prepare()
        mult_series_list = []
        for m in self.multipliers:
            builder = RelativeValuationStrategy.MULTIPLIER_BUILDERS.get(m)
            if builder is None:
                self.signals_df = self._signals_to_df([])
                return
            s = builder(self.income_q, self.balance_q, self.multipliers_q, self.data['close'])
            mult_series_list.append(s)
        if any((s.empty for s in mult_series_list)):
            self.signals_df = self._signals_to_df([])
            return
        common_idx = mult_series_list[0].index
        for s in mult_series_list[1:]:
            common_idx = common_idx.intersection(s.index)
        if len(common_idx) < self.history_window + 1:
            self.signals_df = self._signals_to_df([])
            return
        aligned = [s.reindex(common_idx) for s in mult_series_list]
        if 'earnings' not in self.income_q.columns:
            self.signals_df = self._signals_to_df([])
            return
        earn_ttm = _series_ttm(self.income_q['earnings']).dropna()
        earn_change = earn_ttm.pct_change()
        signals = []
        for i in range(self.history_window, len(common_idx)):
            period_end = common_idx[i]
            score = 0.0
            for j, s in enumerate(aligned):
                current = s.iloc[i]
                hist_avg = s.iloc[i - self.history_window:i].mean()
                if pd.isna(current) or pd.isna(hist_avg) or hist_avg == 0:
                    continue
                deviation = (current - hist_avg) / hist_avg
                score += self.weights[j] * deviation
            fund_growth = earn_change.asof(period_end) if not earn_change.empty else 0.0
            if pd.isna(fund_growth):
                fund_growth = 0.0
            score += self.lambda_coef * -fund_growth
            sig_date = _signal_date(period_end, 'Q', self.trading_days)
            if sig_date is None:
                continue
            price = self.data['close'].asof(sig_date)
            if score < -self.threshold:
                self._emit_in_range(signals, sig_date, 'buy', price)
            elif score > self.threshold:
                self._emit_in_range(signals, sig_date, 'sell', price)
        self.signals_df = self._signals_to_df(signals)

class PriceDisconnectStrategy(_FundamentalStrategy):

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, fundamental='earnings', threshold=0.15):
        super().__init__('PriceDisconnect', ticker, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.fundamental = fundamental
        self.threshold = float(threshold)

    def generate_signals(self):
        self._prepare()
        if self.fundamental not in self.income_q.columns:
            self.signals_df = self._signals_to_df([])
            return
        fund_ttm = _series_ttm(self.income_q[self.fundamental]).dropna()
        if len(fund_ttm) < 2:
            self.signals_df = self._signals_to_df([])
            return
        fund_change = fund_ttm.pct_change()
        prices_at_period = pd.Series({pe: self.data['close'].asof(pe) for pe in fund_ttm.index}).dropna()
        price_change = prices_at_period.pct_change()
        signals = []
        for period_end in fund_change.index:
            fc = fund_change.get(period_end)
            pc = price_change.get(period_end)
            if pd.isna(fc) or pd.isna(pc):
                continue
            disconnect = pc - fc
            sig_date = _signal_date(period_end, 'Q', self.trading_days)
            if sig_date is None:
                continue
            price = self.data['close'].asof(sig_date)
            if disconnect > self.threshold:
                self._emit_in_range(signals, sig_date, 'sell', price)
            elif disconnect < -self.threshold:
                self._emit_in_range(signals, sig_date, 'buy', price)
        self.signals_df = self._signals_to_df(signals)
