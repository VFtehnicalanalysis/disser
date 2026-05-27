import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from .base import Strategy
warnings.filterwarnings('ignore')
ARIMA_ORDER = (1, 0, 1)
ARIMA_MIN_TRAIN = 252
ARIMA_REFIT_EVERY = 30
ARIMA_SIGNAL_ZTHRESHOLD = 0.0
GARCH_MIN_TRAIN = 252
GARCH_REFIT_EVERY = 60
GARCH_VOL_WINDOW = 252
GARCH_LOW_Q = 0.25
GARCH_HIGH_Q = 0.75

def _arima_walkforward_forecast(log_returns, order=ARIMA_ORDER, min_train=ARIMA_MIN_TRAIN, refit_every=ARIMA_REFIT_EVERY):
    n = len(log_returns)
    mu_fc = np.full(n, np.nan)
    sigma_fc = np.full(n, np.nan)
    if n < min_train + 1:
        return (pd.Series(mu_fc, index=log_returns.index), pd.Series(sigma_fc, index=log_returns.index))
    ret_vals = log_returns.values
    const = phi = theta = None
    sigma = None
    prev_eps = 0.0
    prev_mu = 0.0
    for t in range(min_train, n):
        need_refit = const is None or (t - min_train) % refit_every == 0
        if need_refit:
            try:
                train = log_returns.iloc[:t]
                model = ARIMA(train, order=order).fit(method_kwargs={'warn_convergence': False})
                params = model.params
                const = float(params.get('const', params.get('intercept', 0.0)))
                phi = float(params.get('ar.L1', 0.0))
                theta = float(params.get('ma.L1', 0.0))
                sigma2 = float(params.get('sigma2', np.nan))
                sigma = float(np.sqrt(sigma2)) if sigma2 > 0 else None
                resid = model.resid
                prev_eps = float(resid.iloc[-1]) if len(resid) else 0.0
                prev_mu = float(ret_vals[t - 1] - prev_eps)
            except Exception:
                continue
        if const is None or sigma is None:
            continue
        r_last = ret_vals[t - 1]
        eps_t = r_last - prev_mu
        mu_next = const + phi * r_last + theta * eps_t
        if not np.isfinite(mu_next):
            continue
        mu_fc[t] = mu_next
        sigma_fc[t] = sigma
        prev_mu = mu_next
        prev_eps = eps_t
    return (pd.Series(mu_fc, index=log_returns.index), pd.Series(sigma_fc, index=log_returns.index))

class ARIMASignalStrategy(Strategy):

    def __init__(self, ticker, timeframe='1D', initial_account_size=1000000, entry_percentage=100, pyramiding=1, allowshort=True, order=ARIMA_ORDER, min_train=ARIMA_MIN_TRAIN, refit_every=ARIMA_REFIT_EVERY, signal_zthreshold=ARIMA_SIGNAL_ZTHRESHOLD):
        super().__init__('ARIMA_Signal', ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.order = tuple(order)
        self.min_train = int(min_train)
        self.refit_every = int(refit_every)
        self.signal_zthreshold = float(signal_zthreshold)

    def generate_signals(self):
        self.load_data()
        close = self.data['close']
        log_ret = np.log(close / close.shift(1)).dropna()
        mu_fc, sigma_fc = _arima_walkforward_forecast(log_ret, order=self.order, min_train=self.min_train, refit_every=self.refit_every)
        signals = []
        state = 'flat'
        z = self.signal_zthreshold
        for ts in mu_fc.index:
            mu = mu_fc.loc[ts]
            sg = sigma_fc.loc[ts]
            if pd.isna(mu) or pd.isna(sg) or sg <= 0:
                continue
            price = close.loc[ts]
            if pd.isna(price):
                continue
            thresh = z * sg
            if mu > thresh and state != 'long':
                signals.append((ts, 'long', float(price)))
                state = 'long'
            elif mu < -thresh and state != 'short':
                if self.allowshort:
                    signals.append((ts, 'short', float(price)))
                    state = 'short'
                elif state == 'long':
                    signals.append((ts, 'close long', float(price)))
                    state = 'flat'
        if signals:
            self.signals_df = pd.DataFrame(signals, columns=['time', 'signal', 'close']).set_index('time').sort_index()
        else:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])

def _garch_expanding_forecast(log_returns, asymmetry=False, min_train=GARCH_MIN_TRAIN, refit_every=GARCH_REFIT_EVERY):
    n = len(log_returns)
    fc_sigma = np.full(n, np.nan)
    if n < min_train + 1:
        return pd.Series(fc_sigma, index=log_returns.index)
    ret_vals = log_returns.values
    omega = alpha = gamma = beta = None
    last_sigma2 = None
    SCALE = 100.0
    for t in range(min_train, n):
        need_refit = last_sigma2 is None or (t - min_train) % refit_every == 0
        if need_refit:
            try:
                train = log_returns.iloc[:t] * SCALE
                kwargs = dict(vol='GARCH', p=1, q=1, dist='normal')
                if asymmetry:
                    kwargs['o'] = 1
                gm = arch_model(train, **kwargs)
                res = gm.fit(disp='off', show_warning=False)
                params = res.params
                omega = float(params.get('omega', np.nan))
                alpha = float(params.get('alpha[1]', 0.0))
                gamma = float(params.get('gamma[1]', 0.0)) if asymmetry else 0.0
                beta = float(params.get('beta[1]', 0.0))
                last_sigma2 = float(res.conditional_volatility.iloc[-1] ** 2)
            except Exception:
                fc_sigma[t] = np.nan
                continue
        if omega is None or last_sigma2 is None:
            continue
        r_prev = ret_vals[t - 1] * SCALE
        indicator = 1.0 if r_prev < 0 else 0.0
        sigma2_next = omega + alpha * r_prev ** 2 + gamma * indicator * r_prev ** 2 + beta * last_sigma2
        if sigma2_next <= 0 or not np.isfinite(sigma2_next):
            fc_sigma[t] = np.nan
        else:
            sigma_next_pct = np.sqrt(sigma2_next)
            fc_sigma[t] = sigma_next_pct / SCALE
            last_sigma2 = sigma2_next
    return pd.Series(fc_sigma, index=log_returns.index)

class _VolTargetBaseStrategy(Strategy):
    ASYMMETRIC = False
    STRATEGY_NAME = None

    def __init__(self, ticker, initial_account_size, entry_percentage, pyramiding=1, allowshort=True, min_train=GARCH_MIN_TRAIN, refit_every=GARCH_REFIT_EVERY, vol_window=GARCH_VOL_WINDOW, low_q=GARCH_LOW_Q, high_q=GARCH_HIGH_Q):
        if not self.STRATEGY_NAME:
            raise ValueError('STRATEGY_NAME must be set on subclass')
        super().__init__(self.STRATEGY_NAME, ticker, '1D', initial_account_size, entry_percentage, pyramiding, allowshort)
        self.min_train = min_train
        self.refit_every = refit_every
        self.vol_window = vol_window
        self.low_q = low_q
        self.high_q = high_q

    def generate_signals(self):
        self.load_data()
        close = self.data['close']
        log_ret = np.log(close / close.shift(1)).dropna()
        sigma_fc = _garch_expanding_forecast(log_ret, asymmetry=self.ASYMMETRIC, min_train=self.min_train, refit_every=self.refit_every)
        lo_q = sigma_fc.rolling(self.vol_window, min_periods=self.vol_window // 2).quantile(self.low_q)
        hi_q = sigma_fc.rolling(self.vol_window, min_periods=self.vol_window // 2).quantile(self.high_q)
        signals = []
        state = 'flat'
        for ts in sigma_fc.index:
            s = sigma_fc.loc[ts]
            lo = lo_q.loc[ts]
            hi = hi_q.loc[ts]
            if pd.isna(s) or pd.isna(lo) or pd.isna(hi):
                continue
            price = close.loc[ts]
            if pd.isna(price):
                continue
            if s < lo and state != 'long':
                signals.append((ts, 'long', float(price)))
                state = 'long'
            elif s > hi and state != 'short':
                if self.allowshort:
                    signals.append((ts, 'short', float(price)))
                    state = 'short'
                elif state == 'long':
                    signals.append((ts, 'close long', float(price)))
                    state = 'flat'
        if signals:
            self.signals_df = pd.DataFrame(signals, columns=['time', 'signal', 'close']).set_index('time').sort_index()
        else:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])

class GARCHVolTargetStrategy(_VolTargetBaseStrategy):
    ASYMMETRIC = False
    STRATEGY_NAME = 'GARCH_VolTarget'

class TGARCHVolTargetStrategy(_VolTargetBaseStrategy):
    ASYMMETRIC = True
    STRATEGY_NAME = 'TGARCH_VolTarget'
VAR_MIN_TRAIN = 252
VAR_REFIT_EVERY = 20
VAR_MAX_LAGS = 5
ES_WINDOW = 63

def _load_market_close(ticker_imoex_path: str='data/IMOEX_data.csv') -> pd.Series | None:
    import os
    if not os.path.exists(ticker_imoex_path):
        return None
    try:
        df = pd.read_csv(ticker_imoex_path, parse_dates=['begin']).set_index('begin').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        return df['close'] if 'close' in df.columns else None
    except Exception:
        return None

def _var_walkforward_forecast(endog: pd.DataFrame, min_train: int, refit_every: int, max_lags: int, use_varmax: bool=False, exog: pd.DataFrame | None=None) -> pd.Series:
    n = len(endog)
    pred = np.full(n, np.nan)
    if n < min_train + 1:
        return pd.Series(pred, index=endog.index)
    results = None
    target_col = endog.columns[0]
    for t in range(min_train, n):
        need_refit = results is None or (t - min_train) % refit_every == 0
        if need_refit:
            try:
                train = endog.iloc[:t]
                if use_varmax:
                    train_exog = exog.iloc[:t] if exog is not None else None
                    model = VARMAX(train, exog=train_exog, order=(1, 0))
                    results = model.fit(disp=False, maxiter=50)
                else:
                    model = VAR(train)
                    results = model.fit(maxlags=max_lags, ic='aic')
            except Exception:
                results = None
                continue
        if results is None:
            continue
        try:
            if use_varmax:
                step_exog = exog.iloc[[t]] if exog is not None else None
                fc = results.forecast(steps=1, exog=step_exog)
                pred[t] = float(fc.iloc[0][target_col])
            else:
                lag = results.k_ar
                if lag == 0:
                    pred[t] = float(endog[target_col].iloc[:t].mean())
                else:
                    forecast_input = endog.values[t - lag:t]
                    fc_arr = results.forecast(y=forecast_input, steps=1)
                    pred[t] = float(fc_arr[0, 0])
        except Exception:
            continue
    return pd.Series(pred, index=endog.index)

class _VARBaseStrategy(Strategy):
    STRATEGY_NAME = 'VAR_Signal'
    USE_VARMAX = False
    USE_IMPROVED = False

    def __init__(self, ticker, timeframe='1D', initial_account_size=1000000, entry_percentage=100, pyramiding=1, allowshort=True, min_train=VAR_MIN_TRAIN, refit_every=VAR_REFIT_EVERY, max_lags=VAR_MAX_LAGS):
        super().__init__(self.STRATEGY_NAME, ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.min_train = int(min_train)
        self.refit_every = int(refit_every)
        self.max_lags = int(max_lags)

    def _build_endog(self, close: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        ret_ticker = np.log(close / close.shift(1))
        market = _load_market_close()
        if market is None:
            return (pd.DataFrame({'ret': ret_ticker}).dropna(), None)
        market = market.reindex(close.index).ffill()
        ret_market = np.log(market / market.shift(1))
        endog = pd.DataFrame({'ret_ticker': ret_ticker, 'ret_market': ret_market}).dropna()
        return (endog, None)

    def generate_signals(self):
        self.load_data()
        close = self.data['close']
        endog, exog = self._build_endog(close)
        if endog.empty:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])
            return
        try:
            pred = _var_walkforward_forecast(endog, min_train=self.min_train, refit_every=self.refit_every, max_lags=self.max_lags, use_varmax=self.USE_VARMAX, exog=exog)
        except Exception:
            if not self.USE_IMPROVED:
                self.signals_df = pd.DataFrame(columns=['signal', 'close'])
                return
            simple = endog.iloc[:, 0].rolling(self.refit_every, min_periods=1).mean()
            pred = simple.shift(1)
        signals = []
        state = 'flat'
        for ts in pred.index:
            mu = pred.loc[ts]
            if pd.isna(mu):
                continue
            price = close.asof(ts)
            if pd.isna(price):
                continue
            if mu > 0 and state != 'long':
                signals.append((ts, 'long', float(price)))
                state = 'long'
            elif mu < 0 and state != 'short':
                if self.allowshort:
                    signals.append((ts, 'short', float(price)))
                    state = 'short'
                elif state == 'long':
                    signals.append((ts, 'close long', float(price)))
                    state = 'flat'
        if signals:
            self.signals_df = pd.DataFrame(signals, columns=['time', 'signal', 'close']).set_index('time').sort_index()
        else:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])

class VARSignalStrategy(_VARBaseStrategy):
    STRATEGY_NAME = 'VAR_Signal'
    USE_VARMAX = False

class VARMAXSignalStrategy(_VARBaseStrategy):
    STRATEGY_NAME = 'VARMAX_Signal'
    USE_VARMAX = True

class ImprovedVARStrategy(_VARBaseStrategy):
    STRATEGY_NAME = 'ImprovedVAR_Signal'
    USE_VARMAX = False
    USE_IMPROVED = True

class ExponentialSmoothingStrategy(Strategy):

    def __init__(self, ticker, timeframe='1D', initial_account_size=1000000, entry_percentage=100, pyramiding=1, allowshort=True, min_train=VAR_MIN_TRAIN, refit_every=VAR_REFIT_EVERY, trend=None, seasonal=None, seasonal_periods=None, damped=False):
        super().__init__('ExpSmoothing_Signal', ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.min_train = int(min_train)
        self.refit_every = int(refit_every)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped = damped

    def generate_signals(self):
        self.load_data()
        close = self.data['close']
        log_ret = np.log(close / close.shift(1)).dropna()
        n = len(log_ret)
        pred = np.full(n, np.nan)
        if n < self.min_train + 1:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])
            return
        results = None
        for t in range(self.min_train, n):
            need_refit = results is None or (t - self.min_train) % self.refit_every == 0
            if need_refit:
                try:
                    train = log_ret.iloc[:t]
                    model = ExponentialSmoothing(train, trend=self.trend, seasonal=self.seasonal, seasonal_periods=self.seasonal_periods, damped_trend=self.damped)
                    results = model.fit()
                except Exception:
                    results = None
                    continue
            if results is None:
                continue
            try:
                fc = results.forecast(steps=1)
                pred[t] = float(fc.iloc[0] if hasattr(fc, 'iloc') else fc[0])
            except Exception:
                continue
        pred_s = pd.Series(pred, index=log_ret.index)
        signals = []
        state = 'flat'
        for ts in pred_s.index:
            mu = pred_s.loc[ts]
            if pd.isna(mu):
                continue
            price = close.asof(ts)
            if pd.isna(price):
                continue
            if mu > 0 and state != 'long':
                signals.append((ts, 'long', float(price)))
                state = 'long'
            elif mu < 0 and state != 'short':
                if self.allowshort:
                    signals.append((ts, 'short', float(price)))
                    state = 'short'
                elif state == 'long':
                    signals.append((ts, 'close long', float(price)))
                    state = 'flat'
        if signals:
            self.signals_df = pd.DataFrame(signals, columns=['time', 'signal', 'close']).set_index('time').sort_index()
        else:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])

class ExpectedShortfallStrategy(Strategy):

    def __init__(self, ticker, timeframe='1D', initial_account_size=1000000, entry_percentage=100, pyramiding=1, allowshort=True, min_train=VAR_MIN_TRAIN, window=ES_WINDOW, confidence=0.95, low_q=0.33, high_q=0.67):
        super().__init__('ES_Signal', ticker, timeframe, initial_account_size, entry_percentage, pyramiding, allowshort)
        self.min_train = int(min_train)
        self.window = int(window)
        self.confidence = float(confidence)
        self.low_q = float(low_q)
        self.high_q = float(high_q)

    def generate_signals(self):
        self.load_data()
        close = self.data['close']
        log_ret = np.log(close / close.shift(1)).dropna()
        n = len(log_ret)
        es_values = np.full(n, np.nan)
        if n < self.min_train + 1:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])
            return
        alpha = 1 - self.confidence
        for t in range(self.window, n):
            window_data = log_ret.iloc[t - self.window:t].values
            if len(window_data) == 0:
                continue
            sorted_w = np.sort(window_data)
            idx = max(int(len(sorted_w) * alpha), 1)
            tail = sorted_w[:idx]
            es_values[t] = abs(tail.mean()) if len(tail) > 0 else np.nan
        es_s = pd.Series(es_values, index=log_ret.index)
        lo = es_s.rolling(self.window * 4, min_periods=self.window).quantile(self.low_q)
        hi = es_s.rolling(self.window * 4, min_periods=self.window).quantile(self.high_q)
        signals = []
        state = 'flat'
        for ts in es_s.index:
            v = es_s.loc[ts]
            lv, hv = (lo.loc[ts], hi.loc[ts])
            if pd.isna(v) or pd.isna(lv) or pd.isna(hv):
                continue
            price = close.asof(ts)
            if pd.isna(price):
                continue
            if v < lv and state != 'long':
                signals.append((ts, 'long', float(price)))
                state = 'long'
            elif v > hv and state != 'short':
                if self.allowshort:
                    signals.append((ts, 'short', float(price)))
                    state = 'short'
                elif state == 'long':
                    signals.append((ts, 'close long', float(price)))
                    state = 'flat'
        if signals:
            self.signals_df = pd.DataFrame(signals, columns=['time', 'signal', 'close']).set_index('time').sort_index()
        else:
            self.signals_df = pd.DataFrame(columns=['signal', 'close'])
