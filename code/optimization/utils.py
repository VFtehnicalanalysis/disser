import json
import os
import numpy as np
import pandas as pd
PARAM_SPACE = {'ema': [('short_ema', 'int', 5, 100), ('long_ema', 'int', 50, 200), ('allowshort', 'cat', [True, False])], 'rsi': [('rsi_lower', 'int', 10, 45), ('rsi_upper', 'int', 55, 90), ('rsi_window', 'int', 7, 30), ('allowshort', 'cat', [True, False])], 'macd': [('macd_fast', 'int', 5, 20), ('macd_slow', 'int', 20, 50), ('macd_signal', 'int', 5, 15), ('allowshort', 'cat', [True, False])], 'breakout_pivot': [('lookback', 'int', 100, 300), ('pivot_window', 'int', 3, 12), ('min_touches', 'int', 2, 5), ('tolerance_atr_mult', 'float', 0.3, 1.2, 0.05), ('cluster_atr_mult', 'float', 0.3, 1.2, 0.05), ('confirmation_bars', 'int', 1, 3), ('cooldown_bars', 'int', 3, 20), ('allowshort', 'cat', [True, False])], 'breakout_ols': [('lookback', 'int', 100, 300), ('pivot_window', 'int', 3, 12), ('min_touches', 'int', 2, 5), ('tolerance_atr_mult', 'float', 0.3, 1.2, 0.05), ('cluster_atr_mult', 'float', 0.3, 1.2, 0.05), ('confirmation_bars', 'int', 1, 3), ('cooldown_bars', 'int', 3, 20), ('allowshort', 'cat', [True, False])], 'retest_pivot': [('lookback', 'int', 100, 300), ('pivot_window', 'int', 3, 12), ('min_touches', 'int', 2, 5), ('tolerance_atr_mult', 'float', 0.3, 1.2, 0.05), ('cluster_atr_mult', 'float', 0.3, 1.2, 0.05), ('confirmation_bars', 'int', 1, 3), ('cooldown_bars', 'int', 3, 20), ('retest_window', 'int', 10, 40), ('cancel_atr_mult', 'float', 0.5, 2.5, 0.1), ('allowshort', 'cat', [True, False])], 'retest_ols': [('lookback', 'int', 100, 300), ('pivot_window', 'int', 3, 12), ('min_touches', 'int', 2, 5), ('tolerance_atr_mult', 'float', 0.3, 1.2, 0.05), ('cluster_atr_mult', 'float', 0.3, 1.2, 0.05), ('confirmation_bars', 'int', 1, 3), ('cooldown_bars', 'int', 3, 20), ('retest_window', 'int', 10, 40), ('cancel_atr_mult', 'float', 0.5, 2.5, 0.1), ('allowshort', 'cat', [True, False])], 'obv': [('obv_ema_window', 'int', 5, 30), ('confirmation_bars', 'int', 1, 5), ('allowshort', 'cat', [True, False])], 'vwap': [('vwap_window', 'int', 20, 200), ('vwap_tol', 'float', 0.001, 0.02, 0.001), ('vwap_mode', 'cat', ['rolling', 'yearly']), ('allowshort', 'cat', [True, False])], 'engulfing': [('doji_tol', 'float', 0.03, 0.2, 0.01), ('wick_body_ratio', 'float', 1.5, 3.5, 0.1), ('allowshort', 'cat', [True, False])], 'hammerdoji': [('doji_tol', 'float', 0.03, 0.2, 0.01), ('wick_body_ratio', 'float', 1.5, 3.5, 0.1), ('allowshort', 'cat', [True, False])], 'harami': [('doji_tol', 'float', 0.03, 0.2, 0.01), ('wick_body_ratio', 'float', 1.5, 3.5, 0.1), ('allowshort', 'cat', [True, False])], 'candle_ensemble': [('ensemble_window', 'int', 3, 10), ('min_votes', 'int', 2, 4), ('doji_tol', 'float', 0.03, 0.2, 0.01), ('wick_body_ratio', 'float', 1.5, 3.5, 0.1), ('allowshort', 'cat', [True, False])]}

def get_step(stype, param_name):
    for entry in PARAM_SPACE.get(stype, []):
        if entry[0] == param_name and entry[1] == 'float':
            return entry[4] if len(entry) >= 5 else None
    return None

def _round_to_step(x, lo, step):
    if step is None or step <= 0:
        return x
    k = round((x - lo) / step)
    return lo + k * step

def list_bounds(stype):
    space = PARAM_SPACE.get(stype)
    if space is None:
        raise ValueError(f'Unknown strategy type: {stype}')
    bounds = []
    for entry in space:
        name = entry[0]
        kind = entry[1]
        if kind == 'int':
            lo, hi = (entry[2], entry[3])
            bounds.append((name, 'int', float(lo), float(hi)))
        elif kind == 'float':
            lo, hi = (entry[2], entry[3])
            bounds.append((name, 'float', float(lo), float(hi)))
        elif kind == 'cat':
            values = entry[2]
            bounds.append((name, 'cat', 0.0, float(len(values) - 1)))
        else:
            raise ValueError(f'Unknown kind {kind} for {stype}.{name}')
    return bounds

def vec_to_params(stype, vec):
    space = PARAM_SPACE[stype]
    if len(vec) != len(space):
        raise ValueError(f'Wrong vec length for {stype}: expected {len(space)}, got {len(vec)}')
    params = {}
    for entry, x in zip(space, vec):
        name = entry[0]
        kind = entry[1]
        if kind == 'int':
            lo, hi = (entry[2], entry[3])
            params[name] = int(round(float(np.clip(x, lo, hi))))
        elif kind == 'float':
            lo, hi = (entry[2], entry[3])
            step = entry[4] if len(entry) >= 5 else None
            val = float(np.clip(x, lo, hi))
            if step is not None:
                val = round(_round_to_step(val, lo, step), 10)
                val = float(np.clip(val, lo, hi))
            params[name] = val
        elif kind == 'cat':
            values = entry[2]
            idx = int(round(float(np.clip(x, 0, len(values) - 1))))
            params[name] = values[idx]
        else:
            raise ValueError(f'Unknown kind {kind}')
    return params

def params_to_constraint_ok(stype, params):
    if stype == 'ema':
        return params['short_ema'] < params['long_ema']
    if stype == 'macd':
        return params['macd_fast'] < params['macd_slow']
    return True

def save_topn(rows, path):
    if not rows:
        print(f'  (нет строк для сохранения в {path})')
        return
    df_new = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df_out = merge_append(path, df_new)
    df_out.to_csv(path, index=False)
    print(f'  Сохранено: {path} (+{len(df_new)}, всего {len(df_out)} строк)')

def merge_append(path, df_new, key_cols=('ticker', 'strategy_type', 'timeframe', 'method')):
    if not os.path.exists(path):
        return df_new
    try:
        df_old = pd.read_csv(path)
    except Exception:
        return df_new
    if df_old.empty:
        return df_new
    keys_in_new = set()
    for _, row in df_new[list(key_cols)].iterrows():
        keys_in_new.add(tuple(row.tolist()))
    if all((c in df_old.columns for c in key_cols)):

        def is_old_row_kept(row):
            k = tuple((row[c] for c in key_cols))
            return k not in keys_in_new
        mask_keep = df_old.apply(is_old_row_kept, axis=1)
        df_old = df_old[mask_keep]
    return pd.concat([df_old, df_new], ignore_index=True, sort=False)
