import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_combined_riskfree():
    repo = pd.read_csv('data/MOEXREPO_data.csv')
    rusfar = pd.read_csv('data/RUSFAR_data.csv')
    for df in (repo, rusfar):
        df['begin'] = pd.to_datetime(df['begin'])
    cutoff = pd.Timestamp('2018-01-01')
    repo_trim = repo[repo['begin'] < cutoff].copy()
    rusfar_trim = rusfar[rusfar['begin'] >= cutoff].copy()
    combined = pd.concat([repo_trim[['begin', 'close']], rusfar_trim[['begin', 'close']]], ignore_index=True)
    combined = combined.sort_values('begin').drop_duplicates('begin', keep='first')
    combined = combined.set_index('begin')
    combined.index = pd.to_datetime(combined.index)
    combined['open'] = combined['close']
    combined['high'] = combined['close']
    combined['low'] = combined['close']
    combined['value'] = 0
    combined['volume'] = 0
    combined['end'] = combined.index + pd.Timedelta(seconds=86399)
    combined = combined[['open', 'close', 'high', 'low', 'value', 'volume', 'end']]
    out_path = 'data/RISKFREE_data.csv'
    combined.to_csv(out_path)
    print(f'Объединённая risk-free кривая сохранена в {out_path}')
    print(f'  Период: {combined.index[0].date()} ... {combined.index[-1].date()}')
    print(f'  Строк: {len(combined)}')
    print(f"  Первое close: {combined['close'].iloc[0]}")
    print(f"  Последнее close: {combined['close'].iloc[-1]}")
    print(f"  Среднее close: {combined['close'].mean():.2f}")
if __name__ == '__main__':
    build_combined_riskfree()
