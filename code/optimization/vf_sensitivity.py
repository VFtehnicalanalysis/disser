import os
import sys
import json
import argparse
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.importance import FanovaImportanceEvaluator, get_param_importances

def reconstruct_studies_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    studies = {}
    param_cols = [c for c in df.columns if c.startswith('p_')]
    for fold_id in df['fold'].unique():
        fold_df = df[df['fold'] == fold_id]
        studies[fold_id] = fold_df
    return studies

def build_distributions_from_search_space():
    return {'rsi_period': optuna.distributions.IntDistribution(7, 30), 'rsi_low': optuna.distributions.IntDistribution(15, 40), 'rsi_high': optuna.distributions.IntDistribution(60, 85), 'pivot_period': optuna.distributions.IntDistribution(2, 10), 'L_short_period': optuna.distributions.IntDistribution(20, 150), 'L_long_period': optuna.distributions.IntDistribution(40, 300), 'H_short_period': optuna.distributions.IntDistribution(20, 150), 'H_long_period': optuna.distributions.IntDistribution(40, 300), 'is_using_macd': optuna.distributions.CategoricalDistribution([True, False]), 'is_smart_stop_activated': optuna.distributions.CategoricalDistribution([True, False]), 'smart_stop': optuna.distributions.FloatDistribution(1.0, 10.0, step=0.5), 'smart_stop_activation': optuna.distributions.FloatDistribution(5.0, 25.0, step=1.0), 'sell_count': optuna.distributions.IntDistribution(5, 30), 'pyramiding': optuna.distributions.IntDistribution(1, 3), 'allowshort': optuna.distributions.CategoricalDistribution([True, False])}

def rerun_and_compute_importance(ticker: str, timeframe: str, n_trials: int=200, output_dir: str='results/vf_strategy'):
    from .vf_optimizer import WalkForwardOptimizer, get_search_space, run_strategy_on_slice, composite_score, STATIC_PARAMS
    if timeframe == '1D':
        data_file = f'data/{ticker}_data_new.csv'
        htf = '1W'
    else:
        data_file = f'data/{ticker}_hourly_data_new.csv'
        htf = '1D'
    df = pd.read_csv(data_file)
    df['begin'] = pd.to_datetime(df['begin'])
    df.set_index('begin', inplace=True)
    df.sort_index(inplace=True)
    print(f'[{ticker} {timeframe}] {len(df)} баров, {df.index[0].date()} .. {df.index[-1].date()}')
    print(f'Запуск {n_trials} trials для importance analysis...')
    train_data = df.copy()

    def objective(trial):
        params = get_search_space(trial)
        stats, _ = run_strategy_on_slice(ticker, timeframe, params, train_data, htf)
        if stats is None:
            return -1000000000.0
        return composite_score(stats)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f'Завершено. Best score: {study.best_value:.4f}')
    print(f'Best params: {study.best_params}')
    print('\nВычисление importance (fANOVA)...')
    try:
        importances = get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=42))
    except Exception as e:
        print(f'fANOVA не удалось: {e}. Используем mean decrease impurity (RandomForest).')
        from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
        importances = get_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator(seed=42))
    print(f"\n{'=' * 60}")
    print(f'IMPORTANCE RANKING ({ticker} {timeframe})')
    print(f"{'=' * 60}")
    print(f"{'Параметр':<28} {'Importance':>12}")
    print('-' * 42)
    for param, score in sorted(importances.items(), key=lambda x: -x[1]):
        bar = '█' * int(score * 50)
        print(f'{param:<28} {score:>12.4f}  {bar}')
    out = {'ticker': ticker, 'timeframe': timeframe, 'n_trials': n_trials, 'best_score': study.best_value, 'best_params': study.best_params, 'importance': dict(importances)}
    json_path = os.path.join(output_dir, f'{ticker}_importance_{timeframe}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    print(f'\nSaved: {json_path}')
    rows = [{'param': p, 'importance': score} for p, score in sorted(importances.items(), key=lambda x: -x[1])]
    csv_path = os.path.join(output_dir, f'{ticker}_importance_{timeframe}.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f'Saved: {csv_path}')
    return (importances, study.best_params)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ticker', default='SBER', nargs='?')
    parser.add_argument('timeframe', choices=['1D', '1h'], default='1h', nargs='?')
    parser.add_argument('--trials', type=int, default=200)
    args = parser.parse_args()
    rerun_and_compute_importance(args.ticker, args.timeframe, args.trials)
if __name__ == '__main__':
    main()
