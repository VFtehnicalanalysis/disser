import os
import sys
import time
import json
import subprocess
import argparse
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
import pandas as pd
ALL_TICKERS = ['LKOH', 'SBER', 'GAZP', 'ROSN', 'VKCO', 'AFKS', 'VTBR', 'SNGSP', 'GMKN', 'PLZL']

def regenerate_data(tickers: list, end_date: str='2026-04-19'):
    print(f"\n{'=' * 70}\n[STEP 1] Регенерация _new.csv для {len(tickers)} тикеров\n{'=' * 70}")
    from TA.ta_loader import change_data, change_hourly_data
    change_data(tickers, '2014-01-01', end_date)
    change_hourly_data(tickers)

def run_optimization_pipeline(ticker: str, timeframe: str, trials: int, output_dir: str, top_ensemble: int=5):
    htf = '1W' if timeframe == '1D' else '1D'
    print(f"\n{'#' * 70}")
    print(f'#  PIPELINE: {ticker} {timeframe} (htf={htf}, trials={trials})')
    print(f"{'#' * 70}")
    py = sys.executable
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    steps = [('Sensitivity analysis (200 trials TPE)', [py, '-m', 'optimization.vf_sensitivity', ticker, timeframe, '--trials', '200']), ('NSGA-II multi-objective optimization', [py, '-m', 'optimization.vf_optimizer_v2', ticker, timeframe, '--trials', str(trials), '--folds', '3']), ('Apply Pareto-optimal points', [py, '-m', 'optimization.vf_pareto', ticker, timeframe, '--output', output_dir]), ('Build ensembles (equal + sharpe-weighted)', [py, '-m', 'optimization.vf_ensemble', ticker, timeframe, '--top', str(top_ensemble), '--output', output_dir]), ('Generate visual HTML report', [py, '-m', 'reports.vf_visual', ticker, timeframe, '--output', output_dir])]
    results = {'ticker': ticker, 'timeframe': timeframe, 'steps': []}
    for step_name, cmd in steps:
        print(f'\n--- [{ticker} {timeframe}] {step_name} ---')
        t0 = time.time()
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=3600)
            elapsed = time.time() - t0
            ok = r.returncode == 0
            print(f"  {('OK' if ok else 'FAILED')} ({elapsed:.1f}s, exit={r.returncode})")
            if not ok:
                print('  STDERR (last 10 lines):')
                for line in r.stderr.splitlines()[-10:]:
                    print(f'    {line}')
            results['steps'].append({'name': step_name, 'elapsed_sec': elapsed, 'ok': ok, 'returncode': r.returncode})
            if not ok and 'NSGA-II' in step_name:
                print(f'  ✗ Прерываем pipeline для {ticker} {timeframe}')
                break
        except subprocess.TimeoutExpired:
            print(f'  ✗ TIMEOUT после {time.time() - t0:.1f}s')
            results['steps'].append({'name': step_name, 'ok': False, 'timeout': True})
            break
        except Exception as e:
            print(f'  ✗ Exception: {e}')
            results['steps'].append({'name': step_name, 'ok': False, 'error': str(e)})
            break
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', default=ALL_TICKERS, help='Список тикеров для оптимизации')
    parser.add_argument('--timeframes', nargs='+', default=['1h', '1D'], choices=['1h', '1D'], help='Таймфреймы')
    parser.add_argument('--trials', type=int, default=300, help='Trials для NSGA-II')
    parser.add_argument('--top-ensemble', type=int, default=5, help='Top-N Pareto-точек для ансамбля')
    parser.add_argument('--output', default='results/vf_strategy', help='Папка для результатов')
    parser.add_argument('--skip-data-regen', action='store_true', help='Пропустить регенерацию _new.csv')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    print(f"\n{'#' * 70}")
    print(f'#  VF STRATEGY MULTI-TICKER OPTIMIZATION')
    print(f'#  Tickers: {args.tickers}')
    print(f'#  Timeframes: {args.timeframes}')
    print(f'#  Trials per (ticker, TF): {args.trials}')
    print(f'#  Output: {args.output}')
    print(f"{'#' * 70}")
    overall_t0 = time.time()
    if not args.skip_data_regen:
        regenerate_data(args.tickers)
    else:
        print('Регенерация данных пропущена (--skip-data-regen)')
    all_results = []
    for ticker in args.tickers:
        for tf in args.timeframes:
            r = run_optimization_pipeline(ticker, tf, args.trials, args.output, args.top_ensemble)
            all_results.append(r)
    overall_elapsed = time.time() - overall_t0
    summary = {'tickers': args.tickers, 'timeframes': args.timeframes, 'trials': args.trials, 'total_time_sec': overall_elapsed, 'results': all_results}
    summary_path = os.path.join(args.output, 'OPTIMIZE_ALL_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'=' * 70}\nИТОГОВАЯ СВОДКА\n{'=' * 70}")
    print(f'Общее время: {overall_elapsed / 60:.1f} мин')
    print(f"\n{'Ticker':<8} {'TF':<5} {'Status':<10}")
    for r in all_results:
        all_ok = all((s.get('ok', False) for s in r['steps']))
        print(f"{r['ticker']:<8} {r['timeframe']:<5} {('OK' if all_ok else 'PARTIAL')}")
if __name__ == '__main__':
    main()
