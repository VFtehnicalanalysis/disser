import os
import sys
import time
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from strategies.vf import MyStrategy
from strategies.ema_longterm import LongTermHold
from reports.vf_report import generate_vf_report, export_vf_csv
VF_CONFIGS_1D = [{'name': 'VF_D_default', 'comment': 'PineScript-defaults: long-only, htf=1W, без стопов', 'params': dict(timeframe='1D', pyramiding=2, allowshort=False, entry_percentage=50.0, htf='1W', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=False, is_using_trend_analysis=True, is_smart_stop_activated=False, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, sell_count=10, is_price_step=False)}, {'name': 'VF_D_short', 'comment': 'С шортами + htf=1W', 'params': dict(timeframe='1D', pyramiding=2, allowshort=True, entry_percentage=50.0, htf='1W', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=False, is_using_trend_analysis=True, is_smart_stop_activated=False, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, sell_count=10, is_price_step=False)}, {'name': 'VF_D_stops_TP', 'comment': 'Со стандартными стопами 3%/1% и TP 6%/3%', 'params': dict(timeframe='1D', pyramiding=2, allowshort=True, entry_percentage=50.0, htf='1W', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=False, is_using_trend_analysis=True, is_smart_stop_activated=False, is_using_stops=True, stop_l_percent=3.0, stop_s_percent=1.0, is_using_take_profits=True, TP_l_percent=6.0, TP_s_percent=3.0, is_using_trailing_stop=False, sell_count=10, is_price_step=False)}, {'name': 'VF_D_smart_stop', 'comment': 'Smart-стоп: активация +10%, стоп +5%', 'params': dict(timeframe='1D', pyramiding=2, allowshort=False, entry_percentage=50.0, htf='1W', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=False, is_using_trend_analysis=True, is_smart_stop_activated=True, smart_stop=5.0, smart_stop_activation=10.0, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, sell_count=10, is_price_step=False)}]
VF_CONFIGS_1H = [{'name': 'VF_H_TV_exact', 'comment': 'ТОЧНОЕ соответствие TradingView: MACD ON, sell_count=5, smart_stop ON, allowshort', 'params': dict(timeframe='1h', pyramiding=2, allowshort=True, entry_percentage=50.0, htf='1D', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=True, macd_fast=12, macd_slow=26, macd_signal=9, is_using_trend_analysis=True, check_info='BUY & SELL', is_smart_stop_activated=True, smart_stop=5.0, smart_stop_activation=10.0, is_rsi_changer=False, rsi_changer=11, sell_count=5, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, is_price_step=False)}, {'name': 'VF_H_default', 'comment': 'Часовая, long-only, htf=1D, без стопов', 'params': dict(timeframe='1h', pyramiding=2, allowshort=False, entry_percentage=50.0, htf='1D', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=False, is_using_trend_analysis=True, is_smart_stop_activated=False, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, sell_count=10, is_price_step=False)}, {'name': 'VF_H_short', 'comment': 'Часовая с шортами, htf=1D', 'params': dict(timeframe='1h', pyramiding=2, allowshort=True, entry_percentage=50.0, htf='1D', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=False, is_using_trend_analysis=True, is_smart_stop_activated=False, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, sell_count=10, is_price_step=False)}, {'name': 'VF_H_smart_stop', 'comment': 'Часовая с шортами + smart-stop', 'params': dict(timeframe='1h', pyramiding=2, allowshort=True, entry_percentage=50.0, htf='1D', rsi_period=14, rsi_low=30, rsi_high=70, L_short_period=100, L_long_period=200, H_short_period=100, H_long_period=200, pivot_period=3, is_using_macd=False, is_using_trend_analysis=True, is_smart_stop_activated=True, smart_stop=5.0, smart_stop_activation=10.0, is_using_stops=False, is_using_take_profits=False, is_using_trailing_stop=False, sell_count=10, is_price_step=False)}]

def run_one(ticker, config, initial_account=1000000):
    name = config['name']
    print(f"\n{'=' * 70}\n>> {name}: {config['comment']}\n{'=' * 70}")
    t0 = time.time()
    strat = MyStrategy(name=name, ticker=ticker, initial_account_size=initial_account, **config['params'])
    strat.load_data()
    strat.generate_signals()
    strat.execute_trades()
    strat.finalize()
    elapsed = time.time() - t0
    stats = strat.get_stats()
    print(f'  Завершено за {elapsed:.1f}s')
    print(f"  Сделок: {stats.get('Количество сделок', 0)} | Win%: {stats.get('Процент прибыльных сделок', 0)} | Доходность: {stats.get('Процент роста', 0)}% | Sharpe: {stats.get('Sharpe', '—')} | MaxDD: {stats.get('Макс. просадка (%)', 0)}%")
    return strat

def run_benchmark(ticker, initial_account=1000000):
    print(f"\n{'=' * 70}\n>> BENCHMARK: LongTermHold (1D Buy&Hold)\n{'=' * 70}")
    bh = LongTermHold(ticker, '1D', initial_account, 100)
    bh.generate_signals()
    bh.execute_trades()
    bh.equity_df = bh.get_equity_curve()
    stats = bh.get_stats()
    print(f"  Доходность: {stats.get('Процент роста', 0)}%")
    return bh

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'SBER'
    output_dir = 'results/vf_strategy'
    print(f"\n{'#' * 70}\n#  VF STRATEGY RUNNER — {ticker}\n{'#' * 70}")
    print(f'Конфигураций 1D: {len(VF_CONFIGS_1D)}')
    print(f'Конфигураций 1h: {len(VF_CONFIGS_1H)}')
    print(f'+ 1 Buy&Hold benchmark')
    print(f'Output: {output_dir}/')
    strategies = []
    for cfg in VF_CONFIGS_1D:
        try:
            strategies.append(run_one(ticker, cfg))
        except Exception as e:
            print(f'  ОШИБКА: {e}')
            import traceback
            traceback.print_exc()
    for cfg in VF_CONFIGS_1H:
        try:
            strategies.append(run_one(ticker, cfg))
        except Exception as e:
            print(f'  ОШИБКА: {e}')
            import traceback
            traceback.print_exc()
    try:
        strategies.append(run_benchmark(ticker))
    except Exception as e:
        print(f'  Benchmark ОШИБКА: {e}')
    print(f"\n{'=' * 70}\nЭкспорт результатов в {output_dir}/\n{'=' * 70}")
    export_vf_csv(ticker, strategies, output_dir=output_dir)
    generate_vf_report(ticker, strategies, output_dir=output_dir, report_suffix='vf_comparison', price_data_tf='1h')
    print(f"\n{'=' * 70}\nСВОДНАЯ ТАБЛИЦА\n{'=' * 70}")
    print(f"{'Стратегия':<25} {'TF':<5} {'Сделок':>7} {'Win%':>6} {'Доход%':>8} {'MaxDD%':>8} {'Sharpe':>7}")
    print('-' * 75)
    for s in strategies:
        st = s.get_stats()
        print(f"{s.name:<25} {s.timeframe:<5} {st.get('Количество сделок', 0):>7} {st.get('Процент прибыльных сделок', 0):>6} {st.get('Процент роста', 0):>8} {st.get('Макс. просадка (%)', 0):>8} {str(st.get('Sharpe', '—')):>7}")
if __name__ == '__main__':
    main()
