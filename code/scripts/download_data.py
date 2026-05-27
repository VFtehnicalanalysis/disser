import argparse
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import load_moex_data, load_moex_hourly_data
TICKERS = ['LKOH', 'SBER', 'GAZP', 'ROSN', 'VKCO', 'AFKS', 'VTBR', 'SNGSP', 'GMKN', 'PLZL']
INDEX = 'IMOEX'
RISK_FREE_RATE = 'MOEXREPO'
START_DATE = '2014-01-01'
END_DATE = '2026-04-20'

def download_daily():
    print('=' * 60)
    print(f'Загрузка ДНЕВНЫХ данных: {START_DATE} ... {END_DATE}')
    print('=' * 60)
    t0 = time.time()
    load_moex_data(TICKERS, START_DATE, END_DATE)
    load_moex_data([INDEX, RISK_FREE_RATE], START_DATE, END_DATE, market='index')
    print(f'\nДневные данные загружены за {time.time() - t0:.1f} сек.')

def download_hourly():
    print('=' * 60)
    print(f'Загрузка ЧАСОВЫХ данных: {START_DATE} ... {END_DATE}')
    print('  (это займёт 30-60 минут, блоки по 30 дней)')
    print('=' * 60)
    t0 = time.time()
    load_moex_hourly_data(TICKERS, START_DATE, END_DATE)
    load_moex_hourly_data([INDEX, RISK_FREE_RATE], START_DATE, END_DATE, market='index')
    print(f'\nЧасовые данные загружены за {time.time() - t0:.1f} сек.')

def main():
    parser = argparse.ArgumentParser(description='Загрузка MOEX-данных')
    parser.add_argument('--daily-only', action='store_true', help='Только дневные')
    parser.add_argument('--hourly-only', action='store_true', help='Только часовые')
    parser.add_argument('--all', action='store_true', help='Все (по умолчанию)')
    args = parser.parse_args()
    if args.daily_only:
        download_daily()
    elif args.hourly_only:
        download_hourly()
    else:
        download_daily()
        download_hourly()
    print('\nЗагрузка завершена.')
if __name__ == '__main__':
    main()
