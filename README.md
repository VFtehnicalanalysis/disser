# Роль методов технического анализа в формировании инвестиционной стратегии

Эмпирическое исследование 37 торговых стратегий (16 ТА + 8 СА + 12 ФА + 1 авторская VF Strategy) на 10 ликвидных акциях Московской биржи за период **2014 — 2026**.

> Магистерская диссертация. Экономический факультет МГУ имени М. В. Ломоносова.
> Магистрант — **Фурасов Владислав Дмитриевич**. Научный руководитель — **Маркина Вероника Сергеевна.**

---

## Что внутри

| Папка | Что содержит |
|---|---|
| `code/` | Воспроизводимый код на Python (стратегии, бэктест, оптимизация, отчёты) + Pine Script |
| `data/` | Котировки OHLCV: дневные и часовые, 10 тикеров + IMOEX/MOEXREPO/RUSFAR |
| `finance_data/` | Финансовая отчётность, дивиденды, мультипликаторы (для ФА-стратегий) |
| `ta_data/` | Кэш предрасчитанных технических индикаторов (EMA, RSI, MACD, OBV, VWAP, ATR) |
| `results/` | CSV-результаты всех бэктестов и оптимизаций, на которые ссылается работа |

## Тикеры

`LKOH`, `SBER`, `GAZP`, `ROSN`, `VKCO`, `AFKS`, `VTBR`, `SNGSP`, `GMKN`, `PLZL`, индекс `IMOEX`. Безрисковая ставка — композит `MOEXREPO` (до 2018 г.) и `RUSFAR` (с 2018 г.).

## Источники данных

- **Котировки** — официальный [MOEX ISS API](https://iss.moex.com/) через библиотеку `apimoex` (дневные и часовые свечи).
- **Финансовая отчётность и мультипликаторы** — [financemarker.ru](https://financemarker.ru/), исторические данные TradingView.

---

## Установка

```bash
# 1. Клонировать / разархивировать содержимое в рабочий каталог
cd for_publication

# 2. Создать виртуальное окружение и поставить зависимости
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate       # Linux / macOS
pip install -r code/requirements.txt
```

Требуется Python ≥ 3.10.

---

## Воспроизведение исследования

Все команды запускаются из корня архива (`for_publication/`) c `code/` в `PYTHONPATH`:

```bash
set PYTHONPATH=%CD%\code         # Windows (cmd)
$env:PYTHONPATH = "$PWD\code"    # Windows (PowerShell)
export PYTHONPATH=$PWD/code      # Linux / macOS
```

### 1. Подготовка данных

Если хочется загрузить котировки заново (иначе CSV уже лежат в `data/`):

```bash
python code/scripts/download_data.py            # все таймфреймы
python code/scripts/download_data.py --daily-only
python code/scripts/download_data.py --hourly-only
```

Сборка композитной безрисковой ставки (MOEXREPO + RUSFAR):

```bash
python code/scripts/build_riskfree.py
```

### 2. Бэктест 37 стратегий

Прогон всех классических стратегий на всех тикерах (TA + FA + SA + VF):

```bash
python code/main.py
```

На выходе в `results/` появятся файлы `{ТИКЕР}_period_stats_full.csv`, `{ТИКЕР}_strategies_equity.csv`, `{ТИКЕР}_strategies_trades.csv` и `{ТИКЕР}_strategies_stats.csv`.

### 3. Оптимизация параметров

**Bayesian TPE (Optuna)** — байесовская оптимизация по 13 классам ТА-стратегий:

```bash
python -m optimization.bayesian_ta --type all --timeframe 1D --trials 50
python -m optimization.bayesian_ta --type ema --ticker SBER --timeframe 1h --trials 50
```

Результаты — `results/optimization/{type}_top5.csv`, `..._all_trials.csv` и сводный `all_top5.csv`.

**Differential Evolution (SciPy)** — параллельный прогон для проверки устойчивости:

```bash
python -m optimization.diffevolution_ta --type all --timeframe 1D --maxiter 30 --popsize 12
```

Результаты — `results/optimization_de/{type}_de_top5.csv` и `all_de_top5.csv`.

**NSGA-II multi-objective для VF Strategy** (3 цели: Sharpe ↑, MaxDD ↓, std-Sharpe ↓):

```bash
python -m optimization.vf_optimizer_v2 SBER 1h --trials 400 --folds 3
```

Результаты — `results/vf_strategy/{ТИКЕР}_v2_pareto_{ТФ}.csv` и интерактивный HTML-фронт.

**Полнопараметрический Bayesian с тёплым стартом от Pareto** (14 параметров VF):

```bash
python -m optimization.vf_bayesian SBER 1h --trials 500
```

Результаты — `results/vf_strategy/{ТИКЕР}_bayesian_full_{ТФ}.csv`.

**Прогон всех тикеров в одном цикле**:

```bash
python -m optimization.vf_optimize_all
```

### 4. Walk-forward валидация (10 окон: 3 года обучения + 1 год тест)

```bash
python -m backtest.walk_forward --type ema --ticker SBER --trials 30
python -m backtest.walk_forward --type macd
python -m backtest.walk_forward --type rsi
```

Результаты — `results/walkforward/{ТИКЕР}_{strategy}_vf_params.csv` и `_vf_equity.csv`.

### 5. Применение лучших параметров и сборка ансамбля

```bash
python -m optimization.vf_apply_optimized       # apply top-1 Pareto
python -m optimization.vf_ensemble              # equal-weight ensemble (top-5)
```

### 6. Сводный отчёт и проверка гипотез H1 / H2

```bash
python -m reports.consolidated
```

На выходе — `results/THESIS_CONSOLIDATED_REPORT.html` (сводная картина по 10 тикерам) и расчёт статистических тестов (t-критерий Уэлча для H1, U-критерий Манна-Уитни для H2, BCa block-bootstrap для доверительных интервалов).

### 7. Авторская стратегия в TradingView

Pine Script реализация VF Strategy лежит в [`code/pinescript/vf_strategy.pine`](code/pinescript/vf_strategy.pine) — её можно скопировать в редактор TradingView и запустить на любом инструменте.

---

## Структура результатов (`results/`)

```
results/
├── {ТИКЕР}_period_stats_full.csv     # все 25 метрик по каждой стратегии за полный период
├── {ТИКЕР}_period_stats_IS.csv       # то же на in-sample (2014-2024)
├── {ТИКЕР}_period_stats_OOS.csv      # то же на out-of-sample (2025-2026)
├── {ТИКЕР}_strategies_equity.csv     # кривые капитала по всем стратегиям
├── {ТИКЕР}_strategies_stats.csv      # агрегированная статистика по стратегиям
├── {ТИКЕР}_strategies_trades.csv     # лог отдельных сделок
├── walkforward/                      # 10-оконная walk-forward валидация
├── optimization/                     # Bayesian TPE: top-5 и полные трейлы по стратегиям
├── optimization_de/                  # Differential Evolution: то же для проверки устойчивости
└── vf_strategy/                      # VF Strategy: NSGA-II Pareto, fANOVA, full Bayesian, ансамбли
    ├── ALL_TICKERS_summary.csv       # сводная таблица VF по 10 тикерам
    ├── {ТИКЕР}_v2_pareto_{ТФ}.csv    # точки Pareto-фронта
    ├── {ТИКЕР}_importance_{ТФ}.csv   # fANOVA важность параметров
    ├── {ТИКЕР}_bayesian_full_{ТФ}.csv# полнопараметрический Bayesian
    └── {ТИКЕР}_ensemble_summary_{ТФ}.csv
```

---

## Структура кода (`code/`)

```
code/
├── config.py                  # пути проекта
├── main.py                    # сквозной прогон: загрузка → индикаторы → бэктест
├── data/                      # MOEX ISS API и financemarker.ru загрузчики
├── strategies/
│   ├── base.py                # базовый интерфейс Strategy и движок Trade
│   ├── technical.py           # 16 ТА-стратегий
│   ├── fundamental.py         # 12 ФА-стратегий (DDM, DCF, DivYield, MR и др.)
│   ├── statistical.py         # 8 СА-моделей (ARIMA, GARCH, T-GARCH, VAR, ES)
│   ├── vf.py                  # авторская VF Strategy (14 параметров)
│   └── ema_longterm.py        # бенчмарки Long-Term Hold / EMA-долгосрочная
├── backtest/
│   ├── portfolio.py           # запуск всех стратегий по тикеру
│   ├── walk_forward.py        # walk-forward валидация (3y train + 1y test)
│   ├── vf_runner.py           # обёртка для VF-прогонов
│   └── vf_oos_check.py        # OOS-проверка лучших конфигураций
├── optimization/
│   ├── bayesian_ta.py         # Optuna TPE по 13 классам стратегий
│   ├── diffevolution_ta.py    # SciPy Differential Evolution
│   ├── vf_optimizer_v2.py     # NSGA-II Pareto для VF
│   ├── vf_bayesian.py         # 14-параметрический Bayesian с warm-start
│   ├── vf_pareto.py           # построение и сериализация Pareto-фронта
│   ├── vf_ensemble.py         # ансамбль top-5 Pareto точек
│   ├── vf_apply_optimized.py  # применение лучших параметров
│   └── vf_sensitivity.py      # анализ чувствительности по транзакционным издержкам
├── reports/
│   ├── consolidated.py        # сводный отчёт + проверка гипотез H1/H2
│   ├── comparative.py         # ТА vs ФА vs СА таблицы
│   └── report.py              # per-ticker HTML-отчёты
├── TA/                        # расчёт индикаторов и их кэш
├── modules/                   # CAPM / Sharpe / Sortino / Treynor / Jensen и пр.
├── scripts/
│   ├── download_data.py       # MOEX ISS API → data/
│   └── build_riskfree.py      # сшивка MOEXREPO + RUSFAR
└── pinescript/
    └── vf_strategy.pine       # TradingView-версия VF Strategy
```
