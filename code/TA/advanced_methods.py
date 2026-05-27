from TA.strategy_old import crossover_strategy, ema_cross_strategy, ma_cross_strategy, rsi_strategy, macd_strategy

def add_signals_to_data(df):
    df['Signal_Price_SMA_50'] = crossover_strategy(df, price_column='close', indicator_column='SMA_50')
    df['Signal_Price_SMA_100'] = crossover_strategy(df, price_column='close', indicator_column='SMA_100')
    df['Signal_Price_EMA_21'] = crossover_strategy(df, price_column='close', indicator_column='EMA_21')
    df['Signal_Price_EMA_50'] = crossover_strategy(df, price_column='close', indicator_column='EMA_50')
    df['Signal_EMA21_50'] = ema_cross_strategy(df, short_ema='EMA_21', long_ema='EMA_50')
    df['Signal_EMA50_200'] = ema_cross_strategy(df, short_ema='EMA_50', long_ema='EMA_200')
    df['Signal_MA50_100'] = ma_cross_strategy(df, short_ma='SMA_50', long_ma='SMA_100')
    df['Signal_RSI3070'] = rsi_strategy(df, rsi_column='RSI_14', buy_level=30, sell_level=70)
    df['Signal_RSI2080'] = rsi_strategy(df, rsi_column='RSI_14', buy_level=20, sell_level=80)
    df['Signal_MACD1226'] = macd_strategy(df, macd_column='MACD', signal_column='MACD_Signal')
    return df
