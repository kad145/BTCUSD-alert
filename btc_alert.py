import os
import requests
import pandas as pd
import numpy as np

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
LIMIT = 500

UT_KEY = 2.0
ATR_PERIOD = 300

STC_LENGTH = 80
STC_FAST = 27
STC_SLOW = 50
STC_FACTOR = 0.5

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]


def get_data():
    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close",
        "volume", "close_time", "quote_volume", "trades",
        "buy_volume", "buy_quote_volume", "ignore"
    ])

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = df[column].astype(float)

    df["close_time"] = pd.to_datetime(
        df["close_time"], unit="ms"
    )

    # Csak lezárt gyertyák
    now = pd.Timestamp.utcnow().tz_localize(None)
    df = df[df["close_time"] <= now]

    return df.reset_index(drop=True)


def calculate_atr(df, period):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    previous_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()


def calculate_ut_bot(df):
    close = df["close"]
    atr = calculate_atr(df, ATR_PERIOD)
    loss = UT_KEY * atr

    stop = pd.Series(np.nan, index=df.index)

    for i in range(1, len(df)):

        previous_stop = stop.iloc[i - 1]
        current_close = close.iloc[i]
        previous_close = close.iloc[i - 1]

        if pd.isna(previous_stop):
            stop.iloc[i] = current_close - loss.iloc[i]
            continue

        if (
            current_close > previous_stop
            and previous_close > previous_stop
        ):
            stop.iloc[i] = max(
                previous_stop,
                current_close - loss.iloc[i]
            )

        elif (
            current_close < previous_stop
            and previous_close < previous_stop
        ):
            stop.iloc[i] = min(
                previous_stop,
                current_close + loss.iloc[i]
            )

        elif current_close > previous_stop:
            stop.iloc[i] = current_close - loss.iloc[i]

        else:
            stop.iloc[i] = current_close + loss.iloc[i]

    buy = (
        (close > stop) &
        (close.shift(1) <= stop.shift(1))
    )

    sell = (
        (close < stop) &
        (close.shift(1) >= stop.shift(1))
    )

    return buy, sell


def calculate_stc(df):
    close = df["close"]

    macd = (
        close.ewm(span=STC_FAST, adjust=False).mean()
        -
        close.ewm(span=STC_SLOW, adjust=False).mean()
    )

    lowest = macd.rolling(STC_LENGTH).min()
    highest = macd.rolling(STC_LENGTH).max()

    denominator = highest - lowest

    stoch = pd.Series(
        np.where(
            denominator != 0,
            100 * (macd - lowest) / denominator,
            0
        ),
        index=df.index
    )

    stc = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):

        if pd.isna(stoch.iloc[i]):
            stc.iloc[i] = stc.iloc[i - 1]
        else:
            stc.iloc[i] = (
                stc.iloc[i - 1]
                +
                STC_FACTOR *
                (stoch.iloc[i] - stc.iloc[i - 1])
            )

    return stc


def send_discord(message):
    response = requests.post(
        DISCORD_WEBHOOK_URL,
        json={"content": message},
        timeout=20
    )

    response.raise_for_status()


def main():

    df = get_data()

    if len(df) < ATR_PERIOD + STC_LENGTH:
        print("Nincs elegendő adat.")
        return

    ut_buy, ut_sell = calculate_ut_bot(df)
    stc = calculate_stc(df)

    i = len(df) - 1

    current_stc = stc.iloc[i]
    previous_stc = stc.iloc[i - 1]

    stc_up = current_stc > previous_stc
    stc_down = current_stc < previous_stc

    buy_signal = (
        current_stc < 50
        and stc_up
        and ut_buy.iloc[i]
    )

    sell_signal = (
        current_stc > 80
        and stc_down
        and ut_sell.iloc[i]
    )

    price = df["close"].iloc[i]
    candle_time = df["close_time"].iloc[i]

    print("--------------------------------")
    print("BTCUSD 15M")
    print(f"Gyertya: {candle_time}")
    print(f"Ár: {price:.2f}")
    print(f"STC: {current_stc:.2f}")
    print(f"Előző STC: {previous_stc:.2f}")
    print(f"UT BUY: {ut_buy.iloc[i]}")
    print(f"UT SELL: {ut_sell.iloc[i]}")
    print("--------------------------------")

    if buy_signal:

        send_discord(
            "🟢 BTCUSD BUY\n\n"
            f"15M\n"
            f"Price: {price:.2f}\n"
            f"STC: {current_stc:.2f}\n"
            "STC < 50 + felfelé váltás\n"
            "UT Bot BUY"
        )

        print("BUY JELZÉS ELKÜLDVE")

    elif sell_signal:

        send_discord(
            "⚪ BTCUSD SELL\n\n"
            f"15M\n"
            f"Price: {price:.2f}\n"
            f"STC: {current_stc:.2f}\n"
            "STC > 80 + lefelé váltás\n"
            "UT Bot SELL"
        )

        print("SELL JELZÉS ELKÜLDVE")

    else:
        print("No new BTC signal")


if __name__ == "__main__":
    main()
