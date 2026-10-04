import os
import requests
import pandas as pd
import numpy as np

SYMBOL = "BTC/USD"
INTERVAL = "15min"
LIMIT = 500

UT_KEY = 2.0
ATR_PERIOD = 300

STC_LENGTH = 80
STC_FAST = 27
STC_SLOW = 50
STC_FACTOR = 0.5

DISCORD_WEBHOOK_URL = os.environ["DISCORD_WEBHOOK_URL"]
TWELVE_DATA_API_KEY = os.environ["TWELVE_DATA_API_KEY"]


def get_data():
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": LIMIT,
        "apikey": TWELVE_DATA_API_KEY,
        "timezone": "UTC"
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(data.get("message", "Twelve Data API error"))

    if "values" not in data:
        raise RuntimeError(f"Nincs adat a Twelve Data válaszban: {data}")

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    for column in ["open", "high", "low", "close", "volume"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("datetime").reset_index(drop=True)

    # Csak lezárt gyertyák
    now = pd.Timestamp.now(tz="UTC")
    df = df[df["datetime"] < now]

    df = df.rename(columns={"datetime": "close_time"})

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
        print(
            f"Nincs elegendő adat. "
            f"Kapott gyertyák: {len(df)}, "
            f"Szükséges: {ATR_PERIOD + STC_LENGTH}"
        )
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
            "15M\n"
            f"Price: {price:.2f}\n"
            f"STC: {current_stc:.2f}\n"
            "STC < 50 + felfelé váltás\n"
            "UT Bot BUY"
        )

        print("BUY JELZÉS ELKÜLDVE")

    elif sell_signal:

        send_discord(
            "⚪ BTCUSD SELL\n\n"
            "15M\n"
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
