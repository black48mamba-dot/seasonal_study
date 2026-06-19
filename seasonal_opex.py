from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import yfinance as yf

MONTH_NAMES = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


@dataclass(frozen=True)
class OpexCycleReturn:
    ticker: str
    cycle_year: int
    cycle_month: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    start_close: float
    end_close: float
    return_pct: float


def third_friday(year: int, month: int) -> pd.Timestamp:
    first_day = pd.Timestamp(year=year, month=month, day=1)
    first_friday_offset = (4 - first_day.weekday()) % 7
    return first_day + pd.Timedelta(days=first_friday_offset + 14)


def get_previous_trading_day(target_date: pd.Timestamp, trading_index: pd.DatetimeIndex) -> pd.Timestamp:
    eligible = trading_index[trading_index <= target_date]
    if len(eligible) == 0:
        raise ValueError(f"No trading day found on or before {target_date.date()}.")
    return eligible[-1]


def build_opex_schedule(
    trading_index: pd.DatetimeIndex,
    start_year: int,
    end_year: int,
    start_month: int = 1,
    end_month: int = 12,
    max_completed_date: pd.Timestamp | None = None,
) -> list[tuple[int, int, pd.Timestamp]]:
    schedule: list[tuple[int, int, pd.Timestamp]] = []
    for year in range(start_year, end_year + 1):
        first_month = start_month if year == start_year else 1
        last_month = end_month if year == end_year else 12
        for month in range(first_month, last_month + 1):
            nominal = third_friday(year, month)
            actual = get_previous_trading_day(nominal, trading_index)
            if max_completed_date is not None and actual > max_completed_date:
                continue
            schedule.append((year, month, actual))
    schedule.sort(key=lambda item: item[2])
    return schedule


def download_prices(ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    start = f"{start_year - 1}-12-01"
    end = f"{end_year + 1}-01-15"

    attempts: list[tuple[str, object]] = []

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Trying to detect encoding from a tiny portion")
        for pause_seconds in (0.0, 1.0):
            if pause_seconds:
                time.sleep(pause_seconds)

            try:
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                if not data.empty:
                    break
                attempts.append(("download", "empty dataframe"))
            except Exception as exc:
                attempts.append(("download", exc))
        else:
            data = pd.DataFrame()

        if data.empty:
            try:
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(start=start, end=end, auto_adjust=False)
                if data.empty:
                    attempts.append(("history", "empty dataframe"))
            except Exception as exc:
                attempts.append(("history", exc))

    if data.empty:
        details = "; ".join(f"{name}: {value}" for name, value in attempts) or "unknown fetch error"
        raise ValueError(f"No data returned for {ticker}. Fetch attempts: {details}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.index = pd.to_datetime(data.index).tz_localize(None)
    return data.sort_index()


def get_close_series(prices: pd.DataFrame) -> pd.Series:
    close = prices["Adj Close"] if "Adj Close" in prices.columns else prices["Close"]
    return close.dropna()


def compute_opex_cycle_returns(
    ticker: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    prices = download_prices(ticker, start_year, end_year)
    close = get_close_series(prices)
    first_date = close.index.min()
    last_date = close.index.max()

    schedule = build_opex_schedule(
        close.index,
        first_date.year,
        last_date.year,
        start_month=first_date.month,
        end_month=last_date.month,
        max_completed_date=last_date,
    )

    rows: list[OpexCycleReturn] = []
    for i in range(1, len(schedule)):
        prev_year, prev_month, prev_opex = schedule[i - 1]
        cycle_year, cycle_month, curr_opex = schedule[i]

        if cycle_year < start_year or cycle_year > end_year:
            continue

        if prev_opex < first_date or curr_opex > last_date:
            continue

        start_close = float(close.loc[prev_opex])
        end_close = float(close.loc[curr_opex])
        return_pct = (end_close / start_close - 1.0) * 100.0

        rows.append(
            OpexCycleReturn(
                ticker=ticker.upper(),
                cycle_year=cycle_year,
                cycle_month=cycle_month,
                start_date=prev_opex,
                end_date=curr_opex,
                start_close=start_close,
                end_close=end_close,
                return_pct=return_pct,
            )
        )

    return pd.DataFrame([row.__dict__ for row in rows])


def summarize_by_cycle_month(cycle_returns: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cycle_returns.groupby(["ticker", "cycle_month"], as_index=False)["return_pct"]
        .agg(
            observations="count",
            avg_return_pct="mean",
            median_return_pct="median",
            win_rate=lambda s: (s > 0).mean() * 100.0,
            best_return_pct="max",
            worst_return_pct="min",
        )
        .sort_values(["ticker", "cycle_month"])
    )
    summary["cycle_label"] = summary["cycle_month"].map(MONTH_NAMES)
    return summary[
        [
            "ticker",
            "cycle_month",
            "cycle_label",
            "observations",
            "avg_return_pct",
            "median_return_pct",
            "win_rate",
            "best_return_pct",
            "worst_return_pct",
        ]
    ]


def parse_tickers(raw: Iterable[str]) -> list[str]:
    tickers: list[str] = []
    for item in raw:
        tickers.extend(part.strip().upper() for part in item.split(",") if part.strip())
    if not tickers:
        raise ValueError("At least one ticker is required.")
    return tickers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate seasonal returns between monthly US equity option expiration cycles."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        required=True,
        help='Ticker list, for example: --tickers SPY QQQ or --tickers "SPY,QQQ"',
    )
    parser.add_argument("--start-year", type=int, required=True, help="First cycle-ending year to include.")
    parser.add_argument("--end-year", type=int, required=True, help="Last cycle-ending year to include.")
    parser.add_argument(
        "--details-csv",
        default="opex_cycle_returns.csv",
        help="Path for detailed cycle-by-cycle results.",
    )
    parser.add_argument(
        "--summary-csv",
        default="opex_cycle_summary.csv",
        help="Path for monthly summary results.",
    )
    args = parser.parse_args()

    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year.")

    tickers = parse_tickers(args.tickers)

    all_details: list[pd.DataFrame] = []
    for ticker in tickers:
        details = compute_opex_cycle_returns(
            ticker=ticker,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        all_details.append(details)

    details_df = pd.concat(all_details, ignore_index=True)
    summary_df = summarize_by_cycle_month(details_df)

    details_df.to_csv(args.details_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", None)

    print("\nDetailed cycle returns:")
    print(details_df.to_string(index=False))

    print("\nSummary by cycle-ending month:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print(f"\nWrote detailed results to: {args.details_csv}")
    print(f"Wrote summary results to: {args.summary_csv}")


if __name__ == "__main__":
    main()
