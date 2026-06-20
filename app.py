from __future__ import annotations

import math
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse

from seasonal_opex import (
    MONTH_NAMES,
    build_opex_schedule,
    download_prices,
    get_close_series,
)

DEFAULT_HISTORY_YEARS = 30
DEFAULT_LOOKBACK_YEARS = 10

app = FastAPI(title="OPEX Seasonality Dashboard")


def compute_cycle_returns_with_prices(
    ticker: str,
    history_years: int,
) -> tuple[pd.DataFrame, pd.Series]:
    current_year = datetime.now().year
    start_year = current_year - history_years
    prices = download_prices(ticker=ticker, start_year=start_year, end_year=current_year)
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

    rows: list[dict] = []
    for i in range(1, len(schedule)):
        _, _, start_date = schedule[i - 1]
        cycle_year, cycle_month, end_date = schedule[i]

        if start_date < first_date or end_date > last_date:
            continue

        start_close = float(close.loc[start_date])
        end_close = float(close.loc[end_date])
        rows.append(
            {
                "ticker": ticker.upper(),
                "cycle_year": cycle_year,
                "cycle_month": cycle_month,
                "cycle_label": MONTH_NAMES[cycle_month],
                "start_date": start_date,
                "end_date": end_date,
                "start_close": start_close,
                "end_close": end_close,
                "return_pct": (end_close / start_close - 1.0) * 100.0,
            }
        )

    return pd.DataFrame(rows), close


def collect_window_daily_returns(close: pd.Series, cycle_rows: pd.DataFrame) -> pd.Series:
    all_daily_returns: list[pd.Series] = []
    for row in cycle_rows.itertuples(index=False):
        window = close.loc[row.start_date : row.end_date]
        daily_returns = window.pct_change().dropna()
        if not daily_returns.empty:
            all_daily_returns.append(daily_returns)

    if not all_daily_returns:
        return pd.Series(dtype=float)

    return pd.concat(all_daily_returns, ignore_index=True)


def build_metrics(selected_cycles: pd.DataFrame, daily_returns: pd.Series) -> dict[str, float | int | str]:
    observations = int(len(selected_cycles))
    mean_return = float(selected_cycles["return_pct"].mean()) if observations else math.nan
    std_return = float(selected_cycles["return_pct"].std(ddof=1)) if observations > 1 else math.nan

    if daily_returns.empty:
        daily_vol_pct = math.nan
        annualized_sharpe = math.nan
    else:
        daily_std = float(daily_returns.std(ddof=1))
        daily_mean = float(daily_returns.mean())
        daily_vol_pct = daily_std * 100.0
        annualized_sharpe = math.nan if daily_std == 0 else (daily_mean / daily_std) * math.sqrt(252.0)

    return {
        "observations": observations,
        "mean_return_pct": mean_return,
        "std_return_pct": std_return,
        "daily_vol_pct": daily_vol_pct,
        "annualized_sharpe": annualized_sharpe,
    }


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, channel)):02x}" for channel in rgb)


def interpolate_color(start_hex: str, end_hex: str, ratio: float) -> str:
    start = hex_to_rgb(start_hex)
    end = hex_to_rgb(end_hex)
    ratio = max(0.0, min(1.0, ratio))
    rgb = tuple(round(start[i] + (end[i] - start[i]) * ratio) for i in range(3))
    return rgb_to_hex(rgb)


def gradient_colors_for_returns(values: pd.Series) -> list[str]:
    if values.empty:
        return []

    max_positive = float(values[values >= 0].max()) if (values >= 0).any() else 0.0
    min_negative = float(values[values < 0].min()) if (values < 0).any() else 0.0

    colors: list[str] = []
    for value in values:
        if value >= 0:
            ratio = 0.0 if max_positive <= 0 else float(value) / max_positive
            colors.append(interpolate_color("#dcfce7", "#15803d", ratio))
        else:
            ratio = 0.0 if min_negative >= 0 else abs(float(value) / min_negative)
            colors.append(interpolate_color("#fecaca", "#b91c1c", ratio))
    return colors


def build_selected_month_chart(selected_cycles: pd.DataFrame, month_name: str, ticker: str) -> str:
    if selected_cycles.empty:
        return "<p>No completed cycles available for the selected month in the current lookback window.</p>"

    bar_colors = gradient_colors_for_returns(selected_cycles["return_pct"])
    fig = px.bar(
        selected_cycles,
        x="cycle_year",
        y="return_pct",
        title=f"{ticker.upper()} - {month_name} OPEX-to-OPEX Returns by Year",
        labels={"cycle_year": "Cycle year", "return_pct": "Return %"},
    )
    fig.update_traces(marker_color=bar_colors)
    mean_value = selected_cycles["return_pct"].mean()
    fig.add_hline(
        y=mean_value,
        line_dash="dash",
        line_color="#1f77b4",
        annotation_text=f"Mean: {mean_value:.2f}%",
    )
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=60, b=40))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def summarize_monthly_comparison(cycle_returns: pd.DataFrame) -> pd.DataFrame:
    summary = (
        cycle_returns.groupby("cycle_month", as_index=False)["return_pct"]
        .agg(
            observations="count",
            mean_return_pct="mean",
            std_return_pct=lambda s: s.std(ddof=1),
        )
        .sort_values("cycle_month")
    )
    summary["cycle_label"] = summary["cycle_month"].map(MONTH_NAMES)
    return summary[
        ["cycle_month", "cycle_label", "observations", "mean_return_pct", "std_return_pct"]
    ]


def get_previous_close_date(target_date: pd.Timestamp, trading_index: pd.DatetimeIndex) -> pd.Timestamp | None:
    eligible = trading_index[trading_index < target_date]
    if len(eligible) == 0:
        return None
    return eligible[-1]


def compute_week_bucket_returns(close: pd.Series, cycle_returns: pd.DataFrame) -> pd.DataFrame:
    trading_index = close.index
    bucket_order = {
        "week_1": 1,
        "week_2": 2,
        "opex_week": 3,
        "post_opex_week": 4,
    }
    bucket_labels = {
        "week_1": "Week 1",
        "week_2": "Week 2",
        "opex_week": "OPEX Week",
        "post_opex_week": "Post-OPEX Week",
    }

    rows: list[dict] = []
    month_rows = cycle_returns[["ticker", "cycle_year", "cycle_month", "cycle_label", "end_date"]].dropna().drop_duplicates()

    for row in month_rows.itertuples(index=False):
        month_start = pd.Timestamp(year=int(row.cycle_year), month=int(row.cycle_month), day=1)
        month_end = month_start + pd.offsets.MonthEnd(1)
        week_cursor = month_start - pd.Timedelta(days=month_start.weekday())

        week_segments: list[dict] = []
        while week_cursor <= month_end:
            segment_start = max(week_cursor, month_start)
            segment_end = min(week_cursor + pd.Timedelta(days=6), month_end)
            week_days = trading_index[(trading_index >= segment_start) & (trading_index <= segment_end)]
            if len(week_days) > 0:
                week_segments.append(
                    {
                        "start_date": week_days[0],
                        "end_date": week_days[-1],
                    }
                )
            week_cursor += pd.Timedelta(days=7)

        if not week_segments:
            continue

        bucket_to_segment: dict[str, dict] = {}
        if len(week_segments) >= 1:
            bucket_to_segment["week_1"] = week_segments[0]
        if len(week_segments) >= 2:
            bucket_to_segment["week_2"] = week_segments[1]

        opex_index = next(
            (idx for idx, segment in enumerate(week_segments) if segment["start_date"] <= row.end_date <= segment["end_date"]),
            None,
        )
        if opex_index is not None:
            bucket_to_segment["opex_week"] = week_segments[opex_index]
            if opex_index + 1 < len(week_segments):
                bucket_to_segment["post_opex_week"] = week_segments[opex_index + 1]

        for bucket_key, segment in bucket_to_segment.items():
            previous_close_date = get_previous_close_date(segment["start_date"], trading_index)
            if previous_close_date is None:
                continue

            start_close = float(close.loc[previous_close_date])
            end_close = float(close.loc[segment["end_date"]])
            rows.append(
                {
                    "ticker": row.ticker,
                    "cycle_year": int(row.cycle_year),
                    "cycle_month": int(row.cycle_month),
                    "cycle_label": row.cycle_label,
                    "bucket_key": bucket_key,
                    "bucket_label": bucket_labels[bucket_key],
                    "bucket_order": bucket_order[bucket_key],
                    "start_date": segment["start_date"],
                    "end_date": segment["end_date"],
                    "return_pct": (end_close / start_close - 1.0) * 100.0,
                }
            )

    return pd.DataFrame(rows)


def summarize_weekly_buckets(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    if weekly_returns.empty:
        return pd.DataFrame(
            columns=["bucket_key", "bucket_label", "bucket_order", "observations", "mean_return_pct", "std_return_pct", "win_rate"]
        )

    summary = (
        weekly_returns.groupby(["bucket_key", "bucket_label", "bucket_order"], as_index=False)["return_pct"]
        .agg(
            observations="count",
            mean_return_pct="mean",
            std_return_pct=lambda s: s.std(ddof=1),
            win_rate=lambda s: (s > 0).mean() * 100.0,
        )
        .sort_values("bucket_order")
    )
    return summary


def summarize_weekly_buckets_by_month(weekly_returns: pd.DataFrame) -> pd.DataFrame:
    if weekly_returns.empty:
        return pd.DataFrame(
            columns=[
                "cycle_month",
                "cycle_label",
                "bucket_key",
                "bucket_label",
                "bucket_order",
                "observations",
                "mean_return_pct",
                "std_return_pct",
            ]
        )

    summary = (
        weekly_returns.groupby(
            ["cycle_month", "cycle_label", "bucket_key", "bucket_label", "bucket_order"],
            as_index=False,
        )["return_pct"]
        .agg(
            observations="count",
            mean_return_pct="mean",
            std_return_pct=lambda s: s.std(ddof=1),
        )
        .sort_values(["cycle_month", "bucket_order"])
    )
    return summary


def build_ytd_seasonality_chart(close: pd.Series, lookback_years: int, ticker: str) -> str:
    daily_returns = close.pct_change().dropna()
    if daily_returns.empty:
        return "<p>No daily data available for the YTD seasonality overlay.</p>"

    daily_df = daily_returns.to_frame(name="daily_return")
    daily_df["year"] = daily_df.index.year

    available_years = sorted(daily_df["year"].unique())
    current_year = max(available_years)
    prior_years = [year for year in available_years if year < current_year]
    prior_years = prior_years[-lookback_years:]

    if not prior_years:
        return "<p>Not enough prior-year history to build the YTD seasonality overlay.</p>"

    compare_years = prior_years + [current_year]
    filtered = daily_df[daily_df["year"].isin(compare_years)].copy()
    filtered["trading_day_of_year"] = filtered.groupby("year").cumcount() + 1
    filtered["cum_return_pct"] = (
        filtered.groupby("year")["daily_return"].transform(lambda s: ((1.0 + s).cumprod() - 1.0) * 100.0)
    )

    prior_mean = (
        filtered[filtered["year"].isin(prior_years)]
        .groupby("trading_day_of_year", as_index=False)["cum_return_pct"]
        .mean()
    )
    current_path = filtered[filtered["year"] == current_year][["trading_day_of_year", "cum_return_pct"]].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prior_mean["trading_day_of_year"],
            y=prior_mean["cum_return_pct"],
            mode="lines",
            name=f"Mean prior years ({prior_years[0]}-{prior_years[-1]})",
            line=dict(color="#1f77b4", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=current_path["trading_day_of_year"],
            y=current_path["cum_return_pct"],
            mode="lines",
            name=f"{current_year} current year",
            line=dict(color="#111827", width=3, dash="dash"),
        )
    )
    fig.add_hline(y=0, line_color="#9ca3af", line_width=1)
    fig.update_layout(
        title=f"{ticker.upper()} - YTD Cumulative Return: Prior-Year Mean vs Current Year",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(title="Trading day of year"),
        yaxis=dict(title="Cumulative return %"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_weekly_global_chart(weekly_summary: pd.DataFrame, ticker: str) -> str:
    if weekly_summary.empty:
        return "<p>No weekly bucket data available.</p>"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weekly_summary["bucket_label"],
            y=weekly_summary["mean_return_pct"],
            name="Mean return %",
            marker_color=gradient_colors_for_returns(weekly_summary["mean_return_pct"]),
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weekly_summary["bucket_label"],
            y=weekly_summary["std_return_pct"],
            name="Std dev %",
            mode="lines+markers",
            marker=dict(color="#1f77b4"),
            line=dict(color="#1f77b4", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"{ticker.upper()} - Weekly Bucket Mean Return and Standard Deviation",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(title="Mean return %"),
        yaxis2=dict(title="Std dev %", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_weekly_monthly_heatmap(weekly_monthly_summary: pd.DataFrame, ticker: str, show_std: bool) -> str:
    if weekly_monthly_summary.empty:
        return "<p>No monthly weekly-bucket data available.</p>"

    mean_pivot = (
        weekly_monthly_summary.pivot(index="cycle_label", columns="bucket_label", values="mean_return_pct")
        .reindex([MONTH_NAMES[m] for m in range(1, 13)])
    )
    std_pivot = (
        weekly_monthly_summary.pivot(index="cycle_label", columns="bucket_label", values="std_return_pct")
        .reindex([MONTH_NAMES[m] for m in range(1, 13)])
    )
    bucket_columns = ["Week 1", "Week 2", "OPEX Week", "Post-OPEX Week"]
    bucket_columns = [column for column in bucket_columns if column in mean_pivot.columns]
    mean_pivot = mean_pivot[bucket_columns]
    std_pivot = std_pivot[bucket_columns]
    colorscale, zmin, zmax = build_return_colorscale(mean_pivot.values)

    text_values = []
    hover_text = []
    for month_label in mean_pivot.index:
        text_row = []
        hover_row = []
        for bucket_label in mean_pivot.columns:
            mean_value = mean_pivot.loc[month_label, bucket_label]
            std_value = std_pivot.loc[month_label, bucket_label]
            if pd.notna(mean_value):
                std_text = "N/A" if pd.isna(std_value) else f"{std_value:.2f}%"
                text_row.append(f"{mean_value:.2f}%<br>({std_text})" if show_std else f"{mean_value:.2f}%")
                hover_row.append(
                    f"Month {month_label}<br>Bucket {bucket_label}<br>Mean return {mean_value:.2f}%<br>Std dev {std_text}"
                )
            else:
                text_row.append("")
                hover_row.append(f"Month {month_label}<br>Bucket {bucket_label}<br>No data")
        text_values.append(text_row)
        hover_text.append(hover_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=mean_pivot.values,
            x=list(mean_pivot.columns),
            y=list(mean_pivot.index),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Mean return %"),
            text=text_values,
            texttemplate="%{text}",
            customdata=hover_text,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{ticker.upper()} - Weekly Bucket Mean Return by Month",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_monthly_comparison_chart(monthly_summary: pd.DataFrame, ticker: str) -> str:
    if monthly_summary.empty:
        return "<p>No monthly comparison data available.</p>"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=monthly_summary["cycle_label"],
            y=monthly_summary["mean_return_pct"],
            name="Mean return %",
            marker_color=gradient_colors_for_returns(monthly_summary["mean_return_pct"]),
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_summary["cycle_label"],
            y=monthly_summary["std_return_pct"],
            name="Std dev %",
            mode="lines+markers",
            marker=dict(color="#1f77b4"),
            line=dict(color="#1f77b4", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"{ticker.upper()} - Monthly Mean Return and Standard Deviation",
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(title="Mean return %"),
        yaxis2=dict(title="Std dev %", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def build_return_colorscale(values: pd.DataFrame) -> tuple[list[list[float | str]], float, float]:
    numeric = pd.Series(values.ravel()).dropna()
    if numeric.empty:
        return [[0.0, "#dc2626"], [0.5, "#fecaca"], [0.5, "#dcfce7"], [1.0, "#15803d"]], -1.0, 1.0

    zmin = float(numeric.min())
    zmax = float(numeric.max())

    if zmin >= 0:
        if zmax == 0:
            zmax = 1.0
        return [[0.0, "#dcfce7"], [1.0, "#15803d"]], 0.0, zmax

    if zmax <= 0:
        if zmin == 0:
            zmin = -1.0
        return [[0.0, "#b91c1c"], [1.0, "#fecaca"]], zmin, 0.0

    zero_point = abs(zmin) / (zmax - zmin)
    colorscale = [
        [0.0, "#b91c1c"],
        [max(zero_point * 0.7, 0.0), "#ef4444"],
        [max(zero_point - 1e-6, 0.0), "#fecaca"],
        [min(zero_point + 1e-6, 1.0), "#dcfce7"],
        [1.0, "#15803d"],
    ]
    return colorscale, zmin, zmax


def build_heatmap(cycle_returns: pd.DataFrame, ticker: str) -> str:
    pivot = (
        cycle_returns.pivot(index="cycle_year", columns="cycle_month", values="return_pct")
        .sort_index()
        .rename(columns=MONTH_NAMES)
    )
    pivot = pivot[[MONTH_NAMES[m] for m in range(1, 13) if MONTH_NAMES[m] in pivot.columns]]
    colorscale, zmin, zmax = build_return_colorscale(pivot.values)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=[str(v) for v in pivot.index],
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="Return %"),
            text=[[f"{value:.2f}%" if pd.notna(value) else "" for value in row] for row in pivot.values],
            texttemplate="%{text}",
            hovertemplate="Year %{y}<br>Month %{x}<br>Return %{z:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{ticker.upper()} - Monthly OPEX Cycle Return Heatmap",
        height=540,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def select_heatmap_window(cycle_returns: pd.DataFrame, lookback_years: int) -> pd.DataFrame:
    if cycle_returns.empty:
        return cycle_returns

    max_year = int(cycle_returns["cycle_year"].max())
    min_year = max_year - lookback_years + 1
    years = list(range(min_year, max_year + 1))

    filtered = cycle_returns[cycle_returns["cycle_year"].between(min_year, max_year)].copy()

    placeholder_rows = []
    existing_years = set(filtered["cycle_year"].unique())
    ticker = str(cycle_returns["ticker"].iloc[0])
    for year in years:
        if year not in existing_years:
            placeholder_rows.append(
                {
                    "ticker": ticker,
                    "cycle_year": year,
                    "cycle_month": 1,
                    "cycle_label": MONTH_NAMES[1],
                    "start_date": pd.NaT,
                    "end_date": pd.NaT,
                    "start_close": math.nan,
                    "end_close": math.nan,
                    "return_pct": math.nan,
                }
            )

    if placeholder_rows:
        filtered = pd.concat([filtered, pd.DataFrame(placeholder_rows)], ignore_index=True)

    return filtered


def format_value(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}{suffix}"


def render_dashboard(
    ticker: str,
    selected_month: int,
    lookback_years: int,
    show_weekly_std: bool,
) -> str:
    history_years = max(lookback_years + 2, 12)
    cycle_returns, close = compute_cycle_returns_with_prices(ticker, history_years=history_years)
    month_name = MONTH_NAMES[selected_month]

    filtered = cycle_returns[cycle_returns["cycle_month"] == selected_month].sort_values("cycle_year")
    selected_cycles = filtered.tail(lookback_years).copy()
    actual_lookback_years = int(len(selected_cycles))
    daily_returns = collect_window_daily_returns(close, selected_cycles)
    metrics = build_metrics(selected_cycles, daily_returns)

    heatmap_source = select_heatmap_window(cycle_returns, lookback_years=lookback_years)
    monthly_comparison = summarize_monthly_comparison(heatmap_source)
    weekly_returns = compute_week_bucket_returns(close, heatmap_source)
    weekly_global_summary = summarize_weekly_buckets(weekly_returns)
    weekly_monthly_summary = summarize_weekly_buckets_by_month(weekly_returns)
    ytd_seasonality_chart = build_ytd_seasonality_chart(close, lookback_years=lookback_years, ticker=ticker)

    selected_cycles_table = selected_cycles[
        ["cycle_year", "start_date", "end_date", "start_close", "end_close", "return_pct"]
    ].copy()
    for col in ["start_date", "end_date"]:
        selected_cycles_table[col] = selected_cycles_table[col].dt.strftime("%Y-%m-%d")
    for col in ["start_close", "end_close", "return_pct"]:
        selected_cycles_table[col] = selected_cycles_table[col].map(lambda x: f"{x:.2f}")

    selected_month_chart = build_selected_month_chart(selected_cycles, month_name, ticker=ticker)
    heatmap_chart = build_heatmap(heatmap_source, ticker=ticker)
    monthly_comparison_chart = build_monthly_comparison_chart(monthly_comparison, ticker=ticker)
    weekly_global_chart = build_weekly_global_chart(weekly_global_summary, ticker=ticker)
    weekly_monthly_heatmap = build_weekly_monthly_heatmap(
        weekly_monthly_summary,
        ticker=ticker,
        show_std=show_weekly_std,
    )
    monthly_comparison_table = monthly_comparison.copy()
    for col in ["mean_return_pct", "std_return_pct"]:
        monthly_comparison_table[col] = monthly_comparison_table[col].map(
            lambda x: "" if pd.isna(x) else f"{x:.2f}"
        )
    weekly_global_table = weekly_global_summary.copy()
    for col in ["mean_return_pct", "std_return_pct", "win_rate"]:
        weekly_global_table[col] = weekly_global_table[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    weekly_monthly_table = weekly_monthly_summary.copy()
    for col in ["mean_return_pct", "std_return_pct"]:
        weekly_monthly_table[col] = weekly_monthly_table[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")

    options_html = "".join(
        f'<option value="{month}" {"selected" if month == selected_month else ""}>{label}</option>'
        for month, label in MONTH_NAMES.items()
    )
    show_std_checked = "checked" if show_weekly_std else ""

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OPEX Seasonality Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f8; color: #111827; }}
    h1, h2 {{ margin-bottom: 8px; }}
    form {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: end; margin-bottom: 20px; }}
    label {{ display: flex; flex-direction: column; font-size: 14px; gap: 6px; }}
    input, select, button {{ padding: 8px 10px; font-size: 14px; }}
    button {{ cursor: pointer; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }}
    .card {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
    .card .label {{ font-size: 13px; color: #4b5563; margin-bottom: 6px; }}
    .card .value {{ font-size: 24px; font-weight: 700; }}
    .panel {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); margin-bottom: 20px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>OPEX Seasonality Dashboard</h1>
  <p>Cycle definition: return from the prior monthly OPEX close to the selected month's OPEX close.</p>
  <p>Showing the latest available {actual_lookback_years} completed {month_name} cycles for {ticker.upper()}.</p>

  <form method="get" action="/">
    <label>
      Ticker
      <input type="text" name="ticker" value="{ticker.upper()}" />
    </label>
    <label>
      Ending month
      <select name="month">{options_html}</select>
    </label>
    <label>
      Lookback years
      <input type="number" name="lookback" min="3" max="30" value="{lookback_years}" />
    </label>
    <label style="flex-direction: row; align-items: center; gap: 8px; padding-bottom: 8px;">
      <input type="checkbox" name="show_weekly_std" value="1" {show_std_checked} />
      Show weekly std-dev labels
    </label>
    <button type="submit">Update</button>
  </form>

  <div class="cards">
    <div class="card"><div class="label">{month_name} mean return</div><div class="value">{format_value(metrics["mean_return_pct"], "%")}</div></div>
    <div class="card"><div class="label">{month_name} return std dev</div><div class="value">{format_value(metrics["std_return_pct"], "%")}</div></div>
    <div class="card"><div class="label">{month_name} daily volatility</div><div class="value">{format_value(metrics["daily_vol_pct"], "%")}</div></div>
    <div class="card"><div class="label">{month_name} annualized Sharpe</div><div class="value">{format_value(metrics["annualized_sharpe"])}</div></div>
    <div class="card"><div class="label">Observations</div><div class="value">{metrics["observations"]}</div></div>
  </div>

  <div class="panel">{selected_month_chart}</div>
  <div class="panel">{ytd_seasonality_chart}</div>
  <div class="panel">{monthly_comparison_chart}</div>
  <div class="panel">{weekly_global_chart}</div>
  <div class="panel">{weekly_monthly_heatmap}</div>
  <div class="panel">{heatmap_chart}</div>

  <div class="panel">
    <h2>Monthly comparison</h2>
    {monthly_comparison_table.to_html(index=False, escape=False)}
  </div>

  <div class="panel">
    <h2>Weekly bucket summary</h2>
    <p>Definition used: Week 1 and Week 2 are the first two Monday-Friday market weeks touching the month. OPEX Week is the market week containing that month's actual OPEX trading day. Post-OPEX Week is the next market week after that.</p>
    {weekly_global_table.to_html(index=False, escape=False)}
  </div>

  <div class="panel">
    <h2>Weekly bucket monthly breakdown</h2>
    {weekly_monthly_table.to_html(index=False, escape=False)}
  </div>

  <div class="panel">
    <h2>{month_name} cycle details</h2>
    {selected_cycles_table.to_html(index=False, escape=False)}
  </div>
</body>
</html>
"""


def render_error_page(
    ticker: str,
    selected_month: int,
    lookback_years: int,
    error_message: str,
    show_weekly_std: bool,
) -> str:
    options_html = "".join(
        f'<option value="{month}" {"selected" if month == selected_month else ""}>{label}</option>'
        for month, label in MONTH_NAMES.items()
    )
    show_std_checked = "checked" if show_weekly_std else ""

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OPEX Seasonality Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f8; color: #111827; }}
    form {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: end; margin-bottom: 20px; }}
    label {{ display: flex; flex-direction: column; font-size: 14px; gap: 6px; }}
    input, select, button {{ padding: 8px 10px; font-size: 14px; }}
    button {{ cursor: pointer; }}
    .panel {{ background: white; padding: 16px; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); margin-bottom: 20px; }}
    code {{ white-space: pre-wrap; word-break: break-word; }}
  </style>
</head>
<body>
  <h1>OPEX Seasonality Dashboard</h1>
  <form method="get" action="/">
    <label>
      Ticker
      <input type="text" name="ticker" value="{ticker.upper()}" />
    </label>
    <label>
      Ending month
      <select name="month">{options_html}</select>
    </label>
    <label>
      Lookback years
      <input type="number" name="lookback" min="3" max="30" value="{lookback_years}" />
    </label>
    <label style="flex-direction: row; align-items: center; gap: 8px; padding-bottom: 8px;">
      <input type="checkbox" name="show_weekly_std" value="1" {show_std_checked} />
      Show weekly std-dev labels
    </label>
    <button type="submit">Update</button>
  </form>
  <div class="panel">
    <h2>Data load failed</h2>
    <p>The app could not retrieve price history for <strong>{ticker.upper()}</strong>.</p>
    <code>{error_message}</code>
    <p>Common causes: temporary Yahoo response failure, network filtering, or a local package issue in yfinance/requests.</p>
  </div>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home(
    ticker: str = Query("SPY"),
    month: int = Query(datetime.now().month, ge=1, le=12),
    lookback: int = Query(DEFAULT_LOOKBACK_YEARS, ge=3, le=30),
    show_weekly_std: bool = Query(False),
) -> HTMLResponse:
    try:
        html = render_dashboard(
            ticker=ticker,
            selected_month=month,
            lookback_years=lookback,
            show_weekly_std=show_weekly_std,
        )
    except Exception as exc:
        html = render_error_page(
            ticker=ticker,
            selected_month=month,
            lookback_years=lookback,
            error_message=str(exc),
            show_weekly_std=show_weekly_std,
        )
    return HTMLResponse(html)
