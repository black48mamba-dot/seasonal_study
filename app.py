from __future__ import annotations

import math
from datetime import datetime
from urllib.parse import urlencode

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

TICKER_PRESETS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]

BASE_CSS = """
  :root {
    --bg: #f1f5f9;
    --surface: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #e2e8f0;
    --accent: #2563eb;
    --accent-weak: #eff6ff;
    --pos: #15803d;
    --pos-weak: #dcfce7;
    --neg: #b91c1c;
    --neg-weak: #fee2e2;
    --shadow: 0 1px 2px rgba(15,23,42,.06), 0 4px 12px rgba(15,23,42,.05);
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
    line-height: 1.45;
  }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 0 24px 64px; }

  /* sticky header */
  header.app-header {
    position: sticky; top: 0; z-index: 50;
    background: rgba(255,255,255,.82);
    backdrop-filter: saturate(180%) blur(10px);
    -webkit-backdrop-filter: saturate(180%) blur(10px);
    border-bottom: 1px solid var(--line);
  }
  .header-inner {
    max-width: 1200px; margin: 0 auto; padding: 12px 24px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 18px; letter-spacing: -.01em; }
  .brand .dot { width: 11px; height: 11px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 4px var(--accent-weak); }
  nav.section-nav { display: flex; gap: 4px; margin-left: auto; flex-wrap: wrap; }
  nav.section-nav a {
    color: var(--muted); text-decoration: none; font-size: 14px; font-weight: 500;
    padding: 8px 12px; border-radius: 8px; transition: all .15s ease;
  }
  nav.section-nav a:hover { color: var(--accent); background: var(--accent-weak); }

  /* control bar */
  .controls {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: var(--radius); box-shadow: var(--shadow);
    padding: 18px 20px; margin: 24px 0;
  }
  form.controls-form { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-end; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field > label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  input[type=text], input[type=number], select {
    padding: 9px 12px; font-size: 14px; font-family: inherit; color: var(--ink);
    background: #fff; border: 1px solid var(--line); border-radius: 9px;
    transition: border-color .15s ease, box-shadow .15s ease; min-width: 96px;
  }
  input:focus, select:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-weak); }
  .field-check { flex-direction: row; align-items: center; gap: 8px; padding-bottom: 10px; cursor: pointer; font-size: 14px; }
  .field-check input { width: 16px; height: 16px; cursor: pointer; }
  button.primary {
    background: var(--accent); color: #fff; border: none; cursor: pointer;
    padding: 10px 22px; font-size: 14px; font-weight: 600; border-radius: 9px;
    transition: background .15s ease, transform .05s ease; font-family: inherit;
  }
  button.primary:hover { background: #1d4ed8; }
  button.primary:active { transform: translateY(1px); }

  /* presets */
  .presets { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 16px; padding-top: 16px; border-top: 1px dashed var(--line); }
  .presets .label { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; margin-right: 4px; }
  .chip {
    display: inline-block; text-decoration: none; font-size: 13px; font-weight: 600;
    padding: 6px 14px; border-radius: 999px; border: 1px solid var(--line);
    color: var(--ink); background: #fff; transition: all .15s ease; cursor: pointer; font-family: inherit;
  }
  .chip:hover { border-color: var(--accent); color: var(--accent); background: var(--accent-weak); }
  .chip.active { background: var(--accent); border-color: var(--accent); color: #fff; }

  /* subhead + context */
  .subhead { font-size: 13px; color: var(--muted); margin: 4px 0 0; }

  section { scroll-margin-top: 76px; margin-top: 36px; }
  .section-title { display: flex; align-items: baseline; gap: 12px; margin: 0 0 4px; }
  .section-title h2 { font-size: 19px; margin: 0; letter-spacing: -.01em; }
  .section-title .tag { font-size: 12px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  .section-desc { color: var(--muted); font-size: 13px; margin: 0 0 16px; max-width: 760px; }

  /* KPI cards */
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-top: 16px; }
  .card {
    background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
    padding: 18px 20px; box-shadow: var(--shadow); position: relative; overflow: hidden;
  }
  .card::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: var(--accent); opacity: .9; }
  .card.pos::before { background: var(--pos); }
  .card.neg::before { background: var(--neg); }
  .card.neutral::before { background: var(--muted); }
  .card .label { font-size: 12px; color: var(--muted); margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }
  .card .value { font-size: 26px; font-weight: 700; letter-spacing: -.02em; }

  /* panels */
  .panel { background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: var(--shadow); padding: 20px; margin-top: 16px; }
  .panel h3 { margin: 0 0 4px; font-size: 15px; letter-spacing: -.01em; }
  .panel > p:first-child { margin-top: 0; }

  /* tables */
  .table-wrap { overflow-x: auto; }
  table.data { width: 100%; border-collapse: collapse; font-size: 13px; }
  table.data thead th {
    background: #f8fafc; color: var(--muted); font-weight: 600; text-align: right;
    padding: 10px 12px; border-bottom: 2px solid var(--line); white-space: nowrap; font-size: 12px;
    text-transform: uppercase; letter-spacing: .03em;
  }
  table.data thead th:first-child, table.data tbody td:first-child { text-align: left; }
  table.data tbody td { padding: 9px 12px; border-bottom: 1px solid var(--line); white-space: nowrap; font-variant-numeric: tabular-nums; }
  table.data tbody tr:nth-child(even) td { background: #f8fafc; }
  table.data tbody tr:hover td { background: var(--accent-weak); }
  .num.pos { color: var(--pos); font-weight: 600; }
  .num.neg { color: var(--neg); font-weight: 600; }
  .num.na { color: var(--muted); }

  /* loading overlay */
  #loader {
    position: fixed; inset: 0; background: rgba(241,245,249,.72); backdrop-filter: blur(2px);
    display: none; align-items: center; justify-content: center; z-index: 100;
    font-size: 15px; font-weight: 600; color: var(--accent); gap: 12px;
  }
  #loader.show { display: flex; }
  #loader .spinner { width: 22px; height: 22px; border: 3px solid var(--accent-weak); border-top-color: var(--accent); border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  footer { margin-top: 40px; color: var(--muted); font-size: 12px; text-align: center; }
  @media (max-width: 640px) {
    .header-inner { gap: 12px; }
    nav.section-nav { margin-left: 0; width: 100%; overflow-x: auto; }
  }
"""


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


def apply_chart_theme(fig: go.Figure) -> go.Figure:
    """Apply the shared fintech-light theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif", size=13, color="#0f172a"),
        title_font=dict(size=17, color="#0f172a"),
        margin=dict(l=48, r=24, t=64, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1, font=dict(size=12)),
        hoverlabel=dict(bgcolor="#0f172a", font_size=12, font_color="#ffffff"),
    )
    fig.update_xaxes(gridcolor="#eef2f7", zerolinecolor="#e5e7eb", tickfont=dict(size=12))
    fig.update_yaxes(gridcolor="#eef2f7", zerolinecolor="#e5e7eb", tickfont=dict(size=12))
    return fig


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
    apply_chart_theme(fig)
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
    apply_chart_theme(fig)
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
    apply_chart_theme(fig)
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
    apply_chart_theme(fig)
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
    apply_chart_theme(fig)
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
    apply_chart_theme(fig)
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


def color_signed_value(value: float, suffix: str = "") -> str:
    """Return an HTML span with green/red coloring for a signed numeric value.

    Used inside tables rendered with escape=False. NaN/empty become a muted dash.
    """
    if pd.isna(value):
        return '<span class="num na">&mdash;</span>'
    cls = "pos" if value > 0 else ("neg" if value < 0 else "")
    return f'<span class="num {cls}">{value:+.2f}{suffix}</span>' if suffix else f'<span class="num {cls}">{value:.2f}</span>'


def format_value_signed(value: float) -> str:
    """Colored span form used for KPI cards that want a leading sign on returns."""
    if pd.isna(value):
        return '<span class="num na">N/A</span>'
    cls = "pos" if value > 0 else ("neg" if value < 0 else "")
    return f'<span class="num {cls}">{value:+.2f}%</span>'


def build_query_string(ticker: str, month: int, lookback: int, show_weekly_std: bool) -> str:
    """Build a GET query string preserving the active dashboard params."""
    std_flag = "1" if show_weekly_std else "0"
    return "?" + urlencode(
        {
            "ticker": ticker.upper(),
            "month": month,
            "lookback": lookback,
            "show_weekly_std": std_flag,
        }
    )


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
    for col in ["start_close", "end_close"]:
        selected_cycles_table[col] = selected_cycles_table[col].map(lambda x: f"{x:.2f}")
    selected_cycles_table["return_pct"] = selected_cycles_table["return_pct"].map(color_signed_value)

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
    monthly_comparison_table["mean_return_pct"] = monthly_comparison_table["mean_return_pct"].map(color_signed_value)
    monthly_comparison_table["std_return_pct"] = monthly_comparison_table["std_return_pct"].map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )
    weekly_global_table = weekly_global_summary.copy()
    weekly_global_table["mean_return_pct"] = weekly_global_table["mean_return_pct"].map(color_signed_value)
    weekly_global_table["win_rate"] = weekly_global_table["win_rate"].map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}%"
    )
    weekly_global_table["std_return_pct"] = weekly_global_table["std_return_pct"].map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )
    weekly_monthly_table = weekly_monthly_summary.copy()
    weekly_monthly_table["mean_return_pct"] = weekly_monthly_table["mean_return_pct"].map(color_signed_value)
    weekly_monthly_table["std_return_pct"] = weekly_monthly_table["std_return_pct"].map(
        lambda x: "" if pd.isna(x) else f"{x:.2f}"
    )

    options_html = "".join(
        f'<option value="{month}" {"selected" if month == selected_month else ""}>{label}</option>'
        for month, label in MONTH_NAMES.items()
    )
    show_std_checked = "checked" if show_weekly_std else ""

    active_ticker = ticker.upper()
    preset_chips = "".join(
        f'<a class="chip{" active" if preset == active_ticker else ""}" '
        f'href="{build_query_string(preset, selected_month, lookback_years, show_weekly_std)}">{preset}</a>'
        for preset in TICKER_PRESETS
    )

    mean_return = metrics["mean_return_pct"]
    mean_card_cls = "pos" if (not pd.isna(mean_return) and mean_return > 0) else (
        "neg" if (not pd.isna(mean_return) and mean_return < 0) else "neutral"
    )

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OPEX Seasonality Dashboard - {active_ticker} {month_name}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>{BASE_CSS}</style>
</head>
<body>
  <div id="loader"><span class="spinner"></span>Loading data&hellip;</div>

  <header class="app-header">
    <div class="header-inner">
      <div class="brand"><span class="dot"></span>OPEX Seasonality</div>
      <nav class="section-nav">
        <a href="#overview">Overview</a>
        <a href="#monthly">Monthly</a>
        <a href="#weekly">Weekly</a>
        <a href="#details">Details</a>
      </nav>
    </div>
  </header>

  <main class="wrap">
    <div class="controls">
      <form class="controls-form" method="get" action="/" data-loader>
        <div class="field">
          <label for="ticker">Ticker</label>
          <input type="text" id="ticker" name="ticker" value="{active_ticker}" autocomplete="off" spellcheck="false" />
        </div>
        <div class="field">
          <label for="month">Ending month</label>
          <select id="month" name="month">{options_html}</select>
        </div>
        <div class="field">
          <label for="lookback">Lookback years</label>
          <input type="number" id="lookback" name="lookback" min="3" max="30" value="{lookback_years}" />
        </div>
        <label class="field-check">
          <input type="checkbox" name="show_weekly_std" value="1" {show_std_checked} />
          Show weekly std-dev labels
        </label>
        <button class="primary" type="submit">Update</button>
      </form>
      <div class="presets">
        <span class="label">Quick tickers</span>
        {preset_chips}
      </div>
    </div>

    <p class="subhead">
      Showing the latest available {actual_lookback_years} completed {month_name} cycles for {active_ticker}.
      Cycle = return from the prior monthly OPEX close to the selected month's OPEX close.
    </p>

    <section id="overview">
      <div class="section-title"><h2>Overview</h2><span class="tag">{active_ticker} · {month_name} · {lookback_years}y</span></div>
      <p class="section-desc">Headline seasonality metrics for the selected month's OPEX cycle, plus how the current year is tracking against the prior-year average.</p>
      <div class="cards">
        <div class="card {mean_card_cls}"><div class="label">{month_name} mean return</div><div class="value">{format_value_signed(metrics["mean_return_pct"])}</div></div>
        <div class="card neutral"><div class="label">{month_name} return std dev</div><div class="value">{format_value(metrics["std_return_pct"], "%")}</div></div>
        <div class="card neutral"><div class="label">{month_name} daily volatility</div><div class="value">{format_value(metrics["daily_vol_pct"], "%")}</div></div>
        <div class="card neutral"><div class="label">{month_name} annualized Sharpe</div><div class="value">{format_value(metrics["annualized_sharpe"])}</div></div>
        <div class="card neutral"><div class="label">Observations</div><div class="value">{metrics["observations"]}</div></div>
      </div>
      <div class="panel">{selected_month_chart}</div>
      <div class="panel">{ytd_seasonality_chart}</div>
    </section>

    <section id="monthly">
      <div class="section-title"><h2>Monthly</h2><span class="tag">all months · {lookback_years}y window</span></div>
      <p class="section-desc">How every calendar month compares on average OPEX-cycle return and dispersion, with a full year-by-year heatmap.</p>
      <div class="panel">{monthly_comparison_chart}</div>
      <div class="panel">{heatmap_chart}</div>
      <div class="panel">
        <h3>Monthly comparison</h3>
        <div class="table-wrap">{monthly_comparison_table.to_html(index=False, escape=False, classes="data")}</div>
      </div>
    </section>

    <section id="weekly">
      <div class="section-title"><h2>Weekly</h2><span class="tag">within-month buckets</span></div>
      <p class="section-desc">Performance split into Week 1, Week 2, OPEX Week, and Post-OPEX Week. Week 1 &amp; Week 2 are the first two Mon-Fri market weeks touching the month; OPEX Week contains that month's OPEX day; Post-OPEX Week is the next market week after.</p>
      <div class="panel">{weekly_global_chart}</div>
      <div class="panel">{weekly_monthly_heatmap}</div>
      <div class="panel">
        <h3>Weekly bucket summary</h3>
        <div class="table-wrap">{weekly_global_table.to_html(index=False, escape=False, classes="data")}</div>
      </div>
      <div class="panel">
        <h3>Weekly bucket monthly breakdown</h3>
        <div class="table-wrap">{weekly_monthly_table.to_html(index=False, escape=False, classes="data")}</div>
      </div>
    </section>

    <section id="details">
      <div class="section-title"><h2>Details</h2><span class="tag">{month_name} cycle-by-cycle</span></div>
      <p class="section-desc">Underlying OPEX start/end prices and resulting returns for each of the {actual_lookback_years} completed {month_name} cycles.</p>
      <div class="panel">
        <h3>{month_name} cycle details</h3>
        <div class="table-wrap">{selected_cycles_table.to_html(index=False, escape=False, classes="data")}</div>
      </div>
    </section>

    <footer>Prices via yfinance. Past seasonality is descriptive, not predictive.</footer>
  </main>

  <script>
    (function () {{
      var form = document.querySelector('form[data-loader]');
      var loader = document.getElementById('loader');
      if (form && loader) {{
        form.addEventListener('submit', function () {{ loader.classList.add('show'); }});
        window.addEventListener('pageshow', function () {{ loader.classList.remove('show'); }});
      }}
    }})();
  </script>
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
    active_ticker = ticker.upper()

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OPEX Seasonality Dashboard - {active_ticker}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>{BASE_CSS}</style>
</head>
<body>
  <header class="app-header">
    <div class="header-inner">
      <div class="brand"><span class="dot"></span>OPEX Seasonality</div>
    </div>
  </header>

  <main class="wrap">
    <div class="controls">
      <form class="controls-form" method="get" action="/" data-loader>
        <div class="field">
          <label for="ticker">Ticker</label>
          <input type="text" id="ticker" name="ticker" value="{active_ticker}" autocomplete="off" spellcheck="false" />
        </div>
        <div class="field">
          <label for="month">Ending month</label>
          <select id="month" name="month">{options_html}</select>
        </div>
        <div class="field">
          <label for="lookback">Lookback years</label>
          <input type="number" id="lookback" name="lookback" min="3" max="30" value="{lookback_years}" />
        </div>
        <label class="field-check">
          <input type="checkbox" name="show_weekly_std" value="1" {show_std_checked} />
          Show weekly std-dev labels
        </label>
        <button class="primary" type="submit">Update</button>
      </form>
    </div>

    <section id="error" style="margin-top:24px;">
      <div class="section-title"><h2>Data load failed</h2><span class="tag" style="color:var(--neg);">error</span></div>
      <div class="panel" style="border-left:4px solid var(--neg);">
        <p>The app could not retrieve price history for <strong>{active_ticker}</strong>.</p>
        <p style="margin-top:12px;"><code style="background:#f8fafc;border:1px solid var(--line);border-radius:6px;padding:8px 10px;white-space:pre-wrap;word-break:break-word;font-size:12px;color:var(--neg);">{error_message}</code></p>
        <p style="margin-top:12px;color:var(--muted);">Common causes: temporary Yahoo response failure, network filtering, or a local package issue in yfinance/requests.</p>
      </div>
    </section>

    <footer>Prices via yfinance. Past seasonality is descriptive, not predictive.</footer>
  </main>
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
