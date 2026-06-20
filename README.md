# OPEX Seasonality Dashboard

FastAPI dashboard for analyzing seasonal US ETF or stock performance using monthly option expiration cycles.

Cycle definition:

- `Jul` cycle = return from June OPEX close to July OPEX close
- if the third Friday is a market holiday, the app uses the previous trading day

## Features

- OPEX-to-OPEX seasonal return analysis
- selected month summary:
  - mean return
  - return standard deviation
  - daily volatility
  - annualized Sharpe
- year-by-year selected-month return chart
- monthly mean return and standard deviation comparison
- heatmap of monthly OPEX-cycle returns across years
- YTD cumulative return overlay:
  - average prior-year path
  - current-year path to latest available date

## Requirements

- Python 3.10+
- internet access for market data

## Install

```bash
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Run on a server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://YOUR_SERVER_IP:8000
```

## Inputs

The web UI supports:

- ticker, for example `SPY` or `QQQ`
- ending month
- lookback years

The app automatically:

- uses only the recent history needed for the requested lookback
- clamps to the latest available completed cycles
- keeps the current year visible in the heatmap
- leaves future incomplete months blank

## Notes on data

- price data is fetched with `yfinance`
- if Yahoo access is blocked by local firewall, proxy, antivirus, or network policy, data loading can fail
- common symptom: `possibly delisted` for valid tickers like `SPY` or `QQQ`
- that message is usually upstream fetch failure, not a real delisting

## Repository structure

- `app.py` - FastAPI dashboard
- `seasonal_opex.py` - OPEX date and return calculation logic
- `requirements.txt` - Python dependencies

## Example

To compare the July OPEX cycle for SPY over the last 10 years:

1. start the app
2. open the browser
3. enter `SPY`
4. choose `Jul`
5. set lookback to `10`

## Clone

```bash
git clone git@github.com:black48mamba-dot/seasonal_study.git
cd seasonal_study
```
