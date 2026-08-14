# Daily Market Scanner

Beginner-friendly **ranked daily report** for:

- **Forex:** EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF  
- **Commodities:** Gold (XAUUSD), Silver (XAGUSD), Oil (USOIL)  
- **Crypto:** BTC/ETH/SOL code is kept but **disabled by default**

**Alerts & analysis only** — no brokerage connection, no order placement, no API keys, no passwords.

## Active universe

Default scans = **forex + commodities only**.  
Re-enable crypto for a single run with `--include-crypto` or `--assets forex,crypto,commodity`.

## What you get each run

For every scanned market the report shows:

- instrument & current price  
- bullish / bearish / **neutral** direction  
- timeframe  
- confidence: **HIGH / MEDIUM / LOW** or **NO STRONG SETUP**  
- plain-English reason  
- RSI, SMA20/SMA50 relationship, MACD condition  
- ATR / volatility note  
- nearby support & resistance  

Results are ranked strongest-first. Weak/mixed markets show **NO STRONG SETUP** instead of forcing a signal.

## Quick start

```bash
cd market_scanner

# Beginner daily report (default timeframe: 1d)
python3 run_scanner.py --live
python3 run_scanner.py --demo

# Multi-timeframe
python3 run_scanner.py --live --tf 1h,1d

# One example from each asset class
python3 run_scanner.py --live --symbols EURUSD,BTCUSD,XAUUSD --tf 1d
```

Outputs:

- `output/daily_summary.txt` — plain-English beginner report  
- `output/daily_summary.json` — structured opportunities  
- `output/latest_alerts.json` / `.csv` — machine-readable copy  

## Requirements

| Item | Required? |
|---|---|
| API key | **No** (Yahoo Finance public chart data) |
| Brokerage account | **No** |
| Secrets / passwords | **Never paste these into chat** |
| Extra packages here | **No** (`numpy` + `requests` already present) |

On your own PC, if needed: `pip install -r requirements.txt`

## Confidence validation (out-of-sample)

```bash
cd market_scanner
python3 run_validation.py --live --tf 1d,1wk
```

- Chronological 70/30 train/test split (no shuffle, no look-ahead in features)
- Explains why HIGH is rare under original rules
- Measures feature predictive lift on TRAIN only
- Proposes a frozen revised candidate, then compares ORIGINAL vs REVISED on unseen TEST
- Live scanner keeps ORIGINAL rules unless you explicitly approve a change
- Writes `output/validation_report.txt`

## Historical backtest (confidence ratings)

Before trusting HIGH / MEDIUM / LOW labels, run an unbiased historical check:

```bash
cd market_scanner
python3 run_backtest.py --live --tf 1d,1wk
python3 run_backtest.py --demo   # offline synthetic
```

- Uses the **same** `evaluate_opportunity()` rules as the live scanner (not tuned for prettier results)
- No look-ahead: each signal uses only bars available at that time
- Reports win rate, average return, winners/losers, profit factor, max drawdown by confidence, asset class, and timeframe
- Writes `output/backtest_report.txt` and `output/backtest_report.json`

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Safety

Educational analysis only. Not financial advice. Public data may be delayed. Gold/silver/oil use public futures proxies (GC=F, SI=F, CL=F).
