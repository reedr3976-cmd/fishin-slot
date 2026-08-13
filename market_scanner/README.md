# Daily Market Scanner

Beginner-friendly **ranked daily report** for:

- **Forex:** EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF  
- **Crypto:** BTCUSD, ETHUSD, SOLUSD  
- **Commodities:** Gold (XAUUSD), Silver (XAGUSD), Oil (USOIL)

**Alerts & analysis only** — no brokerage connection, no order placement, no API keys, no passwords.

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

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Safety

Educational analysis only. Not financial advice. Public data may be delayed. Gold/silver/oil use public futures proxies (GC=F, SI=F, CL=F).
