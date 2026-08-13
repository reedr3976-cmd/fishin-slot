# Market Scanner

Beginner-friendly **alerts & analysis only** scanner for:

- **Forex:** EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF  
- **Crypto:** BTCUSD, ETHUSD, SOLUSD  
- **Commodities:** Gold (XAUUSD), Silver (XAGUSD), Oil (USOIL)

It **does not** connect to a brokerage, place orders, or require passwords / API keys / seed phrases.

## What this environment already had

| Capability | Status |
|---|---|
| Python 3.12, NumPy, Requests | Available |
| Network egress | Open (public HTTPS works) |
| Yahoo Finance public chart API | Works **without API key** (forex, crypto, gold/silver/oil) |
| CoinGecko / Coinbase / Kraken public endpoints | Reachable (not required for v1) |
| Binance | Blocked in this region (not used) |
| Brokerage / trading APIs | **Not used** (by design) |

## Requirements — nothing paid, no secrets

| Item | Required now? | Why |
|---|---|---|
| API key | **No** | Uses Yahoo Finance public chart endpoints |
| Paid data vendor | **No** | Public delayed/quote data is enough for scanning |
| Brokerage account | **No** | Alerts only — no execution |
| Extra downloads on this Cloud Agent | **No** | Uses already-installed `numpy` + `requests` |
| Extra downloads on your own PC | Only if missing | `pip install -r requirements.txt` (numpy, requests) |

If you later want denser intraday data, news, or broker-synced positions, that would need a **separate** discussion before any signup — and you should **never** paste passwords, private keys, seed phrases, or brokerage credentials into chat.

## Quick start

```bash
cd market_scanner

# Offline / synthetic historical (always works)
python3 run_scanner.py --demo

# Public live quotes via Yahoo (no key)
python3 run_scanner.py --live

# Focused scan
python3 run_scanner.py --live --symbols EURUSD,BTCUSD,XAUUSD --tf 1h,1d
python3 run_scanner.py --live --assets forex,commodity --tf 1d --min-strength medium
```

## Prove it works

A successful run prints:

1. A banner stating **NO brokerage / NO orders**
2. A **PRICE SNAPSHOTS** table (instrument, timeframe, last price, RSI)
3. **SETUP ALERTS** (RSI extremes, SMA crosses, MACD crosses, Bollinger touches, trend alignment)
4. Paths to saved files: `output/latest_alerts.json` and `output/latest_alerts.csv`

Unit tests (offline):

```bash
python3 -m unittest discover -s tests -v
```

## Setups detected

| Setup | Meaning (watchlist only) |
|---|---|
| `rsi_oversold` / `rsi_overbought` | RSI ≤ 30 or ≥ 70 |
| `rsi_exit_oversold` / `rsi_exit_overbought` | RSI leaving extreme zone |
| `sma_golden_cross` / `sma_death_cross` | SMA20 crosses SMA50 |
| `trend_aligned_bullish` / `bearish` | Price + MA structure aligned |
| `macd_bullish_cross` / `macd_bearish_cross` | MACD vs signal cross |
| `bb_lower_touch` / `bb_upper_touch` | Price at Bollinger band |

## Action you may need to take

**On this Cloud Agent:** none for the initial scanner — it is already runnable.

**On your own computer (optional):**

1. Install Python 3.10+  
2. `cd market_scanner && pip install -r requirements.txt`  
3. Run `python3 run_scanner.py --live` or `--demo`

No account signup is required for this version.

## Safety

- Output is **analysis / alerts only**
- Not financial advice
- Public data can be delayed or differ from a broker’s feed
- Futures symbols (GC=F, SI=F, CL=F) are **proxies** for gold, silver, and oil education — not a substitute for your broker’s contract specs
