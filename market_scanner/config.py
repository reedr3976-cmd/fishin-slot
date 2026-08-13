"""Instrument universe and scanner settings.

Alerts and analysis only — no brokerage connection, no order placement.
"""

from __future__ import annotations

# Public Yahoo Finance chart symbols (no API key).
INSTRUMENTS: dict[str, dict[str, str]] = {
    # Forex
    "EURUSD": {"symbol": "EURUSD=X", "asset_class": "forex", "name": "Euro / US Dollar"},
    "GBPUSD": {"symbol": "GBPUSD=X", "asset_class": "forex", "name": "British Pound / US Dollar"},
    "USDJPY": {"symbol": "USDJPY=X", "asset_class": "forex", "name": "US Dollar / Japanese Yen"},
    "AUDUSD": {"symbol": "AUDUSD=X", "asset_class": "forex", "name": "Australian Dollar / US Dollar"},
    "USDCAD": {"symbol": "USDCAD=X", "asset_class": "forex", "name": "US Dollar / Canadian Dollar"},
    "USDCHF": {"symbol": "USDCHF=X", "asset_class": "forex", "name": "US Dollar / Swiss Franc"},
    # Cryptocurrency
    "BTCUSD": {"symbol": "BTC-USD", "asset_class": "crypto", "name": "Bitcoin / US Dollar"},
    "ETHUSD": {"symbol": "ETH-USD", "asset_class": "crypto", "name": "Ethereum / US Dollar"},
    "SOLUSD": {"symbol": "SOL-USD", "asset_class": "crypto", "name": "Solana / US Dollar"},
    # Commodities (futures proxies — public quotes only)
    "XAUUSD": {"symbol": "GC=F", "asset_class": "commodity", "name": "Gold (COMEX)"},
    "XAGUSD": {"symbol": "SI=F", "asset_class": "commodity", "name": "Silver (COMEX)"},
    "USOIL": {"symbol": "CL=F", "asset_class": "commodity", "name": "Crude Oil WTI (NYMEX)"},
}

# Yahoo interval -> (chart interval, range string)
TIMEFRAMES: dict[str, dict[str, str]] = {
    "1h": {"interval": "60m", "range": "30d"},
    "4h": {"interval": "60m", "range": "60d"},  # aggregated from 1h bars
    "1d": {"interval": "1d", "range": "1y"},
    "1wk": {"interval": "1wk", "range": "5y"},
}

DEFAULT_TIMEFRAMES = ["1h", "1d"]

# Indicator / setup thresholds
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
SMA_FAST = 20
SMA_SLOW = 50
EMA_FAST = 12
EMA_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2.0
ATR_PERIOD = 14

# Polite delay between Yahoo requests (seconds)
REQUEST_DELAY_SEC = 0.35

# Output
OUTPUT_DIR = "output"
ALERTS_JSON = "output/latest_alerts.json"
ALERTS_CSV = "output/latest_alerts.csv"
