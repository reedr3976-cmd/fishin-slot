"""Instrument universe and scanner settings.

Alerts and analysis only — no brokerage connection, no order placement.
"""

from __future__ import annotations

# Public Yahoo Finance chart symbols (no API key).
# Full catalog kept here — crypto remains available but is disabled by default.
INSTRUMENTS: dict[str, dict[str, str]] = {
    # Forex
    "EURUSD": {"symbol": "EURUSD=X", "asset_class": "forex", "name": "Euro / US Dollar"},
    "GBPUSD": {"symbol": "GBPUSD=X", "asset_class": "forex", "name": "British Pound / US Dollar"},
    "USDJPY": {"symbol": "USDJPY=X", "asset_class": "forex", "name": "US Dollar / Japanese Yen"},
    "AUDUSD": {"symbol": "AUDUSD=X", "asset_class": "forex", "name": "Australian Dollar / US Dollar"},
    "USDCAD": {"symbol": "USDCAD=X", "asset_class": "forex", "name": "US Dollar / Canadian Dollar"},
    "USDCHF": {"symbol": "USDCHF=X", "asset_class": "forex", "name": "US Dollar / Swiss Franc"},
    # Cryptocurrency (kept in catalog; excluded from active/live universe by default)
    "BTCUSD": {"symbol": "BTC-USD", "asset_class": "crypto", "name": "Bitcoin / US Dollar"},
    "ETHUSD": {"symbol": "ETH-USD", "asset_class": "crypto", "name": "Ethereum / US Dollar"},
    "SOLUSD": {"symbol": "SOL-USD", "asset_class": "crypto", "name": "Solana / US Dollar"},
    # Commodities (futures proxies — public quotes only)
    "XAUUSD": {"symbol": "GC=F", "asset_class": "commodity", "name": "Gold (COMEX)"},
    "XAGUSD": {"symbol": "SI=F", "asset_class": "commodity", "name": "Silver (COMEX)"},
    "USOIL": {"symbol": "CL=F", "asset_class": "commodity", "name": "Crude Oil WTI (NYMEX)"},
    # Stocks / ETFs (catalog + analysis studies; live defaults unchanged until approved)
    "SPY": {"symbol": "SPY", "asset_class": "stocks", "name": "SPDR S&P 500 ETF"},
    "QQQ": {"symbol": "QQQ", "asset_class": "stocks", "name": "Invesco QQQ Trust"},
    "AAPL": {"symbol": "AAPL", "asset_class": "stocks", "name": "Apple Inc."},
    "MSFT": {"symbol": "MSFT", "asset_class": "stocks", "name": "Microsoft Corp."},
    "XOM": {"symbol": "XOM", "asset_class": "stocks", "name": "Exxon Mobil Corp."},
    "JPM": {"symbol": "JPM", "asset_class": "stocks", "name": "JPMorgan Chase & Co."},
}

# Active LIVE scan universe: Forex + commodities only (crypto off; stocks not live yet).
# Pass asset_classes explicitly or use STUDY_ASSET_CLASSES for analysis runners.
ENABLED_ASSET_CLASSES: tuple[str, ...] = ("forex", "commodity")

# Target product focus for analysis/diagnostics (not enabled live until approved).
STUDY_ASSET_CLASSES: tuple[str, ...] = ("forex", "commodity", "stocks")


def active_instruments(
    asset_classes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Return instruments in the active universe (crypto off by default)."""
    classes = tuple(asset_classes) if asset_classes is not None else ENABLED_ASSET_CLASSES
    allowed = set(classes)
    return {k: v for k, v in INSTRUMENTS.items() if v["asset_class"] in allowed}


def default_asset_classes() -> list[str]:
    return list(ENABLED_ASSET_CLASSES)


def study_instruments() -> dict[str, dict[str, str]]:
    """Forex + commodities + stocks for analysis-only studies (no crypto)."""
    return active_instruments(STUDY_ASSET_CLASSES)


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

# Daily scanner ranking thresholds (0–100 confidence score)
# Kept in sync with scanner.scoring.ORIGINAL_RULES for the live scanner.
# Below SCORE_LOW => "NO STRONG SETUP" (do not force a trade idea)
SCORE_HIGH = 60
SCORE_MEDIUM = 40
SCORE_LOW = 25

# Out-of-sample validation defaults (chronological split — no shuffle)
VALIDATION_TRAIN_FRACTION = 0.70  # first 70% of each series by time = train
# Pre-specified share of TRAIN actionable scores labeled HIGH / MEDIUM
# (not optimized for return — reduces overfitting vs maximizing train PnL)
REVISED_HIGH_QUANTILE = 0.85   # top 15% of train actionable scores → HIGH cutoff
REVISED_MEDIUM_QUANTILE = 0.50  # above median → MEDIUM cutoff (else LOW if ≥ score_low)
MIN_FEATURE_HITS_FOR_EDGE = 30
VALIDATION_REPORT_TXT = "output/validation_report.txt"
VALIDATION_REPORT_JSON = "output/validation_report.json"

# 4H trending diagnostics (analysis only — live scanner unchanged)
FOURH_DIAG_REPORT_TXT = "output/fourh_diagnostics_report.txt"
FOURH_DIAG_REPORT_JSON = "output/fourh_diagnostics_report.json"

# Default timeframes for the beginner daily report
DAILY_TIMEFRAMES = ["1d"]

# Swing lookback for support / resistance
SR_LOOKBACK = 40
SR_PIVOT_LEFT = 2
SR_PIVOT_RIGHT = 2

# Output
OUTPUT_DIR = "output"
ALERTS_JSON = "output/latest_alerts.json"
ALERTS_CSV = "output/latest_alerts.csv"
DAILY_SUMMARY_TXT = "output/daily_summary.txt"
DAILY_SUMMARY_JSON = "output/daily_summary.json"

# ---------------------------------------------------------------------------
# Historical backtest settings (analysis only — does not change live rules)
# ---------------------------------------------------------------------------
# Longer Yahoo ranges used only by the backtester (live scanner ranges unchanged).
BACKTEST_TIMEFRAMES: dict[str, dict[str, str]] = {
    "1h": {"interval": "60m", "range": "730d"},
    "4h": {"interval": "60m", "range": "730d"},  # longer history for OOS 4H studies
    "1d": {"interval": "1d", "range": "5y"},
    "1wk": {"interval": "1wk", "range": "10y"},
}

# Forward holding period in bars after a signal (close-to-close).
FORWARD_BARS: dict[str, int] = {
    "1h": 6,   # ~6 hours
    "4h": 4,   # ~16 hours
    "1d": 5,   # ~1 trading week
    "1wk": 4,  # ~1 month
}

# Round-trip cost assumptions (fraction of price): spread + slippage proxy.
# Applied once per signal (entry+exit). Not a broker model — educational only.
ROUND_TRIP_COST: dict[str, float] = {
    "forex": 0.0004,      # ~4 bps
    "crypto": 0.0020,     # ~20 bps
    "commodity": 0.0010,  # ~10 bps
    "stocks": 0.0010,     # ~10 bps (ETF/large-cap proxy)
}

BACKTEST_WARMUP_BARS = 60  # need SMA50 + buffer before first signal
MIN_SIGNALS_FOR_CONCLUSION = 30  # below this, report is marked unreliable
BACKTEST_DEFAULT_TIMEFRAMES = ["1d", "1wk"]
BACKTEST_REPORT_TXT = "output/backtest_report.txt"
BACKTEST_REPORT_JSON = "output/backtest_report.json"
