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
    # Cryptocurrency (kept in catalog; excluded from active universe by default)
    "BTCUSD": {"symbol": "BTC-USD", "asset_class": "crypto", "name": "Bitcoin / US Dollar"},
    "ETHUSD": {"symbol": "ETH-USD", "asset_class": "crypto", "name": "Ethereum / US Dollar"},
    "SOLUSD": {"symbol": "SOL-USD", "asset_class": "crypto", "name": "Solana / US Dollar"},
    # Commodities (futures proxies — public quotes only)
    "XAUUSD": {"symbol": "GC=F", "asset_class": "commodity", "name": "Gold (COMEX)"},
    "XAGUSD": {"symbol": "SI=F", "asset_class": "commodity", "name": "Silver (COMEX)"},
    "USOIL": {"symbol": "CL=F", "asset_class": "commodity", "name": "Crude Oil WTI (NYMEX)"},
    # Stocks / ETFs (catalog only — not in live ENABLED_ASSET_CLASSES)
    "SPY": {"symbol": "SPY", "asset_class": "stock", "name": "S&P 500 ETF"},
    "QQQ": {"symbol": "QQQ", "asset_class": "stock", "name": "Nasdaq 100 ETF"},
    "AAPL": {"symbol": "AAPL", "asset_class": "stock", "name": "Apple"},
    "MSFT": {"symbol": "MSFT", "asset_class": "stock", "name": "Microsoft"},
    "XOM": {"symbol": "XOM", "asset_class": "stock", "name": "Exxon Mobil"},
    # V5 held-out stocks (research validation only — not used to tune V4_S1_STOCK)
    "AMZN": {"symbol": "AMZN", "asset_class": "stock", "name": "Amazon"},
    "GOOGL": {"symbol": "GOOGL", "asset_class": "stock", "name": "Alphabet"},
    "META": {"symbol": "META", "asset_class": "stock", "name": "Meta Platforms"},
    "NVDA": {"symbol": "NVDA", "asset_class": "stock", "name": "NVIDIA"},
    "JPM": {"symbol": "JPM", "asset_class": "stock", "name": "JPMorgan Chase"},
    "JNJ": {"symbol": "JNJ", "asset_class": "stock", "name": "Johnson & Johnson"},
    "WMT": {"symbol": "WMT", "asset_class": "stock", "name": "Walmart"},
    "BA": {"symbol": "BA", "asset_class": "stock", "name": "Boeing"},
    "DIS": {"symbol": "DIS", "asset_class": "stock", "name": "Disney"},
}

# Active scan universe: Forex + commodities only (crypto disabled by default).
# Pass asset_classes=["crypto"] or include crypto explicitly to re-enable.
# Live scanner unchanged — stocks remain research/backtest catalog only.
ENABLED_ASSET_CLASSES: tuple[str, ...] = ("forex", "commodity")

# Scanner V2 / research studies only (live defaults unchanged).
STUDY_ASSET_CLASSES: tuple[str, ...] = ("forex", "commodity", "stock")


def active_instruments(
    asset_classes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Return instruments in the active universe (crypto off by default)."""
    classes = tuple(asset_classes) if asset_classes is not None else ENABLED_ASSET_CLASSES
    allowed = set(classes)
    return {k: v for k, v in INSTRUMENTS.items() if v["asset_class"] in allowed}


def default_asset_classes() -> list[str]:
    return list(ENABLED_ASSET_CLASSES)


def study_instruments(
    asset_classes: tuple[str, ...] | list[str] | None = None,
) -> dict[str, dict[str, str]]:
    """Research universe (forex + commodity + stock). Live ENABLED unchanged."""
    classes = tuple(asset_classes) if asset_classes is not None else STUDY_ASSET_CLASSES
    return active_instruments(classes)


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
    "1h": {"interval": "60m", "range": "60d"},
    # Longer 4h history for research/walk-forward (live TIMEFRAMES["4h"] unchanged).
    "4h": {"interval": "60m", "range": "730d"},
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
    "stock": 0.0010,      # ~10 bps (research catalog)
}

BACKTEST_WARMUP_BARS = 60  # need SMA50 + buffer before first signal
MIN_SIGNALS_FOR_CONCLUSION = 30  # below this, report is marked unreliable
BACKTEST_DEFAULT_TIMEFRAMES = ["1d", "1wk"]
BACKTEST_REPORT_TXT = "output/backtest_report.txt"
BACKTEST_REPORT_JSON = "output/backtest_report.json"

# Scanner V2 research (analysis only — does not enable live trading)
SCANNER_V2_REPORT_TXT = "output/scanner_v2_report.txt"
SCANNER_V2_REPORT_JSON = "output/scanner_v2_report.json"
V2_RISK_FRACTION = 0.01  # 1% equity risk per trade (normalized across assets)
V2_ATR_STOP_MULT = 1.5
V2_ADX_MIN = 20.0
V2_MAX_HOLD_BARS = 24  # safety cap on 4H (~4 days)
V2_TRAIN_FRACTION = 0.70
V2_N_FOLDS = 4

# Scanner V3 research (analysis only — live ORIGINAL unchanged; V2 not merged live)
SCANNER_V3_REPORT_TXT = "output/scanner_v3_report.txt"
SCANNER_V3_REPORT_JSON = "output/scanner_v3_report.json"
V3_BREAKOUT_LOOKBACK = 20
V3_RR_TARGET = 2.0  # reward/risk for optional target exits
V3_STRUCT_PIVOT = 2
V3_MAX_HOLD_BARS = 24
V3_ATR_STOP_MULT = 1.5
V3_TRAIN_FRACTION = 0.70
V3_N_FOLDS = 4
# Promotion gates (OOS-focused; TRAIN is diagnostic only)
V3_MAX_DD_ACCEPT = 0.35
V3_MIN_FOLDS_POSITIVE = 3
V3_MIN_TRADES = 30
V3_MIN_SYMBOLS_POSITIVE = 2

# Scanner V4 research (analysis only — live ORIGINAL untouched; do not merge V3)
SCANNER_V4_REPORT_TXT = "output/scanner_v4_report.txt"
SCANNER_V4_REPORT_JSON = "output/scanner_v4_report.json"
V4_TRAIN_FRACTION = 0.70
V4_N_FOLDS = 4
V4_MAX_DD_ACCEPT = 0.35
V4_MIN_FOLDS_POSITIVE = 3
V4_MIN_TRADES = 30
V4_MIN_SYMBOLS_POSITIVE = 2
V4_STRUCT_PIVOT = 2
V4_ATR_STOP_MULT = 1.5
V4_MAX_HOLD_BARS = 24
# Pre-specified (not OOS-tuned) false-break distance in ATR units
V4_MIN_BREAK_ATR = 0.25
V4_PERSIST_BARS = 5
# Stage 1 diagnostic: stock/commodity must beat FX OOS by this margin to justify Stage 2
V4_CLASS_EDGE_MARGIN = 0.0  # expectancy units (fraction); any positive gap counts if SC>FX

# Scanner V5 — independent robustness validation of frozen V4_S1_STOCK (no retune)
SCANNER_V5_REPORT_TXT = "output/scanner_v5_report.txt"
SCANNER_V5_REPORT_JSON = "output/scanner_v5_report.json"
V5_TRAIN_FRACTION = 0.70
V5_N_FOLDS = 4
V5_MAX_DD_ACCEPT = 0.35
V5_MIN_FOLDS_POSITIVE = 3
V5_MIN_TRADES = 30
V5_MIN_SYMBOLS_POSITIVE = 2
V5_MC_RUNS = 500
V5_MC_SEED = 42
# Frozen V4_S1_STOCK parameters (DO NOT change based on V5 OOS)
V5_FROZEN_LOOKBACK = 20
V5_FROZEN_ATR_STOP_MULT = 1.5
V5_FROZEN_MAX_HOLD = 24
V5_FROZEN_STRUCT_PIVOT = 2
V5_FROZEN_SMA_SLOPE_BARS = 3
# V4 original stock universe vs V5 held-out validation names
V5_V4_STOCKS: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "XOM")
V5_HELD_OUT_STOCKS: tuple[str, ...] = (
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "JPM",
    "JNJ",
    "WMT",
    "BA",
    "DIS",
)
V5_COMMODITIES: tuple[str, ...] = ("XAUUSD", "XAGUSD", "USOIL")
V5_ENTRY_SLIP_ATR = 0.05  # modest adverse entry slippage for stress

# Scanner V6 — clean strategy-family reset (research only; V4/V5 falsified)
SCANNER_V6_REPORT_TXT = "output/scanner_v6_report.txt"
SCANNER_V6_REPORT_JSON = "output/scanner_v6_report.json"
V6_TRAIN_FRACTION = 0.70
V6_N_FOLDS = 4
V6_MAX_DD_ACCEPT = 0.35
V6_MIN_FOLDS_POSITIVE = 3
V6_MIN_TRADES = 25
V6_MIN_SYMBOLS_POSITIVE = 2
V6_MC_RUNS = 200
V6_MC_SEED = 7
V6_ATR_STOP_MULT = 1.5
V6_MAX_HOLD = 24
V6_LOOKBACK = 20
V6_VOL_ATR_MULT = 1.2
V6_ENTRY_SLIP_ATR = 0.05
# Discovery vs held-out (held-out NEVER used for family selection)
V6_STOCK_DISCOVERY: tuple[str, ...] = ("SPY", "QQQ", "AAPL", "MSFT", "XOM")
V6_STOCK_HELDOUT: tuple[str, ...] = (
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "JPM",
    "JNJ",
    "WMT",
    "BA",
    "DIS",
)
V6_COMMODITY_DISCOVERY: tuple[str, ...] = ("XAUUSD", "XAGUSD")
V6_COMMODITY_HELDOUT: tuple[str, ...] = ("USOIL",)
V6_FX_DISCOVERY: tuple[str, ...] = ("EURUSD", "GBPUSD", "USDJPY")
V6_FX_HELDOUT: tuple[str, ...] = ("AUDUSD", "USDCAD", "USDCHF")
