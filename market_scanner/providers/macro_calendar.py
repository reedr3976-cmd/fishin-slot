"""Official macroeconomic calendars and policy-rate series (research only).

NEWS SAFETY:
- Never fabricate events, forecasts, speeches, or prints.
- Missing fields are UNKNOWN and must not drive trading decisions.
- Historical backtests use only information dated on or before the bar.
- BLS/BEA headline *values* from current APIs are revised; they are NOT used
  as vintage actuals or surprises in signals.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from html import unescape
from typing import Any, Optional
from zoneinfo import ZoneInfo

import requests

USER_AGENT = (
    "Mozilla/5.0 (compatible; MarketScanner/1.0; educational-research)"
)
ET = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
UTC = timezone.utc
UNKNOWN = "UNKNOWN"

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


@dataclass
class MacroEvent:
    event_id: str
    name: str
    country: str
    category: str
    importance: str  # HIGH | MEDIUM | LOW | UNKNOWN
    ts_unix: Optional[int]
    time_precision: str  # exact | convention | date_only | UNKNOWN
    provenance: str
    affected_assets: tuple[str, ...]
    previous: Any = UNKNOWN
    consensus: Any = UNKNOWN
    actual: Any = UNKNOWN
    surprise: Any = UNKNOWN  # actual - consensus; UNKNOWN unless both known at the time
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["affected_assets"] = list(self.affected_assets)
        return d


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/json,*/*"})
    return s


def _plain_row(html: str) -> str:
    text = unescape(re.sub(r"(?is)<[^>]+>", " ", html))
    return re.sub(r"\s+", " ", text).strip()


def _et(year: int, month: int, day: int, hour: int, minute: int) -> int:
    dt = datetime(year, month, day, hour, minute, tzinfo=ET)
    return int(dt.timestamp())


def _parse_bls_datetime(plain: str) -> Optional[tuple[int, str]]:
    """Parse 'Friday, January 05, 2024 08:30 AM ...' → unix ts ET."""
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+(\d{1,2}),\s+(20\d{2})\s+(\d{1,2}):(\d{2})\s*(AM|PM)",
        plain,
        re.I,
    )
    if not m:
        return None
    month = MONTHS[m.group(1).lower()]
    day = int(m.group(2))
    year = int(m.group(3))
    hour = int(m.group(4)) % 12
    minute = int(m.group(5))
    if m.group(6).upper() == "PM":
        hour += 12
    return _et(year, month, day, hour, minute), "exact"


# Asset impact maps (research tagging — not directional calls)
US_RISK = (
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "AUDUSD",
    "USDCAD",
    "USDCHF",
    "XAUUSD",
    "XAGUSD",
    "USOIL",
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "XOM",
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
EUR_ASSETS = ("EURUSD", "XAUUSD")
GBP_ASSETS = ("GBPUSD",)
JPY_ASSETS = ("USDJPY",)
AUD_ASSETS = ("AUDUSD", "XAUUSD", "COPPER")
CAD_ASSETS = ("USDCAD", "USOIL")
NZD_ASSETS = ()  # no NZD pair in catalog
CN_ASSETS = ("XAUUSD", "XAGUSD", "USOIL", "COPPER", "CORN")

BLS_NAME_MAP = (
    ("Employment Situation", "US_NFP", "HIGH", "US", US_RISK, "NFP / unemployment / AHE (single release)"),
    ("Consumer Price Index", "US_CPI", "HIGH", "US", US_RISK, "CPI / core CPI (single release)"),
    ("Producer Price Index", "US_PPI", "MEDIUM", "US", US_RISK, "PPI"),
    ("Job Openings and Labor Turnover", "US_JOLTS", "MEDIUM", "US", US_RISK, "JOLTS"),
    ("Employment Cost Index", "US_ECI", "MEDIUM", "US", US_RISK, "ECI"),
)


def fetch_bls_yearly_events(years: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)) -> list[MacroEvent]:
    sess = _session()
    events: list[MacroEvent] = []
    for year in years:
        url = f"https://www.bls.gov/schedule/{year}/home.htm"
        try:
            resp = sess.get(url, timeout=30)
        except requests.RequestException:
            continue
        if resp.status_code != 200:
            continue
        rows = re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", resp.text)
        for row in rows:
            plain = _plain_row(row)
            parsed = _parse_bls_datetime(plain)
            if not parsed:
                continue
            ts, prec = parsed
            for needle, cat, imp, country, assets, note in BLS_NAME_MAP:
                if needle in plain:
                    # Skip non-headline variants (veterans, etc.)
                    if needle == "Employment Situation" and "Veterans" in plain:
                        continue
                    events.append(
                        MacroEvent(
                            event_id=f"{cat}_{ts}",
                            name=needle,
                            country=country,
                            category=cat,
                            importance=imp,
                            ts_unix=ts,
                            time_precision=prec,
                            provenance=url,
                            affected_assets=assets,
                            notes=note,
                        )
                    )
                    break
    return events


def fetch_fomc_events() -> list[MacroEvent]:
    url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    sess = _session()
    try:
        resp = sess.get(url, timeout=30)
    except requests.RequestException:
        return []
    if resp.status_code != 200:
        return []
    events: list[MacroEvent] = []
    # Statement HTML/PDF dated YYYYMMDD is the announcement day (official).
    # Clock time: FOMC statements are customarily 14:00 ET — recorded as convention.
    for ymd in sorted(set(re.findall(r"monetary(20\d{6})a", resp.text))):
        year, month, day = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
        ts = _et(year, month, day, 14, 0)
        events.append(
            MacroEvent(
                event_id=f"FOMC_STMT_{ymd}",
                name="FOMC statement / rate decision / chair press conference (same-day convention)",
                country="US",
                category="FOMC_DECISION",
                importance="HIGH",
                ts_unix=ts,
                time_precision="convention",
                provenance=url,
                affected_assets=US_RISK,
                notes=(
                    "Official statement date from Fed calendar URL. 14:00 ET is the "
                    "published FOMC release convention, not a scraped clock field. "
                    "Expected rate path / consensus = UNKNOWN (no vintage survey)."
                ),
            )
        )
    return events


def fetch_boe_bank_rate() -> tuple[list[tuple[int, float]], list[MacroEvent]]:
    """Official BOE Bank Rate history. Rate-change dates are events; surprise UNKNOWN."""
    url = (
        "https://www.bankofengland.co.uk/boeapps/iadb/fromshowcolumns.asp"
        "?csv.x=yes&Datefrom=01/Jan/2021&Dateto=01/Jan/2028"
        "&SeriesCodes=IUDBEDR&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
    )
    sess = _session()
    try:
        resp = sess.get(url, timeout=30)
    except requests.RequestException:
        return [], []
    if resp.status_code != 200 or "IUDBEDR" not in resp.text:
        return [], []
    series: list[tuple[int, float]] = []
    events: list[MacroEvent] = []
    prev: Optional[float] = None
    for line in resp.text.splitlines():
        if line.startswith("DATE") or not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            dt = datetime.strptime(parts[0], "%d %b %Y").replace(tzinfo=LONDON)
            val = float(parts[1])
        except ValueError:
            continue
        ts = int(dt.timestamp())
        series.append((ts, val))
        if prev is not None and val != prev:
            events.append(
                MacroEvent(
                    event_id=f"BOE_RATE_{dt.strftime('%Y%m%d')}",
                    name="Bank of England Bank Rate change",
                    country="UK",
                    category="BOE_DECISION",
                    importance="HIGH",
                    ts_unix=ts,
                    time_precision="date_only",
                    provenance=url,
                    affected_assets=GBP_ASSETS,
                    previous=prev,
                    actual=val,
                    consensus=UNKNOWN,
                    surprise=UNKNOWN,
                    notes="Change vs prior official rate; MPC-day clock time UNKNOWN.",
                )
            )
        prev = val
    return series, events


def fetch_ecb_refi_rate() -> tuple[list[tuple[int, float]], list[MacroEvent]]:
    url = (
        "https://data-api.ecb.europa.eu/service/data/FM/"
        "D.U2.EUR.4F.KR.MRR_FR.LEV?startPeriod=2021-01-01&format=jsondata"
    )
    sess = _session()
    try:
        resp = sess.get(url, timeout=30)
        data = resp.json() if resp.status_code == 200 else {}
    except (requests.RequestException, ValueError):
        return [], []
    try:
        obs = data["dataSets"][0]["series"]
        series_key = next(iter(obs))
        obs_map = obs[series_key]["observations"]
        dates = data["structure"]["dimensions"]["observation"][0]["values"]
    except (KeyError, StopIteration, IndexError):
        return [], []
    series: list[tuple[int, float]] = []
    events: list[MacroEvent] = []
    prev: Optional[float] = None
    for idx, meta in enumerate(dates):
        rec = obs_map.get(str(idx))
        if not rec:
            continue
        try:
            dt = datetime.strptime(meta["id"], "%Y-%m-%d").replace(tzinfo=UTC)
            val = float(rec[0])
        except (KeyError, ValueError, TypeError):
            continue
        ts = int(dt.timestamp())
        series.append((ts, val))
        if prev is not None and val != prev:
            events.append(
                MacroEvent(
                    event_id=f"ECB_RATE_{meta['id']}",
                    name="ECB main refinancing rate change",
                    country="EA",
                    category="ECB_DECISION",
                    importance="HIGH",
                    ts_unix=ts,
                    time_precision="date_only",
                    provenance=url,
                    affected_assets=EUR_ASSETS,
                    previous=prev,
                    actual=val,
                    consensus=UNKNOWN,
                    surprise=UNKNOWN,
                    notes="Official ECB SDW series; intraday announcement clock UNKNOWN.",
                )
            )
        prev = val
    return series, events


def weekly_us_claims_events(start: datetime, end: datetime) -> list[MacroEvent]:
    """DOL Employment Situation companion: Initial Claims typically Thursday 8:30 ET.

    This is a published schedule convention, not a scraped per-week calendar.
    Holiday weeks may differ — those exceptions are UNKNOWN and unadjusted.
    """
    events: list[MacroEvent] = []
    d = start
    # advance to Thursday
    while d.weekday() != 3:
        d += timedelta(days=1)
    while d <= end:
        ts = _et(d.year, d.month, d.day, 8, 30)
        events.append(
            MacroEvent(
                event_id=f"US_CLAIMS_{d.strftime('%Y%m%d')}",
                name="US Initial / Continuing Jobless Claims (Thursday convention)",
                country="US",
                category="US_CLAIMS",
                importance="MEDIUM",
                ts_unix=ts,
                time_precision="convention",
                provenance="DOL/BLS weekly claims release convention (Thursday 8:30 ET)",
                affected_assets=US_RISK,
                notes="Holiday reschedules UNKNOWN; consensus/actual UNKNOWN (no vintage).",
            )
        )
        d += timedelta(days=7)
    return events


def unknown_catalog() -> list[dict[str, str]]:
    """Requested series with no reliable timestamped public source in this environment."""
    return [
        {"item": "US Core CPI as a separate timestamp", "status": UNKNOWN, "reason": "Released with headline CPI"},
        {"item": "US PCE / Core PCE historical event times", "status": UNKNOWN, "reason": "BEA public page is upcoming-only; no vintage calendar fetched"},
        {"item": "ADP employment", "status": UNKNOWN, "reason": "No official free timestamped calendar"},
        {"item": "ISM Manufacturing / Services PMI", "status": UNKNOWN, "reason": "No official free historical calendar"},
        {"item": "Conference Board Consumer Confidence", "status": UNKNOWN, "reason": "No official free historical calendar"},
        {"item": "University of Michigan sentiment / inflation expectations", "status": UNKNOWN, "reason": "No official free historical calendar"},
        {"item": "Retail Sales historical event times", "status": UNKNOWN, "reason": "Census calendar not fetched with exact times"},
        {"item": "Scheduled Fed speeches (non-FOMC)", "status": UNKNOWN, "reason": "Would require a speech calendar we do not have"},
        {"item": "FOMC minutes release clock", "status": UNKNOWN, "reason": "Fed page dates meetings, not minutes publication time"},
        {"item": "Consensus / forecast for all prints", "status": UNKNOWN, "reason": "No vintage survey (Bloomberg/Econoday) available; will not invent"},
        {"item": "BLS first-print actuals", "status": UNKNOWN, "reason": "Current BLS API is revised; using it would leak revisions"},
        {"item": "BOC / RBA / RBNZ / BOJ / SNB policy-rate histories", "status": UNKNOWN, "reason": "No reliable parseable series in this run"},
        {"item": "Chinese official releases", "status": UNKNOWN, "reason": "No reliable timestamped NBS calendar fetched"},
        {"item": "Expected future rate path (dots / OIS)", "status": UNKNOWN, "reason": "No vintage expectation dataset"},
    ]


def load_macro_bundle() -> dict[str, Any]:
    """Fetch official calendars/rates. Failures become empty lists, not invented events."""
    errors: list[str] = []
    bls: list[MacroEvent] = []
    fomc: list[MacroEvent] = []
    boe_series: list[tuple[int, float]] = []
    boe_events: list[MacroEvent] = []
    ecb_series: list[tuple[int, float]] = []
    ecb_events: list[MacroEvent] = []
    try:
        bls = fetch_bls_yearly_events()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"BLS calendar: {type(exc).__name__}: {exc}")
    try:
        fomc = fetch_fomc_events()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"FOMC calendar: {type(exc).__name__}: {exc}")
    try:
        boe_series, boe_events = fetch_boe_bank_rate()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"BOE rate: {type(exc).__name__}: {exc}")
    try:
        ecb_series, ecb_events = fetch_ecb_refi_rate()
    except Exception as exc:  # noqa: BLE001
        errors.append(f"ECB rate: {type(exc).__name__}: {exc}")

    start = datetime(2022, 1, 1)
    end = datetime(2026, 12, 31)
    claims = weekly_us_claims_events(start, end)
    events = bls + fomc + boe_events + ecb_events + claims
    events.sort(key=lambda e: e.ts_unix or 0)
    return {
        "events": events,
        "boe_rate": boe_series,
        "ecb_rate": ecb_series,
        "unknown": unknown_catalog(),
        "errors": errors,
        "counts": {
            "bls": len(bls),
            "fomc": len(fomc),
            "boe_decisions": len(boe_events),
            "ecb_decisions": len(ecb_events),
            "us_claims_convention": len(claims),
            "total_events": len(events),
        },
    }
