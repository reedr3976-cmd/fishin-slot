#!/usr/bin/env python3
"""MTF group study: AGREE / DISAGREE / WEEKLY_UNKNOWN and policies A/B/C.

Analysis only. Does not change live scanner defaults or scoring thresholds.
Crypto remains disabled by default.

  python3 run_mtf_group_study.py --live
  python3 run_mtf_group_study.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import INSTRUMENTS, OUTPUT_DIR, VALIDATION_TRAIN_FRACTION, active_instruments
from backtest.mtf_groups import run_mtf_group_study
from backtest.report_mtf_groups import build_mtf_group_report

REPORT_TXT = "output/mtf_group_study.txt"
REPORT_JSON = "output/mtf_group_study.json"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="MTF AGREE/DISAGREE/UNKNOWN group study")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true")
    mode.add_argument("--live", action="store_true")
    p.add_argument("--symbols", type=str, default=None)
    p.add_argument("--train-frac", type=float, default=VALIDATION_TRAIN_FRACTION)
    args = p.parse_args(argv)

    symbols = None
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        bad = [s for s in symbols if s not in INSTRUMENTS]
        if bad:
            print(f"Unknown: {bad}")
            return 2

    print("MTF group study (analysis only; live scanner unchanged)...")
    print(f"Universe: {symbols or list(active_instruments().keys())}")
    print("Policies: A=original, B=suppress DISAGREE only, C=require AGREE")

    result = run_mtf_group_study(
        demo=bool(args.demo),
        instruments=symbols,
        train_fraction=args.train_frac,
    )
    report = build_mtf_group_report(result)
    print(report)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(REPORT_TXT).write_text(report, encoding="utf-8")

    def bag(b):
        return b.to_dict()

    payload = {
        "mode": result["mode"],
        "train_fraction": result["train_fraction"],
        "instruments": result["instruments"],
        "focus": result["focus"],
        "train_counts": result["train_counts"],
        "test_counts": result["test_counts"],
        "groups_test": {
            g: {
                "all_mh": bag(v["all_mh"]),
                "HIGH": bag(v["HIGH"]),
                "MEDIUM": bag(v["MEDIUM"]),
                "by_asset_class": {c: bag(x) for c, x in v["by_asset_class"].items()},
            }
            for g, v in result["groups_test"].items()
        },
        "policies_test": {
            name: {
                "kept_mh_groups": pol["kept_mh_groups"],
                "overall": bag(pol["overall"]),
                "medium_high": bag(pol["medium_high"]),
                "by_confidence": {c: bag(x) for c, x in pol["by_confidence"].items()},
                "by_asset_class": {c: bag(x) for c, x in pol["by_asset_class"].items()},
            }
            for name, pol in result["policies_test"].items()
        },
        "errors": result["errors"],
        "disclaimer": (
            "Educational analysis only. Live scanner unchanged. Do not merge without approval."
        ),
    }
    Path(REPORT_JSON).write_text(
        json.dumps(payload, indent=2, default=str).replace("Infinity", "null"),
        encoding="utf-8",
    )
    print(f"Saved: {REPORT_TXT}")
    print(f"Saved: {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
