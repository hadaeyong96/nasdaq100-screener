"""시장 온도계 오늘 자료 받기 (입출력 전용) — 기존 함수 재사용.

- QQQ·^VIX·나스닥100 구성 종목 종가: data.prices.fetch_one / fetch_universe_prices (확정 봉만, 분할 반영 Close)
- 구성 종목: data.universe.get_universe (오늘 명단 — 과거 날짜를 이 함수로 계산하면 오늘 명단 기준이 된다)
- 공포·탐욕: data.macro.get_fear_greed (아침 편지와 같은 함수·같은 캐시)
항목 하나가 실패하면 그 항목만 비우고(온도계에서 "자료 없음") 실패 이유를 prices["errors"]에 모은다. QQQ 실패는 그대로 올린다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _main_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_live(as_of: date, cfg: dict) -> dict:
    """오늘(as_of 기준) 온도계 자료 → core.get_temperature의 prices dict (+ "errors": [str], "failed": {티커: 이유})."""
    from data import macro
    from data.prices import fetch_one, fetch_universe_prices
    from data.universe import get_universe

    main_cfg = _main_config()
    errors: list[str] = []
    out: dict = {"errors": errors, "failed": {}}
    out["qqq"] = fetch_one(cfg["sma"]["ticker"], main_cfg)["close"].astype(float)

    try:
        tickers = list(get_universe()["ticker"])
        res = fetch_universe_prices(tickers, main_cfg)
        out["failed"] = dict(res.failed)
        if res.failed:
            errors.append(f"구성 종목 가격 실패 {len(res.failed)}개: {', '.join(sorted(res.failed))}")
        out["stocks"] = pd.DataFrame({t: df["close"].astype(float) for t, df in res.prices.items()}).sort_index()
        out["members"] = set(tickers)  # 실패 종목도 명단에는 남긴다(확보율이 낮으면 시장 폭 자료 없음)
    except Exception as exc:
        errors.append(f"시장 폭 자료 실패: {exc}")

    try:
        vdf = fetch_one(cfg["vix"]["ticker"], main_cfg)
        out["vix"] = vdf[["close", "high"]].astype(float)
    except Exception as exc:
        errors.append(f"VIX 실패: {exc}")

    try:
        fg = macro.get_fear_greed(as_of, stale_fallback_days=int(cfg["fear_greed"]["stale_days"]))
        if fg.get("warning"):
            errors.append(f"공포·탐욕: {fg['warning']}")
        out["fear_greed"] = macro._load_fear_greed_cache()
    except Exception as exc:
        errors.append(f"공포·탐욕 실패: {exc}")
    return out
