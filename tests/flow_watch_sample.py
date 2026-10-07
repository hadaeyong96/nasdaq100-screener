"""실제 캐시 일봉(data/cache/*.parquet)으로 "돈이 몰리는 종목 TOP 10"·"섹터별 자금 흐름" 견본 보고서를 만든다.

네트워크 없음. 종목 = 캐시에 있는 매일 실행 종목(ETF 제외는 core.flow_watch가 함), 이름 = data/name_kr.csv,
업종 = data/cache/sectors.parquet(읽기만), 우리 신호 = 실전 DB(data/state.db)의 보유 상태와 그 기준일 매수 이벤트(읽기만,
DB가 없으면 "없음"). 보고서의 다른 구역은 비워 둔다(_empty_run_summary) — 이 견본은 새 표 확인용이다.
캐시가 없으면 합성 데이터로 만든다.

사용법: python -m tests.flow_watch_sample [--screenshot]  → outputs/flow_watch_sample_live_<기준일>.html (+ png)
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd
import yaml

from engine import daily
from notify import report_html
from store import db

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "outputs"
_KIND_BUCKET = {"A1": "b1", "A2": "b2", "A3": "b3", "B": "b9"}


def load_cached_prices() -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(CACHE.glob("*.parquet")):
        if p.stem in ("earnings", "sectors"):
            continue
        df = pd.read_parquet(p)
        if {"close", "volume"} <= set(df.columns):
            df.index = pd.to_datetime(df.index)
            out[p.stem] = df
    return out


def name_map() -> dict[str, str]:
    path = ROOT / "data" / "name_kr.csv"
    if not path.exists():
        return {}
    kr = pd.read_csv(path, dtype=str).fillna("")
    return dict(zip(kr.iloc[:, 0], kr.iloc[:, 1]))


def live_signals(as_of: pd.Timestamp) -> tuple[dict, dict, str]:
    """실전 DB에서 보유 상태·그 기준일 매수 이벤트를 읽기만 한다."""
    if not db.DB_PATH.exists():
        return {}, {k: [] for k in _KIND_BUCKET.values()}, "실전 DB 없음"
    conn = sqlite3.connect(f"file:{db.DB_PATH.as_posix()}?mode=ro", uri=True)
    positions = db.load_all_positions(conn)
    events = db.get_events_for_date(conn, as_of.date().isoformat())
    last = db.get_meta(conn, "last_processed_date")
    conn.close()
    groups = {k: [] for k in _KIND_BUCKET.values()}
    for e in events:
        if e.get("kind") in _KIND_BUCKET:
            groups[_KIND_BUCKET[e["kind"]]].append({"ticker": e["ticker"]})
    return positions, groups, f"실전 DB 마지막 처리일 {last}"


def main(screenshot: bool = False) -> list[Path]:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    names = name_map()
    # 캐시에는 다른 스크립트가 받아 둔 ETF(QQQ·QQEW 등)도 있다 — 매일 실행처럼 구성 종목(name_kr.csv)만 쓴다
    prices = {t: df for t, df in load_cached_prices().items() if not names or t in names}
    source = "실제 캐시"
    if len(prices) < 10:
        from tests.test_flow_watch import _random_prices

        prices, source = _random_prices(20, n=300), "합성"
    skip = set(cfg.get("flow_watch", {}).get("exclude_tickers", []))
    as_of = max(df.index[-1] for t, df in prices.items() if t not in skip)  # 매일 실행처럼 구성 종목 기준일
    states, groups, db_note = live_signals(as_of)
    view = daily.build_flow_watch_view(prices, as_of, cfg, names, groups, states, daily._flow_sector_map())
    summary = daily._empty_run_summary("live", as_of.date(), [f"견본: {source} 일봉 {len(prices)}종목 · 우리 신호는 {db_note} 기준"])
    summary["as_of"] = as_of
    summary["flow_watch"] = view
    html_path = report_html.render_report(summary, cfg, OUT)
    sample = OUT / f"flow_watch_sample_live_{as_of.date()}.html"
    html_path.replace(sample)
    paths = [sample]
    print(f"견본: {sample} ({source}, 기준일 {as_of.date()}, 대상 {view['scan_count']}종목, 제외 {len(view['excluded'])})")
    if screenshot:
        paths += take_screenshots(sample, as_of)
    return paths


def take_screenshots(html: Path, as_of) -> list[Path]:
    from playwright.sync_api import sync_playwright

    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for label, width in (("desktop", 1200), ("390", 390)):
            page = browser.new_page(viewport={"width": width, "height": 900})
            page.goto(html.resolve().as_uri())
            overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            target = OUT / f"flow_watch_{label}_{as_of.date()}.png"
            page.locator('section[aria-label="돈이 몰리는 종목"]').screenshot(path=str(target))
            print(f"스크린숏: {target} (가로 넘침 {overflow}px)")
            out.append(target)
            page.close()
        browser.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", action="store_true")
    main(ap.parse_args().screenshot)
