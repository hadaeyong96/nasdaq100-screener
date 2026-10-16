"""가상 장부 월간 성적표 — 읽기·연결 층 (전략공장 F2).

가상 장부 DB(data/virtual_books.db — Actions에서는 드라이브 state 폴더에서 내려받은 사본)를 **읽기만** 한다.
engine/daily.py는 attach_scorecard 한 곳만 부르고, 여기서 난 오류는 모두 잡아 브리핑 본문 발송을 막지 않는다.

성적표를 넣는 날: 실행일(한국 날짜)이 그달 첫 NYSE 거래일일 때. 성적을 매기는 달 = 지난달.
예: 2026-11-02(월) 아침 실행 → 2026-10 성적표 (기록 2026-10-14 시작이라 "기록 짧음").
"""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from factory import scorecard as sc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "virtual_books.db"
OUT_DIR = ROOT / "outputs" / "scorecard"


def load_ledger(db_path: Path, strats: list[sc.Strategy]) -> tuple[dict, dict]:
    """가상 장부 DB(읽기 전용으로 연다) → ({장부 이름: 일별 DataFrame}, {장부 이름: 매매 DataFrame})."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        daily, trades = {}, {}
        for s in strats:
            for dt, tt in (("daily", "trades"), ("picks_daily", "picks_trades")):
                if dt not in tables:
                    continue
                d = pd.read_sql_query(f"SELECT date, value, value_after_tax_liq FROM {dt} WHERE book = ? ORDER BY date",
                                      conn, params=(s.name,))
                if len(d):
                    d.index = pd.to_datetime(d.pop("date"))
                    daily[s.name] = d
                    if tt in tables:
                        trades[s.name] = pd.read_sql_query(f"SELECT fill_date, commission FROM {tt} WHERE book = ?",
                                                           conn, params=(s.name,))
                    break
        return daily, trades
    finally:
        conn.close()


def scorecard_month(run_date: date, trading_days: pd.DatetimeIndex) -> pd.Period | None:
    """실행일이 그달 첫 거래일이면 성적을 매길 달(지난달), 아니면 None."""
    if not sc.is_first_trading_day(run_date, trading_days):
        return None
    return pd.Period(pd.Timestamp(run_date), freq="M") - 1


def make(cfg: dict, run_date: date, qqq_close: pd.Series, *, db_path: Path = DEFAULT_DB,
         trading_days: pd.DatetimeIndex | None = None) -> dict | None:
    """그날이 성적표 날이면 성적표 dict(+ md·briefing_lines), 아니면 None. 오류는 그대로 올린다(attach_scorecard가 잡음)."""
    if trading_days is None:
        from data.market_calendar import trading_days_between

        trading_days = trading_days_between(run_date - timedelta(days=40), run_date + timedelta(days=5))
    month = scorecard_month(run_date, trading_days)
    if month is None:
        return None
    strats = sc.strategies_from_cfg(cfg)
    daily, trades = load_ledger(db_path, strats)
    card = sc.build(strats, daily, trades, qqq_close, month)
    card["markdown"] = sc.render_markdown(card)
    card["lines"] = sc.briefing_lines(card)
    return card


def save_local(card: dict, out_dir: Path = OUT_DIR) -> Path:
    """성적표 md를 outputs/scorecard/YYYY-MM.md에 남긴다(git 제외 폴더). 외부 저장 위치는 사용자 결정 뒤 연결."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{card['month']}.md"
    p.write_text(card["markdown"], encoding="utf-8")
    return p


def attach_scorecard(summary: dict, cfg: dict, run_date: date, qqq_fetch, render) -> None:
    """engine/daily.py가 부르는 유일한 연결점. 성적표 날이면 summary["scorecard"]를 넣고 보고서를 다시 그린다.

    입력: qqq_fetch() → QQQ 종가 Series, render(summary) → 보고서 경로. 어떤 오류든 잡아 로그만 남기고
    summary·보고서를 원래대로 둔다(브리핑 본문 발송 보호).
    """
    original = summary.get("report_path")
    try:
        card = make(cfg, run_date, qqq_fetch())
        if card is None:
            return
        summary["scorecard"] = {"month": card["month"], "lines": card["lines"]}
        summary["report_path"] = render(summary)
        path = save_local(card)
        print(f"[scorecard] {card['month']} 월간 성적표를 보고서 접기에 넣었습니다 ({path})")
    except Exception as exc:  # noqa: BLE001 — 성적표 실패가 브리핑을 막으면 안 된다
        print(f"[scorecard] 월간 성적표를 만들지 못해 넣지 않습니다: {type(exc).__name__}: {exc}")
        summary.pop("scorecard", None)
        summary["report_path"] = original
