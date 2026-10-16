"""가상 장부 월간 성적표 — 읽기·연결 층 (전략공장 F2).

가상 장부 DB(data/virtual_books.db — Actions에서는 드라이브 state 폴더에서 내려받은 사본)를 **읽기만** 한다.
engine/daily.py는 attach_scorecard 한 곳만 부르고, 여기서 난 오류는 모두 잡아 브리핑 본문 발송을 막지 않는다.

넣는 날 (2026-10-16 사용자 결정):
- 성적을 매기는 달 = 지난달. 그달 첫 NYSE 거래일(한국 날짜 기준 실행일)부터 넣을 수 있다.
- 원본 시트 「월간성적표」 탭(덧붙이기 전용)에 그 달이 아직 없으면 그날 넣는다 → 첫 거래일 실행이 실패해도
  다음 실행에서 한 번만 들어간다. 시트를 쓸 수 없으면(시트 ID 없음 등) 첫 거래일에만 넣는다(중복 확인 불가).
- 접기는 개인 채팅 보고서에만. 투자클럽 단체방에는 성적표를 넣기 전 보고서 사본(summary["group_report_path"])을 보낸다.
예: 2026-11-02(월) 아침 실행 → 2026-10 성적표 (기록 2026-10-14 시작이라 "기록 짧음").
"""

from __future__ import annotations

import shutil
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from factory import scorecard as sc

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "virtual_books.db"
OUT_DIR = ROOT / "outputs" / "scorecard"
KST = ZoneInfo("Asia/Seoul")


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


def _first_trading_day_of(run_date: date, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    td = pd.DatetimeIndex(trading_days).normalize()
    same = td[(td.year == run_date.year) & (td.month == run_date.month)]
    return same.min() if len(same) else None


def scorecard_month(run_date: date, trading_days: pd.DatetimeIndex, done: set[str] | None = None) -> pd.Period | None:
    """성적을 매길 달(지난달) 또는 None.

    done(시트에 이미 있는 달)이 None이면 실행일이 그달 첫 거래일일 때만. done이 있으면 그달 첫 거래일 이후
    어느 실행이든 지난달이 done에 없을 때(실패한 날 다시 넣기, 같은 달 두 번 없음)."""
    month = pd.Period(pd.Timestamp(run_date), freq="M") - 1
    if done is None:
        return month if sc.is_first_trading_day(run_date, trading_days) else None
    first = _first_trading_day_of(run_date, trading_days)
    if first is None or pd.Timestamp(run_date) < first or str(month) in done:
        return None
    return month


def make(cfg: dict, run_date: date, qqq_close: pd.Series, *, db_path: Path = DEFAULT_DB,
         trading_days: pd.DatetimeIndex | None = None, done: set[str] | None = None) -> dict | None:
    """성적표 날이면 성적표 dict(+ markdown·lines), 아니면 None. 오류는 그대로 올린다(attach_scorecard가 잡음)."""
    if trading_days is None:
        from data.market_calendar import trading_days_between

        trading_days = trading_days_between(run_date - timedelta(days=40), run_date + timedelta(days=5))
    month = scorecard_month(run_date, trading_days, done)
    if month is None:
        return None
    strats = sc.strategies_from_cfg(cfg)
    daily, trades = load_ledger(db_path, strats)
    card = sc.build(strats, daily, trades, qqq_close, month)
    card["markdown"] = sc.render_markdown(card)
    card["lines"] = sc.briefing_lines(card)
    return card


def save_local(card: dict, out_dir: Path = OUT_DIR) -> Path:
    """성적표 md를 outputs/scorecard/YYYY-MM.md에 남긴다(git 제외 폴더, 실행 기록용)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{card['month']}.md"
    p.write_text(card["markdown"], encoding="utf-8")
    return p


class SheetTarget:
    """원본 시트 「월간성적표」 탭 (data.virtual_books_sheet). 시트 ID·인증이 없으면 available=False."""

    def __init__(self, cfg: dict, client=None):
        from data import virtual_books_sheet as vbsheet

        self.vbsheet = vbsheet
        env = ((cfg.get("virtual_books") or {}).get("sheet_id_env")) or "VIRTUAL_BOOKS_SHEETS_ID"
        self.sheet_id = vbsheet.sheet_id_from_env(env)
        self.client = client
        if self.sheet_id and self.client is None:
            from data.sheets import get_client

            self.client = get_client()
        self.available = bool(self.sheet_id and self.client is not None)

    def months(self) -> set[str]:
        return self.vbsheet.scorecard_months(self.client, self.sheet_id)

    def append(self, rows: list[list]) -> int:
        return self.vbsheet.append_scorecard(self.client, self.sheet_id, rows)


def attach_scorecard(summary: dict, cfg: dict, run_date: date, qqq_fetch, render, *, sheet=None, notify=None) -> None:
    """engine/daily.py가 부르는 유일한 연결점. 성적표 날이면 시트에 덧붙이고, 개인 보고서에 접기를 넣어 다시 그린다.

    입력: qqq_fetch() → QQQ 종가 Series, render(summary) → 보고서 경로, sheet(SheetTarget 대용, 테스트 주입),
         notify(event, summary) → 전략분석기 알림(기본 factory.notify.send_safely).
    단체방용으로 성적표를 넣기 전 보고서 사본 경로를 summary["group_report_path"]에 둔다.
    어떤 오류든 잡아 로그 한 줄만 남기고 summary·보고서를 원래대로 둔다(브리핑 본문 발송 보호).
    """
    original = summary.get("report_path")
    if notify is None:
        from factory.notify import send_safely as notify
    stage = "월간 성적표 준비"
    try:
        if sheet is None:
            try:
                sheet = SheetTarget(cfg)
            except Exception as exc:  # noqa: BLE001 — 시트 없이도 첫 거래일 보고서 접기는 넣는다
                print(f"[scorecard] 시트 연결 실패 — 중복 확인 없이 첫 거래일만 봅니다: {type(exc).__name__}")
                sheet = None
        done = sheet.months() if sheet is not None and sheet.available else None
        stage = "월간 성적표 계산"
        card = make(cfg, run_date, qqq_fetch(), done=done)
        if card is None:
            return
        stage = "월간 성적표 시트 기록"
        if sheet is not None and sheet.available:
            n = sheet.append(sc.sheet_rows(card, datetime.now(KST).isoformat(timespec="seconds")))
            print(f"[scorecard] 원본 시트 「월간성적표」에 {card['month']} {n}줄 덧붙임")
        else:
            print("[scorecard] 시트 ID가 없어 「월간성적표」 탭 기록은 건너뜁니다")
        stage = "월간 성적표 보고서"
        if original and Path(original).exists():
            group_copy = Path(original).with_name(Path(original).stem + "_group" + Path(original).suffix)
            shutil.copyfile(original, group_copy)  # 성적표 없는 판 → 투자클럽 단체방
            summary["group_report_path"] = group_copy
        summary["scorecard"] = {"month": card["month"], "lines": card["lines"]}
        summary["report_path"] = render(summary)
        path = save_local(card)
        print(f"[scorecard] {card['month']} 월간 성적표를 개인 보고서 접기에 넣었습니다 ({path})")
        notify("scorecard_done", {"month": card["month"],
                                  "stop": [n for _, n, p in card["proposals"] if p == "중단 제안"],
                                  "review": [n for _, n, p in card["proposals"] if p == "실제 돈 일부 검토"]})
    except Exception as exc:  # noqa: BLE001 — 성적표 실패가 브리핑을 막으면 안 된다
        print(f"[scorecard] 월간 성적표를 넣지 못했습니다({stage}): {type(exc).__name__}: {exc}")
        summary.pop("scorecard", None)
        summary.pop("group_report_path", None)
        summary["report_path"] = original
        try:
            notify("error", {"stage": stage})
        except Exception:  # noqa: BLE001
            pass
