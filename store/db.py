"""8장 저장소(SQLite): 종목별 상태, 이벤트 기록, 실행 로그.

파일: data/state.db (live 모드), data/paper_state.db (paper 모드). 둘 다
.gitignore의 *.db로 이미 제외한다. 두 모드는 절대 같은 DB 파일을 쓰지 않는다
(P3 — 운용 모드 정리).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "state.db"
PAPER_DB_PATH = ROOT / "data" / "paper_state.db"


def db_path_for_mode(mode: str) -> Path:
    """운용 모드에 맞는 DB 파일 경로. paper는 별도 파일로 완전히 분리한다."""
    return PAPER_DB_PATH if mode == "paper" else DB_PATH

_POSITION_FIELDS = [
    "ticker",
    "name_kr",
    "state",
    "units",
    "entries",
    "stop",
    "a1_date",
    "cooldown_until",
    "sent_alerts",
    "updated_at",
    "b_entry_date",
    "b_total_qty",
    "pending",
    "last_judgment",
]
_JSON_FIELDS = {"units", "entries", "sent_alerts", "pending"}
_DATE_FIELDS = {"a1_date", "cooldown_until", "updated_at", "b_entry_date"}


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """DB에 연결하고 테이블이 없으면 만든다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            ticker TEXT PRIMARY KEY,
            name_kr TEXT,
            state TEXT,
            units TEXT,
            entries TEXT,
            stop REAL,
            a1_date TEXT,
            cooldown_until TEXT,
            sent_alerts TEXT,
            updated_at TEXT,
            b_entry_date TEXT,
            b_total_qty INTEGER,
            pending TEXT,
            last_judgment TEXT
        )
        """
    )
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
    if "pending" not in existing_columns:  # 기존 DB(P3.1 이전) 마이그레이션
        conn.execute("ALTER TABLE positions ADD COLUMN pending TEXT")
    if "last_judgment" not in existing_columns:  # 기존 DB(라이브 어드바이저 1단계 이전) 마이그레이션
        conn.execute("ALTER TABLE positions ADD COLUMN last_judgment TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            ticker TEXT,
            kind TEXT,
            detail TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT,
            as_of_date TEXT,
            ticker_count INTEGER,
            warning_count INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT,
            ticker TEXT,
            date TEXT,
            close REAL,
            close_source TEXT,
            meta_time TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            as_of_date TEXT PRIMARY KEY,
            sent_at TEXT
        )
        """
    )
    # 라이브 어드바이저 1단계(docs/design/live_advisor.md 9번): 월간 비교(2단계)에
    # 쓸 데이터를 1단계부터 append-only로 쌓아 둔다. 기존 positions·events는 안 건드림.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fill_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            mode TEXT,
            ticker TEXT,
            side TEXT,
            qty INTEGER,
            price_usd REAL,
            fx_rate REAL,
            amount_krw REAL,
            qqqm_close REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_mark (
            date TEXT,
            mode TEXT,
            ticker TEXT,
            qty INTEGER,
            price_usd REAL,
            fx_rate REAL,
            PRIMARY KEY (date, mode, ticker)
        )
        """
    )
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """운용 시작일 등 이 DB 하나에 딸린 작은 키-값을 읽는다."""
    cur = conn.execute("SELECT value FROM meta WHERE key = ?", (key,))
    row = cur.fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def has_notified(conn: sqlite3.Connection, as_of_date: str) -> bool:
    """같은 기준일 텔레그램 발송 기록이 이미 있는지 본다 (중복 발송 방지)."""
    cur = conn.execute("SELECT 1 FROM notifications WHERE as_of_date = ?", (as_of_date,))
    return cur.fetchone() is not None


def record_notified(conn: sqlite3.Connection, as_of_date: str, sent_at: str) -> None:
    conn.execute(
        "INSERT INTO notifications (as_of_date, sent_at) VALUES (?, ?) "
        "ON CONFLICT(as_of_date) DO UPDATE SET sent_at=excluded.sent_at",
        (as_of_date, sent_at),
    )
    conn.commit()


def _serialize(state: dict) -> dict:
    row = {}
    for field in _POSITION_FIELDS:
        value = state.get(field)
        if field in _JSON_FIELDS:
            default = [] if field == "sent_alerts" else ({} if field != "pending" else None)
            row[field] = json.dumps(value if value is not None else default, default=str)
        elif field in _DATE_FIELDS:
            row[field] = None if value is None else str(value)
        else:
            row[field] = value
    return row


def _deserialize(row: dict) -> dict:
    state = {}
    for field in _POSITION_FIELDS:
        value = row[field]
        if field in _JSON_FIELDS:
            state[field] = json.loads(value) if value else ({} if field != "sent_alerts" else [])
        else:
            state[field] = value
    return state


def save_position(conn: sqlite3.Connection, state: dict) -> None:
    """종목 상태 하나를 upsert한다."""
    row = _serialize(state)
    columns = ", ".join(_POSITION_FIELDS)
    placeholders = ", ".join(f":{f}" for f in _POSITION_FIELDS)
    updates = ", ".join(f"{f}=excluded.{f}" for f in _POSITION_FIELDS if f != "ticker")
    conn.execute(
        f"INSERT INTO positions ({columns}) VALUES ({placeholders}) "
        f"ON CONFLICT(ticker) DO UPDATE SET {updates}",
        row,
    )
    conn.commit()


def load_position(conn: sqlite3.Connection, ticker: str) -> dict | None:
    cur = conn.execute(f"SELECT {', '.join(_POSITION_FIELDS)} FROM positions WHERE ticker = ?", (ticker,))
    row = cur.fetchone()
    if row is None:
        return None
    return _deserialize(dict(zip(_POSITION_FIELDS, row)))


def load_all_positions(conn: sqlite3.Connection) -> dict[str, dict]:
    cur = conn.execute(f"SELECT {', '.join(_POSITION_FIELDS)} FROM positions")
    return {row[0]: _deserialize(dict(zip(_POSITION_FIELDS, row))) for row in cur.fetchall()}


def get_events_for_date(conn: sqlite3.Connection, date_str: str) -> list[dict]:
    """그 날짜에 기록된 이벤트를 모두 읽는다 (상태를 다시 계산하지 않고 보고서만
    다시 만들 때 씀 — engine.daily.build_report_summary가 today_events로 받는
    형태와 같은 dict 목록을 돌려준다: {date, ticker, kind, ...detail}).

    record_events가 event["date"]를 str()로 그대로 저장해, pd.Timestamp였던 값은
    "2026-09-23 00:00:00"처럼 시각까지 붙어 저장되고 순수 날짜 문자열("FUNNEL"
    이벤트 등)은 "2026-09-23"로 저장된다 — 두 표기가 섞여 있어 SQLite의 date()로
    날짜 부분만 비교한다 (== 로는 시각이 붙은 쪽을 놓친다).
    FUNNEL 이벤트(ticker="")는 보고서 조립에 쓰이지 않으므로 뺀다.
    """
    cur = conn.execute(
        "SELECT date, ticker, kind, detail FROM events WHERE date(date) = ? AND kind != 'FUNNEL' ORDER BY id",
        (date_str,),
    )
    events = []
    seen: set[str] = set()
    for date, ticker, kind, detail in cur.fetchall():
        # 같은 날짜를 실수로 두 번 실제 처리했을 때(가드를 우회한 --replay 재실행 등)
        # 완전히 같은 내용의 행이 중복 기록될 수 있다 — 보고서에 두 번 나오지 않도록 접는다.
        # detail의 값(reasons 리스트 등)이 해시 불가능할 수 있어 JSON 문자열로 비교한다.
        dedup_key = f"{ticker}|{kind}|{detail}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        event = {"date": pd.Timestamp(date), "ticker": ticker, "kind": kind}
        event.update(json.loads(detail) if detail else {})
        events.append(event)
    return events


def record_events(conn: sqlite3.Connection, events: list[dict]) -> None:
    """이벤트 목록을 events 테이블에 남긴다 (브리핑·백테스트 검증용, P3~)."""
    for event in events:
        date = str(event.get("date"))
        ticker = event.get("ticker", "")
        kind = event.get("kind", "")
        detail = json.dumps({k: v for k, v in event.items() if k not in ("date", "ticker", "kind")}, default=str)
        conn.execute(
            "INSERT INTO events (date, ticker, kind, detail) VALUES (?, ?, ?, ?)",
            (date, ticker, kind, detail),
        )
    conn.commit()


def delete_events_since(conn: sqlite3.Connection, date_str: str) -> None:
    """date_str(YYYY-MM-DD) 이후 날짜의 이벤트를 지운다 — 체결 기록이 바뀌어 그 날짜부터
    상태를 다시 계산할 때 같은 이벤트가 두 번 쌓이지 않게 (engine.daily 라이브 재계산)."""
    conn.execute("DELETE FROM events WHERE substr(date, 1, 10) >= ?", (date_str,))
    conn.commit()


def record_price_snapshots(conn: sqlite3.Connection, run_at: str, rows: list[dict]) -> None:
    """실행마다 종목별 (close, close_source, meta_time)을 남긴다 (재현성 확인용, P2.1 보완 3번).

    입력: run_at(실행 시각), rows({ticker, date, close, close_source, meta_time})
    """
    for row in rows:
        conn.execute(
            "INSERT INTO price_snapshots (run_at, ticker, date, close, close_source, meta_time) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_at, row["ticker"], row["date"], row.get("close"), row.get("close_source"), row.get("meta_time")),
        )
    conn.commit()


def record_run(conn: sqlite3.Connection, run_at: str, as_of_date: str, ticker_count: int, warning_count: int) -> None:
    conn.execute(
        "INSERT INTO runs (run_at, as_of_date, ticker_count, warning_count) VALUES (?, ?, ?, ?)",
        (run_at, as_of_date, ticker_count, warning_count),
    )
    conn.commit()


_FILL_LEDGER_FIELDS = ["date", "mode", "ticker", "side", "qty", "price_usd", "fx_rate", "amount_krw", "qqqm_close"]
_DAILY_MARK_FIELDS = ["date", "mode", "ticker", "qty", "price_usd", "fx_rate"]


def record_fill_ledger(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """체결/판정 이벤트마다 한 줄을 append한다 (실제 체결 live, 가상 체결 paper
    둘 다 여기 쌓는다 — docs/design/live_advisor.md 9번, 2단계 월간 비교용).

    입력: rows([{date, mode(live|paper), ticker, side(buy|sell), qty, price_usd,
         fx_rate, amount_krw, qqqm_close(그날 QQQM 종가, 그림자 계산용 — 없으면
         None)}, ...])
    """
    for r in rows:
        row = {f: r.get(f) for f in _FILL_LEDGER_FIELDS}
        conn.execute(
            "INSERT INTO fill_ledger (date, mode, ticker, side, qty, price_usd, fx_rate, amount_krw, qqqm_close) "
            "VALUES (:date, :mode, :ticker, :side, :qty, :price_usd, :fx_rate, :amount_krw, :qqqm_close)",
            row,
        )
    conn.commit()


def delete_fill_ledger(conn: sqlite3.Connection, mode: str) -> None:
    """그 모드의 fill_ledger를 모두 지운다 — 체결 기록 소급 반영으로 상태를 다시 계산할 때
    반영 목록을 새로 쓰기 위해 (engine.daily)."""
    conn.execute("DELETE FROM fill_ledger WHERE mode = ?", (mode,))
    conn.commit()


def get_fill_ledger(conn: sqlite3.Connection, mode: str | None = None) -> list[dict]:
    """fill_ledger를 id 순서(기록 순)로 읽는다. mode를 주면 그 모드만."""
    if mode is None:
        cur = conn.execute(f"SELECT {', '.join(_FILL_LEDGER_FIELDS)} FROM fill_ledger ORDER BY id")
    else:
        cur = conn.execute(
            f"SELECT {', '.join(_FILL_LEDGER_FIELDS)} FROM fill_ledger WHERE mode = ? ORDER BY id", (mode,)
        )
    return [dict(zip(_FILL_LEDGER_FIELDS, row)) for row in cur.fetchall()]


def record_daily_mark(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """보유 중인 종목마다 매일 한 줄을 남긴다 (같은 date+mode+ticker로 다시
    실행하면 덮어쓴다 — docs/design/live_advisor.md 9번). 새 체결이 없는 날에도
    매일 찍어 수익률 곡선을 거래일 사이에도 매끈하게 그릴 수 있게 한다.

    입력: rows([{date, mode(live|paper), ticker, qty, price_usd, fx_rate}, ...])
    """
    for r in rows:
        row = {f: r.get(f) for f in _DAILY_MARK_FIELDS}
        conn.execute(
            "INSERT INTO daily_mark (date, mode, ticker, qty, price_usd, fx_rate) "
            "VALUES (:date, :mode, :ticker, :qty, :price_usd, :fx_rate) "
            "ON CONFLICT(date, mode, ticker) DO UPDATE SET "
            "qty=excluded.qty, price_usd=excluded.price_usd, fx_rate=excluded.fx_rate",
            row,
        )
    conn.commit()


def get_daily_mark(conn: sqlite3.Connection, mode: str | None = None) -> list[dict]:
    """daily_mark를 (date, ticker) 순으로 읽는다. mode를 주면 그 모드만."""
    if mode is None:
        cur = conn.execute(f"SELECT {', '.join(_DAILY_MARK_FIELDS)} FROM daily_mark ORDER BY date, ticker")
    else:
        cur = conn.execute(
            f"SELECT {', '.join(_DAILY_MARK_FIELDS)} FROM daily_mark WHERE mode = ? ORDER BY date, ticker", (mode,)
        )
    return [dict(zip(_DAILY_MARK_FIELDS, row)) for row in cur.fetchall()]
