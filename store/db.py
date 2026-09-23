"""8장 저장소(SQLite): 종목별 상태, 이벤트 기록, 실행 로그.

파일: data/state.db (live 모드), data/paper_state.db (paper 모드). 둘 다
.gitignore의 *.db로 이미 제외한다. 두 모드는 절대 같은 DB 파일을 쓰지 않는다
(P3 — 운용 모드 정리).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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
]
_JSON_FIELDS = {"units", "entries", "sent_alerts"}
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
            b_total_qty INTEGER
        )
        """
    )
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
            row[field] = json.dumps(value if value is not None else ({} if field != "sent_alerts" else []))
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
