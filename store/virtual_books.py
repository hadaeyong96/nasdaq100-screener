"""가상 장부 저장소 (SQLite, data/virtual_books.db) — docs/design/virtual_books.md.

live·paper DB(store/db.py)와 완전히 분리한 파일이다. GitHub Actions에서는 드라이브 state
폴더의 virtual_books.db를 scripts/drive_sync.py(--mode vbooks)로 내려받고 다시 올린다.

장부는 매 실행마다 시작일부터 다시 계산한다(core/virtual_books.py — 같은 입력이면 같은 결과).
그래서 일별·체결 표는 실행마다 통째로 바꿔 쓰고(같은 날 두 번 실행해도 중복 행이 생기지 않음),
실행 기록(runs)은 run_id·기준일로 남긴다. 알림(전환 신호·낙폭)은 alerts 표에 (종류, 키)로
기록해 한 번만 보낸다 — 실제로 보낸 뒤에만 기록한다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VBOOKS_DB_PATH = ROOT / "data" / "virtual_books.db"

_DAILY_COLS = [
    "book", "date", "holding", "qty", "close", "cash", "value", "cum_return_pct", "drawdown_pct",
    "realized_ytd", "tax_cum", "value_after_tax_liq", "dividends_cum",
]
_TRADE_COLS = [
    "book", "signal_date", "fill_date", "side", "ticker", "qty", "price", "commission", "amount",
    "realized_pnl", "reason",
]


def connect(db_path: Path = VBOOKS_DB_PATH) -> sqlite3.Connection:
    """DB에 연결하고 테이블이 없으면 만든다."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, base_date TEXT, run_at TEXT, status TEXT, error TEXT
        );
        CREATE TABLE IF NOT EXISTS daily (
            book TEXT, date TEXT, holding TEXT, qty INTEGER, close REAL, cash REAL, value REAL,
            cum_return_pct REAL, drawdown_pct REAL, realized_ytd REAL, tax_cum REAL,
            value_after_tax_liq REAL, dividends_cum REAL,
            PRIMARY KEY (book, date)
        );
        CREATE TABLE IF NOT EXISTS trades (
            book TEXT, signal_date TEXT, fill_date TEXT, side TEXT, ticker TEXT, qty INTEGER,
            price REAL, commission REAL, amount REAL, realized_pnl REAL, reason TEXT,
            PRIMARY KEY (book, fill_date, side, ticker)
        );
        CREATE TABLE IF NOT EXISTS alerts (kind TEXT, key TEXT, sent_at TEXT, PRIMARY KEY (kind, key));
        """
    )
    return conn


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    """meta 값 (없으면 None)."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """meta 값을 쓴다(덮어쓰기)."""
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
    conn.commit()


def record_run(conn: sqlite3.Connection, run_id: str, base_date: str, run_at: str, status: str, error: str = "") -> None:
    """실행 기록 한 줄 (run_id가 같으면 상태만 갱신)."""
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, base_date, run_at, status, error) VALUES (?, ?, ?, ?, ?)",
        (run_id, base_date, run_at, status, error),
    )
    conn.commit()


def runs_for_base_date(conn: sqlite3.Connection, base_date: str) -> list[dict]:
    """같은 기준일의 실행 기록 목록."""
    cur = conn.execute("SELECT run_id, base_date, run_at, status, error FROM runs WHERE base_date = ?", (base_date,))
    return [dict(zip(["run_id", "base_date", "run_at", "status", "error"], r)) for r in cur.fetchall()]


def _iso(v):
    return v.date().isoformat() if hasattr(v, "date") and callable(v.date) else v


def replace_ledger(conn: sqlite3.Connection, daily_rows: list[dict], trades: list[dict]) -> None:
    """일별·체결 표를 통째로 바꾼다(한 트랜잭션). 같은 날 다시 실행해도 행이 늘지 않는다."""
    with conn:
        conn.execute("DELETE FROM daily")
        conn.execute("DELETE FROM trades")
        conn.executemany(
            f"INSERT INTO daily ({', '.join(_DAILY_COLS)}) VALUES ({', '.join('?' * len(_DAILY_COLS))})",
            [tuple(_iso(r[c]) for c in _DAILY_COLS) for r in daily_rows],
        )
        conn.executemany(
            f"INSERT INTO trades ({', '.join(_TRADE_COLS)}) VALUES ({', '.join('?' * len(_TRADE_COLS))})",
            [tuple(_iso(t[c]) for c in _TRADE_COLS) for t in trades],
        )


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    """표 행 수 (테스트·점검용). table은 daily|trades|runs|alerts."""
    if table not in ("daily", "trades", "runs", "alerts"):
        raise ValueError(table)
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def has_alert(conn: sqlite3.Connection, kind: str, key: str) -> bool:
    """이미 보낸 알림인지."""
    return conn.execute("SELECT 1 FROM alerts WHERE kind = ? AND key = ?", (kind, key)).fetchone() is not None


def record_alert(conn: sqlite3.Connection, kind: str, key: str, sent_at: str) -> None:
    """보낸 알림 기록 (실제로 보낸 뒤에만 부른다)."""
    conn.execute("INSERT OR IGNORE INTO alerts (kind, key, sent_at) VALUES (?, ?, ?)", (kind, key, sent_at))
    conn.commit()
