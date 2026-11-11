"""실적급등 추천 장부(pead_watch) 기록 — 가상 장부 DB(virtual_books.db)의 pead_watch_* 표 (docs/prereg/pead_watch.md 8장).

표: members(달 말 명단) · filings(색인 Accepted 캐시·반응일) · signals(판정 기록, 판정 불가 포함) · trades(장부별 매매) · daily(실행별 요약).
메타: pead_watch_start(시작일, 첫 정규 실행이 고정), pead_watch_last(마지막으로 진행한 기준일).
"""

from __future__ import annotations

import json
import sqlite3

from store import virtual_books as vbstore

META_START, META_LAST = "pead_watch_start", "pead_watch_last"
TRADE_COLS = ("ledger", "ticker", "d0", "label", "status", "buy_day", "plan_day", "buy_px", "q_buy", "sell_day", "sell_px",
              "q_sell", "fx_buy", "fx_sell", "skip")
SIGNAL_COLS = ("ticker", "d0", "accn", "label", "accepted_et", "ret", "excess", "vol_ratio", "signal", "status", "judged_as_of")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pead_watch_members (month_end TEXT, idx TEXT, tickers TEXT, source TEXT,
                                                       PRIMARY KEY (month_end, idx));
        CREATE TABLE IF NOT EXISTS pead_watch_filings (accn TEXT PRIMARY KEY, ticker TEXT, cik INTEGER, filing_date TEXT,
                                                       accepted_et TEXT, d0 TEXT);
        CREATE TABLE IF NOT EXISTS pead_watch_signals (ticker TEXT, d0 TEXT, accn TEXT, label TEXT, accepted_et TEXT,
                                                       ret REAL, excess REAL, vol_ratio REAL, signal INTEGER, status TEXT,
                                                       judged_as_of TEXT, PRIMARY KEY (ticker, d0));
        CREATE TABLE IF NOT EXISTS pead_watch_trades (ledger TEXT, ticker TEXT, d0 TEXT, label TEXT, status TEXT,
                                                      buy_day TEXT, plan_day TEXT, buy_px REAL, q_buy REAL, sell_day TEXT,
                                                      sell_px REAL, q_sell REAL, fx_buy REAL, fx_sell REAL, skip TEXT,
                                                      PRIMARY KEY (ledger, ticker, d0));
        CREATE TABLE IF NOT EXISTS pead_watch_daily (date TEXT PRIMARY KEY, row_json TEXT);
    """)
    conn.commit()


def get_members(conn: sqlite3.Connection, month_end: str) -> dict[str, set[str]] | None:
    """달 말 명단 {'ndx': 집합, 'sp100': 집합} — 둘 다 있어야 돌려준다."""
    rows = {r[0]: set(json.loads(r[1])) for r in conn.execute(
        "SELECT idx, tickers FROM pead_watch_members WHERE month_end = ?", (month_end,))}
    return rows if {"ndx", "sp100"} <= set(rows) else None


def put_members(conn: sqlite3.Connection, month_end: str, idx: str, tickers: set[str], source: str) -> None:
    conn.execute("INSERT OR REPLACE INTO pead_watch_members VALUES (?, ?, ?, ?)",
                 (month_end, idx, json.dumps(sorted(tickers)), source))


def get_accepted(conn: sqlite3.Connection, accn: str) -> str | None:
    r = conn.execute("SELECT accepted_et FROM pead_watch_filings WHERE accn = ?", (accn,)).fetchone()
    return r[0] if r else None


def put_filing(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO pead_watch_filings VALUES (?, ?, ?, ?, ?, ?)",
                 (row["accn"], row["ticker"], int(row["cik"]), row["filing_date"], row["accepted_et"], row.get("d0")))


def load_trades(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(f"SELECT {', '.join(TRADE_COLS)} FROM pead_watch_trades ORDER BY ledger, d0, ticker")
    return [dict(zip(TRADE_COLS, r)) for r in cur]


def replace_trades(conn: sqlite3.Connection, trades: list[dict]) -> None:
    """장부 매매 전체를 이번 실행 결과로 바꾼다(끝난 매매는 이전 값과 같아야 한다 — 엔진이 대조)."""
    conn.execute("DELETE FROM pead_watch_trades")
    conn.executemany(f"INSERT INTO pead_watch_trades VALUES ({', '.join('?' * len(TRADE_COLS))})",
                     [tuple(t.get(c) for c in TRADE_COLS) for t in trades])


def load_signals(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(f"SELECT {', '.join(SIGNAL_COLS)} FROM pead_watch_signals ORDER BY d0, ticker")
    return [dict(zip(SIGNAL_COLS, r)) for r in cur]


def put_signals(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """판정 기록은 한 번만 — 이미 있는 (종목, 반응일)은 덮지 않는다."""
    conn.executemany(f"INSERT OR IGNORE INTO pead_watch_signals VALUES ({', '.join('?' * len(SIGNAL_COLS))})",
                     [tuple(r.get(c) for c in SIGNAL_COLS) for r in rows])


def put_daily(conn: sqlite3.Connection, date: str, row: dict) -> None:
    conn.execute("INSERT OR REPLACE INTO pead_watch_daily VALUES (?, ?)", (date, json.dumps(row, ensure_ascii=False)))


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    return vbstore.get_meta(conn, key)


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    vbstore.set_meta(conn, key, value)
