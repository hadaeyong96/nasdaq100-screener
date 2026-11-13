"""실적급등 매도 조건 비교(pead_exits) 기록 — 가상 장부 DB(virtual_books.db) (docs/prereg/pead_exits.md 8장).

표: pead_exit_a_trades · pead_exit_b_trades(V1 매매 칸 + 조건 칸) · pead_exits_plan(예정일·출처·꼬리표, 처음 값 고정)
    · pead_exits_portfolio · pead_exits_pf_events(15칸 V1·A·B). 메타: pead_exits_start, pead_exits_pf_state_<장부>.
V1 표(pead_watch_*)는 건드리지 않는다.
"""

from __future__ import annotations

import json
import sqlite3

from store import pead_watch as pwst

META_START = "pead_exits_start"
TABLES = {"exit_a": "pead_exit_a_trades", "exit_b": "pead_exit_b_trades"}
BOOKS = ("main", "exit_a", "exit_b")
EXIT_COLS = pwst.TRADE_COLS + ("exit_kind", "sell_at", "next_date", "next_source", "target_px", "hit_day", "hit_days")
PLAN_COLS = ("ticker", "d0", "next_date", "source", "n_gaps", "yf_dates", "fetched_as_of", "fetched_kst", "guid_raise",
             "guid_text", "prev_below_sma200", "errors")
_REAL = {"buy_px", "q_buy", "sell_px", "q_sell", "fx_buy", "fx_sell", "target_px"}


def pf_state_key(book: str) -> str:
    return f"pead_exits_pf_state_{book}"


def ensure_schema(conn: sqlite3.Connection) -> None:
    cols = ", ".join(f"{c} {'REAL' if c in _REAL else 'INTEGER' if c == 'hit_days' else 'TEXT'}" for c in EXIT_COLS)
    for t in TABLES.values():
        conn.execute(f"CREATE TABLE IF NOT EXISTS {t} ({cols}, PRIMARY KEY (ticker, d0))")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS pead_exits_plan (ticker TEXT, d0 TEXT, next_date TEXT, source TEXT, n_gaps INTEGER,
                                                    yf_dates TEXT, fetched_as_of TEXT, fetched_kst TEXT, guid_raise INTEGER,
                                                    guid_text TEXT, prev_below_sma200 INTEGER, errors TEXT,
                                                    PRIMARY KEY (ticker, d0));
        CREATE TABLE IF NOT EXISTS pead_exits_portfolio (book TEXT, date TEXT, value_usd REAL, qqq100_usd REAL,
                                                         vs_qqq_pct REAL, fx REAL, value_krw REAL, qqq100_krw REAL,
                                                         row_json TEXT, PRIMARY KEY (book, date));
        CREATE TABLE IF NOT EXISTS pead_exits_reports (milestone TEXT PRIMARY KEY, as_of TEXT, counts TEXT, report_json TEXT);
        CREATE TABLE IF NOT EXISTS pead_exits_pf_events (book TEXT, date TEXT, ticker TEXT, d0 TEXT, action TEXT,
                                                         shares INTEGER, px REAL, amount_usd REAL, note TEXT,
                                                         PRIMARY KEY (book, date, ticker, d0, action));
    """)
    conn.commit()


def load_trades(conn: sqlite3.Connection, rule: str) -> list[dict]:
    cur = conn.execute(f"SELECT {', '.join(EXIT_COLS)} FROM {TABLES[rule]} ORDER BY d0, ticker")
    return [dict(zip(EXIT_COLS, r)) for r in cur]


def replace_trades(conn: sqlite3.Connection, rule: str, trades: list[dict]) -> None:
    conn.execute(f"DELETE FROM {TABLES[rule]}")
    conn.executemany(f"INSERT INTO {TABLES[rule]} VALUES ({', '.join('?' * len(EXIT_COLS))})",
                     [tuple(t.get(c) for c in EXIT_COLS) for t in trades])


def load_plans(conn: sqlite3.Connection) -> dict[tuple[str, str], dict]:
    cur = conn.execute(f"SELECT {', '.join(PLAN_COLS)} FROM pead_exits_plan")
    return {(r[0], r[1]): dict(zip(PLAN_COLS, r)) for r in cur}


def put_plans(conn: sqlite3.Connection, rows: list[dict]) -> None:
    """예정일은 처음 값 고정 — 이미 있는 (종목, 반응일)은 덮지 않는다."""
    conn.executemany(f"INSERT OR IGNORE INTO pead_exits_plan VALUES ({', '.join('?' * len(PLAN_COLS))})",
                     [tuple(r.get(c) for c in PLAN_COLS) for r in rows])


def load_reports(conn: sqlite3.Connection) -> dict[str, dict]:
    return {r[0]: {"as_of": r[1], "counts": json.loads(r[2]), "report": json.loads(r[3])}
            for r in conn.execute("SELECT milestone, as_of, counts, report_json FROM pead_exits_reports")}


def put_report(conn: sqlite3.Connection, milestone: str, as_of: str, counts: dict, report: dict) -> None:
    """성적 시점마다 한 번만(사전 등록 7장) — 이미 있으면 덮지 않는다."""
    conn.execute("INSERT OR IGNORE INTO pead_exits_reports VALUES (?, ?, ?, ?)",
                 (milestone, as_of, json.dumps(counts), json.dumps(report, ensure_ascii=False, default=str)))


def put_portfolio_rows(conn: sqlite3.Connection, book: str, rows: list[dict]) -> None:
    conn.executemany("INSERT OR REPLACE INTO pead_exits_portfolio VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     [(book, r["date"], r["value_usd"], r["qqq100_usd"], r["vs_qqq_pct"], r.get("fx"), r.get("value_krw"),
                       r.get("qqq100_krw"), json.dumps(r, ensure_ascii=False)) for r in rows])


def load_portfolio(conn: sqlite3.Connection, book: str) -> list[dict]:
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT row_json FROM pead_exits_portfolio WHERE book = ? ORDER BY date", (book,))]


def put_pf_events(conn: sqlite3.Connection, book: str, rows: list[dict]) -> None:
    conn.executemany(f"INSERT OR REPLACE INTO pead_exits_pf_events VALUES ({', '.join('?' * (len(pwst.PF_EVENT_COLS) + 1))})",
                     [(book,) + tuple(r.get(c) for c in pwst.PF_EVENT_COLS) for r in rows])


def load_pf_events(conn: sqlite3.Connection, book: str) -> list[dict]:
    cur = conn.execute(f"SELECT {', '.join(pwst.PF_EVENT_COLS)} FROM pead_exits_pf_events WHERE book = ? ORDER BY date, ticker",
                       (book,))
    return [dict(zip(pwst.PF_EVENT_COLS, r)) for r in cur]
