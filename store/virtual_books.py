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


# ── 추천 따라하기 장부 (core/picks_book.py) — 하루씩 이어 가므로 표를 통째로 바꾸지 않고 쌓는다 ──────────

import json  # noqa: E402

import pandas as pd  # noqa: E402

_PICKS_DAILY_COLS = [
    "book", "date", "holding", "qty", "cash", "value", "cum_return_pct", "drawdown_pct", "realized_ytd",
    "tax_cum", "value_after_tax_liq", "dividends_cum", "sgov_value",
]
_PICKS_TRADE_COLS = [
    "book", "signal_date", "fill_date", "side", "ticker", "qty", "price", "commission", "amount", "realized_pnl",
    "reason", "kind", "in_briefing", "entry_date",
]
_PICKS_SIGNAL_COLS = ["book", "date", "ticker", "kind", "stage", "score", "note"]


def _ensure_picks_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS picks_state (book TEXT PRIMARY KEY, state_json TEXT, last_date TEXT);
        CREATE TABLE IF NOT EXISTS picks_daily (
            book TEXT, date TEXT, holding TEXT, qty INTEGER, cash REAL, value REAL, cum_return_pct REAL,
            drawdown_pct REAL, realized_ytd REAL, tax_cum REAL, value_after_tax_liq REAL, dividends_cum REAL,
            sgov_value REAL, PRIMARY KEY (book, date)
        );
        CREATE TABLE IF NOT EXISTS picks_trades (
            book TEXT, signal_date TEXT, fill_date TEXT, side TEXT, ticker TEXT, qty INTEGER, price REAL,
            commission REAL, amount REAL, realized_pnl REAL, reason TEXT, kind TEXT, in_briefing INTEGER,
            entry_date TEXT, PRIMARY KEY (book, fill_date, ticker, side, kind)
        );
        CREATE TABLE IF NOT EXISTS picks_signals (
            book TEXT, date TEXT, ticker TEXT, kind TEXT, stage TEXT, score REAL, note TEXT,
            PRIMARY KEY (book, date, ticker, kind, stage)
        );
        CREATE TABLE IF NOT EXISTS rec_log (date TEXT, side TEXT, ticker TEXT, PRIMARY KEY (date, side, ticker));
        CREATE TABLE IF NOT EXISTS universe_log (date TEXT PRIMARY KEY, tickers TEXT);
        """
    )


def _enc(o):
    if isinstance(o, pd.Timestamp):
        return {"__ts__": o.isoformat()}
    if hasattr(o, "item"):  # numpy 숫자
        return o.item()
    raise TypeError(f"직렬화할 수 없는 값: {type(o)}")


def _dec(d: dict):
    return pd.Timestamp(d["__ts__"]) if set(d) == {"__ts__"} else d


def dump_book(book: dict) -> str:
    """장부 상태 dict → JSON 문자열 (Timestamp 보존)."""
    return json.dumps(book, default=_enc, ensure_ascii=False)


def load_book_json(text: str) -> dict:
    """JSON 문자열 → 장부 상태 dict (Timestamp 복원)."""
    return json.loads(text, object_hook=_dec)


def load_picks_book(conn: sqlite3.Connection, book: str) -> dict | None:
    """저장된 장부 상태 (없으면 None)."""
    _ensure_picks_schema(conn)
    row = conn.execute("SELECT state_json FROM picks_state WHERE book = ?", (book,)).fetchone()
    return load_book_json(row[0]) if row else None


def _cell(v):
    if isinstance(v, pd.Timestamp):
        return v.date().isoformat()
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float) and v != v:  # NaN
        return None
    return v.item() if hasattr(v, "item") else v


def save_picks_day(conn: sqlite3.Connection, book: dict, daily_rows: list[dict], trades: list[dict],
                   signals: list[dict]) -> None:
    """장부 상태 덮어쓰기 + 일별·매매·신호 행 쌓기(같은 키면 바꿔 쓴다) — 한 트랜잭션."""
    _ensure_picks_schema(conn)
    with conn:
        conn.execute("INSERT OR REPLACE INTO picks_state (book, state_json, last_date) VALUES (?, ?, ?)",
                     (book["name"], dump_book(book), _cell(book.get("last_date"))))
        for table, cols, rows in (("picks_daily", _PICKS_DAILY_COLS, daily_rows),
                                  ("picks_trades", _PICKS_TRADE_COLS, trades),
                                  ("picks_signals", _PICKS_SIGNAL_COLS, signals)):
            conn.executemany(
                f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [tuple(_cell(r.get(c)) if c != "stage" else (r.get(c) or "") for c in cols) for r in rows],
            )


def picks_rows(conn: sqlite3.Connection, table: str, book: str) -> list[dict]:
    """picks_daily|picks_trades|picks_signals 의 한 장부 행 (날짜 오름차순)."""
    cols = {"picks_daily": _PICKS_DAILY_COLS, "picks_trades": _PICKS_TRADE_COLS, "picks_signals": _PICKS_SIGNAL_COLS}[table]
    _ensure_picks_schema(conn)
    order = {"picks_daily": "date", "picks_trades": "fill_date, rowid", "picks_signals": "date, ticker"}[table]
    cur = conn.execute(f"SELECT {', '.join(cols)} FROM {table} WHERE book = ? ORDER BY {order}", (book,))
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def save_rec_log(conn: sqlite3.Connection, date: str, buys: list[str], sells: list[str]) -> None:
    """그날 브리핑 매수·매도 추천 티커 (같은 날 다시 실행하면 바꿔 쓴다)."""
    _ensure_picks_schema(conn)
    with conn:
        conn.execute("DELETE FROM rec_log WHERE date = ?", (date,))
        conn.executemany("INSERT OR IGNORE INTO rec_log (date, side, ticker) VALUES (?, ?, ?)",
                         [(date, "day", "")] + [(date, "buy", t) for t in buys] + [(date, "sell", t) for t in sells])


def save_universe(conn: sqlite3.Connection, date: str, tickers: list[str]) -> None:
    """그날 스크리너 명단(나스닥 100)을 남긴다 — 따라잡기 때 지난 날짜의 명단으로 쓴다(같은 날 다시 쓰면 바꿔 쓴다)."""
    _ensure_picks_schema(conn)
    with conn:
        conn.execute("INSERT OR REPLACE INTO universe_log (date, tickers) VALUES (?, ?)", (date, ",".join(sorted(set(tickers)))))


def universe_on(conn: sqlite3.Connection, date: str) -> tuple[str, list[str]] | None:
    """date 이하로 가장 최근에 남긴 명단 (기록 날짜, 티커 목록). 없으면 None."""
    _ensure_picks_schema(conn)
    row = conn.execute("SELECT date, tickers FROM universe_log WHERE date <= ? ORDER BY date DESC LIMIT 1", (date,)).fetchone()
    return (row[0], [t for t in row[1].split(",") if t]) if row else None


def load_rec_log(conn: sqlite3.Connection) -> dict[str, dict]:
    """{날짜: {"buy": set, "sell": set}} — 기록한 날짜만(추천 0건인 날도 "day" 표시 행으로 남는다)."""
    _ensure_picks_schema(conn)
    out: dict[str, dict] = {}
    for d, side, t in conn.execute("SELECT date, side, ticker FROM rec_log").fetchall():
        rec = out.setdefault(d, {"buy": set(), "sell": set()})
        if side in rec:
            rec[side].add(t)
    return out


# ── 8번 장부 「가치평가 따라하기」 월말 점수표·SIC (core/value_book.py, engine/value_book.py) ─────────


def _ensure_value_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS value_scores (book TEXT, date TEXT, scores_json TEXT, PRIMARY KEY (book, date));
        """
    )


def save_value_scores(conn: sqlite3.Connection, book: str, date: str, scores: pd.DataFrame) -> None:
    """그 월말 점수표(index=티커)를 남긴다 — 다음 실행·12월 실행일이 같은 점수를 다시 쓴다(같은 키면 바꿔 쓴다)."""
    _ensure_value_schema(conn)
    with conn:
        conn.execute("INSERT OR REPLACE INTO value_scores (book, date, scores_json) VALUES (?, ?, ?)",
                     (book, date, scores.to_json(orient="split")))


def load_value_scores(conn: sqlite3.Connection, book: str, date: str) -> pd.DataFrame | None:
    """저장된 그 월말 점수표 (없으면 None)."""
    import io

    _ensure_value_schema(conn)
    row = conn.execute("SELECT scores_json FROM value_scores WHERE book = ? AND date = ?", (book, date)).fetchone()
    return pd.read_json(io.StringIO(row[0]), orient="split") if row else None


def latest_value_scores_date(conn: sqlite3.Connection, book: str) -> str | None:
    _ensure_value_schema(conn)
    row = conn.execute("SELECT MAX(date) FROM value_scores WHERE book = ?", (book,)).fetchone()
    return row[0] if row and row[0] else None


def load_sic_cache(conn: sqlite3.Connection) -> dict:
    """{티커: SIC} — SEC submissions에서 한 번 받은 값(meta 표 value_sic_cache)."""
    raw = get_meta(conn, "value_sic_cache")
    return json.loads(raw) if raw else {}


def save_sic_cache(conn: sqlite3.Connection, cache: dict) -> None:
    set_meta(conn, "value_sic_cache", json.dumps(cache, ensure_ascii=False))


# ── 10번 동전주 장부 선정 기록 (종목명은 저장소에 커밋하지 않는다 — 이 DB와 시트에만) ─────────────────

_PENNY_PICK_COLS = ["book", "date", "rank", "ticker", "ratio", "width", "close", "dollar_volume"]


def _ensure_penny_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS penny_picks (
            book TEXT, date TEXT, rank INTEGER, ticker TEXT, ratio REAL, width REAL, close REAL, dollar_volume REAL,
            PRIMARY KEY (book, date, ticker)
        );
        """
    )


def save_penny_picks(conn: sqlite3.Connection, book: str, date: str, rows: list[dict]) -> None:
    """그날 선정 명단(순위·수렴 비율·가격 폭·P일 종가·거래대금). 같은 날 다시 쓰면 바꿔 쓴다."""
    _ensure_penny_schema(conn)
    with conn:
        conn.execute("DELETE FROM penny_picks WHERE book = ? AND date = ?", (book, date))
        conn.executemany(f"INSERT INTO penny_picks ({', '.join(_PENNY_PICK_COLS)}) VALUES ({', '.join('?' * len(_PENNY_PICK_COLS))})",
                         [tuple(_cell({**r, "book": book, "date": date}.get(c)) for c in _PENNY_PICK_COLS) for r in rows])


def penny_picks(conn: sqlite3.Connection, book: str) -> list[dict]:
    """선정 기록 전체 (날짜·순위 순)."""
    _ensure_penny_schema(conn)
    cur = conn.execute(f"SELECT {', '.join(_PENNY_PICK_COLS)} FROM penny_picks WHERE book = ? ORDER BY date, rank", (book,))
    return [dict(zip(_PENNY_PICK_COLS, r)) for r in cur.fetchall()]
