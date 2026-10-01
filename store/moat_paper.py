"""해자 paper 월말 기록 저장소 (docs/design/moat_paper.md, scripts/moat_paper.py·moat_alert.py 전용).

store/trials.py와 같은 관례(append 위주, sqlite3). 두 테이블:
- monthly_records: P1·P2·QQQM·QQEW·무작위 각 포트폴리오의 월말 평가 배수(세전, 마크투마켓)
- grade_history: 보유 종목의 월별 해자 등급 재계산 기록(core/moat_alert.grade_alert가
  "직전 등급"을 찾는 데 쓴다)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "store" / "moat_paper.db"


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """DB에 연결하고 테이블이 없으면 만든다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS monthly_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            period TEXT NOT NULL,
            portfolio TEXT NOT NULL,
            value_factor REAL,
            status TEXT NOT NULL,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS grade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT NOT NULL,
            period TEXT NOT NULL,
            ticker TEXT NOT NULL,
            grade TEXT NOT NULL
        )
        """
    )
    conn.commit()


def record_monthly(
    conn: sqlite3.Connection, recorded_at: str, period: str, portfolio: str,
    value_factor: float | None, status: str, notes: str | None = None,
) -> int:
    """포트폴리오 하나의 월말 평가를 기록한다 (append-only).

    입력: recorded_at(실행 시각 ISO), period("YYYY-MM"), portfolio("P1"|"P2"|"QQQM"|"QQEW"|
         "RANDOM_MEDIAN" 등), value_factor(진입 대비 배수, 진입 대기 중이면 None),
         status("진입 대기"|"평가"), notes
    출력: 새로 만들어진 행 id
    """
    cur = conn.execute(
        "INSERT INTO monthly_records (recorded_at, period, portfolio, value_factor, status, notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (recorded_at, period, portfolio, value_factor, status, notes),
    )
    conn.commit()
    return cur.lastrowid


def record_grade(conn: sqlite3.Connection, recorded_at: str, period: str, ticker: str, grade: str) -> int:
    """종목 하나의 그 달 해자 등급을 기록한다 (append-only)."""
    cur = conn.execute(
        "INSERT INTO grade_history (recorded_at, period, ticker, grade) VALUES (?, ?, ?, ?)",
        (recorded_at, period, ticker, grade),
    )
    conn.commit()
    return cur.lastrowid


def latest_grade_before(conn: sqlite3.Connection, ticker: str, before_period: str) -> str | None:
    """이번 달(before_period) 전에 기록된 그 종목의 가장 최근 등급 (core.moat_alert.grade_alert의
    "직전 등급" 입력용).

    출력: 등급 문자열. 기록이 없으면(첫 실행) None.
    """
    row = conn.execute(
        "SELECT grade FROM grade_history WHERE ticker = ? AND period < ? ORDER BY period DESC, id DESC LIMIT 1",
        (ticker, before_period),
    ).fetchone()
    return row[0] if row else None


def load_monthly_records(conn: sqlite3.Connection, portfolio: str | None = None) -> list[dict]:
    """월말 기록을 전부 읽는다(보고·검증용)."""
    if portfolio is None:
        cur = conn.execute("SELECT recorded_at, period, portfolio, value_factor, status, notes FROM monthly_records ORDER BY period, portfolio")
    else:
        cur = conn.execute(
            "SELECT recorded_at, period, portfolio, value_factor, status, notes FROM monthly_records WHERE portfolio = ? ORDER BY period",
            (portfolio,),
        )
    cols = ("recorded_at", "period", "portfolio", "value_factor", "status", "notes")
    return [dict(zip(cols, row)) for row in cur.fetchall()]
