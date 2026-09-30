"""시도 로그 저장소 (AI 펀드 F2, docs/design/fund_sim.md 2.2 "시도 기록", 5.9).

백테스트로 시험한 **모든 설정**(과거 v3 개발 중 시험분 포함)을 한 줄씩 기록한다.
다중 검정 보정(Deflated Sharpe Ratio 등)의 N_raw는 이 로그의 행 수에서 나온다 —
그래서 트라이얼은 절대 지우지 않는다(실패한 실험도 "시도했다"는 사실 자체가 값어치가
있다). variant_id는 선택 항목이다: F2 시점에는 아직 strategies/variants 체계로
정식 편입되지 않은 임시 실험(예: 과거 P5 실험, B0.5 재현 테스트)도 기록해야 하므로.

파일: data/trials.db (store/db.py와 같은 관례 — 설계 문서의 "store/trials.db" 표기와
실제 파일 위치는 다르다).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "trials.db"


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """DB에 연결하고 테이블이 없으면 만든다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trials (
            trial_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at TEXT NOT NULL,
            label TEXT NOT NULL,
            variant_id TEXT,
            config_hash TEXT,
            date_range_start TEXT,
            date_range_end TEXT,
            universe_mode TEXT,
            metrics TEXT,
            notes TEXT
        )
        """
    )
    conn.commit()


def record_trial(
    conn: sqlite3.Connection,
    run_at: str,
    label: str,
    metrics: dict,
    variant_id: str | None = None,
    config_hash: str | None = None,
    date_range_start: str | None = None,
    date_range_end: str | None = None,
    universe_mode: str | None = None,
    notes: str | None = None,
) -> int:
    """시도 하나를 기록한다 (append-only — 수정·삭제 함수는 일부러 두지 않는다).

    입력: run_at(실행 시각 ISO 문자열), label(사람이 읽을 이름 — 예: "B0.5",
         "P5-3 E1 ATR=2.5"), metrics(dict — 세전·세후 CAGR·MDD 등 이 실행의 결과
         요약, JSON으로 저장), variant_id(store.fund의 변형과 연결되면 그 id, 아직
         변형 체계 밖이면 None), config_hash, date_range_start/end, universe_mode
         ("POINT_IN_TIME"|"CURRENT_CONSTITUENTS"), notes
    출력: 새로 만들어진 trial_id
    """
    cur = conn.execute(
        "INSERT INTO trials (run_at, label, variant_id, config_hash, date_range_start, date_range_end, "
        "universe_mode, metrics, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_at, label, variant_id, config_hash, date_range_start, date_range_end,
            universe_mode, json.dumps(metrics, default=str), notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def _row_to_trial(row: tuple) -> dict:
    (
        trial_id, run_at, label, variant_id, config_hash, date_range_start, date_range_end,
        universe_mode, metrics_json, notes,
    ) = row
    return {
        "trial_id": trial_id,
        "run_at": run_at,
        "label": label,
        "variant_id": variant_id,
        "config_hash": config_hash,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "universe_mode": universe_mode,
        "metrics": json.loads(metrics_json) if metrics_json else {},
        "notes": notes,
    }


_COLUMNS = (
    "trial_id", "run_at", "label", "variant_id", "config_hash", "date_range_start", "date_range_end",
    "universe_mode", "metrics", "notes",
)


def load_trial(conn: sqlite3.Connection, trial_id: int) -> dict | None:
    row = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
    return None if row is None else _row_to_trial(row)


def load_all_trials(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM trials ORDER BY trial_id")
    return [_row_to_trial(row) for row in cur.fetchall()]


def count_trials(conn: sqlite3.Connection) -> int:
    """N_raw(설계 5.9) — 이 로그에 기록된 전체 시도 수."""
    row = conn.execute("SELECT COUNT(*) FROM trials").fetchone()
    return row[0] if row else 0
