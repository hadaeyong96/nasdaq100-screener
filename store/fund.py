"""전략(Strategy)·변형(Variant)·직원(Employee) 3계층 저장소 (AI 펀드 F2,
docs/design/fund_sim.md 3.4).

F2 범위는 스키마와 최소 CRUD뿐이다 — 실제 8개 전략·41개 변형·직원은 F3에서 채운다.

| 계층 | 뜻 | 불변 여부 |
| --- | --- | --- |
| Strategy | 전략 계열(팀) | 이름 등은 갱신 가능 |
| Variant | 파라미터 조합 하나 | **불변** — 한 번 만들면 다시 못 만든다(파라미터가 바뀌면 새 variant_id) |
| Employee | 지금 운용 중인 변형의 실행 주체 | active_to로 은퇴 표시, 새 이력은 그대로 남는다 |

파일: data/fund.db (store/db.py와 같은 관례 — store/ 밑 모듈이 data/ 밑 DB 파일을 관리,
설계 문서의 "store/fund.db" 표기와 실제 파일 위치는 다르다).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "fund.db"


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """DB에 연결하고 테이블이 없으면 만든다."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS strategies (
            strategy_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            signal_fn_path TEXT,
            signal_type TEXT,
            created_at TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS variants (
            variant_id TEXT PRIMARY KEY,
            strategy_id TEXT NOT NULL,
            params TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            created_by TEXT NOT NULL,
            parent_variant_id TEXT,
            created_at TEXT,
            FOREIGN KEY (strategy_id) REFERENCES strategies (strategy_id),
            FOREIGN KEY (parent_variant_id) REFERENCES variants (variant_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS employees (
            employee_id TEXT PRIMARY KEY,
            variant_id TEXT NOT NULL,
            active_from TEXT NOT NULL,
            active_to TEXT,
            allocation_history TEXT,
            FOREIGN KEY (variant_id) REFERENCES variants (variant_id)
        )
        """
    )
    conn.commit()


def params_hash(params: dict) -> str:
    """파라미터 dict의 결정적 해시 (순수 함수) — 키 순서와 무관하게 같은 값이면 같은 해시.

    입력: params(JSON 직렬화 가능한 dict)
    출력: 16자리 16진 해시 문자열
    """
    canon = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


# ── strategies ───────────────────────────────────────────────────────────


def upsert_strategy(
    conn: sqlite3.Connection, strategy_id: str, name: str, signal_fn_path: str | None, signal_type: str | None, created_at: str
) -> None:
    """전략 하나를 upsert한다 (이름·신호 함수 경로는 갱신 가능 — strategy_id만 고정 키)."""
    conn.execute(
        "INSERT INTO strategies (strategy_id, name, signal_fn_path, signal_type, created_at) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(strategy_id) DO UPDATE SET name=excluded.name, signal_fn_path=excluded.signal_fn_path, signal_type=excluded.signal_type",
        (strategy_id, name, signal_fn_path, signal_type, created_at),
    )
    conn.commit()


def load_strategy(conn: sqlite3.Connection, strategy_id: str) -> dict | None:
    row = conn.execute(
        "SELECT strategy_id, name, signal_fn_path, signal_type, created_at FROM strategies WHERE strategy_id = ?",
        (strategy_id,),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(("strategy_id", "name", "signal_fn_path", "signal_type", "created_at"), row))


def load_all_strategies(conn: sqlite3.Connection) -> list[dict]:
    cols = ("strategy_id", "name", "signal_fn_path", "signal_type", "created_at")
    cur = conn.execute(f"SELECT {', '.join(cols)} FROM strategies ORDER BY strategy_id")
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ── variants ─────────────────────────────────────────────────────────────


def insert_variant(
    conn: sqlite3.Connection,
    variant_id: str,
    strategy_id: str,
    params: dict,
    created_by: str,
    created_at: str,
    parent_variant_id: str | None = None,
) -> str:
    """변형 하나를 새로 기록한다. 변형은 불변이라 같은 variant_id를 다시 넣으면 오류를 낸다
    (실수로 다른 파라미터로 덮어쓰는 사고를 막는다 — 파라미터가 바뀌면 새 variant_id를 써야 한다).

    입력: variant_id, strategy_id, params(dict), created_by("LHS"|"manual"|"PBT"),
         created_at, parent_variant_id(PBT로 파생됐으면 그 부모, 아니면 None)
    출력: 이 변형의 params_hash
    예외: variant_id가 이미 있으면 ValueError
    """
    existing = conn.execute("SELECT 1 FROM variants WHERE variant_id = ?", (variant_id,)).fetchone()
    if existing is not None:
        raise ValueError(f"variant_id {variant_id!r}가 이미 있습니다 — 변형은 불변이라 다시 만들 수 없습니다.")
    h = params_hash(params)
    conn.execute(
        "INSERT INTO variants (variant_id, strategy_id, params, params_hash, created_by, parent_variant_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (variant_id, strategy_id, json.dumps(params, default=str), h, created_by, parent_variant_id, created_at),
    )
    conn.commit()
    return h


def _row_to_variant(row: tuple) -> dict:
    variant_id, strategy_id, params_json, h, created_by, parent_variant_id, created_at = row
    return {
        "variant_id": variant_id,
        "strategy_id": strategy_id,
        "params": json.loads(params_json),
        "params_hash": h,
        "created_by": created_by,
        "parent_variant_id": parent_variant_id,
        "created_at": created_at,
    }


def load_variant(conn: sqlite3.Connection, variant_id: str) -> dict | None:
    row = conn.execute(
        "SELECT variant_id, strategy_id, params, params_hash, created_by, parent_variant_id, created_at "
        "FROM variants WHERE variant_id = ?",
        (variant_id,),
    ).fetchone()
    return None if row is None else _row_to_variant(row)


def load_variants_for_strategy(conn: sqlite3.Connection, strategy_id: str) -> list[dict]:
    cur = conn.execute(
        "SELECT variant_id, strategy_id, params, params_hash, created_by, parent_variant_id, created_at "
        "FROM variants WHERE strategy_id = ? ORDER BY variant_id",
        (strategy_id,),
    )
    return [_row_to_variant(row) for row in cur.fetchall()]


def variant_lineage(conn: sqlite3.Connection, variant_id: str) -> list[dict]:
    """variant_id에서 parent_variant_id를 따라 원본까지 거슬러 올라간 계보를 돌려준다
    (PBT가 만든 변형의 "세대"를 알아내는 데 쓴다 — 설계 3.4·5.9의 실효 N 계산 기반).

    출력: [자기 자신, 부모, 조부모, ...] (variant_id가 없으면 빈 목록)
    예외: 순환 참조가 있으면 RuntimeError(데이터 오류 — 조용히 무한 루프에 빠지지 않는다)
    """
    lineage: list[dict] = []
    seen: set[str] = set()
    current = variant_id
    while current is not None:
        if current in seen:
            raise RuntimeError(f"variant_id {current!r}에서 부모 계보가 순환합니다 — 데이터 오류")
        seen.add(current)
        v = load_variant(conn, current)
        if v is None:
            break
        lineage.append(v)
        current = v["parent_variant_id"]
    return lineage


# ── employees ────────────────────────────────────────────────────────────


def insert_employee(
    conn: sqlite3.Connection,
    employee_id: str,
    variant_id: str,
    active_from: str,
    active_to: str | None = None,
    allocation_history: list[dict] | None = None,
) -> None:
    conn.execute(
        "INSERT INTO employees (employee_id, variant_id, active_from, active_to, allocation_history) VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(employee_id) DO UPDATE SET variant_id=excluded.variant_id, active_from=excluded.active_from, "
        "active_to=excluded.active_to, allocation_history=excluded.allocation_history",
        (employee_id, variant_id, active_from, active_to, json.dumps(allocation_history or [], default=str)),
    )
    conn.commit()


def _row_to_employee(row: tuple) -> dict:
    employee_id, variant_id, active_from, active_to, allocation_json = row
    return {
        "employee_id": employee_id,
        "variant_id": variant_id,
        "active_from": active_from,
        "active_to": active_to,
        "allocation_history": json.loads(allocation_json) if allocation_json else [],
    }


def load_employee(conn: sqlite3.Connection, employee_id: str) -> dict | None:
    row = conn.execute(
        "SELECT employee_id, variant_id, active_from, active_to, allocation_history FROM employees WHERE employee_id = ?",
        (employee_id,),
    ).fetchone()
    return None if row is None else _row_to_employee(row)


def load_active_employees(conn: sqlite3.Connection, as_of: str) -> list[dict]:
    """as_of(YYYY-MM-DD) 날짜에 활동 중인(active_to가 없거나 as_of 이후인) 직원 목록."""
    cur = conn.execute(
        "SELECT employee_id, variant_id, active_from, active_to, allocation_history FROM employees "
        "WHERE active_from <= ? AND (active_to IS NULL OR active_to >= ?) ORDER BY employee_id",
        (as_of, as_of),
    )
    return [_row_to_employee(row) for row in cur.fetchall()]
