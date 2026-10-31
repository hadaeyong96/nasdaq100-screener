"""시장 온도계 기록 — 가상 장부 DB(data/virtual_books.db, 드라이브 동기화 중)의 thermometer_history 표에 하루 한 줄
(기준일·항목별 원값·항목별 도수·합계 온도·구간·자료 없음·sent_at). 2026-10-29부터 아침 편지 실행(engine.daily)이 기준일마다
한 줄 남긴다(sent_at = 기록 시각). 독립 메시지를 다시 켜면 같은 기준일 중복 발송 막기도 이 표를 읽는다(2026-10-28).
백필(연구용)은 같은 열 형식의 CSV(write_rows).
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from thermometer.core import KEYS, fear_greed_label, interpretation, zone_of

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "virtual_books.db"  # store.virtual_books.VBOOKS_DB_PATH와 같은 파일
TABLE = "thermometer_history"
COLUMNS = ["date", *[f"raw_{k}" for k in KEYS], *[f"deg_{k}" for k in KEYS], "temp", "zone", "missing"]


def _db(path) -> Path:
    p = Path(path) if path is not None else DEFAULT_DB
    return p if p.is_absolute() else ROOT / p


def _connect(path) -> sqlite3.Connection:
    p = _db(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    cols = ", ".join(["date TEXT PRIMARY KEY"] + [f"{c} TEXT" for c in COLUMNS[1:]] + ["sent_at TEXT"])
    conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} ({cols})")
    return conn


def row_of(result: dict) -> dict:
    """get_temperature 결과 → 기록 한 줄(dict, 글자). 기준일은 data_date(QQQ 마지막 거래일)."""
    row = {"date": result["data_date"] or result["date"]}
    for k in KEYS:
        raw, deg = result["parts"][k]["raw"], result["parts"][k]["deg"]
        row[f"raw_{k}"] = "" if raw is None else f"{raw:.4f}"
        row[f"deg_{k}"] = "" if deg is None else f"{deg:.4f}"
    row["temp"] = "" if result["temp"] is None else str(result["temp"])
    row["zone"] = result["zone"] or ""
    row["missing"] = "|".join(result["missing"])
    return row


def _read(path, where: str = "", args: tuple = (), order: str = "date") -> list[dict]:
    """읽기 전용 조회 — DB 파일이나 표가 없으면 빈 목록(파일·표를 만들지 않는다)."""
    p = _db(path)
    if not p.exists():
        return []
    conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    try:
        cur = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM {TABLE} {where} ORDER BY {order}", args)
        return [dict(zip(COLUMNS, r)) for r in cur.fetchall()]
    except sqlite3.OperationalError:  # 표가 아직 없음
        return []
    finally:
        conn.close()


def load(path=None) -> list[dict]:
    """기록 전체 → [행 dict] (날짜순). DB 파일이 없으면 빈 목록(파일을 만들지 않는다)."""
    return _read(path)


def find(day: str, path=None) -> dict | None:
    """기준일(YYYY-MM-DD) 행. 없으면 None."""
    rows = _read(path, "WHERE date = ?", (day,))
    return rows[0] if rows else None


def previous(day: str, path=None) -> dict | None:
    """기준일보다 앞선 가장 최근 기록 행(어제 대비용). 없으면 None."""
    rows = _read(path, "WHERE date < ?", (day,), order="date DESC LIMIT 1")
    return rows[0] if rows else None


def _num(v) -> float | None:
    return None if v in (None, "") else float(v)


def result_of(row: dict, cfg: dict, prev: dict | None = None) -> dict:
    """기록 한 줄 → get_temperature 결과 모양(보고서·설명 한 줄용, 2026-10-31) — 다시 계산하지 않고 기록 값 그대로.
    어제 대비(delta)는 앞선 기록 행(prev)의 온도와 비교한다(없으면 None). 해석 한 줄은 기록된 이격률 부호로 정한다."""
    parts = {}
    for k in KEYS:
        raw, deg = _num(row.get(f"raw_{k}")), _num(row.get(f"deg_{k}"))
        parts[k] = {"deg": deg, "raw": raw}
        if k == "fear_greed" and raw is not None:
            parts[k]["label"] = fear_greed_label(raw, cfg)
    temp = int(row["temp"]) if row.get("temp") not in (None, "") else None
    prev_temp = int(prev["temp"]) if prev and prev.get("temp") not in (None, "") else None
    return {"date": row["date"], "data_date": row["date"], "temp": temp, "zone": row.get("zone") or None,
            "zone_desc": zone_of(temp, cfg)["desc"] if temp is not None else None, "parts": parts,
            "note": interpretation(parts["sma_fast"]["raw"], parts["sma_slow"]["raw"], cfg),
            "missing": [m for m in (row.get("missing") or "").split("|") if m],
            "prev_date": prev["date"] if prev else None, "prev_temp": prev_temp,
            "delta": temp - prev_temp if temp is not None and prev_temp is not None else None, "source": "history"}


def append(result: dict, path=None) -> dict:
    """결과 한 줄을 기록한다(같은 기준일이 이미 있으면 그대로 두고 그 행을 돌려준다)."""
    row = row_of(result)
    old = find(row["date"], path)
    if old is not None:
        return old
    conn = _connect(path)
    try:
        cols = COLUMNS + ["sent_at"]
        conn.execute(f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                     [row[c] for c in COLUMNS] + [datetime.now(timezone.utc).isoformat(timespec="seconds")])
        conn.commit()
    finally:
        conn.close()
    return row


def write_rows(rows: list[dict], path) -> None:
    """여러 줄을 새 CSV로 쓴다(백필용)."""
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
