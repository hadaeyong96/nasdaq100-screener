"""시장 온도계 기록 data/thermometer/history.csv — 하루 한 줄(기준일·항목별 원값·항목별 도수·합계 온도·구간).

같은 기준일 중복 발송 막기와 아침 편지 한 줄(letter_line)이 이 파일을 읽는다. Actions에서는 드라이브에 보관
(scripts/drive_sync.py --mode thermo). 백필(연구용)도 같은 열 형식(write_rows).
"""

from __future__ import annotations

import csv
from pathlib import Path

from thermometer.core import KEYS, interpretation, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "thermometer" / "history.csv"
COLUMNS = ["date", *[f"raw_{k}" for k in KEYS], *[f"deg_{k}" for k in KEYS], "temp", "zone", "missing"]


def _path(path) -> Path:
    p = Path(path) if path is not None else DEFAULT_PATH
    return p if p.is_absolute() else ROOT / p


def row_of(result: dict) -> dict:
    """get_temperature 결과 → 기록 한 줄(dict). 기준일은 data_date(QQQ 마지막 거래일)."""
    row = {"date": result["data_date"] or result["date"]}
    for k in KEYS:
        raw, deg = result["parts"][k]["raw"], result["parts"][k]["deg"]
        row[f"raw_{k}"] = "" if raw is None else f"{raw:.4f}"
        row[f"deg_{k}"] = "" if deg is None else f"{deg:.4f}"
    row["temp"] = "" if result["temp"] is None else str(result["temp"])
    row["zone"] = result["zone"] or ""
    row["missing"] = "|".join(result["missing"])
    return row


def load(path=None) -> list[dict]:
    """기록 전체 → [행 dict]. 파일이 없으면 빈 목록."""
    p = _path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def find(day: str, path=None) -> dict | None:
    """기준일(YYYY-MM-DD) 행. 없으면 None."""
    for r in load(path):
        if r.get("date") == day:
            return r
    return None


def append(result: dict, path=None) -> dict:
    """결과 한 줄을 덧붙인다(같은 기준일이 이미 있으면 덧붙이지 않고 그 행을 돌려준다)."""
    row = row_of(result)
    old = find(row["date"], path)
    if old is not None:
        return old
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists()
    with open(p, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


def write_rows(rows: list[dict], path) -> None:
    """여러 줄을 새 파일로 쓴다(백필용)."""
    p = _path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def letter_line(as_of, path=None) -> str | None:
    """아침 편지 ■ 미국 증시 마감 칸 끝 한 줄 "시장 온도계 {T}도 · {구간}." — 그 기준일 기록이 없으면 None.
    어떤 오류도 밖으로 내보내지 않는다(편지 실패 금지)."""
    try:
        if as_of is None:
            return None
        day = str(getattr(as_of, "date", lambda: as_of)())[:10]
        r = find(day, path)
        if not r or not r.get("temp"):
            return None
        return f"시장 온도계 {int(r['temp'])}도 · {r['zone']}."
    except Exception:
        return None


def detail(as_of, path=None, cfg: dict | None = None) -> dict | None:
    """보고서 「시장 온도계」 칸 — 그 기준일 기록 → {temp, zone, rows[{key, name, raw, deg}], note, missing[이름]}. 없거나 오류면 None."""
    try:
        if as_of is None:
            return None
        day = str(getattr(as_of, "date", lambda: as_of)())[:10]
        r = find(day, path)
        if not r or not r.get("temp"):
            return None
        cfg = cfg or load_config()
        num = lambda v: float(v) if v not in (None, "") else None  # noqa: E731
        fmt = {"sma_fast": lambda v: f"{v:+.1f}%", "sma_slow": lambda v: f"{v:+.1f}%", "breadth": lambda v: f"{v:.0f}%",
               "vix": lambda v: f"{v:.1f}", "fear_greed": lambda v: f"{v:.0f}"}
        rows = []
        for k in KEYS:
            raw, deg = num(r.get(f"raw_{k}")), num(r.get(f"deg_{k}"))
            if deg is not None:
                rows.append({"key": k, "name": cfg["names"][k], "raw": fmt[k](raw) if raw is not None else "", "deg": f"{deg:.1f}"})
        missing = [cfg["names"][k] for k in (r.get("missing") or "").split("|") if k]
        return {"temp": int(r["temp"]), "zone": r["zone"], "rows": rows, "missing": missing,
                "note": interpretation(num(r.get("raw_sma_fast")), num(r.get("raw_sma_slow")), cfg)}
    except Exception:
        return None
