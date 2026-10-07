"""F2 자금 흐름 관찰 판정 (사전 등록: docs/design/flow_watch.md, configs/f2_preregistration.yaml, tag f2-prereg).

실전 DB(data/state.db)의 flow_watch 기록만 읽는다(과거 데이터로 판정하지 않음). 시작할 때 사전 등록 두 파일의
sha256(LF 정규화)을 확인하고 다르면 멈춘다. 2027-10-01 전에는 기록 수·결과 채움 수만 보여주고 수익은 출력하지 않는다.

실행: python -m scripts.f2_review            (오늘 날짜 기준)
      python -m scripts.f2_review --db PATH  (다른 DB 파일)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from store import db  # noqa: E402

PREREG_FILES = {
    "docs/design/flow_watch.md": "8bd3dadee0d2c415a1a188ebabbf6fd3478a091675bf378e1d08785bd4a58ddb",
    "configs/f2_preregistration.yaml": "d7cf02931c0fdc6051b7095ae87dc4d38f31ac96dfed083d3fe9a6a126ff9770",
}


def normalized_sha256(path: Path) -> str:
    """줄바꿈을 LF로 맞춘 내용의 sha256 (Windows 체크아웃의 CRLF 변환에 흔들리지 않게)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_preregistration(root: Path = ROOT) -> dict:
    """두 파일 해시가 다르면 SystemExit. 같으면 yaml 내용을 돌려준다."""
    for rel, expected in PREREG_FILES.items():
        got = normalized_sha256(root / rel)
        if got != expected:
            raise SystemExit(f"[f2] 사전 등록 파일이 바뀌었습니다: {rel} (기대 {expected[:12]}…, 실제 {got[:12]}…) — 멈춤")
    with open(root / "configs" / "f2_preregistration.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def control_distribution(universe: pd.DataFrame, per_date: int, iterations: int, seed: int) -> np.ndarray:
    """대조군: 매회 기준일마다 대상 행 중 per_date개(적으면 전부) 비복원 추출 → 모아 평균 (구현 해석 23번).

    입력: universe(base_date, excess_20d), per_date, iterations, seed / 출력: 평균 iterations개
    """
    rng = np.random.default_rng(seed)
    groups = [g["excess_20d"].to_numpy(dtype=float) for _, g in universe.sort_values(["base_date", "ticker"]).groupby("base_date")]
    out = np.empty(iterations)
    for i in range(iterations):
        picks = [rng.choice(g, size=min(per_date, len(g)), replace=False) for g in groups if len(g)]
        out[i] = float(np.mean(np.concatenate(picks))) if picks else np.nan
    return out


def judge(records: pd.DataFrame, prereg: dict) -> dict:
    """판정 표본·대조군·합격 조건 (구현 해석 21~24번). 입력: flow_watch 표 / 출력: dict"""
    rv = prereg["review"]
    s = rv["sample_base_dates"]
    in_range = records[(records["base_date"] >= s["from"]) & (records["base_date"] <= s["to"])]
    filled = in_range[in_range["filled_20d_at"].notna()]
    inflow, dump, univ = (filled[filled["list"] == k] for k in ("유입", "투매", "대상"))
    c = rv["control"]
    dist = control_distribution(univ, c["per_base_date"], c["iterations"], c["seed"])
    inflow_mean = float(inflow["excess_20d"].mean()) if len(inflow) else float("nan")
    dump_mean = float(dump["excess_20d"].mean()) if len(dump) else float("nan")
    p75 = float(np.nanpercentile(dist, 75)) if np.isfinite(dist).any() else float("nan")
    cond = {
        "유입 평균 > 0": bool(len(inflow)) and inflow_mean > 0,
        "유입 평균 ≥ 대조군 75백분위": bool(len(inflow)) and np.isfinite(p75) and inflow_mean >= p75,
        "투매 평균 < 유입 평균": bool(len(inflow)) and bool(len(dump)) and dump_mean < inflow_mean,
    }
    return {
        "base_dates": int(in_range["base_date"].nunique()), "unfilled_rows": int(in_range["filled_20d_at"].isna().sum()),
        "inflow_n": int(len(inflow)), "dump_n": int(len(dump)), "universe_n": int(len(univ)),
        "inflow_mean": inflow_mean, "dump_mean": dump_mean, "control_p75": p75,
        "control_mean": float(np.nanmean(dist)) if np.isfinite(dist).any() else float("nan"),
        "inflow_percentile": float((dist < inflow_mean).mean() * 100) if len(inflow) and np.isfinite(dist).any() else float("nan"),
        "conditions": cond, "pass": all(cond.values()),
    }


def status_only(records: pd.DataFrame) -> dict:
    """판정일 전: 기록 수·채움 수만 (수익 출력 안 함, 구현 해석 20번)."""
    if records.empty:
        return {"base_dates": 0, "rows": {}, "filled_5d": 0, "filled_20d": 0}
    return {
        "base_dates": int(records["base_date"].nunique()), "first": records["base_date"].min(), "last": records["base_date"].max(),
        "rows": records.groupby("list").size().to_dict(),
        "filled_5d": int(records["filled_5d_at"].notna().sum()), "filled_20d": int(records["filled_20d_at"].notna().sum()),
    }


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(db.DB_PATH))
    ap.add_argument("--today", default=None, help="테스트용 오늘 날짜(YYYY-MM-DD)")
    args = ap.parse_args(argv)
    prereg = verify_preregistration()
    print("[f2] 사전 등록 해시 확인 통과")
    today = date.fromisoformat(args.today) if args.today else date.today()
    conn = db.connect(Path(args.db))
    records = db.load_flow_watch(conn)
    conn.close()
    if today < date.fromisoformat(prereg["review"]["verdict_not_before"]):
        st = status_only(records)
        print(f"[f2] 판정일({prereg['review']['verdict_not_before']}) 전 — 기록 현황만 출력합니다(수익은 출력하지 않음).")
        print(st)
        return {"mode": "status", **st}
    res = judge(records, prereg)
    print(f"[f2] 판정 표본: 기준일 {res['base_dates']}개, 유입 {res['inflow_n']}행, 투매 {res['dump_n']}행, 대상 {res['universe_n']}행, 20일 미채움 {res['unfilled_rows']}행")
    print(f"[f2] 유입 20일 초과수익 평균 {res['inflow_mean']:+.4%} · 투매 {res['dump_mean']:+.4%} · 대조군 평균 {res['control_mean']:+.4%} · 대조군 75백분위 {res['control_p75']:+.4%} (유입 위치 {res['inflow_percentile']:.1f}백분위)")
    for k, v in res["conditions"].items():
        print(f"  {'✓' if v else '✗'} {k}")
    print(f"[f2] 판정: {'합격 — 신호로 쓰는 것을 새 사전 등록으로 검토' if res['pass'] else '불합격 — 참고 표시만 유지하거나 제거'}")
    return {"mode": "verdict", **res}


if __name__ == "__main__":
    main()
