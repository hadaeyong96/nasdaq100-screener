"""검색량 실적 예고 — 순수 계산 (docs/prereg/search_forecast.md, 구현 해석 docs/prereg/search_forecast_impl.md).

네트워크·파일·DB·현재 시각 접근 없음(시각은 인자로 받는다). 앞으로 기록 전용 — 과거 수익률 계산 없음.
- YoY는 검색어 하나의 한 번 받은 값(같은 요청의 0~100) 안에서만 계산한다. 서로 다른 검색어·요청 값을 섞지 않는다.
- 회사 YoY = 검색어 YoY들의 중앙값(v1.1, 2026-11-02 — v1은 평균). 점수 = 이번 분기 회사 YoY − 직전 8개 분기 회사 YoY 중앙값.
- v1 평균 방식 점수도 같은 값으로 함께 낸다(*_v1, 판정 때 참고 보고만 — 판정 제외).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

ACCEL, DECEL, NONE = "매출 가속", "매출 감속", "예측 없음"


# ── 검색량 YoY·점수 ──────────────────────────────────────────────────────────


def window_mean(weekly: pd.Series, start: pd.Timestamp, end: pd.Timestamp, min_weeks: int = 8) -> float:
    """주간 값(인덱스 = 주 시작일) 중 [start, end] 안의 평균. 값이 min_weeks개보다 적으면 NaN."""
    w = weekly[(weekly.index >= start) & (weekly.index <= end)].dropna()
    return float(w.mean()) if len(w) >= min_weeks else np.nan


def term_yoy(weekly: pd.Series, start: pd.Timestamp, end: pd.Timestamp, shift_days: int = 364, min_weeks: int = 8) -> float:
    """검색어 하나(한 번 받은 값)의 분기 평균 전년 동기 대비 변화율. 전년 창 = 같은 창을 shift_days 앞으로."""
    now = window_mean(weekly, start, end, min_weeks)
    sh = pd.Timedelta(days=shift_days)
    prev = window_mean(weekly, start - sh, end - sh, min_weeks)
    if np.isnan(now) or np.isnan(prev) or prev <= 0:
        return np.nan
    return now / prev - 1.0


AGG = {"median": np.median, "mean": np.mean}


def company_yoy(terms: dict[str, pd.Series], start: pd.Timestamp, end: pd.Timestamp, shift_days: int = 364, min_weeks: int = 8,
                agg: str = "median") -> float:
    """검색어별 YoY(각자 자기 값 안에서)를 agg(중앙값 v1.1 / 평균 v1)로 합친 값. 구할 수 있는 검색어가 없으면 NaN."""
    ys = [term_yoy(s, start, end, shift_days, min_weeks) for s in terms.values()]
    ys = [y for y in ys if not np.isnan(y)]
    return float(AGG[agg](ys)) if ys else np.nan


def _score(terms, quarters, n_base, shift_days, min_weeks, agg) -> tuple[float, float, float, int]:
    *hist, cur = quarters
    y = company_yoy(terms, *cur, shift_days, min_weeks, agg)
    base = [company_yoy(terms, *q, shift_days, min_weeks, agg) for q in hist[-n_base:]]
    base = [b for b in base if not np.isnan(b)]
    med = float(np.median(base)) if base else np.nan
    return y, med, (y - med if not (np.isnan(y) or np.isnan(med)) else np.nan), len(base)


def company_score(terms: dict[str, pd.Series], quarters: list[tuple[pd.Timestamp, pd.Timestamp]], *, n_base: int = 8,
                  shift_days: int = 364, min_weeks: int = 8, agg: str = "median") -> dict:
    """quarters = 오래된 순 [(시작, 끝)], 마지막이 점수를 낼 분기.
    출력: {yoy, baseline, score, base_n} (agg 방식) + {yoy_v1, baseline_v1, score_v1} (v1 평균 방식, 참고용)."""
    y, med, sc, n = _score(terms, quarters, n_base, shift_days, min_weeks, agg)
    y1, med1, sc1, _ = _score(terms, quarters, n_base, shift_days, min_weeks, "mean")
    return {"yoy": y, "baseline": med, "score": sc, "base_n": n, "yoy_v1": y1, "baseline_v1": med1, "score_v1": sc1}


# ── 회계 분기 ────────────────────────────────────────────────────────────────


def _is_month_end(d: pd.Timestamp) -> bool:
    return (d + pd.Timedelta(days=1)).day == 1


def project_quarter_ends(known_ends: list[pd.Timestamp], until: pd.Timestamp) -> list[pd.Timestamp]:
    """EDGAR로 아는 분기 끝들 → until까지 다음 분기 끝 추정. 4분기 전 끝 + 1년(모두 달 말이면 12개월 뒤 달 말, 아니면 364일).

    입력: 오름차순 실제 분기 끝(최소 4개). 출력: 실제 + 추정 분기 끝(오름차순, until 이하까지 + 다음 하나).
    """
    ends = sorted(pd.Timestamp(e) for e in known_ends)
    if len(ends) < 4:
        return ends
    month_end = all(_is_month_end(e) for e in ends[-8:])
    while ends[-1] <= until:
        base = ends[-4]
        nxt = (base + pd.offsets.MonthEnd(12)) if month_end else base + pd.Timedelta(days=364)
        ends.append(pd.Timestamp(nxt))
    return ends


def quarter_windows(ends: list[pd.Timestamp]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """분기 끝 목록 → [(시작 = 앞 끝 + 1일, 끝)] (첫 끝은 시작을 몰라 뺀다)."""
    return [(ends[i - 1] + pd.Timedelta(days=1), ends[i]) for i in range(1, len(ends))]


def season_of(end: pd.Timestamp) -> str:
    """분기 끝이 속한 달력 분기(시즌) 'YYYY-Qn'."""
    return f"{end.year}-Q{(end.month - 1) // 3 + 1}"


def season_end(season: str) -> pd.Timestamp:
    y, q = season.split("-Q")
    return pd.Timestamp(int(y), int(q) * 3, 1) + pd.offsets.MonthEnd(0)


# ── 시즌 예측 ────────────────────────────────────────────────────────────────


def terciles(scores: dict[str, float]) -> dict[str, str]:
    """같은 시즌 점수 → 상위 1/3 매출 가속, 하위 1/3 매출 감속, 가운데 예측 없음. k = 회사 수 // 3. 점수 없음(NaN)은 예측 없음."""
    ok = sorted(((t, s) for t, s in scores.items() if s is not None and not np.isnan(s)), key=lambda x: (-x[1], x[0]))
    k = len(ok) // 3
    out = {t: NONE for t in scores}
    for t, _ in ok[:k]:
        out[t] = ACCEL
    for t, _ in ok[len(ok) - k:] if k else []:
        out[t] = DECEL
    return out


# ── 잠금 ────────────────────────────────────────────────────────────────────


def lock_hash(rows: list[dict]) -> str:
    """잠그는 행들의 SHA-256(키 정렬 JSON 한 줄씩, 행도 정렬). 같은 행이면 순서와 관계없이 같은 값."""
    lines = sorted(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def lock_line(locked_at: str, digest: str) -> str:
    """잠금 파일 한 줄 — 시각과 해시만(종목·방향·점수 없음)."""
    return f"{locked_at} sha256={digest}"


def is_valid(locked_at: pd.Timestamp, announced: pd.Timestamp | None) -> bool | None:
    """잠금이 실적 발표보다 먼저였는지. 발표일(미 동부 날짜)을 모르면 None. 같은 날이면 무효(시각 불확실)."""
    if announced is None or pd.isna(announced):
        return None
    lk = pd.Timestamp(locked_at)
    lk = lk.tz_convert("America/New_York").tz_localize(None) if lk.tzinfo else lk
    return lk.normalize() < pd.Timestamp(announced).normalize()


# ── 채점 ────────────────────────────────────────────────────────────────────


def quarterly_revenue(facts: list[dict]) -> pd.DataFrame:
    """EDGAR 매출 사실(start·end·val·filed) → 분기 매출. 분기 = 기간 77~119일(12·16주 분기 포함).
    4분기가 연간(350~380일)으로만 있으면 연간 − 같은 시작의 누적(240~290일) 또는 − 안의 세 분기.

    출력: DataFrame[start, end, revenue] (end 오름차순, 같은 기간은 처음 공시 값).
    """
    if not facts:
        return pd.DataFrame(columns=["start", "end", "revenue"])
    f = pd.DataFrame(facts)
    f["start"], f["end"], f["filed"] = pd.to_datetime(f["start"]), pd.to_datetime(f["end"]), pd.to_datetime(f["filed"])
    f = f.sort_values("filed").drop_duplicates(["start", "end"], keep="first")
    f["days"] = (f["end"] - f["start"]).dt.days
    q = f[(f["days"] >= 77) & (f["days"] <= 119)][["start", "end", "val"]]
    add = []
    for _, y in f[(f["days"] >= 350) & (f["days"] <= 380)].iterrows():
        if (abs(q["end"] - y["end"]) <= pd.Timedelta(days=5)).any():
            continue
        ytd = f[(f["start"] == y["start"]) & (f["days"] >= 240) & (f["days"] <= 290)]
        if len(ytd):
            r = ytd.sort_values("end").iloc[-1]
            add.append({"start": r["end"] + pd.Timedelta(days=1), "end": y["end"], "val": y["val"] - r["val"]})
            continue
        inside = q[(q["start"] >= y["start"] - pd.Timedelta(days=5)) & (q["end"] <= y["end"] - pd.Timedelta(days=60))]
        if len(inside) == 3:
            add.append({"start": inside["end"].max() + pd.Timedelta(days=1), "end": y["end"], "val": y["val"] - inside["val"].sum()})
    out = pd.concat([q, pd.DataFrame(add)], ignore_index=True) if add else q
    out = out.rename(columns={"val": "revenue"}).sort_values("end").drop_duplicates("end", keep="first")
    return out.reset_index(drop=True)


def revenue_yoy(rev: pd.DataFrame, end: pd.Timestamp, tol_days: int = 20) -> float:
    """분기 끝 end(±tol) 매출 ÷ 1년 전(±tol) 같은 분기 매출 − 1. 못 찾으면 NaN."""
    def at(d):
        m = rev[(rev["end"] - d).abs() <= pd.Timedelta(days=tol_days)]
        return float(m.iloc[0]["revenue"]) if len(m) else np.nan

    a, b = at(end), at(end - pd.Timedelta(days=364))
    return a / b - 1 if not (np.isnan(a) or np.isnan(b)) and b > 0 else np.nan


def accelerated(rev: pd.DataFrame, end: pd.Timestamp) -> bool | None:
    """정답: 이번 분기 매출 YoY > 직전 분기 YoY. 둘 중 하나라도 없으면 None."""
    prev = rev[rev["end"] < end - pd.Timedelta(days=20)]
    if prev.empty:
        return None
    y1, y0 = revenue_yoy(rev, end), revenue_yoy(rev, prev["end"].iloc[-1])
    if np.isnan(y1) or np.isnan(y0):
        return None
    return bool(y1 > y0)


def announcement_date(filings: list[dict], period_end: pd.Timestamp, max_days: int = 100) -> pd.Timestamp | None:
    """실적 발표일: 분기 끝 뒤 max_days일 안의 첫 8-K(항목 2.02) 또는 10-Q·10-K 공시일 중 이른 날."""
    cands = []
    for f in filings:
        d = pd.Timestamp(f["filingDate"])
        if not (period_end < d <= period_end + pd.Timedelta(days=max_days)):
            continue
        if (f["form"] == "8-K" and "2.02" in str(f.get("items") or "")) or f["form"] in ("10-Q", "10-K"):
            cands.append(d)
    return min(cands) if cands else None


def excess_returns(stock: pd.Series, bench: pd.Series, announced: pd.Timestamp, windows: dict[str, list[int]]) -> dict[str, float | None]:
    """발표일(그날 또는 다음 거래일 = 위치 0) 기준 [a, b] 거래일 종가 수익률 − QQQ 같은 기간. 끝 위치가 아직 없으면 None(나중에 채점)."""
    idx = stock.dropna().index.intersection(bench.dropna().index)
    p0 = int(np.searchsorted(idx, pd.Timestamp(announced)))
    out = {}
    for name, (a, b) in windows.items():
        i, j = p0 + a, p0 + b
        if i < 0 or j >= len(idx):
            out[name] = None
            continue
        s = stock[idx[j]] / stock[idx[i]] - 1
        q = bench[idx[j]] / bench[idx[i]] - 1
        out[name] = float(s - q)
    return out
