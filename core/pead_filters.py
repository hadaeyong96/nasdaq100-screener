"""실적급등 보완 필터 탐색 — 순수 함수 (docs/research/pead_filters_plan.md, 태그 pead-filters-plan).

필터 F1~F6 판정과 고르는 규칙만 있다. 매매·수익은 core/pead_timing·core/sw1을 그대로 쓴다.
판정 결과는 (True 통과 | False 걸러짐 | None 판정 불가, 사유) 짝. 네트워크·파일 접근 없음.
모든 판정은 반응일 d0 장 마감까지 알 수 있는 값만 본다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FILTERS = ("F1", "F2", "F3", "F4", "F5", "F6")
F1_MIN = 0.75  # 종가 위치 (종가 − 저가) / (고가 − 저가)
F2_WINDOW, F2_RATIO = 252, 0.95
SMA_WINDOW = 200  # F3·F6
REVENUE_TAGS = ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet", "SalesRevenueServicesNet")
OPINC_TAG = "OperatingIncomeLoss"
CUTOFF_HOUR = 16  # d0 16:00 동부
END_TOL = 5  # 이번 분기 끝날 = report_date ± 5일
YAGO_DAYS, YAGO_TOL = 365, 10
PREV_LO, PREV_HI = 80, 105  # 직전 분기 끝날 = 이번 분기 끝날 − 80~105일
Q_DAYS, FY_DAYS, M9_DAYS = (80, 100), (350, 380), (260, 285)
START_TOL = 5

NA_LATE = "분기 보고서 장 마감 뒤"
NA_NOVAL = "XBRL 값 없음"
NA_NOCMP = "비교 분기 값 없음"
NA_BASE = "전년 동기 값 ≤ 0"
NA_OPBASE = "전년 동기 영업이익 ≤ 0"

Verdict = tuple[bool | None, str]


# ── 가격 필터 ────────────────────────────────────────────────────────────────


def f1_hold(high: float, low: float, close: float, thr: float = F1_MIN) -> Verdict:
    """F1 급등 지킴. 입력: d0 같은 출처의 고가·저가·종가. (종가 − 저가)/(고가 − 저가) ≥ thr면 통과."""
    vals = np.array([high, low, close], dtype=float)
    if not np.isfinite(vals).all() or not high > low:
        return None, "고가·저가 없음"
    if not (low <= close <= high):
        return None, "고가·저가·종가 불일치"
    return bool((close - low) / (high - low) >= thr), ""


def _upto(close: pd.Series, d0: pd.Timestamp) -> pd.Series | None:
    """d0까지 자기 행 종가(결측 뺌). d0 행이 없으면 None."""
    h = close.loc[:pd.Timestamp(d0)].dropna()
    return h if len(h) and h.index[-1] == pd.Timestamp(d0) else None


def f2_near_high(close: pd.Series, d0: pd.Timestamp, window: int = F2_WINDOW, ratio: float = F2_RATIO) -> Verdict:
    """F2 신고가 근처. 입력: 그 종목 종가(날짜 오름차순). d0 종가 ≥ d0 직전 window개 종가 최고 × ratio면 통과."""
    h = _upto(close, d0)
    if h is None:
        return None, "반응일 종가 없음"
    prev = h.iloc[:-1].iloc[-window:]
    if len(prev) < window:
        return None, f"직전 종가 {window}개 미만"
    return bool(h.iloc[-1] >= prev.max() * ratio), ""


def above_sma(close: pd.Series, d0: pd.Timestamp, window: int = SMA_WINDOW) -> Verdict:
    """F3(종목)·F6(QQQ) 공통. d0 종가 > d0 포함 최근 window개 종가 평균이면 통과."""
    h = _upto(close, d0)
    if h is None:
        return None, "반응일 종가 없음"
    if len(h) < window:
        return None, f"종가 {window}개 미만"
    return bool(h.iloc[-1] > h.iloc[-window:].mean()), ""


# ── XBRL 필터 ────────────────────────────────────────────────────────────────


def fact_frame(entries: list[dict]) -> pd.DataFrame:
    """companyfacts 한 태그의 USD 사실 목록 → 표(start, end, val, accn, filed, days). start 없는 사실은 버린다."""
    cols = ["start", "end", "val", "accn", "filed"]
    rows = [{k: e.get(k) for k in cols} for e in entries if e.get("start") and e.get("end")]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df["val"] = df["val"].astype(float)
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


def known(df: pd.DataFrame, d0: pd.Timestamp, q_accn: str | None, q_accepted_et: pd.Timestamp | None) -> pd.DataFrame:
    """d0 16:00(동부)까지 알 수 있는 사실만.

    제출일 < d0 → 앎. 제출일 = d0 → accn이 같은 분기 10-Q/10-K(q_accn)이고 그 Accepted < d0 16:00일 때만. 그 밖은 모름.
    """
    d0 = pd.Timestamp(d0).normalize()
    m = df["filed"] < d0
    if q_accn and q_accepted_et is not None and pd.Timestamp(q_accepted_et) < d0 + pd.Timedelta(hours=CUTOFF_HOUR):
        m = m | ((df["filed"] == d0) & (df["accn"] == q_accn))
    return df[m]


def _latest(df: pd.DataFrame) -> float | None:
    return None if df.empty else float(df.sort_values(["filed", "accn"]).iloc[-1]["val"])


def quarter_value(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """끝날이 [lo, hi]인 분기(3개월) 값 → (끝날, 값). 여러 끝날이면 가장 늦은 끝날.

    3개월(80~100일) 사실이 없으면 같은 끝날 연간(350~380일) − 같은 시작일(±5일) 9개월(260~285일)로 만든다.
    같은 기간이 여러 공시에 있으면 가장 늦게 제출된 값.
    """
    w = df[(df["end"] >= lo) & (df["end"] <= hi)]
    q = w[w["days"].between(*Q_DAYS)]
    if len(q):
        e = q["end"].max()
        return e, _latest(q[q["end"] == e])
    fy = w[w["days"].between(*FY_DAYS)]
    for e in sorted(fy["end"].unique(), reverse=True):
        a = fy[fy["end"] == e]
        st = a.sort_values("filed")["start"].iloc[-1]
        m9 = df[df["days"].between(*M9_DAYS) & ((df["start"] - st).abs().dt.days <= START_TOL)
                & (df["end"] >= e - pd.Timedelta(days=PREV_HI)) & (df["end"] <= e - pd.Timedelta(days=PREV_LO))]
        if len(m9):
            return pd.Timestamp(e), _latest(a) - _latest(m9[m9["end"] == m9["end"].max()])
    return None


def _yago(df: pd.DataFrame, end: pd.Timestamp) -> float | None:
    e = pd.Timestamp(end) - pd.Timedelta(days=YAGO_DAYS)
    r = quarter_value(df, e - pd.Timedelta(days=YAGO_TOL), e + pd.Timedelta(days=YAGO_TOL))
    return None if r is None else r[1]


def yoy(df: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[str, pd.Timestamp | None, float | None]:
    """끝날 [lo, hi] 분기의 전년 동기 대비 증가율. 출력: ('' | 판정 불가 사유, 끝날, 증가율)."""
    r = quarter_value(df, lo, hi)
    if r is None:
        return NA_NOVAL, None, None
    base = _yago(df, r[0])
    if base is None:
        return NA_NOCMP, r[0], None
    if not base > 0:
        return NA_BASE, r[0], None
    return "", r[0], r[1] / base - 1


def revenue_yoy(tags: dict[str, pd.DataFrame], lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[str, pd.Timestamp | None, float | None]:
    """매출 태그 우선순위대로, 이번 값·전년 동기 값이 같은 태그로 다 있는 첫 태그의 증가율. 없으면 첫 사유."""
    first = None
    for t in REVENUE_TAGS:
        if t not in tags:
            continue
        why, e, g = yoy(tags[t], lo, hi)
        if not why:
            return why, e, g
        if first is None or (first[0] == NA_NOVAL and why != NA_NOVAL):
            first = (why, e, g)
    return first if first is not None else (NA_NOVAL, None, None)


def _cur_window(report_end: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    e = pd.Timestamp(report_end)
    return e - pd.Timedelta(days=END_TOL), e + pd.Timedelta(days=END_TOL)


def f4_rev_accel(tags: dict[str, pd.DataFrame], report_end: pd.Timestamp, q_late: bool) -> Verdict:
    """F4 매출 가속. 입력: 태그 → 알 수 있는 사실 표(known 적용 뒤), report_end(이번 분기 끝날), q_late(같은 분기 보고서가
    d0 장 마감 뒤 — 사유 표시에만 씀). 이번 분기 매출 YoY > 직전 분기 매출 YoY면 통과."""
    lo, hi = _cur_window(report_end)
    why, e, g1 = revenue_yoy(tags, lo, hi)
    if why:
        return None, NA_LATE if (why == NA_NOVAL and q_late) else why
    why, _, g0 = revenue_yoy(tags, e - pd.Timedelta(days=PREV_HI), e - pd.Timedelta(days=PREV_LO))
    if why:
        return None, NA_NOCMP if why == NA_NOVAL else why
    return bool(g1 > g0), ""


def f5_margin_up(tags: dict[str, pd.DataFrame], report_end: pd.Timestamp, q_late: bool) -> Verdict:
    """F5 이익률 확대. 이번 분기 영업이익 YoY > 이번 분기 매출 YoY면 통과. 전년 동기 영업이익 ≤ 0 → 판정 불가."""
    lo, hi = _cur_window(report_end)
    why, _, gr = revenue_yoy(tags, lo, hi)
    if why:
        return None, NA_LATE if (why == NA_NOVAL and q_late) else why
    if OPINC_TAG not in tags:
        return None, NA_NOVAL
    why, _, go = yoy(tags[OPINC_TAG], lo, hi)
    if why:
        return None, {NA_BASE: NA_OPBASE}.get(why, NA_LATE if (why == NA_NOVAL and q_late) else why)
    return bool(go > gr), ""


# ── 고르는 규칙 (계획 5절) ───────────────────────────────────────────────────


def passes(row: dict, base: dict, min_frac: float) -> dict:
    """한 판(row: mean·p1·p2·n)이 기준(base)보다 두 구간 모두 높고 매매 수 ≥ 기준 × min_frac인지. 출력: 칸별 bool과 ok."""
    a, b = bool(row["p1"] > base["p1"]), bool(row["p2"] > base["p2"])
    c = bool(row["n"] >= base["n"] * min_frac)
    return {"p1_up": a, "p2_up": b, "enough": c, "ok": a and b and c}


def rank(rows: list[dict]) -> list[dict]:
    """평균 초과 큰 순, 동점이면 매매 수 많은 순, 그다음 이름 순."""
    return sorted(rows, key=lambda r: (-r["mean"], -r["n"], r["name"]))


def pick_pair(singles: list[dict], base: dict, min_frac: float = 0.40) -> tuple[list[dict], tuple[str, str] | None]:
    """1단계: 통과 단일 필터 목록과 겹칠 상위 2개 이름(통과 2개 미만이면 None). 동점은 필터 번호 작은 쪽."""
    ok = [r for r in singles if passes(r, base, min_frac)["ok"]]
    top = sorted(ok, key=lambda r: (-r["mean"], r["name"]))[:2]
    return ok, (tuple(sorted(r["name"] for r in top)) if len(top) == 2 else None)


def final_choice(passed: list[dict], combo: dict | None, base: dict, combo_frac: float = 0.25, k: int = 2) -> tuple[list[dict], bool]:
    """최종 선택(최대 k개)과 겹친 판 후보 여부. 겹친 판은 두 구간 > 기준 그리고 매매 수 ≥ 기준 × combo_frac일 때만 후보."""
    combo_ok = combo is not None and passes(combo, base, combo_frac)["ok"]
    cands = list(passed) + ([combo] if combo_ok else [])
    return rank(cands)[:k], combo_ok
