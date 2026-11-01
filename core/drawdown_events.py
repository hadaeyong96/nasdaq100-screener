"""폭락 연구 1 관문 1 — 하락 사건 표·반토막 진입 분석 (docs/prereg/shock_dip.md, 참고 분석·시험 아님).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 음수 shift를 쓰지 않는다.
사건 표 자체는 사후 기술(전체 기간을 보고 고점·바닥을 찾음)이고, 잠정 유형에 쓰는 지표 값은
as-of 함수로 "그날까지 알려진 값"만 고른다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def find_drawdown_events(close: pd.Series, threshold: float = -0.15) -> pd.DataFrame:
    """종가 기준 직전 최고가 대비 threshold 이하가 된 구간을 사건으로 뽑는다.

    한 사건 = 고점 다음 날부터 종가가 다시 그 고점 이상이 되는 날 전까지(물속 구간). 회복 전에 다시 빠지면
    같은 사건으로 묶인다. 입력: 날짜 인덱스 종가 Series(정렬·결측 없음), threshold(음수 비율).
    출력: DataFrame 열 peak_date, peak, hit_date(처음 threshold 이하), trough_date, trough, max_dd,
    peak_to_trough_days, recovery_date(NaT=미회복), peak_to_recovery_days(NaN=미회복),
    fast20(고점 뒤 20거래일 안에 고점 대비 −10% 이하), any20(고점~바닥 사이 20거래일 수익률 ≤ −10%인 날 있음),
    peak_pos, trough_pos, end_pos(회복일 위치 또는 len).
    """
    s = close.astype(float)
    v = s.to_numpy()
    idx = s.index
    n = len(v)
    cummax = np.maximum.accumulate(v)
    under = v < cummax
    ret20 = s / s.shift(20) - 1.0
    rows = []
    i = 0
    while i < n:
        if not under[i]:
            i += 1
            continue
        start = i
        while i < n and under[i]:
            i += 1
        end = i  # 회복일 위치(=len이면 미회복)
        peak_pos = start - 1
        peak = v[peak_pos]
        seg = v[start:end]
        dd = seg / peak - 1.0
        if dd.min() > threshold:
            continue
        trough_pos = start + int(np.argmin(seg))
        hit_pos = start + int(np.argmax(dd <= threshold))
        fast_win = v[peak_pos + 1 : min(peak_pos + 21, n)]
        r20 = ret20.iloc[peak_pos + 1 : trough_pos + 1]
        rows.append(
            {
                "peak_date": idx[peak_pos],
                "peak": peak,
                "hit_date": idx[hit_pos],
                "trough_date": idx[trough_pos],
                "trough": v[trough_pos],
                "max_dd": v[trough_pos] / peak - 1.0,
                "peak_to_trough_days": trough_pos - peak_pos,
                "recovery_date": idx[end] if end < n else pd.NaT,
                "peak_to_recovery_days": float(end - peak_pos) if end < n else np.nan,
                "fast20": bool(len(fast_win) and fast_win.min() / peak - 1.0 <= -0.10),
                "any20": bool((r20 <= -0.10).any()),
                "peak_pos": peak_pos,
                "trough_pos": trough_pos,
                "end_pos": end,
            }
        )
    return pd.DataFrame(rows)


def first_touch_pos(close: pd.Series, event: pd.Series, level: float) -> int | None:
    """사건 안에서 종가가 고점 × (1 + level) 이하가 된 첫 위치. 없으면 None. level 예: −0.30."""
    v = close.to_numpy(dtype=float)
    seg = v[event["peak_pos"] + 1 : event["end_pos"]]
    hits = np.nonzero(seg <= event["peak"] * (1.0 + level))[0]
    return None if len(hits) == 0 else int(event["peak_pos"] + 1 + hits[0])


def first_above_sma_pos(close: pd.Series, after_pos: int, window: int = 50) -> int | None:
    """after_pos 다음 날부터 종가 > window일 단순이동평균인 첫 위치. 없으면 None."""
    sma = close.rolling(window).mean().to_numpy()
    v = close.to_numpy(dtype=float)
    for p in range(after_pos + 1, len(v)):
        if not np.isnan(sma[p]) and v[p] > sma[p]:
            return p
    return None


def forward_return(close: pd.Series, pos: int, days: int) -> float:
    """pos 종가에 사서 days거래일 뒤 종가까지 가격 수익률. 데이터가 모자라면 NaN."""
    if pos + days >= len(close):
        return np.nan
    return float(close.iloc[pos + days] / close.iloc[pos] - 1.0)


def further_drawdown(close: pd.Series, pos: int, end_pos: int) -> float:
    """pos 종가 대비, pos부터 사건 회복 전(end_pos 제외)까지의 최저 종가 비율 − 1. pos ≥ end_pos면 NaN."""
    if pos >= end_pos:
        return np.nan
    seg = close.iloc[pos:end_pos]
    return float(seg.min() / close.iloc[pos] - 1.0)


def asof_value(values: pd.Series, available_from: pd.Series, day: pd.Timestamp) -> float:
    """available_from(값별 공개일) ≤ day인 값 중 가장 최근 관측의 값. 없으면 NaN.

    입력: values·available_from은 같은 인덱스(관측 시점). 출력: float.
    """
    ok = available_from.reindex(values.index) <= day
    known = values[ok.fillna(False).to_numpy()]
    return float(known.iloc[-1]) if len(known) else np.nan


def expanding_percentile(history: pd.Series, value: float, min_obs: int) -> float:
    """history(그날까지 알려진 값들) 안에서 value 이하 비율(0~1). 표본이 min_obs 미만이거나 value가 NaN이면 NaN."""
    h = history.dropna()
    if np.isnan(value) or len(h) < min_obs:
        return np.nan
    return float((h <= value).mean())


def provisional_type(sahm: float, baa_pct: float, cape_pct: float, dtb3_chg_12m: float, fast: bool,
                     sahm_cut: float = 0.5, pct_cut: float = 0.9, tight_cut: float = 1.0) -> str:
    """그날 정보만으로 붙이는 잠정 유형(참고용 분류, 시험 규칙 아님).

    깃발: 침체(샴 ≥ sahm_cut 또는 BAA10Y 백분위 ≥ pct_cut), 거품(CAPE 백분위 ≥ pct_cut),
    긴축(DTB3 12개월 변화 ≥ tight_cut %p). 깃발 2개 이상 → "혼합(…)", 1개 → 그 이름.
    깃발 0개: 입력이 모두 있고 fast면 "충격", 그 밖은 "UNKNOWN".
    """
    flags = []
    if (not np.isnan(sahm) and sahm >= sahm_cut) or (not np.isnan(baa_pct) and baa_pct >= pct_cut):
        flags.append("침체")
    if not np.isnan(cape_pct) and cape_pct >= pct_cut:
        flags.append("거품")
    if not np.isnan(dtb3_chg_12m) and dtb3_chg_12m >= tight_cut:
        flags.append("긴축")
    if len(flags) >= 2:
        return "혼합(" + "·".join(flags) + ")"
    if len(flags) == 1:
        return flags[0]
    complete = not any(np.isnan(x) for x in (sahm, baa_pct, cape_pct, dtb3_chg_12m))
    return "충격" if (complete and fast) else "UNKNOWN"
