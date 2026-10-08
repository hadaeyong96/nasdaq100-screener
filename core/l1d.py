"""L1d 해외 시장 200일선 2배·3배 전환 — 금리 연결·목표 자산·봉인 (docs/l1d_plan.md, configs/l1d_preregistration.yaml).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 신호·장부·레버리지 합성은 core/l1b.py를 그대로 쓴다
(target_b1, simulate, synthetic_leveraged_returns). 미래 데이터 금지: 음수 shift를 쓰지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import l1b
from core import regime as rg

SEAL_DATE = pd.Timestamp("2021-12-31")
SealError = rg.SealError
ASSET_1X, ASSET_2X, ASSET_3X, ASSET_CASH = "1x", "2x", "3x", "cash"
LEV_ASSET = {"G1": ASSET_2X, "G2": ASSET_3X}


def seal(obj, name: str = "data"):
    """2021-12-31 뒤 행이 있으면 SealError, 없으면 obj 그대로."""
    return rg.enforce_seal(obj, SEAL_DATE, name)


def cut_and_seal(obj, name: str = "data"):
    """2021-12-31까지 자른 뒤 다시 검사한다."""
    return seal(rg.truncate_to_seal(obj, SEAL_DATE), name)


def clean_index(close: pd.Series) -> tuple[pd.Series, dict]:
    """지수 종가 정리(구현 해석 2번): NaN·0 이하 행 제거, 같은 날짜는 마지막 행. 출력: (Series, 제거 수 dict)"""
    s = close.sort_index()
    dup = int(s.index.duplicated(keep="last").sum())
    s = s[~s.index.duplicated(keep="last")]
    nan = int(s.isna().sum())
    s = s.dropna()
    nonpos = int((s <= 0).sum())
    s = s[s > 0]
    return s.astype(float), {"duplicates": dup, "nan": nan, "nonpositive": nonpos}


def chain_monthly_rates(series_by_id: dict[str, pd.Series], priority: list[str], months: pd.PeriodIndex) -> pd.DataFrame:
    """달마다 priority 순서로 처음 값이 있는 시리즈의 연율 %를 고른다(구현 해석 6번).

    입력: series_by_id {시리즈 ID: Series(인덱스 = 월 Period 또는 그달 날짜, 값 = 연율 %)}, priority, months
    출력: DataFrame(인덱스 = months, 열 rate_pct, source). 어느 시리즈에도 없는 달은 rate_pct NaN, source None.
    """
    norm = {}
    for sid, s in series_by_id.items():
        s = s.dropna()
        idx = s.index if isinstance(s.index, pd.PeriodIndex) else pd.DatetimeIndex(s.index).to_period("M")
        norm[sid] = pd.Series(s.values, index=idx).groupby(level=0).mean()
    rate = pd.Series(np.nan, index=months, dtype=float)
    source = pd.Series(None, index=months, dtype=object)
    for sid in priority:
        s = norm.get(sid)
        if s is None:
            continue
        fill = rate.isna() & months.isin(s.index)
        rate[fill] = s.reindex(months[fill]).values
        source[fill] = sid
    return pd.DataFrame({"rate_pct": rate, "source": source})


def source_periods(chain: pd.DataFrame) -> list[dict]:
    """달별 출처를 연속 구간으로 묶는다. 출력: [{source, start(YYYY-MM), end, months}]"""
    out = []
    for m, src in chain["source"].items():
        if out and out[-1]["source"] == src:
            out[-1]["end"] = str(m)
            out[-1]["months"] += 1
        else:
            out.append({"source": src, "start": str(m), "end": str(m), "months": 1})
    return out


def daily_rate(days: pd.DatetimeIndex, chain: pd.DataFrame, divisor: int = 252) -> pd.Series:
    """M월 연율 %를 그달 모든 거래일에: 행당 금리 = 연율/100/divisor (구현 해석 7번). 없는 달은 NaN."""
    m = pd.DatetimeIndex(days).to_period("M")
    return pd.Series(chain["rate_pct"].reindex(m).to_numpy() / 100.0 / divisor, index=days)


def market_returns(close: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """1배·2배·3배·국채 행 수익 표 (레버리지 = core.l1b.synthetic_leveraged_returns). 첫 행 지수 수익 0."""
    r = close.pct_change().fillna(0.0)
    return pd.DataFrame({
        ASSET_1X: r,
        ASSET_2X: l1b.synthetic_leveraged_returns(r, rf, 2.0),
        ASSET_3X: l1b.synthetic_leveraged_returns(r, rf, 3.0),
        ASSET_CASH: rf,
    })


def target(candidate: str, level: pd.Series, sma_days: int = 200) -> pd.Series:
    """G0 늘 1배 / G1 위 2배·아래 국채 / G2 위 3배·아래 국채 (L1b B1 규칙, 같으면 아래). SMA 미산출 None."""
    if candidate == "G0":
        return pd.Series(ASSET_1X, index=level.index, dtype=object)
    base = l1b.target_b1(level, sma_days)  # "2x" | "cash" | None
    lev = LEV_ASSET[candidate]
    return base.map(lambda a: lev if a == l1b.ASSET_2X else a)


def held_after_close(tgt: pd.Series) -> pd.Series:
    """t일 장 마감 후 보유 = t−1일 신호 (다음 거래일 종가 체결, 양수 shift만)."""
    return tgt.shift(1)
