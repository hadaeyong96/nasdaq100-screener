"""내부자 집단 매수 — 2022-01-01 이후 데이터 차단 (docs/prereg/insider_cluster.md 0절).

가격 입구는 gate(), 공시 입구는 gate_filings()(접수일 기준), 계산 입구는 assert_cutoff().
"""

from __future__ import annotations

import pandas as pd

CUTOFF = pd.Timestamp("2021-12-31")  # 이 날 포함
CUTOFF_STR = "2021-12-31"


class CutoffError(RuntimeError):
    """2021-12-31 뒤 날짜가 계산에 들어왔을 때."""


def gate(obj, name: str = ""):
    """가격 입구. 입력: DatetimeIndex Series·DataFrame. 출력: 인덱스 ≤ 2021-12-31 행만 남긴 복사본."""
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise CutoffError(f"{name}: 날짜 인덱스가 아닙니다")
    return obj.loc[obj.index <= CUTOFF].copy()


def gate_filings(df: pd.DataFrame, column: str = "filing_date") -> pd.DataFrame:
    """공시 입구. 입력: column에 Timestamp가 있는 표. 출력: 접수일 ≤ 2021-12-31 행만 남긴 복사본(날짜 없음도 버림)."""
    return df.loc[df[column].notna() & (df[column] <= CUTOFF)].copy()


def assert_cutoff(*objs) -> None:
    """계산 입구. 입력 Series·DataFrame(인덱스)·DatetimeIndex 중 2021-12-31 뒤 날짜가 있으면 CutoffError."""
    for o in objs:
        if o is None:
            continue
        idx = o if isinstance(o, pd.DatetimeIndex) else o.index
        if len(idx) and isinstance(idx, pd.DatetimeIndex) and pd.Timestamp(idx.max()) > CUTOFF:
            raise CutoffError(f"2021-12-31 뒤 날짜가 계산에 들어왔습니다(최대 {pd.Timestamp(idx.max()).date()})")


def assert_dates(*series) -> None:
    """계산 입구(열). 입력 날짜 Series 중 2021-12-31 뒤 값이 있으면 CutoffError."""
    for s in series:
        if len(s) and pd.Timestamp(s.max()) > CUTOFF:
            raise CutoffError(f"2021-12-31 뒤 날짜가 계산에 들어왔습니다(최대 {pd.Timestamp(s.max()).date()})")
