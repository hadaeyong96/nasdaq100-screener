"""사이클 실험실 1단계 — 2013-01-01 이후 데이터 차단 (docs/prereg/cycle_lab_discovery.md 0절).

모든 데이터 입구는 gate() 한 함수를 지난다. 계산 함수는 assert_cutoff()로 입력을 검사한다.
ALFRED·FRED 요청 인자는 check_request()로 요청 전에 검사한다.
"""

from __future__ import annotations

import pandas as pd

CUTOFF = pd.Timestamp("2012-12-31")  # 이 날 포함
CUTOFF_STR = "2012-12-31"


class CutoffError(RuntimeError):
    """2012-12-31 뒤 날짜가 계산·요청에 들어왔을 때."""


def gate(obj, name: str = ""):
    """데이터 입구. 입력: DatetimeIndex Series·DataFrame. 출력: 인덱스 ≤ 2012-12-31 행만 남긴 복사본.

    저장·계산 전에 반드시 이 함수를 지난다. 인덱스가 날짜가 아니면 CutoffError.
    """
    if not isinstance(obj.index, pd.DatetimeIndex):
        raise CutoffError(f"{name}: 날짜 인덱스가 아닙니다")
    return obj.loc[obj.index <= CUTOFF].copy()


def assert_cutoff(*objs) -> None:
    """계산 함수 입구. 입력 Series·DataFrame·DatetimeIndex 중 하나라도 2012-12-31 뒤 날짜가 있으면 CutoffError."""
    for o in objs:
        if o is None:
            continue
        idx = o if isinstance(o, pd.DatetimeIndex) else o.index
        if len(idx) and pd.Timestamp(idx.max()) > CUTOFF:
            raise CutoffError(f"2012-12-31 뒤 날짜가 계산에 들어왔습니다(최대 {pd.Timestamp(idx.max()).date()})")


def check_request(params: dict, alfred: bool) -> dict:
    """FRED·ALFRED 요청 인자 검사. 출력: params 그대로.

    observation_end가 없거나 2012-12-31 뒤면 CutoffError. alfred=True면 realtime_end(vintage 끝)가
    없거나 2012-12-31 뒤여도 CutoffError. alfred=False(3순위 현재값)는 realtime_end를 넣지 않는다.
    """
    obs_end = params.get("observation_end")
    if obs_end is None or pd.Timestamp(obs_end) > CUTOFF:
        raise CutoffError("observation_end가 없거나 2012-12-31 뒤입니다")
    rt_end = params.get("realtime_end")
    if alfred and (rt_end is None or pd.Timestamp(rt_end) > CUTOFF):
        raise CutoffError("ALFRED vintage(realtime_end)가 없거나 2012-12-31 뒤입니다")
    if not alfred and rt_end is not None:
        raise CutoffError("현재값 요청에는 realtime_end를 넣지 않습니다")
    return params
