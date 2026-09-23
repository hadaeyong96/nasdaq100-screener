"""정리본 7장(필터와 신호 등급)의 등급·우선순위 점수·매매 금지 구간을 판정하는
순수 함수 모음. core/의 다른 모듈과 동일하게 네트워크·파일·현재 시각에 접근하지
않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def macd_cross_count(df: pd.DataFrame, date, window: int = 20) -> int:
    """date를 포함해 최근 window거래일 동안의 MACD 교차(gc 또는 dc) 횟수.

    입력: gc·dc 열이 있는 DataFrame, date(df.index에 있는 값)
    출력: 교차 횟수 (정수)
    """
    i = df.index.get_loc(date)
    start = max(0, i - window + 1)
    segment = df.iloc[start : i + 1]
    crosses = segment["gc"].fillna(False) | segment["dc"].fillna(False)
    return int(crosses.sum())


def grade(macd_norm: float, gc_count_20d: int, cfg: dict) -> str | None:
    """A2·B형 등급(S/A/B)을 정한다 (7장).

    입력이 NaN이면 등급을 매길 수 없으므로 None.
    """
    if pd.isna(macd_norm):
        return None
    s_min = cfg["assumptions"]["s_grade_macd_norm_min_pct"]
    b_max = cfg["assumptions"]["b_grade_macd_norm_max_pct"]
    if macd_norm >= s_min:
        return "S"
    if macd_norm >= b_max and gc_count_20d <= 1:
        return "A"
    return "B"


def priority_score(
    grade_letter: str | None,
    vol_ratio: float,
    cloud_thickness_pct: float,
    bb_width_pct: float,
    cfg: dict,
) -> int:
    """우선순위 점수(100점 만점)를 계산한다 (7장).

    - 등급: S 40 / A 25 / B 10
    - 거래량(신호일 ÷ 20일 평균): 1.5배 이상 25 / 1.2배 이상 15
    - 위쪽 구름 두께 ÷ 종가: 3% 이하 20 / 6% 이하 10
    - 볼린저 밴드폭 120일 백분위: 20% 이하 15
    각 항목 값이 NaN이면 그 항목은 0점으로 둔다 (신호 자체를 막지 않는다).
    """
    score = 0
    score += {"S": 40, "A": 25, "B": 10}.get(grade_letter, 0)

    if not pd.isna(vol_ratio):
        if vol_ratio >= 1.5:
            score += 25
        elif vol_ratio >= 1.2:
            score += 15

    if not pd.isna(cloud_thickness_pct):
        if cloud_thickness_pct <= 3:
            score += 20
        elif cloud_thickness_pct <= 6:
            score += 10

    if not pd.isna(bb_width_pct) and bb_width_pct <= 0.20:
        score += 15

    return score


def cloud_thickness_pct(cloud_top: float, cloud_bot: float, close: float) -> float:
    """위쪽 구름 두께 ÷ 종가 (%). 입력에 NaN이 있으면 NaN."""
    if pd.isna(cloud_top) or pd.isna(cloud_bot) or pd.isna(close) or close == 0:
        return float("nan")
    return (cloud_top - cloud_bot) / close * 100


def is_earnings_within(date, earnings_date, trading_days: int = 3) -> bool:
    """오늘부터 earnings_date까지가 trading_days 거래일 이내(포함)인지 본다.

    실적일을 모르면(earnings_date=None) 필터링하지 않는다 — 호출부가
    earnings_unknown=True로 표시만 한다 (11장 실적 필터, P2 지시문).
    미국 공휴일 캘린더는 반영하지 않는다 (프로젝트 전반의 기존 가정과 동일).
    """
    if earnings_date is None:
        return False
    d = pd.Timestamp(date).normalize()
    e = pd.Timestamp(earnings_date).normalize()
    if e < d:
        return False
    diff = int(np.busday_count(d.date(), e.date()))
    return diff <= trading_days


def ban_reasons(
    *,
    stage: str,
    row: pd.Series,
    gc_count_20d: int,
    prev_close: float,
    cfg: dict,
    earnings_date=None,
) -> list[str]:
    """stage("A2"|"A3"|"B") 신규 매수에 대한 매매 금지 구간(7장)을 확인한다.

    출력: 금지 이유 목록 (빈 리스트면 금지 없음). NaN인 조건은 "금지 아님"으로
    본다 — 데이터가 없다고 신호를 막지 않는다.
    """
    reasons: list[str] = []

    if stage in ("A2", "B"):
        rsi = row.get("rsi")
        if not pd.isna(rsi) and rsi >= 70:
            reasons.append("골든크로스 당일 RSI 70 이상")
        if gc_count_20d >= cfg["assumptions"]["whipsaw_max_crosses_20d"]:
            reasons.append(f"최근 20거래일 MACD 교차 {gc_count_20d}회 이상 (휩소)")

    if stage in ("A3", "B"):
        close, cloud_top, cloud_bot = row.get("close"), row.get("cloud_top"), row.get("cloud_bot")
        if not (pd.isna(close) or pd.isna(cloud_top) or pd.isna(cloud_bot)):
            if cloud_bot <= close <= cloud_top:
                reasons.append("종가가 구름 안에 있음")
        future_yang = row.get("future_yang")
        if not pd.isna(future_yang) and not future_yang:
            reasons.append("앞구름 음운")

    if stage == "A3":
        open_price = row.get("open")
        if not pd.isna(open_price) and not pd.isna(prev_close) and prev_close:
            gap_pct = (open_price / prev_close - 1) * 100
            if gap_pct >= cfg["assumptions"]["gap_filter_pct"]:
                reasons.append(f"당일 시가 갭 {gap_pct:.1f}% (필터 {cfg['assumptions']['gap_filter_pct']}% 이상)")

    date = row.name
    if is_earnings_within(date, earnings_date):
        reasons.append("실적 발표 3거래일 이내")

    return reasons
