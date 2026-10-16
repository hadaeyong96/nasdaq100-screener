"""가상 장부 「볼린저·RSI·MACD 따라하기」 신호 — 일목균형표가 맡던 자리만 볼린저 밴드로 바꾼 규칙.

기존 core/signals.py·core/filters.py는 건드리지 않는다. 이 모듈은 볼린저 열을 더하는 함수와,
core.state.process_day에 넣을 신호 묶음(BOLLINGER_RULES)만 정의한다 (docs/design/virtual_books.md).

규칙 (사용자 확정 2026-10-14):
- A1: RSI 30 상향 돌파 그리고 최근 5거래일(오늘 포함) 안에 종가가 볼린저 하단 아래였던 적이 있음
- A2: 기존과 같음 (core.signals.check_a2 — MACD 골든크로스, RSI 30 이상 70 미만)
- A3: 종가 > 중심선, 중심선 > 5거래일 전 중심선, MACD > 시그널, RSI ≥ 50
- B : 종가 > 중심선, 중심선 > 5거래일 전 중심선, MACD 골든크로스, macd_norm ≥ S등급 하한, RSI 50 이상 70 미만
- E3: 종가 < 중심선 그리고 중심선 < 5거래일 전 중심선
- E1·E2·손절·A1 만료: 기존과 같음 (core.state가 core.signals 함수로 판정)
- 매매 금지: 기존 filters.ban_reasons 그대로(골든크로스 날 RSI 70 이상, 휩소, 갭, 실적) — 구름 관련 금지
  (종가가 구름 안·앞구름 음운)만 뺀다 (구름 자리를 볼린저로 바꿨으므로).
- 우선순위 점수: 등급(S40/A25/B10) + 거래량(1.5배 25 / 1.2배 15) + 밴드폭 수축 후 확장 가점.
  가점 = 전일 밴드폭이 최근 120거래일 중 하위 20% 안이고 오늘 밴드폭이 전일보다 넓음.
  [가정] 가점 점수는 구름 두께 항목의 최대 점수와 같은 20점(config virtual_books.bollinger.squeeze_bonus).
- [가정] A3 확정 때 손절선은 올리지 않는다(일목 규칙의 "구름 하단으로 상향" 자리에 볼린저 대응 규칙이 없음).

미래 데이터 금지: rolling·shift(양수)만 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import filters
from core import signals as sig
from core.state import SignalRules, _entry_score

_CLOUD_FIELDS = ("cloud_top", "cloud_bot", "future_yang")


def add_bollinger_columns(df: pd.DataFrame, bb_cfg: dict) -> pd.DataFrame:
    """지표가 계산된 df에 볼린저 신호용 열을 더한 새 DataFrame.

    입력: df(close 포함, 날짜 오름차순), bb_cfg({period, std_mult, ddof, slope_days, touch_days,
         width_rank_period, width_rank_max, squeeze_bonus})
    출력 열: bb_mid, bb_upper, bb_lower, bb_width(=(상단−하단)/중심선), bb_touch_lower(최근 touch_days
         거래일 안에 종가 < 하단이 있었음), bb_mid_up(중심선 > slope_days 전), bb_mid_down(중심선 < slope_days 전),
         bb_squeeze_release(전일 밴드폭 백분위 ≤ width_rank_max 그리고 오늘 밴드폭 > 전일)
    """
    out = df.copy()
    close = out["close"]
    n = int(bb_cfg["period"])
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=int(bb_cfg["ddof"]))
    upper = mid + float(bb_cfg["std_mult"]) * sd
    lower = mid - float(bb_cfg["std_mult"]) * sd
    width = (upper - lower) / mid
    k = int(bb_cfg["slope_days"])
    below = (close < lower).astype(float).where(lower.notna())
    rank = width.rolling(int(bb_cfg["width_rank_period"])).rank(pct=True)

    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bb_width"] = width
    out["bb_touch_lower"] = below.rolling(int(bb_cfg["touch_days"]), min_periods=1).max() == 1.0
    out["bb_mid_up"] = (mid > mid.shift(k)).where(mid.shift(k).notna())
    out["bb_mid_down"] = (mid < mid.shift(k)).where(mid.shift(k).notna())
    release = (rank.shift(1) <= float(bb_cfg["width_rank_max"])) & (width > width.shift(1))
    out["bb_squeeze_release"] = release.where(rank.shift(1).notna() & width.shift(1).notna())
    out.attrs["squeeze_bonus"] = int(bb_cfg["squeeze_bonus"])
    return out


def _true(v) -> bool:
    return sig._bool(v)


def check_a1_bb(prev_rsi: float, row: pd.Series, cfg: dict) -> bool:
    """A1: RSI 30 상향 돌파 그리고 최근 5거래일 안에 종가 < 볼린저 하단이 있었음."""
    return sig.check_a1(prev_rsi, row.get("rsi"), cfg) and _true(row.get("bb_touch_lower"))


def _trend_up(row: pd.Series) -> bool:
    close, mid = row.get("close"), row.get("bb_mid")
    if pd.isna(close) or pd.isna(mid):
        return False
    return bool(close > mid and _true(row.get("bb_mid_up")))


def check_a3_bb(row: pd.Series, cfg: dict) -> bool:
    """A3: 종가 > 중심선, 중심선 상승(5거래일 전 대비), MACD > 시그널, RSI ≥ 50."""
    if any(pd.isna(row.get(f)) for f in ("macd", "signal", "rsi")):
        return False
    return _trend_up(row) and bool(row["macd"] > row["signal"] and row["rsi"] >= sig.levels(cfg)["rsi_mid"])


def check_b_bb(row: pd.Series, cfg: dict) -> bool:
    """B: 종가 > 중심선 그리고 중심선 상승 + 기존 B의 MACD·RSI 조건(골든크로스, macd_norm ≥ S하한, RSI 50~70)."""
    if any(pd.isna(row.get(f)) for f in ("gc", "macd_norm", "rsi")):
        return False
    lv = sig.levels(cfg)
    return _trend_up(row) and bool(
        row["gc"]
        and row["macd_norm"] >= cfg["assumptions"]["s_grade_macd_norm_min_pct"]
        and lv["rsi_mid"] <= row["rsi"] < lv["rsi_overbought"]
    )


def check_e3_bb(row: pd.Series) -> bool:
    """E3: 종가 < 중심선 그리고 중심선 하락(5거래일 전 대비)."""
    close, mid = row.get("close"), row.get("bb_mid")
    if pd.isna(close) or pd.isna(mid):
        return False
    return bool(close < mid and _true(row.get("bb_mid_down")))


def _no_cloud(row: pd.Series) -> pd.Series:
    out = row.copy()
    for f in _CLOUD_FIELDS:
        if f in out.index:
            out[f] = np.nan
    return out


def ban_reasons_bb(df: pd.DataFrame, date, row: pd.Series, stage: str, cfg: dict, earnings_date) -> list[str]:
    """기존 매매 금지 그대로, 구름 관련 금지만 뺀다(구름 열을 NaN으로 넘기면 filters가 '금지 아님'으로 본다)."""
    idx = df.index.get_loc(date)
    gc_count = filters.macd_cross_count(df, date, cfg=cfg) if stage in ("A2", "B") else 0
    prev_close = float(df.iloc[idx - 1]["close"]) if idx > 0 else float("nan")
    return filters.ban_reasons(stage=stage, row=_no_cloud(row), gc_count_20d=gc_count, prev_close=prev_close,
                               cfg=cfg, earnings_date=earnings_date)


def entry_score_bb(df: pd.DataFrame, date, row: pd.Series, kind: str, cfg: dict) -> tuple[str | None, int]:
    """등급 + 거래량 + 밴드폭 수축 후 확장 가점 (구름 두께·기존 밴드폭 백분위 항목 대신)."""
    grade_letter, _ = _entry_score(df, date, _no_cloud(row), kind, cfg)
    base = filters.priority_score(grade_letter, row.get("vol_ratio"), float("nan"), float("nan"), cfg)
    bonus = int(df.attrs.get("squeeze_bonus", 20)) if _true(row.get("bb_squeeze_release")) else 0
    return grade_letter, base + bonus


BOLLINGER_RULES = SignalRules(
    name="bollinger",
    a1=check_a1_bb,
    a2=sig.check_a2,
    a3=check_a3_bb,
    b=check_b_bb,
    e3=check_e3_bb,
    ban=ban_reasons_bb,
    score=entry_score_bb,
    a3_stop=lambda row, old_stop: old_stop,
)
