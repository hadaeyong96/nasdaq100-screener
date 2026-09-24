"""정리본 4장(진입: A1·A2·A3·B)과 5장(청산: E1·E2·E3, 손절, A1 만료)의 조건을
날짜 하루 단위로 판정하는 순수 함수 모음.

- core/의 다른 모듈과 마찬가지로 네트워크·파일·DB·현재 시각에 접근하지 않는다.
- 필요한 입력값이 NaN이면 그 신호는 "없음"(False)으로 판정한다. 데이터가 부족해
  일목 구름 등을 아직 계산할 수 없는 종목도 A1·A2처럼 짧은 지표만 쓰는 신호는
  판정할 수 있어야 한다 (P2 지시문 2번 — SPCX 사례).
- 상태 전이(어떤 조건에서 실제로 이벤트를 내보낼지, 손절이 최우선인지 등)는
  core/state.py의 책임이다. 이 모듈은 "오늘 그 조건이 성립하는가"만 답한다.
"""

from __future__ import annotations

import pandas as pd


def _bool(value) -> bool:
    """NaN이 섞인 값을 안전하게 bool로 바꾼다. NaN이면 False."""
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def check_a1(prev_rsi: float, rsi: float) -> bool:
    """A1 정찰: RSI가 전일 30 미만에서 당일 30 이상으로 상향 돌파."""
    if pd.isna(prev_rsi) or pd.isna(rsi):
        return False
    return bool(prev_rsi < 30 and rsi >= 30)


def check_a2(row: pd.Series) -> bool:
    """A2 확인: 오늘 MACD 골든크로스, RSI 30 이상 70 미만.

    A1 이후 10거래일 이내인지(유효기간)는 core/state.py가 a1_date와
    cfg["assumptions"]["a1_to_a2_expiry_days"]로 별도 확인한다.

    [해석, P2.1 보완 2번] 정리본의 "RSI 30 이상 70 미만 유지"는 A2 판정 당일의
    RSI만 본다. A1 이후 중간에 RSI가 다시 30 아래로 내려간 날이 있어도 A1을
    무효로 하지 않는다 — 그 구간의 하락 위험은 손절 규칙(5장)이 이미 관리한다.
    """
    if pd.isna(row.get("gc")) or pd.isna(row.get("rsi")):
        return False
    return bool(row["gc"] and 30 <= row["rsi"] < 70)


def check_a1_expired(bars_since_a1: int, expiry_days: int) -> bool:
    """A1 만료: A1 당일을 포함해 expiry_days거래일 안에 A2가 없으면 만료.

    입력: bars_since_a1(A1일=0인 거래일 오프셋), expiry_days(설정값, 기본 10)
    출력: 만료 여부. bars_since_a1 <= expiry_days-1 이면 아직 유효.
    """
    return bars_since_a1 >= expiry_days


def check_a3_breakout(row: pd.Series, cfg: dict) -> bool:
    """A3 확정(돌파형): 4가지 조건이 오늘 모두 충족되는가.

    1) 종가 > 오늘 가격 아래 구름의 상단(cloud_top)
    2) 앞구름 양운(future_yang)
    3) 후행스팬 돌파(chikou_ok)
    4) MACD > 시그널, RSI 50 이상 [확장]
    갭 필터(당일 시가 갭 4% 이상)는 core/filters.py의 매매 금지 구간에서 처리한다.
    """
    fields = ["close", "cloud_top", "future_yang", "chikou_ok", "macd", "signal", "rsi"]
    if any(pd.isna(row.get(f)) for f in fields):
        return False
    return bool(
        row["close"] > row["cloud_top"]
        and row["future_yang"]
        and row["chikou_ok"]
        and row["macd"] > row["signal"]
        and row["rsi"] >= 50
    )


def check_a3_pullback(row: pd.Series, cfg: dict) -> bool:
    """A3 확정(되돌림형): 돌파 조건을 충족한 상태에서, 저가가 구름 상단 또는
    기준선의 pullback_tolerance_pct 이내까지 내려왔다가 양봉으로 마감.
    """
    if not check_a3_breakout(row, cfg):
        return False
    if any(pd.isna(row.get(f)) for f in ["low", "open", "kijun"]):
        return False
    tol = cfg["a3"]["pullback_tolerance_pct"] / 100
    near_cloud_top = row["low"] <= row["cloud_top"] * (1 + tol)
    near_kijun = row["low"] <= row["kijun"] * (1 + tol)
    bullish = row["close"] > row["open"]
    return bool((near_cloud_top or near_kijun) and bullish)


def check_a3(row: pd.Series, cfg: dict) -> bool:
    """A3 확정 판정. cfg["a3"]["mode"]로 돌파형/되돌림형을 고른다."""
    mode = cfg["a3"]["mode"]
    if mode == "pullback":
        return check_a3_pullback(row, cfg)
    return check_a3_breakout(row, cfg)


def check_b(row: pd.Series, cfg: dict) -> bool:
    """B형 추세 재진입: 추세 확인 + MACD 골든크로스(0선 근처 이상) + RSI 50~70.

    - 추세 확인: 종가 > 구름 상단, 앞구름 양운, 후행스팬 돌파
    - MACD: 골든크로스, macd_norm이 S등급 하한(기본 -0.5%) 이상
    - RSI: 50 이상 70 미만 (70 이상 골든크로스는 신규 매수 금지 — filters.py)
    """
    fields = ["close", "cloud_top", "future_yang", "chikou_ok", "gc", "macd_norm", "rsi"]
    if any(pd.isna(row.get(f)) for f in fields):
        return False
    s_min = cfg["assumptions"]["s_grade_macd_norm_min_pct"]
    return bool(
        row["close"] > row["cloud_top"]
        and row["future_yang"]
        and row["chikou_ok"]
        and row["gc"]
        and row["macd_norm"] >= s_min
        and 50 <= row["rsi"] < 70
    )


def check_e1(row: pd.Series) -> bool:
    """E1 모멘텀 약화: 오늘 MACD 데드크로스."""
    return _bool(row.get("dc"))


def check_e2(prev_rsi: float, rsi: float) -> bool:
    """E2 추세 약화: RSI가 전일 50 이상에서 당일 50 미만으로 하향 이탈."""
    if pd.isna(prev_rsi) or pd.isna(rsi):
        return False
    return bool(prev_rsi >= 50 and rsi < 50)


def check_e3(row: pd.Series) -> bool:
    """E3 구조 붕괴: 종가 < 오늘 가격 아래 구름의 하단, 또는 후행스팬 < 선행 B."""
    close = row.get("close")
    cloud_bot = row.get("cloud_bot")
    chikou_broken = row.get("chikou_broken")
    below_cloud = (not pd.isna(close)) and (not pd.isna(cloud_bot)) and close < cloud_bot
    chikou = _bool(chikou_broken)
    return bool(below_cloud or chikou)


def check_stop(close: float, stop_price: float | None) -> bool:
    """손절(최우선): 종가 < 손절가."""
    if stop_price is None or pd.isna(close) or pd.isna(stop_price):
        return False
    return bool(close < stop_price)


def check_gap_filter(open_price: float, prev_close: float, cfg: dict) -> bool:
    """당일 시가 갭이 gap_filter_pct 이상이면 True (A3 진입 보류 대상).

    입력이 NaN이면 갭이 없다고(False) 본다 — "없으면 없음" 원칙.
    """
    if pd.isna(open_price) or pd.isna(prev_close) or prev_close == 0:
        return False
    gap_pct = (open_price / prev_close - 1) * 100
    threshold = cfg["assumptions"]["gap_filter_pct"]
    return bool(gap_pct >= threshold)


def entry_limit_price(signal_close: float, cfg: dict) -> float:
    """매수 지정가 = 신호일 종가 × entry.limit_markup, 소수 둘째 자리 반올림."""
    return round(float(signal_close) * cfg["entry"]["limit_markup"], 2)


def rsi_overheat_relief(prev_rsi: float, rsi: float) -> bool:
    """RSI 과열 해소 경고: RSI가 전일 70 이상이었다가 당일 70 아래로 내려옴."""
    if pd.isna(prev_rsi) or pd.isna(rsi):
        return False
    return bool(prev_rsi >= 70 and rsi < 70)


def kijun_breach(close: float, kijun: float) -> bool:
    """기준선 이탈 경고: 종가가 일목 기준선 아래로 마감."""
    if pd.isna(close) or pd.isna(kijun):
        return False
    return bool(close < kijun)


def stop_near(close: float, stop_price: float | None, pct: float) -> bool:
    """손절 근접 경고(P3.5): 종가가 손절가보다 위에 있고, 그 차이가 pct% 이내.

    경계값 포함(예: 정확히 pct%면 True). 이미 손절 신호(종가 < 손절가)가 난
    날은 core/state.py가 먼저 처리하므로 이 함수는 그 경우까지 신경 쓰지
    않는다(check_stop이 우선한다는 전제로 호출부가 순서를 지킨다).
    """
    if stop_price is None or pd.isna(close) or pd.isna(stop_price) or stop_price <= 0:
        return False
    if close < stop_price:
        return False
    return bool((close - stop_price) / stop_price * 100 <= pct)


def target_reached(avg_entry_price: float, current_price: float, stop_price: float | None) -> bool:
    """목표 도달 경고: 평균 매수가 기준 손익비 2배 도달 (avg_entry - stop = 1R)."""
    if stop_price is None or pd.isna(avg_entry_price) or pd.isna(current_price) or pd.isna(stop_price):
        return False
    risk = avg_entry_price - stop_price
    if risk <= 0:
        return False
    return bool(current_price - avg_entry_price >= 2 * risk)
