"""KJB-1(docs/kjb1_instructions.md 1번, docs/kjb_nasdaq100_plan.md 2·3장) 신호 계산.

김종봉 투자법 종목 선택 핵심부를 나스닥 100에 옮긴 신호: 63거래일 상대수익
"처음 초과" + 거래대금 배율 + 장대양봉이 같은 날 모두 충족하는 첫 발생일을 찾는다.

- core/의 다른 모듈과 마찬가지로 네트워크·파일·DB·현재 시각에 접근하지 않는다.
- 모든 값은 신호일 종가까지의 자료만 쓴다(미래 데이터 금지) — `shift(-n)` 금지.
- 기준값(거래대금 배율·장대양봉 % 등)은 하드코딩하지 않고 호출하는 쪽
  (configs/kjb1_preregistration.yaml을 읽은 값)에서 인자로 받는다.
- 여기서 쓰는 수익률·가격은 모두 배당 반영 전 종가(CLAUDE.md 데이터 규칙 3번과
  같은 기준) — QQQ·종목 양쪽 다 통일해서 비교한다.
"""

from __future__ import annotations

import pandas as pd


def relative_return(close: pd.Series, qqq_close: pd.Series, window_days: int) -> pd.Series:
    """window_days거래일 상대수익 = 종목 window_days일 수익률 − QQQ window_days일 수익률.

    입력: close(종목 종가, 날짜 오름차순 인덱스), qqq_close(QQQ 종가, close와 같은
         인덱스로 미리 맞춰 둔 것), window_days(63)
    출력: close와 같은 인덱스의 Series(비율, 0.05 = +5%). 앞쪽 window_days일은 NaN.
    """
    stock_ret = close / close.shift(window_days) - 1
    qqq_ret = qqq_close / qqq_close.shift(window_days) - 1
    return stock_ret - qqq_ret


def first_excess(rel_return: pd.Series, window_days: int) -> pd.Series:
    """"처음 초과": 오늘 상대수익 > 0, 어제 <= 0, 그리고 직전 window_days거래일
    동안(오늘 제외) 한 번도 0을 넘은 적이 없는 날.

    입력: rel_return(relative_return 결과), window_days(63, "직전 63거래일" 확인용)
    출력: bool Series. NaN이 섞인 날은 False.
    """
    today_positive = rel_return > 0
    prev_not_positive = rel_return.shift(1) <= 0
    # "직전 window_days거래일 동안 한 번도 0을 넘은 적이 없음" — 오늘을 뺀 과거
    # window_days일 중 양수가 하나라도 있으면 안 된다.
    no_prior_excess = ~(rel_return.shift(1) > 0).rolling(window_days, min_periods=1).max().astype(bool)
    return (today_positive & prev_not_positive & no_prior_excess).fillna(False)


def first_score(daily_relative_return: pd.Series, window_days: int) -> pd.Series:
    """"처음" 점수: 직전 window_days거래일(오늘 제외) 중 일간 상대수익이 양수인 날 수.

    작을수록 "처음"에 가깝다(우선순위 정렬용). daily_relative_return은 일간
    등락률 차이(종목 − QQQ, 하루 단위)여야 한다 — relative_return(63일 누적)과는
    다른 입력이다.

    입력: daily_relative_return(하루 등락률 차이 Series), window_days(60)
    출력: 정수 Series(0~window_days). 앞쪽 window_days일은 NaN.
    """
    positive = (daily_relative_return > 0).astype(float)
    return positive.shift(1).rolling(window_days, min_periods=window_days).sum()


def daily_relative_return(close: pd.Series, qqq_close: pd.Series) -> pd.Series:
    """하루 등락률 차이 = 종목 당일 등락률 − QQQ 당일 등락률. first_score의 입력."""
    return close.pct_change() - qqq_close.pct_change()


def dollar_volume_multiplier(close: pd.Series, volume: pd.Series, window_days: int = 20) -> pd.Series:
    """거래대금 배율 = 당일 (종가×거래량) ÷ 직전 window_days일 평균(오늘 제외).

    입력: close, volume(같은 인덱스), window_days(20)
    출력: Series. 앞쪽 window_days일은 NaN.
    """
    dollar_volume = close * volume
    prior_avg = dollar_volume.shift(1).rolling(window_days, min_periods=window_days).mean()
    return dollar_volume / prior_avg


def big_bull_candle(open_: pd.Series, close: pd.Series, pct_min: float) -> pd.Series:
    """장대양봉: 당일 종가 등락률 >= pct_min(%)이고 종가 > 시가(양봉).

    입력: open_, close(같은 인덱스), pct_min(퍼센트, 예: 5.0)
    출력: bool Series. 앞쪽 1일은 NaN 등락률이라 False.
    """
    pct_change = close.pct_change() * 100
    return ((pct_change >= pct_min) & (close > open_)).fillna(False)


def big_bull_candle_volatility_margin(open_: pd.Series, close: pd.Series, window_days: int = 20, mult: float = 2.5) -> pd.Series:
    """장대양봉 주변값: 등락률 >= 직전 window_days일 일간 변동성(표준편차, 오늘
    제외) × mult, 그리고 종가 > 시가.

    입력: open_, close, window_days(20), mult(2.5)
    출력: bool Series.
    """
    pct_change = close.pct_change() * 100
    daily_std = pct_change.shift(1).rolling(window_days, min_periods=window_days).std()
    threshold = daily_std * mult
    return ((pct_change >= threshold) & (close > open_)).fillna(False)


def kjb_entry_signal(
    rel_return: pd.Series,
    first_excess_flag: pd.Series,
    dv_multiplier: pd.Series,
    big_candle_flag: pd.Series,
    dv_multiplier_min: float,
) -> pd.Series:
    """같은 날 "처음 초과" + 거래대금 배율 기준 + 장대양봉을 모두 충족하는 날.

    입력: first_excess_flag(first_excess 결과), dv_multiplier(dollar_volume_multiplier
         결과), big_candle_flag(big_bull_candle 또는 주변값 결과), dv_multiplier_min(2.0)
    출력: bool Series
    """
    return (first_excess_flag & (dv_multiplier >= dv_multiplier_min) & big_candle_flag).fillna(False)


def sector_period_return(close_by_ticker: dict[str, pd.Series], sector_by_ticker: dict[str, str], date: pd.Timestamp, window_days: int = 20) -> dict[str, float]:
    """시점 date 기준, 섹터별 동일가중 window_days일 수익률.

    입력: close_by_ticker({티커: 종가 Series, date를 포함하는 것만}), sector_by_ticker
         ({티커: 섹터명}), date, window_days(20)
    출력: {섹터명: 평균 수익률(비율)} — 그 섹터에 유효한 종목이 하나도 없으면 뺀다.
    """
    by_sector: dict[str, list[float]] = {}
    for ticker, close in close_by_ticker.items():
        sector = sector_by_ticker.get(ticker)
        if sector is None or date not in close.index:
            continue
        loc = close.index.get_loc(date)
        if loc < window_days:
            continue
        ret = close.iloc[loc] / close.iloc[loc - window_days] - 1
        if pd.isna(ret):
            continue
        by_sector.setdefault(sector, []).append(float(ret))
    return {sector: sum(vals) / len(vals) for sector, vals in by_sector.items() if vals}


def sector_relative_strength(sector_returns: dict[str, float], qqq_return: float) -> dict[str, float]:
    """섹터 상대강도 = 섹터 동일가중 수익률 − QQQ 수익률(같은 기간)."""
    return {sector: ret - qqq_return for sector, ret in sector_returns.items()}


def sector_dispersion(close_by_ticker: dict[str, pd.Series], sector_by_ticker: dict[str, str], date: pd.Timestamp, window_days: int = 20) -> dict[str, float]:
    """섹터 확산도 = 그 섹터 안에서 window_days일 수익률이 양수인 종목 비율.

    출력: {섹터명: 0~1 비율} — 그 섹터에 유효한 종목이 하나도 없으면 뺀다.
    """
    by_sector: dict[str, list[bool]] = {}
    for ticker, close in close_by_ticker.items():
        sector = sector_by_ticker.get(ticker)
        if sector is None or date not in close.index:
            continue
        loc = close.index.get_loc(date)
        if loc < window_days:
            continue
        ret = close.iloc[loc] / close.iloc[loc - window_days] - 1
        if pd.isna(ret):
            continue
        by_sector.setdefault(sector, []).append(ret > 0)
    return {sector: sum(vals) / len(vals) for sector, vals in by_sector.items() if vals}


def leading_stock_signal(close_by_ticker: dict[str, pd.Series], volume_by_ticker: dict[str, pd.Series], qqq_close: pd.Series, date: pd.Timestamp, window_days: int = 20, top_n: int = 2) -> float | None:
    """대장주 선행 신호(보고용): 그날 거래대금 상위 top_n종목의 window_days일
    수익률 평균 − QQQ window_days일 수익률. 매매 판정에는 쓰지 않는다.

    출력: 값(비율) 또는 계산할 수 없으면 None.
    """
    dollar_volumes: list[tuple[str, float]] = []
    for ticker, close in close_by_ticker.items():
        volume = volume_by_ticker.get(ticker)
        if volume is None or date not in close.index or date not in volume.index:
            continue
        c, v = close.loc[date], volume.loc[date]
        if pd.isna(c) or pd.isna(v):
            continue
        dollar_volumes.append((ticker, float(c) * float(v)))
    if not dollar_volumes:
        return None
    dollar_volumes.sort(key=lambda x: x[1], reverse=True)
    top_tickers = [t for t, _ in dollar_volumes[:top_n]]

    if date not in qqq_close.index:
        return None
    qqq_loc = qqq_close.index.get_loc(date)
    if qqq_loc < window_days:
        return None
    qqq_ret = qqq_close.iloc[qqq_loc] / qqq_close.iloc[qqq_loc - window_days] - 1

    rets = []
    for ticker in top_tickers:
        close = close_by_ticker[ticker]
        loc = close.index.get_loc(date)
        if loc < window_days:
            continue
        ret = close.iloc[loc] / close.iloc[loc - window_days] - 1
        if not pd.isna(ret):
            rets.append(float(ret))
    if not rets:
        return None
    return sum(rets) / len(rets) - float(qqq_ret)


def dollar_volume_rank_today(close_by_ticker: dict[str, pd.Series], volume_by_ticker: dict[str, pd.Series], date: pd.Timestamp) -> dict[str, int]:
    """그날 거래대금 순위(1이 가장 큼). 우선순위에서 상위 N종목을 맨 뒤로 미루는 데 쓴다.

    출력: {티커: 순위(1부터)} — 그날 종가·거래량이 있는 종목만.
    """
    dollar_volumes: list[tuple[str, float]] = []
    for ticker, close in close_by_ticker.items():
        volume = volume_by_ticker.get(ticker)
        if volume is None or date not in close.index or date not in volume.index:
            continue
        c, v = close.loc[date], volume.loc[date]
        if pd.isna(c) or pd.isna(v):
            continue
        dollar_volumes.append((ticker, float(c) * float(v)))
    dollar_volumes.sort(key=lambda x: x[1], reverse=True)
    return {ticker: rank + 1 for rank, (ticker, _) in enumerate(dollar_volumes)}


def rank_candidates(
    candidates: list[dict],
    sector_rel_strength: dict[str, float],
    dollar_volume_rank: dict[str, int],
    top_n_deprioritized: int = 10,
) -> list[dict]:
    """우선순위 정렬(주 설정): ① 섹터 상대강도 높은 순 -> ② "처음" 점수 낮은 순
    -> ③ 거래대금 배율 높은 순. 거래대금 상위 top_n_deprioritized종목은 맨 뒤로.

    입력: candidates([{"ticker","sector","first_score","dollar_volume_multiplier"}, ...]),
         sector_rel_strength(sector_relative_strength 결과), dollar_volume_rank
         (dollar_volume_rank_today 결과), top_n_deprioritized(10)
    출력: candidates를 우선순위 순서로 정렬한 새 리스트(원본 불변)
    """
    def sort_key(c: dict) -> tuple:
        deprioritized = dollar_volume_rank.get(c["ticker"], 999) <= top_n_deprioritized
        sector_strength = sector_rel_strength.get(c["sector"], float("-inf"))
        return (
            deprioritized,  # False(0) 먼저 — 상위 거래대금 종목은 True(1)라 뒤로 밀림
            -sector_strength,
            c["first_score"],
            -c["dollar_volume_multiplier"],
        )

    return sorted(candidates, key=sort_key)
