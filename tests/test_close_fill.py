"""P1.2: 마지막 봉 close 보완 로직 테스트. 네트워크 없이 합성 데이터로 돈다.

야후가 최근 거래일의 close만 비워둔 채 내려주는 상황을 흉내 내고,
chart API meta.regularMarketPrice로 채우는 조건을 확인한다.
"""

from __future__ import annotations

from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd
import pytest

from data.prices import (
    CLOSE_SOURCE_FALLBACK,
    CLOSE_SOURCE_META,
    CLOSE_SOURCE_YAHOO,
    MARKET_CLOSE_HOUR,
    US_EASTERN,
    ChartMeta,
    _cache_is_fresh,
    _clean_raw,
    _restore_close_from_cache,
    find_mid_series_gaps,
    has_meta_close,
)

LAST_DATE = pd.Timestamp("2026-09-22")
# 장 마감 다음 날 아침에 실행하는 상황 (당일 봉 제거 규칙에 걸리지 않는 시각)
NOW_ET = datetime(2026, 9, 23, 7, 0, tzinfo=US_EASTERN)
# 마지막 봉과 같은 날 정규장 마감 시각
CLOSE_ET = datetime.combine(LAST_DATE.date(), dtime(MARKET_CLOSE_HOUR, 0), tzinfo=US_EASTERN)


def make_raw(last_row: dict) -> pd.DataFrame:
    """yfinance Ticker.history 형식(대문자 열, tz-aware 인덱스) raw 프레임을 만든다.

    입력: 마지막 봉의 {Open, High, Low, Close, Volume} (NaN 허용)
    출력: 3거래일짜리 raw DataFrame
    """
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-09-18"), pd.Timestamp("2026-09-21"), LAST_DATE], name="Date"
    ).tz_localize(US_EASTERN)
    rows = [
        {"Open": 219.0, "High": 222.7, "Low": 218.0, "Close": 222.2, "Volume": 190_287_400},
        {"Open": 222.9, "High": 228.5, "Low": 221.5, "Close": 227.3, "Volume": 109_806_100},
        last_row,
    ]
    return pd.DataFrame(rows, index=index)


MISSING_CLOSE_ROW = {
    "Open": 226.91,
    "High": 229.98,
    "Low": 226.50,
    "Close": np.nan,
    "Volume": 93_296_546,
}


def test_fills_close_when_meta_price_in_range():
    """meta 가격이 그날 저가~고가 안이고 마감 이후면 close를 채운다."""
    meta = ChartMeta(price=228.87, time_et=CLOSE_ET)
    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: meta)

    assert out.index[-1] == LAST_DATE
    assert out.iloc[-1]["close"] == pytest.approx(228.87)
    assert out.iloc[-1]["close_source"] == CLOSE_SOURCE_META
    assert out.attrs["close_filled_from_meta"] is True
    assert has_meta_close(out)
    # 보완하지 않은 봉은 출처 표시가 그대로여야 한다
    assert set(out.iloc[:-1]["close_source"]) == {CLOSE_SOURCE_YAHOO}


def test_drops_bar_when_meta_price_out_of_range():
    """meta 가격이 그날 저가~고가를 벗어나면 채우지 않고 그 봉을 버린다."""
    meta = ChartMeta(price=240.00, time_et=CLOSE_ET)  # high(229.98)보다 높다
    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: meta)

    assert out.index[-1] == pd.Timestamp("2026-09-21")
    assert not has_meta_close(out)
    assert any("범위를 벗어나" in w for w in out.attrs["warnings"])


def test_drops_bar_when_ohl_also_missing():
    """open/high/low까지 NaN이면 보완 대상이 아니므로 그 봉을 버린다."""
    row = {"Open": np.nan, "High": np.nan, "Low": np.nan, "Close": np.nan, "Volume": 0}
    meta = ChartMeta(price=228.87, time_et=CLOSE_ET)
    out = _clean_raw(make_raw(row), NOW_ET, meta_provider=lambda: meta)

    assert out.index[-1] == pd.Timestamp("2026-09-21")
    assert not has_meta_close(out)


def test_drops_bar_when_meta_time_is_before_regular_close():
    """정규장 마감 전 시각(장중 가격)이면 채우지 않는다."""
    meta = ChartMeta(price=228.87, time_et=CLOSE_ET - timedelta(hours=2))
    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: meta)

    assert out.index[-1] == pd.Timestamp("2026-09-21")
    assert any("허용 폭" in w for w in out.attrs["warnings"])


def test_drops_bar_when_meta_time_is_long_after_regular_close():
    """정규장 마감 후 한참 지난 시각(장후·라이브 가격일 수 있음)이면 채우지 않는다.

    재현성을 위해(P2.1 보완 3번) 마감 직후의 좁은 허용 폭만 받아들인다.
    """
    meta = ChartMeta(price=228.87, time_et=CLOSE_ET + timedelta(hours=3))
    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: meta)

    assert out.index[-1] == pd.Timestamp("2026-09-21")
    assert any("허용 폭" in w for w in out.attrs["warnings"])
    assert not has_meta_close(out)


def test_fills_close_within_tolerance_window():
    """마감 후 허용 폭(5분) 이내면 그대로 채운다."""
    meta = ChartMeta(price=228.87, time_et=CLOSE_ET + timedelta(minutes=2))
    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: meta)

    assert out.index[-1] == LAST_DATE
    assert has_meta_close(out)
    assert out.attrs["close_meta_time"] == meta.time_et.isoformat()


def test_drops_bar_when_meta_is_for_another_day():
    """meta 시각이 그 봉과 다른 날이면 채우지 않는다."""
    meta = ChartMeta(price=228.87, time_et=CLOSE_ET + timedelta(days=1))
    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: meta)

    assert out.index[-1] == pd.Timestamp("2026-09-21")


def test_drops_bar_when_meta_unavailable():
    """meta 조회가 실패(None)하면 채우지 않고 그 봉을 버린다."""
    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: None)

    assert out.index[-1] == pd.Timestamp("2026-09-21")
    assert any("meta를 쓸 수 없어" in w for w in out.attrs["warnings"])


def test_does_not_touch_complete_last_bar():
    """close가 이미 있으면 meta를 쓰지 않고 그대로 둔다."""
    row = {**MISSING_CLOSE_ROW, "Close": 228.00}
    calls = []

    def _provider():
        calls.append(1)
        return ChartMeta(price=228.87, time_et=CLOSE_ET)

    out = _clean_raw(make_raw(row), NOW_ET, meta_provider=_provider)

    assert calls == []  # 보완이 필요 없으면 chart API를 부르지 않는다
    assert out.iloc[-1]["close"] == pytest.approx(228.00)
    assert out.iloc[-1]["close_source"] == CLOSE_SOURCE_YAHOO
    assert not has_meta_close(out)


def test_boundary_price_equal_to_low_is_filled():
    """보완값이 저가와 정확히 같아도(범위 경계) 채운다."""
    meta = ChartMeta(price=MISSING_CLOSE_ROW["Low"], time_et=CLOSE_ET)
    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: meta)

    assert out.index[-1] == LAST_DATE
    assert out.iloc[-1]["close"] == pytest.approx(MISSING_CLOSE_ROW["Low"])


# ── 캐시에서 되살리기 (다음 날 실행 시 중간에 구멍이 남지 않도록) ─────────────


def _frame(rows: list[dict], dates: list[str]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    return pd.DataFrame(rows, index=index)


CACHED = _frame(
    [
        {"open": 222.9, "high": 228.5, "low": 221.5, "close": 227.3, "volume": 1, "close_source": CLOSE_SOURCE_YAHOO},
        {"open": 226.91, "high": 229.98, "low": 226.50, "close": 228.87, "volume": 1, "close_source": CLOSE_SOURCE_META},
    ],
    ["2026-09-21", "2026-09-22"],
)


def test_restore_fills_mid_series_hole_from_cached_meta_close():
    """다음 날 다시 받았을 때 중간에 남은 빈 close를 캐시의 meta 보완값으로 되살린다."""
    fresh = _frame(
        [
            {"open": 226.91, "high": 229.98, "low": 226.50, "close": np.nan, "volume": 1, "close_source": CLOSE_SOURCE_YAHOO},
            {"open": 229.0, "high": 233.0, "low": 228.0, "close": 231.5, "volume": 1, "close_source": CLOSE_SOURCE_YAHOO},
        ],
        ["2026-09-22", "2026-09-23"],
    )
    out, restored = _restore_close_from_cache(fresh, CACHED)

    assert restored == 1
    assert out.loc[pd.Timestamp("2026-09-22"), "close"] == pytest.approx(228.87)
    assert out.loc[pd.Timestamp("2026-09-22"), "close_source"] == CLOSE_SOURCE_META
    assert not out["close"].isna().any()


def test_restore_skips_when_cached_value_out_of_fresh_range():
    """캐시 값이 새로 받은 그날 저가~고가를 벗어나면(분할 등) 되살리지 않는다."""
    fresh = _frame(
        [
            {"open": 113.4, "high": 115.0, "low": 113.2, "close": np.nan, "volume": 1, "close_source": CLOSE_SOURCE_YAHOO},
        ],
        ["2026-09-22"],
    )
    out, restored = _restore_close_from_cache(fresh, CACHED)

    assert restored == 0
    assert pd.isna(out.loc[pd.Timestamp("2026-09-22"), "close"])


def test_restore_ignores_cache_without_close_source():
    """close_source가 없는 옛 캐시는 되살리기 대상이 아니다."""
    fresh = _frame(
        [{"open": 226.91, "high": 229.98, "low": 226.50, "close": np.nan, "volume": 1, "close_source": CLOSE_SOURCE_YAHOO}],
        ["2026-09-22"],
    )
    old_cache = CACHED.drop(columns=["close_source"])
    out, restored = _restore_close_from_cache(fresh, old_cache)

    assert restored == 0
    assert pd.isna(out.loc[pd.Timestamp("2026-09-22"), "close"])


# ── P2 P1 마무리 1번: 일봉 종가가 meta 보완값보다 우선 ──────────────────────


def test_cache_not_fresh_when_last_close_is_meta():
    """마지막 봉이 meta 보완값이면, 날짜가 최신이어도 신선하다고 보지 않는다.

    실제 일봉 종가가 그 사이 들어왔는지 다시 확인하기 위해 재요청을 유도한다.
    """
    latest_needed = LAST_DATE.date()
    now_et = datetime.combine(latest_needed, dtime(20, 0), tzinfo=US_EASTERN)
    assert not _cache_is_fresh(CACHED, now_et)  # CACHED의 마지막 행(09-22)은 close_source=meta


def test_cache_fresh_when_last_close_is_yahoo():
    """마지막 봉이 실제 일봉 값(yahoo)이면 그대로 신선하다."""
    yahoo_cache = CACHED.copy()
    yahoo_cache.loc[LAST_DATE, "close_source"] = CLOSE_SOURCE_YAHOO
    now_et = datetime.combine(LAST_DATE.date(), dtime(20, 0), tzinfo=US_EASTERN)
    assert _cache_is_fresh(yahoo_cache, now_et)


def test_daily_bar_value_wins_over_stale_meta_when_refetched():
    """재요청 결과 그 날짜에 실제 종가가 들어 있으면 그 값을 쓰고 출처가 yahoo가 된다."""
    # 이전에 meta로 채웠던 날짜(09-22)가 이번 raw 응답에서는 실제 종가를 갖고 있고,
    # 새 마지막 날짜(09-23)가 추가된 상황을 흉내 낸다.
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-09-21"), pd.Timestamp("2026-09-22"), pd.Timestamp("2026-09-23")], name="Date"
    ).tz_localize(US_EASTERN)
    raw = pd.DataFrame(
        [
            {"Open": 222.9, "High": 228.5, "Low": 221.5, "Close": 227.3, "Volume": 1},
            {"Open": 226.91, "High": 229.98, "Low": 226.50, "Close": 228.87, "Volume": 1},  # 이제 실제 값이 채워짐
            {"Open": 229.0, "High": 233.0, "Low": 228.0, "Close": 231.5, "Volume": 1},
        ],
        index=index,
    )
    now_et = datetime(2026, 9, 24, 7, 0, tzinfo=US_EASTERN)
    cleaned = _clean_raw(raw, now_et, meta_provider=None)
    out, restored = _restore_close_from_cache(cleaned, CACHED)

    assert restored == 0  # 이미 실제 값이 있어 캐시에서 되살릴 필요가 없다
    assert out.loc[pd.Timestamp("2026-09-22"), "close"] == pytest.approx(228.87)
    assert out.loc[pd.Timestamp("2026-09-22"), "close_source"] == CLOSE_SOURCE_YAHOO


# ── P2 P1 마무리 2번: 중간 close 구멍은 채우지 않고 data_gap으로 표시 ─────────


def test_find_mid_series_gaps_detects_hole_not_at_last_row():
    df = _frame(
        [
            {"close": 100.0},
            {"close": np.nan},  # 중간 구멍
            {"close": 102.0},
        ],
        ["2026-09-18", "2026-09-21", "2026-09-22"],
    )
    gaps = find_mid_series_gaps(df)
    assert gaps == [pd.Timestamp("2026-09-21")]


def test_find_mid_series_gaps_ignores_last_row_nan():
    """마지막 봉의 NaN은 close 보완/제거 단계가 따로 처리하므로 구멍으로 세지 않는다."""
    df = _frame(
        [{"close": 100.0}, {"close": np.nan}],
        ["2026-09-18", "2026-09-21"],
    )
    assert find_mid_series_gaps(df) == []


def test_find_mid_series_gaps_empty_when_no_hole():
    df = _frame(
        [{"close": 100.0}, {"close": 101.0}],
        ["2026-09-18", "2026-09-21"],
    )
    assert find_mid_series_gaps(df) == []


# ── P3.2 1번: 마지막 봉이 캐시 확인 전에 버려지는 회귀 재현·수정 확인 ─────────


def test_regression_stale_last_bar_restored_from_cache_before_being_dropped():
    """P3.1 실행에서 기준일이 9/22 -> 9/21로 뒤로 간 회귀의 재현 테스트.

    장중(9/23) 재실행 상황: 9/22 close는 야후가 여전히 비워 둔 채 내려주고,
    meta는 이제 9/23(오늘) 값을 가리켜 9/22 보완에 쓸 수 없다. 이때 캐시에
    9/22의 meta 보완값이 있으면, 그 값으로 되살아나야지 9/22 봉 자체가
    통째로 버려져 기준일이 9/21로 후퇴해서는 안 된다.
    """
    index = pd.DatetimeIndex(
        [pd.Timestamp("2026-09-21"), pd.Timestamp("2026-09-22"), pd.Timestamp("2026-09-23")], name="Date"
    ).tz_localize(US_EASTERN)
    raw = pd.DataFrame(
        [
            {"Open": 222.9, "High": 228.5, "Low": 221.5, "Close": 227.3, "Volume": 1},
            {"Open": 226.91, "High": 229.98, "Low": 226.50, "Close": np.nan, "Volume": 1},  # 여전히 비어 있음
            {"Open": 230.0, "High": 231.0, "Low": 229.0, "Close": 230.5, "Volume": 1},  # 오늘(장중) 미확정 봉
        ],
        index=index,
    )
    now_et_intraday = datetime(2026, 9, 23, 10, 0, tzinfo=US_EASTERN)  # 9/23 장중
    # meta는 이제 9/23 실시간 가격을 가리켜 9/22 보완에는 못 쓴다 (날짜 불일치로 거절됨)
    meta = ChartMeta(price=231.0, time_et=now_et_intraday)

    out = _clean_raw(raw, now_et_intraday, meta_provider=lambda: meta, cache_df=CACHED)

    assert out.index[-1] == pd.Timestamp("2026-09-22")  # 9/21로 후퇴하지 않는다 (수정 전에는 실패했다)
    assert out.loc[pd.Timestamp("2026-09-22"), "close"] == pytest.approx(228.87)
    assert out.loc[pd.Timestamp("2026-09-22"), "close_source"] == CLOSE_SOURCE_META


# ── P3.2 1번: 예비 종가 출처(60분봉) ────────────────────────────────────────


def test_fills_close_from_fallback_when_meta_and_cache_both_fail():
    """meta도 캐시도 못 채우면 예비 출처(fallback_provider)로 채운다."""
    calls: list = []

    def fallback(d):
        calls.append(d)
        return 228.87

    out = _clean_raw(make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: None, fallback_provider=fallback)

    assert out.index[-1] == LAST_DATE
    assert out.iloc[-1]["close"] == pytest.approx(228.87)
    assert out.iloc[-1]["close_source"] == CLOSE_SOURCE_FALLBACK
    assert out.attrs["close_filled_from_meta"] is True
    assert calls == [LAST_DATE.date()]


def test_fallback_out_of_range_is_not_used():
    """예비 출처 값이 그날 저가~고가 범위 밖이면 쓰지 않고 봉을 버린다."""
    out = _clean_raw(
        make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: None, fallback_provider=lambda d: 999.0
    )
    assert out.index[-1] == pd.Timestamp("2026-09-21")


def test_cache_checked_before_fallback_source():
    """캐시로 채울 수 있으면 예비 출처는 부르지 않는다 (우선순위: meta > 캐시 > 예비)."""
    calls: list = []

    def fallback(d):
        calls.append(d)
        return 999.0

    out = _clean_raw(
        make_raw(MISSING_CLOSE_ROW), NOW_ET, meta_provider=lambda: None, cache_df=CACHED, fallback_provider=fallback
    )

    assert out.index[-1] == LAST_DATE
    assert out.iloc[-1]["close"] == pytest.approx(228.87)
    assert out.iloc[-1]["close_source"] == CLOSE_SOURCE_META
    assert calls == []  # 캐시로 채워졌으니 예비 출처는 부르지 않는다
