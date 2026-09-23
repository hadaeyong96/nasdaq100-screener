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
    CLOSE_SOURCE_META,
    CLOSE_SOURCE_YAHOO,
    MARKET_CLOSE_HOUR,
    US_EASTERN,
    ChartMeta,
    _clean_raw,
    _restore_close_from_cache,
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
    assert any("정규장 마감 이후가 아니라" in w for w in out.attrs["warnings"])


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
