"""scripts/fund_fetch_missing_prices.py의 순수 함수 테스트 — 네트워크 없이 돈다.

apply_split_only_adjustment의 AAPL 값은 2026-10-01에 Tiingo(raw)·yfinance(auto_adjust=False
Close)를 실제로 대조해 확인한 실제 숫자다(모듈 docstring 참고) — 회귀 방지용으로 그대로 씀.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts import fund_fetch_missing_prices as ffp


def test_fetch_end_never_crosses_seal_boundary():
    """사용자 지시: 기간은 코드에 고정하고 봉인(2022-01-01)을 넘지 않는다."""
    assert ffp.FETCH_END < date(2022, 1, 1)
    assert ffp.FETCH_START < ffp.FETCH_END


def test_records_to_frame_basic_shape():
    records = [
        {"date": "2020-08-28T00:00:00.000Z", "open": 128.0, "high": 129.0, "low": 127.0, "close": 128.5, "volume": 100, "splitFactor": 1.0},
        {"date": "2020-08-31T00:00:00.000Z", "open": 128.0, "high": 129.5, "low": 126.0, "close": 129.04, "volume": 400, "splitFactor": 4.0},
    ]
    df = ffp.records_to_frame(records)
    assert list(df.index.date) == [date(2020, 8, 28), date(2020, 8, 31)]
    assert df.loc["2020-08-31", "split_factor"] == 4.0


def test_apply_split_only_adjustment_matches_yfinance_aapl_4to1_split():
    """AAPL 2020-08-31 4:1 분할 — 2026-10-01 Tiingo 원본·yfinance 값 실측 대조."""
    records = [
        {"date": "2020-08-25T00:00:00.000Z", "open": 500.0, "high": 505.0, "low": 495.0, "close": 499.3, "volume": 1000, "splitFactor": 1.0},
        {"date": "2020-08-28T00:00:00.000Z", "open": 498.0, "high": 500.0, "low": 496.0, "close": 499.23, "volume": 1000, "splitFactor": 1.0},
        {"date": "2020-08-31T00:00:00.000Z", "open": 128.0, "high": 129.5, "low": 126.0, "close": 129.04, "volume": 4000, "splitFactor": 4.0},
        {"date": "2020-09-01T00:00:00.000Z", "open": 132.0, "high": 135.0, "low": 131.0, "close": 134.18, "volume": 1000, "splitFactor": 1.0},
    ]
    df = ffp.apply_split_only_adjustment(ffp.records_to_frame(records))
    # 실제 확인값(2026-10-01): yfinance auto_adjust=False Close
    assert df.loc["2020-08-25", "close"] == pytest.approx(124.824997, abs=1e-3)  # 499.3/4
    assert df.loc["2020-08-28", "close"] == pytest.approx(124.807503, abs=1e-3)  # 499.23/4
    assert df.loc["2020-08-31", "close"] == pytest.approx(129.039993, abs=1e-3)  # 분할일 이후는 그대로
    assert df.loc["2020-09-01", "close"] == pytest.approx(134.179993, abs=1e-3)
    assert (df["source"] == "tiingo").all()


def test_apply_split_only_adjustment_no_split_leaves_raw_unchanged():
    """MSFT류(분할 없음) — 조정 후 close가 raw_close와 그대로 같아야 한다."""
    records = [
        {"date": "2019-06-03T00:00:00.000Z", "open": 118.0, "high": 120.0, "low": 117.0, "close": 119.84, "volume": 500, "splitFactor": 1.0},
        {"date": "2019-06-04T00:00:00.000Z", "open": 120.0, "high": 124.0, "low": 119.0, "close": 123.16, "volume": 600, "splitFactor": 1.0},
    ]
    df = ffp.apply_split_only_adjustment(ffp.records_to_frame(records))
    assert df["close"].tolist() == pytest.approx([119.84, 123.16])
    assert df["volume"].tolist() == pytest.approx([500.0, 600.0])


def test_apply_split_only_adjustment_empty_frame_returns_empty():
    df = ffp.apply_split_only_adjustment(ffp.records_to_frame([]))
    assert df.empty
    assert list(df.columns) >= []


def test_last_real_trading_day_ignores_frozen_zero_volume_tail():
    """YHOO 2017-06-19~06-26처럼 거래정지로 volume=0인 마지막 며칠은 무시한다(2026-10-01 실측)."""
    idx = pd.DatetimeIndex(["2017-06-15", "2017-06-16", "2017-06-19", "2017-06-20"])
    df = pd.DataFrame({"volume": [100.0, 200.0, 0.0, 0.0]}, index=idx)
    assert ffp.last_real_trading_day(df) == date(2017, 6, 16)


def test_last_real_trading_day_empty_returns_none():
    assert ffp.last_real_trading_day(pd.DataFrame({"volume": []})) is None


def test_last_real_trading_day_all_zero_volume_falls_back_to_last_index():
    idx = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    df = pd.DataFrame({"volume": [0.0, 0.0]}, index=idx)
    assert ffp.last_real_trading_day(df) == date(2020, 1, 3)


def test_classify_reason_delist_vs_trading_vs_unclear():
    assert ffp.classify_reason("Yahoo! Inc. was acquired by Verizon") == "delist"
    assert ffp.classify_reason("Annual index reconstitution.") == "trading"
    assert ffp.classify_reason("Quarterly index reconstitution.") == "trading"
    assert ffp.classify_reason("Something we've never seen before.") == "unclear"


def test_check_alignment_delist_within_tolerance_is_ok():
    verdict, _ = ffp.check_alignment("delist", date(2017, 6, 16), date(2017, 6, 19), tolerance_days=10)
    assert verdict == "OK"


def test_check_alignment_delist_far_off_is_review():
    verdict, _ = ffp.check_alignment("delist", date(2010, 1, 1), date(2017, 6, 19), tolerance_days=10)
    assert verdict == "REVIEW"


def test_check_alignment_trading_reaches_needed_end_is_ok():
    verdict, _ = ffp.check_alignment("trading", date(2016, 12, 19), date(2016, 12, 19), tolerance_days=10)
    assert verdict == "OK"


def test_check_alignment_trading_stops_early_is_review():
    verdict, _ = ffp.check_alignment("trading", date(2015, 1, 1), date(2016, 12, 19), tolerance_days=10)
    assert verdict == "REVIEW"


def test_check_alignment_missing_data_is_confirm_needed():
    verdict, _ = ffp.check_alignment("delist", None, date(2017, 6, 19))
    assert verdict == "확인 필요"


def test_needed_end_date_picks_latest_of_multiple_ranges():
    assert ffp._needed_end_date("2014-01-01~2017-12-18") == date(2017, 12, 18)
    assert ffp._needed_end_date("2014-01-01~2008-12-22; 2012-12-24~2017-12-18") == date(2017, 12, 18)


def test_load_target_tickers_skips_excluded_and_out_of_scope(tmp_path):
    csv_path = tmp_path / "missing_tickers.csv"
    csv_path.write_text(
        "ticker,company_name,membership_periods,removal_reason,needed_price_range,fetch_error\n"
        "ALTR,Altera,2000-01-01~2015-10-07,merged,2014-01-01~2015-10-07,x\n"
        "YHOO,Yahoo,2000-01-01~2017-06-19,acquired,2014-01-01~2017-06-19,x\n"
        "ALAB,Astera Labs,2026-06-22~현재,none,겹치는 구간 없음(연구 구간 밖),x\n",
        encoding="utf-8-sig",
    )
    rows = ffp.load_target_tickers(csv_path, {"ALTR"})
    assert [r.ticker for r in rows] == ["YHOO"]
