"""engine/dataqc.py 테스트 (AI 펀드 F2, docs/design/fund_sim.md 5.1). 네트워크 없음.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine import dataqc as dqc


def _cfg(**overrides):
    base = {
        "dataqc": {
            "expected_constituent_count": {"min": 90, "max": 110},
            "min_coverage_pct": 98,
            "extreme_daily_move_pct": 50,
        }
    }
    base["dataqc"].update(overrides)
    return base


def _df(closes: list[float], start: str = "2020-01-02") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=len(closes), name="date")
    return pd.DataFrame({"close": closes}, index=idx)


# ── check_constituent_counts ────────────────────────────────────────────


def test_check_constituent_counts_flags_out_of_range():
    checkpoints = [
        (date(2020, 1, 1), frozenset(f"T{i}" for i in range(100))),  # 정상
        (date(2020, 6, 1), frozenset(f"T{i}" for i in range(50))),  # 너무 적음
        (date(2021, 1, 1), frozenset(f"T{i}" for i in range(150))),  # 너무 많음
    ]
    issues = dqc.check_constituent_counts(checkpoints, min_count=90, max_count=110)
    assert [i.as_of for i in issues] == ["2020-06-01", "2021-01-01"]
    assert issues[0].count == 50
    assert issues[1].count == 150


def test_check_constituent_counts_empty_when_all_in_range():
    checkpoints = [(date(2020, 1, 1), frozenset(f"T{i}" for i in range(100)))]
    assert dqc.check_constituent_counts(checkpoints, min_count=90, max_count=110) == []


# ── compute_price_coverage ───────────────────────────────────────────────


def test_compute_price_coverage_counts_missing_tickers_and_missing_days():
    idx = pd.bdate_range("2020-01-02", periods=3, name="date")
    trading_days = [d.date() for d in idx]
    checkpoints = [(date(2020, 1, 1), frozenset({"AAA", "BBB"}))]
    indicator_map = {
        "AAA": pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=idx),  # 전부 있음
        # BBB는 아예 없음(가격 못 받은 종목) -> 3칸 모두 결측
    }
    expected, priced = dqc.compute_price_coverage(checkpoints, trading_days, indicator_map)
    assert expected == 6  # 2종목 × 3거래일
    assert priced == 3  # AAA만


def test_compute_price_coverage_counts_nan_close_as_missing():
    idx = pd.bdate_range("2020-01-02", periods=2, name="date")
    trading_days = [d.date() for d in idx]
    checkpoints = [(date(2020, 1, 1), frozenset({"AAA"}))]
    indicator_map = {"AAA": pd.DataFrame({"close": [10.0, float("nan")]}, index=idx)}
    expected, priced = dqc.compute_price_coverage(checkpoints, trading_days, indicator_map)
    assert expected == 2
    assert priced == 1


def test_compute_price_coverage_uses_point_in_time_membership_not_all_tickers():
    """체크포인트가 날짜에 따라 구성이 바뀌면 그 날의 실제 구성 종목만 센다."""
    idx = pd.bdate_range("2020-01-02", periods=2, name="date")
    trading_days = [d.date() for d in idx]
    checkpoints = [
        (date(2020, 1, 1), frozenset({"AAA"})),  # 1/2에는 AAA만
        (idx[1].date(), frozenset({"AAA", "BBB"})),  # 1/3부터 BBB 편입
    ]
    indicator_map = {
        "AAA": pd.DataFrame({"close": [10.0, 11.0]}, index=idx),
        "BBB": pd.DataFrame({"close": [20.0, 21.0]}, index=idx),  # BBB는 1/2에도 가격은 있지만 그날 구성종목이 아니다
    }
    expected, priced = dqc.compute_price_coverage(checkpoints, trading_days, indicator_map)
    assert expected == 3  # 1/2: AAA만(1) + 1/3: AAA·BBB(2)
    assert priced == 3


def test_compute_price_coverage_ignores_dates_outside_trading_days_argument():
    """trading_days에 없는 날짜(미래 등)는 indicator_map에 있어도 안 본다 — 미래 데이터 누출 방지."""
    idx = pd.bdate_range("2020-01-02", periods=5, name="date")
    checkpoints = [(date(2020, 1, 1), frozenset({"AAA"}))]
    indicator_map = {"AAA": pd.DataFrame({"close": [10.0, 11.0, 12.0, float("nan"), 14.0]}, index=idx)}

    expected_all, priced_all = dqc.compute_price_coverage(checkpoints, [d.date() for d in idx], indicator_map)
    expected_prefix, priced_prefix = dqc.compute_price_coverage(checkpoints, [d.date() for d in idx[:2]], indicator_map)

    assert expected_all == 5 and priced_all == 4
    assert expected_prefix == 2 and priced_prefix == 2  # 뒤쪽(결측 포함) 날짜는 아예 안 본다 — 결과가 안 섞인다


# ── detect_extreme_daily_moves ───────────────────────────────────────────


def test_detect_extreme_daily_moves_flags_big_jump():
    df = _df([100.0, 101.0, 160.0, 158.0])  # 100->101(정상), 101->160(+58%, 이상치), 160->158(정상)
    issues = dqc.detect_extreme_daily_moves({"AAA": df}, threshold_pct=50)
    assert len(issues) == 1
    assert issues[0].ticker == "AAA"
    assert issues[0].pct_change == pytest.approx(58.42, abs=0.1)


def test_detect_extreme_daily_moves_flags_big_drop():
    df = _df([100.0, 45.0])  # -55%
    issues = dqc.detect_extreme_daily_moves({"AAA": df}, threshold_pct=50)
    assert len(issues) == 1
    assert issues[0].pct_change < 0


def test_detect_extreme_daily_moves_none_when_under_threshold():
    df = _df([100.0, 110.0, 95.0])
    assert dqc.detect_extreme_daily_moves({"AAA": df}, threshold_pct=50) == []


def test_detect_extreme_daily_moves_skips_missing_close_column_or_too_short():
    assert dqc.detect_extreme_daily_moves({"AAA": pd.DataFrame({"open": [1.0]})}, threshold_pct=50) == []
    assert dqc.detect_extreme_daily_moves({"AAA": _df([100.0])}, threshold_pct=50) == []


# ── build_report ─────────────────────────────────────────────────────────


def test_build_report_inconclusive_when_coverage_below_threshold():
    idx = pd.bdate_range("2020-01-02", periods=10, name="date")
    trading_days = [d.date() for d in idx]
    checkpoints = [(date(2020, 1, 1), frozenset({"AAA", "BBB"}))]
    indicator_map = {
        "AAA": pd.DataFrame({"close": np.full(10, 10.0)}, index=idx),
        # BBB 아예 없음 -> coverage 50%
    }
    report = dqc.build_report(checkpoints, trading_days, indicator_map, _cfg(min_coverage_pct=98))
    assert report.coverage_pct == 50.0
    assert report.inconclusive is True
    assert any("확보율" in r for r in report.reasons)


def test_build_report_not_inconclusive_when_coverage_meets_threshold():
    idx = pd.bdate_range("2020-01-02", periods=10, name="date")
    trading_days = [d.date() for d in idx]
    checkpoints = [(date(2020, 1, 1), frozenset({"AAA"}))]
    indicator_map = {"AAA": pd.DataFrame({"close": np.full(10, 10.0)}, index=idx)}
    report = dqc.build_report(checkpoints, trading_days, indicator_map, _cfg(min_coverage_pct=98))
    assert report.coverage_pct == 100.0
    assert report.inconclusive is False
    assert report.reasons == []


def test_build_report_collects_all_three_checks():
    idx = pd.bdate_range("2020-01-02", periods=3, name="date")
    trading_days = [d.date() for d in idx]
    checkpoints = [(date(2020, 1, 1), frozenset({"AAA"} | {f"T{i}" for i in range(120)}))]  # 121종목 -> 범위 초과
    indicator_map = {"AAA": pd.DataFrame({"close": [100.0, 200.0, 199.0]}, index=idx)}  # +100% 이상치
    report = dqc.build_report(checkpoints, trading_days, indicator_map, _cfg())
    assert len(report.constituent_count_issues) == 1
    assert len(report.extreme_move_issues) == 1
