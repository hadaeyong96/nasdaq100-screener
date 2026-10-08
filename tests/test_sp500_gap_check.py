"""S&P 500 가격 공백 점검 스크립트의 순수 함수 (네트워크 없음)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sp500_identity_check as ic  # noqa: E402
import sp500_tiingo_check as tc  # noqa: E402


def test_next_hour_wait_seconds_goes_past_next_hour():
    assert tc.next_hour_wait_seconds(3600 * 10 + 1800, margin_sec=120) == 1800 + 120
    assert tc.next_hour_wait_seconds(3600 * 10, margin_sec=0) == 3600


def test_judge_period():
    first, last = pd.Timestamp("2014-01-31"), pd.Timestamp("2016-06-30")
    assert tc.judge_period(first, last, "2011-06-01", "2016-06-30", None) == "맞음"
    assert tc.judge_period(first, last, "2017-01-03", "2021-12-31", None) == "다른 회사 의심"
    assert tc.judge_period(first, last, "2015-06-01", "2016-06-30", None) == "앞이 빔"
    assert tc.judge_period(first, last, "2011-06-01", "2014-12-31", None) == "끝이 이름"
    # 편출일이 편입 마지막 달보다 이르면 편출일까지만 있으면 된다
    assert tc.judge_period(first, last, "2011-06-01", "2015-03-02", pd.Timestamp("2015-03-01")) == "맞음"
    assert tc.judge_period(first, last, None, None, None) == "가격 없음"


def test_name_sim():
    assert ic.name_sim("Cerner", "CERNER Corp") == 1.0
    assert ic.name_sim("Spectra Energy", "Spectra Energy Corp.") == 1.0
    assert ic.name_sim("BB&T", "Beacon Financial Corp") < 0.6
    assert ic.name_sim("", "X") == 0.0


def test_filed_in_period():
    sub = {"filings": {"filingDate": ["2014-02-01", "2014-05-01", "2014-05-02", "2020-01-01"],
                       "form": ["10-K", "10-Q", "8-K", "10-Q"]}}
    assert ic.filed_in_period(sub, pd.Timestamp("2014-01-01"), pd.Timestamp("2014-12-31")) == 2


def test_continuity():
    idx = pd.bdate_range("2016-01-01", periods=30)
    px = pd.Series(100.0, index=idx)
    px.iloc[-1] = 150.0
    out = ic.continuity(px, idx[-1])
    assert out["last_vs_removed_days"] == 0
    assert out["tail_max_jump_pct"] == 50.0
    assert ic.continuity(pd.Series(dtype=float), None) == {}


def test_gap_status_and_reason():
    assert ic.gap_status("없음", None) == "공백(가격 없음)"
    assert ic.gap_status("tiingo", "다른 회사 의심") == "공백(다른 회사 제외)"
    assert ic.gap_status("yfinance", "앞이 빔") == "공백(일부만)"
    assert ic.gap_status("yfinance(새 티커)", "맞음") == "있음"
    assert ic.gap_reason("인수·합병") == "인수·합병"
    assert ic.gap_reason("불명") == "불명"
    assert ic.gap_reason("지금도 S&P") == "지금도 S&P(티커 문제)"
