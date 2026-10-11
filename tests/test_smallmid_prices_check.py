"""중소형 가격·발표 시각·신호 건수 조사 스크립트의 순수 함수 (네트워크 없음)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import smallmid_prices_check as pc  # noqa: E402


def test_xbrl_prefix():
    assert pc.xbrl_prefix("ads-20121231.xml") == "ADS"
    assert pc.xbrl_prefix("brkr-20191231.htm") == "BRKR"
    assert pc.xbrl_prefix("bf.b-20150430.xml") == "BF.B"
    assert pc.xbrl_prefix("d123456d10k.htm") is None
    assert pc.xbrl_prefix("Financial_Report.xlsx") is None


def test_candidate_tickers_order_and_dots():
    assert pc.candidate_tickers({"ticker_hist": "BF.B", "ticker_now": "BF-B;BF-A"}) == ["BF-B", "BF-A"]
    assert pc.candidate_tickers({"ticker_hist": None, "ticker_now": "ABC"}) == ["ABC"]
    assert pc.candidate_tickers({"ticker_hist": float("nan"), "ticker_now": float("nan")}) == []
    assert pc.candidate_tickers({"ticker_hist": "AHL", "ticker_now": "AHL-PD;AHL-PE"}) == ["AHL"]
    assert pc.candidate_tickers({"ticker_hist": None, "ticker_now": "CDR-PB;VAL-WT"}) == []


def test_check_seal_stops_on_rows_after_2021():
    ok = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2021-12-31"]))
    pc.check_seal(ok, "x")
    bad = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2021-12-31", "2022-01-03"]))
    with pytest.raises(pc.SealError):
        pc.check_seal(bad, "x")


def test_tiingo_adjust_and_raw_close_roundtrip():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    raw = pd.DataFrame({"raw_open": [100.0, 102.0, 51.0], "raw_close": [101.0, 104.0, 52.0],
                        "volume": [10.0, 10.0, 20.0], "split_factor": [1.0, 1.0, 2.0]}, index=idx)
    adj = pc.tiingo_adjust(raw)
    assert list(adj["close"]) == [50.5, 52.0, 52.0]  # 2:1 분할 전날까지 반으로
    assert list(adj["volume"]) == [20.0, 20.0, 20.0]
    assert list(adj["open"]) == [50.0, 51.0, 51.0]
    assert list(pc.raw_close(adj)) == [101.0, 104.0, 52.0]


def test_overlap_and_gap():
    px = pd.DataFrame({"close": 1.0}, index=pd.to_datetime(["2015-01-02", "2015-01-05", "2015-02-02", "2016-01-04"]))
    assert pc.overlap_days(px, "2015-01-01", "2015-12-31", pad_days=0) == 3
    assert pc.overlap_days(px, "2015-01-01", "2015-12-31") == 4  # 한 분기 넓히면 2016-01-04 포함
    assert pc.overlap_days(px, "2015-01-05", "2015-01-05") == 3  # 보고 한 번만 보인 종목
    assert pc.max_gap_days(px, "2015-01-01", "2015-12-31") == 28


def test_next_hour_wait_seconds():
    assert pc.next_hour_wait_seconds(3600 * 10 + 1800, margin_sec=120) == 1920


def test_exit_reason():
    f = pd.DataFrame({"form": ["10-K", "8-K", "25-NSE"], "filing_date": ["2015-03-01", "2015-09-01", "2015-09-20"],
                      "items": ["", "2.01,9.01", ""]})
    assert pc.exit_reason(f, "2015-08-31", False) == "인수·합병(상장폐지)"
    assert pc.exit_reason(f, "2015-08-31", True) == "끝까지 편입(2021-12)"
    bk = pd.DataFrame({"form": ["10-Q", "8-K"], "filing_date": ["2016-05-01", "2016-06-01"], "items": ["", "1.03"]})
    assert pc.exit_reason(bk, "2016-05-31", False) == "파산(8-K 1.03)"
    alive = pd.DataFrame({"form": ["10-K"], "filing_date": ["2018-03-01"], "items": [""]})
    assert pc.exit_reason(alive, "2016-05-31", False) == "지수 편출만(회사는 계속 공시)"
    assert pc.exit_reason(pd.DataFrame(columns=["form", "filing_date", "items"]), "2016-05-31", False) == "불명"


def test_liquid_before_uses_days_before_d0_only():
    idx = pd.bdate_range("2020-01-01", periods=25)
    px = pd.DataFrame({"close": [10.0] * 24 + [1.0], "volume": [600_000.0] * 24 + [0.0]}, index=idx)
    ok, price, dv = pc.liquid_before(px, idx[-1])
    assert ok and price == 10.0 and dv == 6_000_000.0  # 반응일(종가 1달러) 자체는 안 봄
    px.loc[idx[-2], "close"] = 4.0
    assert not pc.liquid_before(px, idx[-1])[0]


def test_member_on_excludes_single_report_intervals():
    ivs = [(pd.Timestamp("2015-03-31"), pd.Timestamp("2016-06-30"), "sp600", 6),
           (pd.Timestamp("2018-03-31"), pd.Timestamp("2018-03-31"), "sp400", 1)]
    assert pc.member_on(ivs, pd.Timestamp("2015-06-01")) == (True, "sp600")
    assert pc.member_on(ivs, pd.Timestamp("2018-03-31")) == (False, None)
    assert pc.member_on(ivs, pd.Timestamp("2017-01-03")) == (False, None)
