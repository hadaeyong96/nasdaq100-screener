"""scripts/sw1_data_check.py 순수 함수 테스트 (네트워크 없음, 합성 데이터)."""

import numpy as np
import pandas as pd

from scripts import sw1_data_check as sw


def test_acceptance_to_et_matches_edgar_index_pages():
    # AAPL 공시 색인 페이지 Accepted 값과 대조 (여름 8시간·겨울 10시간 차이)
    assert sw.acceptance_to_et("2021-10-29T00:30:23.000Z") == pd.Timestamp("2021-10-28 16:30:23")
    assert sw.acceptance_to_et("2021-07-28T02:03:42.000Z") == pd.Timestamp("2021-07-27 18:03:42")
    assert sw.acceptance_to_et("2021-01-28T04:03:06.000Z") == pd.Timestamp("2021-01-27 18:03:06")


def test_session_and_reaction_day():
    days = pd.DatetimeIndex(["2021-10-27", "2021-10-28", "2021-10-29", "2021-11-01"])
    assert sw.session_of(pd.Timestamp("2021-10-28 07:00")) == "장전"
    assert sw.session_of(pd.Timestamp("2021-10-28 12:00")) == "장중"
    assert sw.session_of(pd.Timestamp("2021-10-28 16:30")) == "장후"
    assert sw.reaction_day(pd.Timestamp("2021-10-28 07:00"), days) == pd.Timestamp("2021-10-28")
    assert sw.reaction_day(pd.Timestamp("2021-10-28 16:30"), days) == pd.Timestamp("2021-10-29")
    # 금요일 장후 → 월요일, 토요일 → 월요일
    assert sw.reaction_day(pd.Timestamp("2021-10-29 16:05"), days) == pd.Timestamp("2021-11-01")
    assert sw.reaction_day(pd.Timestamp("2021-10-30 10:00"), days) == pd.Timestamp("2021-11-01")
    assert sw.next_trading_day(pd.Timestamp("2021-10-28"), days) == pd.Timestamp("2021-10-29")
    assert sw.reaction_day(pd.Timestamp("2021-11-01 17:00"), days) is None


def _synthetic(jump=0.06, vol_mult=3.0, bench_move=0.0):
    days = pd.bdate_range("2021-01-04", periods=30)
    close = pd.Series(100.0, index=days)
    vol = pd.Series(1_000.0, index=days)
    d0 = days[25]
    close.loc[d0:] = 100 * (1 + jump)
    vol.loc[d0] = 1_000 * vol_mult
    bench = pd.Series(300.0, index=days)
    bench.loc[d0:] = 300 * (1 + bench_move)
    return pd.DataFrame({"close": close, "volume": vol}), bench, d0


def test_signal_conditions():
    px, b, d0 = _synthetic()
    assert sw.signal_on(px, b, d0)["signal"]
    px, b, d0 = _synthetic(jump=0.04)
    assert not sw.signal_on(px, b, d0)["signal"]
    px, b, d0 = _synthetic(bench_move=0.02)  # 초과 4%p
    assert not sw.signal_on(px, b, d0)["signal"]
    px, b, d0 = _synthetic(vol_mult=1.5)
    assert not sw.signal_on(px, b, d0)["signal"]


def test_signal_uses_no_future_rows():
    px, b, d0 = _synthetic()
    full = sw.signal_on(px, b, d0)
    cut = sw.signal_on(px.loc[:d0], b.loc[:d0], d0)
    assert full == cut
    # d0 뒤 값을 바꿔도 결과가 같다
    px2 = px.copy()
    px2.loc[px2.index > d0, "close"] = np.nan
    assert sw.signal_on(px2, b, d0) == full


def test_signal_none_when_day_missing():
    px, b, d0 = _synthetic()
    assert sw.signal_on(px.drop(index=d0), b, d0) is None


def test_parse_sp100_wikitext_both_formats():
    old = "{| class=\"wikitable sortable\"\n|-\n! Symbol\n! Name\n|-\n| AA \n| [[Alcoa]]\n|-\n| BRK.B\n| Berkshire\n|}"
    new = ("{| class=\"wikitable\"\n! Category !! Value\n|-\n| Closing || 1 || x\n|}\n"
           "{| id=\"constituents\"\n|-\n! Symbol\n! Name\n! Sector\n|-\n| AAPL\n| [[Apple Inc.|Apple]]\n| IT\n|}")
    assert sw.parse_sp100_wikitext(old) == {"AA", "BRK-B"}
    assert sw.parse_sp100_wikitext(new) == {"AAPL"}
    assert sw.parse_sp100_wikitext("no table") == set()


def test_pick_events_filters_forms_and_items():
    rows = pd.DataFrame({
        "form": ["8-K", "8-K", "10-Q", "10-K/A", "10-K", "4"],
        "filing_date": ["2015-01-27", "2015-02-01", "2015-01-28", "2015-03-01", "2011-01-01", "2015-01-27"],
        "report_date": [""] * 6, "acceptance": [""] * 6,
        "items": ["2.02,9.01", "5.02", "", "", "", ""], "accn": list("abcdef"),
    })
    ev = sw.pick_events(rows, "2012-01-01", "2021-12-31")
    assert list(ev["kind"]) == ["8-K 2.02", "10-Q"]


def test_earnings_dates_one_per_quarter():
    ev = pd.DataFrame({
        "kind": ["8-K 2.02", "8-K 2.02", "10-Q", "8-K 2.02", "10-Q", "10-K"],
        "form": ["8-K", "8-K", "10-Q", "8-K", "10-Q", "10-K"],
        "filing_date": ["2015-03-10", "2015-04-20", "2015-04-30", "2015-05-15", "2015-07-30", "2016-02-20"],
        "report_date": ["", "2015-04-20", "2015-03-31", "2015-05-15", "2015-06-30", "2015-12-31"],
        "accn": ["k0", "k1", "q1", "k2", "q2", "y1"],
    })
    e = sw.earnings_dates(ev)
    # 1분기: 분기말 뒤 첫 2.02(k1). 분기 중 2.02(k0)·10-Q 뒤 2.02(k2)는 버림. 2분기·연간: 2.02 없음 → 10-Q·10-K
    assert list(e["accn"]) == ["k1", "q2", "y1"]
    assert list(e["source"]) == ["8-K 2.02", "10-Q", "10-K"]


def test_parse_index_accepted():
    html = '<div class="infoHead">Accepted</div>\n<div class="info">2021-10-28 16:30:23</div>'
    assert sw.parse_index_accepted(html) == "2021-10-28 16:30:23"
    assert sw.parse_index_accepted("<html></html>") is None


def test_reaction_day_intraday_is_next_day():
    # 2026-10-09 결정 1: 장중 발표도 다음 거래일
    days = pd.DatetimeIndex(["2021-10-28", "2021-10-29"])
    assert sw.reaction_day(pd.Timestamp("2021-10-28 12:00"), days) == pd.Timestamp("2021-10-29")
    assert sw.reaction_day(pd.Timestamp("2021-10-28 09:29"), days) == pd.Timestamp("2021-10-28")
    assert sw.reaction_day(pd.Timestamp("2021-10-28 09:30"), days) == pd.Timestamp("2021-10-29")


def test_name_sim_and_wiki_names():
    assert sw.name_sim("Alexion Pharmaceuticals", "ALEXION PHARMACEUTICALS INC") == 1.0
    assert sw.name_sim("Yahoo!", "Altaba Inc.") < 0.6
    txt = "|-\n|ALXN\n|[[Alexion Pharmaceuticals]]\n|-\n| HNZ\n| [[H. J. Heinz Company|Heinz]]\n|-\n|AAPL\n|AAPL\n"
    assert sw.parse_ticker_names(txt) == {"ALXN": {"Alexion Pharmaceuticals"}, "HNZ": {"Heinz"}}


def test_judge_period():
    f, l = pd.Timestamp("2014-01-31"), pd.Timestamp("2016-06-30")
    assert sw.judge_period(f, l, pd.Timestamp("2011-06-01"), pd.Timestamp("2016-06-20")) == "맞음"
    assert sw.judge_period(f, l, pd.Timestamp("2017-01-03"), pd.Timestamp("2021-12-31")) == "다른 회사 의심"
    assert sw.judge_period(f, l, pd.Timestamp("2015-01-02"), pd.Timestamp("2016-06-20")) == "앞이 빔"
    assert sw.judge_period(f, l, None, None) == "가격 없음"


def test_split_adjust_removes_split_jump():
    idx = pd.bdate_range("2020-01-01", periods=4)
    raw = pd.DataFrame({"raw_close": [100.0, 102.0, 51.0, 52.0], "volume": [10.0, 10.0, 20.0, 20.0],
                        "split_factor": [1.0, 1.0, 2.0, 1.0]}, index=idx)
    adj = sw.split_adjust(raw)
    assert adj["close"].tolist() == [50.0, 51.0, 51.0, 52.0]
    assert adj["volume"].tolist() == [20.0, 20.0, 20.0, 20.0]
