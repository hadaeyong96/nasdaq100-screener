"""core/mu_strategy.py·2단계 날짜 차단 테스트 (합성 데이터, 네트워크 없음)."""

import numpy as np
import pandas as pd
import pytest

from core import mu_strategy as ms
from data import mu_cycle as dm


def _days(start="2013-01-01", end="2016-12-31"):
    return pd.bdate_range(start, end)


def test_tranche_weight_never_above_10pct_even_when_b_rallies():
    d = _days()
    n = len(d)
    pa = np.full(n, 100.0)
    pb = 10.0 * np.cumprod(np.full(n, 1.01))  # B가 매일 1% 올라 비중이 저절로 커짐
    ev = ms.entry_events([pd.Timestamp("2013-01-31")], d, 18)
    b = ms.run_book(pa, pb, np.zeros(n), np.zeros(n), ev, tax_on=False, dates=d)
    assert 0 < b.max_b_weight_after_buy <= ms.MAX_WEIGHT + 1e-9


def test_tranches_reach_10pct_when_prices_flat():
    d = _days()
    n = len(d)
    ev = ms.entry_events([pd.Timestamp("2013-01-31")], d, 18)
    b = ms.run_book(np.full(n, 100.0), np.full(n, 10.0), np.zeros(n), np.zeros(n), ev, cost=0.0, tax_on=False, dates=d)
    assert np.isclose(b.max_b_weight_after_buy, 0.10)


def test_entry_event_dates():
    d = _days()
    ev = ms.entry_events([pd.Timestamp("2013-01-31")], d, 18)
    got = sorted((d[p].date().isoformat(), w, bo) for p, (w, bo) in ev.items())
    assert got == [("2013-02-01", 0.025, True), ("2013-03-01", 0.05, True), ("2013-04-01", 0.075, True),
                   ("2013-05-01", 0.10, True), ("2014-11-03", 0.0, False)]  # 5월 + 18개월 = 2014-11 첫 거래일


def test_accept_signals_ignores_while_holding():
    s = [pd.Timestamp(x) for x in ["2013-01-31", "2013-06-30", "2014-10-31", "2014-11-30", "2016-01-31"]]
    acc = ms.accept_signals(s, pd.Timestamp("2012-12-31"), pd.Timestamp("2021-11-30"), 18)
    # 2013-01 진입 → 복귀 달 2014-11. 2013-06·2014-10 무시, 2014-11부터 다시 받음
    assert acc == [pd.Timestamp("2013-01-31"), pd.Timestamp("2014-11-30")]


@pytest.mark.parametrize("n", [0, 1, 3, 4])
def test_random_paths_have_same_entry_count_and_no_overlap(n):
    months = pd.date_range("2012-12-31", "2021-11-30", freq="ME")
    rng = np.random.default_rng(0)
    seen_first = set()
    for _ in range(200):
        r = ms.random_signal_months(n, months, 18, rng)
        assert len(r) == n
        assert ms.accept_signals(r, months[0], months[-1], 18) == r  # 겹침 없음 = 모두 받아들여짐
        if r:
            seen_first.add(r[0])
    if n == 1:
        assert len(seen_first) > 50  # 시작 달이 고르게 퍼짐


def test_realtime_change_prefers_vintage_then_lagged_current():
    me = pd.DatetimeIndex([pd.Timestamp("2011-05-31"), pd.Timestamp("2011-07-31")])
    obs = pd.date_range("2010-01-01", "2011-06-01", freq="MS")
    current = pd.Series(np.arange(len(obs), dtype=float) + 100, index=obs)
    vint = pd.Series([50.0, 60.0], index=pd.to_datetime(["2010-12-01", "2011-06-01"]))
    x = ms.realtime_change(me, {me[1]: vint}, current)
    # 5월 말: 발표본 없음 → 현재 값 중 3월 관측까지(+2개월) → 3월 / 9월 − 1
    assert not x["vintage"].iloc[0] and x["obs_month"].iloc[0] == pd.Timestamp("2011-03-01")
    assert np.isclose(x["value"].iloc[0], (current["2011-03-01"] / current["2010-09-01"] - 1) * 100)
    assert x["vintage"].iloc[1] and np.isclose(x["value"].iloc[1], 20.0)


def test_annual_rebalance_restores_10pct():
    d = _days("2013-01-01", "2014-12-31")
    n = len(d)
    p = int(np.searchsorted(d, pd.Timestamp("2014-01-01")))
    pb = np.where(np.arange(n) < 200, 10.0, 30.0)  # B 세 배 → 비중 25%로 커짐
    pb[p + 1:] = 60.0  # 재조정 다음 날 B 두 배
    b = ms.run_book(np.full(n, 100.0), pb, np.zeros(n), np.zeros(n), ms.annual_rebalance_events(d), cost=0.0,
                    tax_on=False, dates=d)
    assert np.isclose(b.nav[p + 1] / b.nav[p], 1.10)  # 재조정 뒤 B가 정확히 10%였음


def test_tax_matches_shared_rule_on_hold():
    d = pd.to_datetime(["2013-12-30", "2013-12-31", "2014-05-01", "2014-05-02"])
    n = len(d)
    pa = np.array([100.0, 120.0, 120.0, 120.0])
    ev = {1: (0.0, False)}  # 이벤트 아님 — B 0% 맞추기는 거래 없음
    b = ms.run_book(pa, np.full(n, 10.0), np.zeros(n), np.zeros(n), ev, cost=0.0, fx=np.full(n, 1000.0), dates=pd.DatetimeIndex(d))
    gain = 120_000_000 - 100_000_000  # 끝날 청산(2014년)
    assert np.isclose(b.end_after_tax_krw, 120_000_000 - (gain - 2_500_000) * 0.22)


def test_dividend_withholding_reinvested():
    d = _days("2013-01-01", "2013-01-10")
    n = len(d)
    da = np.zeros(n)
    da[2] = 1.0
    b = ms.run_book(np.full(n, 100.0), np.full(n, 10.0), da, np.zeros(n), {}, cost=0.0, tax_on=False, dates=d)
    assert np.isclose(b.nav[-1] / b.nav[0], 1.0085)  # 주당 1달러 배당(1%) × 85%


# ── 봉인 ⓑ 날짜 차단 ──


def test_guard_s2_blocks_2022():
    ok = pd.Series([1.0], index=pd.to_datetime(["2021-12-31"]))
    assert dm.guard_s2(ok) is ok
    with pytest.raises(dm.FutureDataError, match="봉인"):
        dm.guard_s2(pd.Series([1.0], index=pd.to_datetime(["2022-01-03"])))


def test_s2_loaders_refuse_2022_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "CACHE", tmp_path)
    monkeypatch.setattr(dm, "S2_ALFRED", tmp_path / "s2_alfred")
    pd.DataFrame({"Close": [1.0, 2.0], "Adj Close": [1.0, 2.0], "Volume": [1, 1], "Dividends": [0, 0]},
                 index=pd.to_datetime(["2021-12-31", "2022-01-03"])).to_csv(tmp_path / "s2_yf_MU.csv")
    with pytest.raises(dm.FutureDataError):
        dm.load_yf_s2("MU")
    pd.Series([1.0, 2.0], index=pd.to_datetime(["2021-12-31", "2022-01-03"]), name="value").to_csv(
        tmp_path / "s2_fred_DEXKOUS.csv", index_label="date")
    with pytest.raises(dm.FutureDataError):
        dm.load_fred_s2("DEXKOUS")
    (tmp_path / "s2_alfred").mkdir()
    pd.DataFrame({"date": ["2020-01-01", "2020-03-01"], "value": [1, 2]}).to_csv(
        tmp_path / "s2_alfred" / "A34SNO_2020-02-29.csv", index=False)
    with pytest.raises(dm.FutureDataError):
        dm.load_alfred_asof("A34SNO", pd.Timestamp("2020-02-29"))  # 발표본 날짜 뒤 관측값


def test_alfred_fetch_refuses_sealed_dates():
    with pytest.raises(dm.FutureDataError):
        dm.fetch_alfred_asof("A34SNO", pd.Timestamp("2022-01-31"))


def test_stage2_requests_stop_at_2021():
    src = open(dm.__file__, encoding="utf-8").read()
    assert 'end="2022-01-01"' in src and "coed={S2_CUTOFF.date().isoformat()}" in src
    assert dm.S2_CUTOFF == pd.Timestamp("2021-12-31")
