"""core/shock_dip_perf.py 테스트 (합성 데이터, 네트워크 없음) — 관문 3 장부·세금·무작위 배치."""

import numpy as np
import pandas as pd

from core import shock_dip_perf as sp


def test_hold_only_tracks_price_and_liquidation_tax():
    px = np.array([100.0, 110.0, 120.0])
    r = sp.run_book(px, px.copy(), [], cost=0.0, tax=True, fx=np.full(3, 1000.0), start_krw=100_000_000,
                    years=np.array([2000, 2000, 2000]))
    assert np.allclose(r.nav, [1e5, 1.1e5, 1.2e5])
    gain = 120_000_000 - 100_000_000
    assert np.isclose(r.taxes_krw, (gain - 2_500_000) * 0.22)
    assert np.isclose(r.end_after_tax_krw, 120_000_000 - (gain - 2_500_000) * 0.22)


def test_entry_exit_50_50_with_costs():
    px = np.array([100.0, 100.0, 110.0, 110.0])
    pl = np.array([10.0, 10.0, 12.0, 12.0])
    r = sp.run_book(px, pl, sp.strategy_events([(1, 3)], 4), cost=0.001, start_krw=1000.0)
    # 1일: X 500 팔고(비용 0.5) L 499.5/1.001 삼
    l_val = 499.5 / 1.001
    v2 = 500 / 100 * 110 + l_val / 10 * 12
    assert np.isclose(r.nav[2], v2)
    assert r.trades == 2 and r.costs > 0


def test_may_payment_uses_previous_year_gains():
    dates = pd.to_datetime(["2000-12-28", "2000-12-29", "2001-05-01", "2001-05-02"])
    px = np.array([100.0, 100.0, 100.0, 100.0])
    pl = np.array([10.0, 20.0, 20.0, 20.0])
    may = sp.may_payment_positions(pd.DatetimeIndex(dates))
    assert may == {2: 2000}
    # 0일 진입 → 1일 청산: L이 두 배 → 2000년 실현 이익
    r = sp.run_book(px, pl, [(0, 0.5), (1, 1.0)], cost=0.0, tax=True, fx=np.ones(4), start_krw=10_000_000,
                    may_pay=may, years=dates.year.to_numpy())
    owed = (5_000_000 - 2_500_000) * 0.22
    assert np.isclose(r.nav[2], 15_000_000 - owed)


def test_switch_events_and_leveraged_formula():
    assert sp.switch_events(np.array([True, True, False, False, True])) == [(1, 0.0), (3, 1.0)]
    r1 = pd.Series([0.0, 0.01, -0.02])
    rate = pd.Series([2.52, 2.52, 5.04])
    lev = sp.leveraged_returns(r1, rate, 0.0252)
    assert np.allclose(lev, [2 * 0.0 - 0.0001 - 0.0001, 0.02 - 0.0001 - 0.0001, -0.04 - 0.0001 - 0.0001])


def test_random_placements_keep_holds_and_do_not_overlap():
    rng = np.random.default_rng(0)
    holds = [60, 20, 5, 60]
    for _ in range(200):
        pl = sp.random_placements(holds, 10, 400, rng)
        assert sorted(x - e for e, x in pl) == sorted(holds)
        assert pl[0][0] >= 10 and pl[-1][1] <= 400
        assert all(pl[i + 1][0] >= pl[i][1] + 1 for i in range(len(pl) - 1))
    assert sp.random_placements([300, 300], 0, 100, rng) is None


def test_nav_before_t_unchanged_when_future_prices_change():
    rng = np.random.default_rng(3)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    pl = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    ev = sp.strategy_events([(50, 110), (150, 210)], 300)
    a = sp.run_book(px, pl, ev).nav
    px2, pl2 = px.copy(), pl.copy()
    px2[200:] *= 3
    pl2[200:] *= 0.2
    b = sp.run_book(px2, pl2, ev).nav
    assert np.allclose(a[:200], b[:200])
