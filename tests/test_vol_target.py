"""변동성 맞춤 레버리지 전략·여러 자산 장부 (네트워크 없음, 합성 데이터)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import multi_book as mb
from core import vol_target as vt


def _close(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)), index=pd.bdate_range("2000-01-03", periods=n))


def test_sigma_no_future_data():
    c = _close()
    full = vt.sigma(c)
    t = 250
    assert full.iloc[t] == pytest.approx(vt.sigma(c.iloc[: t + 1]).iloc[-1])
    assert full.iloc[t] == pytest.approx(c.pct_change().iloc[t - 19: t + 1].std(ddof=1) * np.sqrt(252))


def test_weights_and_exposure():
    assert np.allclose(vt.weights_of(0.4), [0.6, 0.4, 0.0]) and np.allclose(vt.weights_of(1.5), [0.0, 0.5, 0.5])
    assert np.allclose(vt.weights_of(2.0), [0, 0, 1]) and np.allclose(vt.weights_of(5), [0, 0, 1])
    for e in (0.0, 0.3, 1.0, 1.7, 2.0):
        assert vt.exposure(vt.weights_of(e)) == pytest.approx(e)


def test_targets_cap_and_v2():
    c = _close()
    pos = vt.week_end_positions(c.index, 210)
    assert all(c.index[p].dayofweek == 4 for p in pos[:-1])  # 합성 영업일 → 금요일
    e1 = vt.target_exposures(c, pos, 0.20)
    s = vt.sigma(c).to_numpy()
    assert e1[0] == pytest.approx(min(2.0, 0.20 / s[pos[0]]))
    down = c.copy()
    down.iloc[300:] = down.iloc[300:] * 0.5  # 200일선 아래로
    e2 = vt.target_exposures(down, pos, 0.20, cap_below_sma=True)
    assert all(e <= 1.0 for p, e in zip(pos, e2) if p >= 300 and down.iloc[p] < down.rolling(200).mean().iloc[p])


def test_block_shuffle_keeps_blocks():
    rng = np.random.default_rng(20261022)
    v = list(range(10))
    out = vt.block_shuffle(v, rng)
    assert sorted(out) == v
    blocks = {tuple(v[i:i + 4]) for i in range(0, 10, 4)}
    i = 0
    while i < len(out):  # 섞인 수열은 원래 블록들을 이어 붙인 것
        b = next(b for b in blocks if tuple(out[i:i + len(b)]) == b)
        i += len(b)


def test_threshold_rebalance_next_day_and_cost():
    n = 6
    prices = np.column_stack([np.ones(n), np.full(n, 100.0), np.full(n, 50.0)])
    dec = {1: vt.weights_of(1.1), 3: vt.weights_of(1.5)}  # 0.1 차이 → 안 맞춤, 0.5 차이 → 다음 날 맞춤
    r = mb.run_multi(prices, vt.weights_of(1.0), dec, vt.should_rebalance(0.2), cost=0.001, start_krw=1000.0)
    assert r.trades == 1
    assert vt.exposure(r.weights[3]) == pytest.approx(1.0) and vt.exposure(r.weights[4]) == pytest.approx(1.5, abs=1e-3)
    assert r.costs == pytest.approx(500 * 0.001 + 499.5 - 499.5 / 1.001)  # 1배 절반 팔기 + 2배 사기
    assert r.nav[-1] == pytest.approx(1000 - r.costs)


def test_tax_common_rule_may_payment_and_liquidation():
    d = pd.bdate_range("2010-12-30", "2011-05-10")
    n = len(d)
    px = np.linspace(100, 200, n)
    prices = np.column_stack([np.ones(n), px, px])
    fx = np.full(n, 1000.0)
    sell_day = 2  # 2011년에 1배를 2배로 반 바꿔 이익 실현
    r = mb.run_multi(prices, [0, 1, 0], {1: np.array([0, 0.5, 0.5])}, lambda a, b: True, cost=0.0, tax=True, fx=fx,
                     start_krw=100_000_000.0, may_pay={}, years=d.year.to_numpy())
    gain = 50_000_000 * (px[sell_day] / 100 - 1)
    assert r.trades == 1
    end_gain = 100_000_000 * (px[-1] / 100 - 1)  # 끝날 전부 팔면 총이익 = 끝 가치 − 1억(비용 0)
    assert r.end_after_tax_krw == pytest.approx(100_000_000 + end_gain - max(end_gain - 2_500_000, 0) * 0.22, rel=1e-6)
    assert gain > 0


def test_cagr_mdd():
    assert mb.cagr(1.0, 2.0, 365.25) == pytest.approx(1.0)
    assert mb.max_drawdown(np.array([1, 2, 1, 3])) == pytest.approx(-0.5)
