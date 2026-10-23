"""상승장 대형주 웅덩이 갈아타기 core/puddle.py — 합성 데이터, 네트워크 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import puddle as pu


def _cal(n=700, start="2010-01-04"):
    return pd.bdate_range(start, periods=n)


def _frame(cal, series: dict):
    return pd.DataFrame(series, index=cal)


def _world(stock_path, n=700, extra=None):
    """QQQ는 꾸준히 오름(지수 조건 늘 참), 종목 A는 stock_path. 구성·상위 10은 A(+extra)."""
    cal = _cal(n)
    q = pd.Series(100 * 1.0004 ** np.arange(n), index=cal)
    cols = {"A": stock_path}
    cols.update(extra or {})
    close = _frame(cal, cols)
    vol = close * 0 + 1e6
    members = {cal[0]: set(cols)}
    fx = pd.Series(1000.0, index=cal)
    return cal, q, close, vol, members, fx


def _puddle_path(n=700, dip_at=450):
    """오르다가 dip_at에서 20일선 아래로 잠깐 빠졌다가 회복해 새 고점."""
    x = 100 * 1.001 ** np.arange(n)
    x[dip_at:dip_at + 10] *= np.linspace(0.97, 0.95, 10)
    x[dip_at + 10:dip_at + 30] *= np.linspace(0.96, 1.02, 20)
    x[dip_at + 30:] *= 1.02
    return x


def test_indicators_no_future_data():
    rng = np.random.default_rng(1)
    cal = _cal(400)
    c = pd.DataFrame({"A": 100 + rng.normal(0, 1, 400).cumsum(), "B": 50 + rng.normal(0, 1, 400).cumsum()}, index=cal)
    v = pd.DataFrame(rng.integers(1, 9, (400, 2)) * 1e5, index=cal, columns=["A", "B"])
    for t in (250, 320, 399):
        cut = c.iloc[:t + 1]
        assert np.allclose(pu.sma(cut, 200).iloc[-1], pu.sma(c, 200).iloc[t], equal_nan=True)
        assert np.allclose(pu.prior_high(cut, 60).iloc[-1], pu.prior_high(c, 60).iloc[t], equal_nan=True)
        assert np.allclose(pu.avg_dollar_volume(cut, v.iloc[:t + 1]).iloc[-1], pu.avg_dollar_volume(c, v).iloc[t], equal_nan=True)


def test_top_merges_share_classes():
    dv = np.array([5.0, 9.0, 8.0, 1.0, np.nan])
    member = np.array([True, True, True, True, True])
    company = np.array([0, 1, 1, 2, 3])  # 열 1·2는 같은 회사(GOOG·GOOGL)
    assert pu.top_by_dollar_volume(dv, member, company, 3) == [1, 0, 3]
    assert pu.top_by_dollar_volume(dv, np.array([True, False, True, False, False]), company, 10) == [2, 0]


def test_qqq_only_matches_buy_and_hold_before_tax():
    cal, q, close, vol, members, fx = _world(_puddle_path())
    inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[-1])
    r = pu.simulate(inp, pu.Params(tax=False), no_stocks=True)
    assert r.nav[-1] / r.nav[0] == pytest.approx(inp.qqq_px[-1] / inp.qqq_px[0])
    assert not r.fills and r.costs_usd == 0


def test_puddle_entry_and_filled_exit():
    cal, q, close, vol, members, fx = _world(_puddle_path())
    inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[-1])
    r = pu.simulate(inp, pu.Params(tax=False))
    assert r.fills, "1차 매수가 있어야 한다"
    f0 = r.fills[0]
    sig = r.entry_signal_days[0]
    assert f0["pos"] == sig + 1 and f0["tranche"] == 1  # 신호 다음 거래일 체결
    a, m20 = inp.sig[sig, 0], inp.sma[20][sig, 0]
    assert a < m20 and inp.sig[sig - 1, 0] >= inp.sma[20][sig - 1, 0]  # 처음 20일선 아래
    assert f0["usd"] == pytest.approx(0.05 * r.nav[sig] * (1 - 0.001), rel=1e-9)  # QQQ를 신호 날 자산 5%만큼 팔고(비용 뺀 돈으로 삼)
    assert r.trades[0]["reason"] in ("메워짐", "구간 끝")
    assert any(tr["reason"] == "메워짐" for tr in r.trades)


def test_no_buy_when_index_below_200():
    cal, q, close, vol, members, fx = _world(_puddle_path())
    qd = pd.Series(200 * 0.999 ** np.arange(len(cal)), index=cal)  # QQQ 내리막 → 200일선 아래
    inp, _ = pu.build_inputs(qd, qd, close, close, vol, members, {}, fx, cal[250], cal[-1])
    assert not pu.simulate(inp, pu.Params(tax=False)).fills


def test_cap_and_max_names():
    n = 700
    base = _puddle_path(n)
    extra = {"B": base * 1.0, "C": base * 1.0}  # 같은 날 세 종목이 동시에 신호
    cal, q, close, vol, members, fx = _world(base, n, extra)
    inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[-1])
    r = pu.simulate(inp, pu.Params(tax=False))
    first_day = r.fills[0]["pos"]
    assert len({f["k"] for f in r.fills if f["pos"] == first_day}) == 2  # 동시 최대 2종목
    assert max(r.stock_w) < 2 * 0.15 + 0.02


def test_cooldown_blocks_rebuy():
    n = 760
    x = _puddle_path(n, dip_at=450)
    x2 = x.copy()
    x2[490:500] *= np.linspace(0.97, 0.95, 10)  # 메워진 직후 다시 웅덩이(20거래일 안)
    cal, q, close, vol, members, fx = _world(x2, n)
    inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[-1])
    r = pu.simulate(inp, pu.Params(tax=False))
    sells = [tr["exit_pos"] for tr in r.trades if tr["reason"] != "구간 끝"]
    for s in sells:
        assert not any(s < f["pos"] - 1 <= s + 20 and f["tranche"] == 1 for f in r.fills)


def test_tax_and_harvest_qqq():
    cal, q, close, vol, members, fx = _world(_puddle_path())
    inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[-1])
    no_harv = pu.simulate(inp, pu.Params(tax=True), no_stocks=True)
    assert no_harv.harvests >= 1 and no_harv.taxes_krw >= 0
    pre = pu.simulate(inp, pu.Params(tax=False), no_stocks=True)
    assert no_harv.end_after_tax_krw < pre.nav[-1] * inp.fx[-1]


def test_signals_do_not_use_future():
    """t까지 자른 데이터로 돌린 체결이 전체 데이터 체결의 t 이전 부분과 같다."""
    cal, q, close, vol, members, fx = _world(_puddle_path())
    full_inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[-1])
    full = pu.simulate(full_inp, pu.Params(tax=False))
    for t in (470, 520, 600):
        inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[t])
        part = pu.simulate(inp, pu.Params(tax=False))
        last = len(inp.dates) - 1
        assert [f for f in part.fills if f["pos"] <= last] == [f for f in full.fills if f["pos"] <= last]
        assert np.allclose(part.nav, full.nav[:last + 1])


def test_random_control_uses_given_days():
    cal, q, close, vol, members, fx = _world(_puddle_path())
    inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[-1])
    r = pu.simulate(inp, pu.Params(tax=False))
    rr = pu.simulate(inp, pu.Params(tax=False), random_days=r.entry_signal_days, rng=np.random.default_rng(0))
    assert [f["pos"] for f in rr.fills if f["tranche"] == 1][:1] == [r.fills[0]["pos"]]  # 후보가 하나면 같은 날 같은 종목
    none = pu.simulate(inp, pu.Params(tax=False), random_days=[], rng=np.random.default_rng(0))
    assert not none.fills


def test_cash30_prototype_keeps_cash():
    cal, q, close, vol, members, fx = _world(_puddle_path())
    inp, _ = pu.build_inputs(q, q, close, close, vol, members, {}, fx, cal[250], cal[-1])
    r = pu.simulate(inp, pu.Params(tax=False, mode="cash30"))
    assert r.nav[0] == pytest.approx(100_000_000 / 1000)
    assert r.nav[-1] / r.nav[0] < inp.qqq_px[-1] / inp.qqq_px[0]  # 현금 30%가 놀아 QQQ보다 낮다
