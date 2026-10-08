"""E1 분기 EPS·TTM·성장 판정·장부 테스트 (네트워크 없음, 합성 데이터)."""

from __future__ import annotations

import inspect
import random
import re

import numpy as np
import pandas as pd
import pytest

from core import e1
from core import moat_backtest as mb


# ── 합성 공시 ───────────────────────────────────────────────────────────────


def _q_ends(n, last="2021-09-30"):
    return list(pd.date_range(end=last, periods=n, freq="QE"))


def _facts(values, ends=None, filed_lag=40, form="10-Q"):
    """3개월 사실 목록. values[i]는 ends[i] 분기 값."""
    ends = ends or _q_ends(len(values))
    out = []
    for v, e in zip(values, ends):
        s = (e - pd.offsets.QuarterBegin(startingMonth=1)).normalize() if False else (e - pd.Timedelta(days=90))
        out.append({"val": v, "start": s.date().isoformat(), "end": e.date().isoformat(),
                    "filed": (e + pd.Timedelta(days=filed_lag)).date().isoformat(), "form": form})
    return out


def test_future_filed_not_used():
    ents = _facts([1.0] * 16)
    last_filed = pd.Timestamp(ents[-1]["filed"])
    q_before = e1.quarterly_values(ents, last_filed - pd.Timedelta(days=1))
    q_after = e1.quarterly_values(ents, last_filed)
    assert len(q_before) == 15 and len(q_after) == 16
    # 같은 기간을 나중에 고쳐 공시한 값은 그 공시일 뒤에만 쓰인다
    fix = {**ents[-1], "val": 9.0, "filed": (last_filed + pd.Timedelta(days=30)).date().isoformat(), "form": "10-Q/A"}
    q = e1.quarterly_values(ents + [fix], last_filed + pd.Timedelta(days=10))
    assert q.iloc[-1] == 1.0
    q = e1.quarterly_values(ents + [fix], last_filed + pd.Timedelta(days=30))
    assert q.iloc[-1] == 9.0


def test_q4_equals_annual_minus_nine_months_and_fallback():
    fy_s, q1, q2, q3, fy_e = "2020-01-01", "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"
    base = [
        {"val": 1.0, "start": fy_s, "end": q1, "filed": "2020-05-01", "form": "10-Q"},
        {"val": 1.2, "start": "2020-04-01", "end": q2, "filed": "2020-08-01", "form": "10-Q"},
        {"val": 1.3, "start": "2020-07-01", "end": q3, "filed": "2020-11-01", "form": "10-Q"},
        {"val": 6.0, "start": fy_s, "end": fy_e, "filed": "2021-02-15", "form": "10-K"},
    ]
    nine = {"val": 3.4, "start": fy_s, "end": q3, "filed": "2020-11-01", "form": "10-Q"}
    q = e1.quarterly_values(base + [nine], "2021-03-01")
    assert q.loc[fy_e] == pytest.approx(6.0 - 3.4)
    q = e1.quarterly_values(base, "2021-03-01")  # 9개월 누계 없음 → 3개월 3개 합
    assert q.loc[fy_e] == pytest.approx(6.0 - 3.5)
    assert len(e1.quarterly_values(base, "2021-02-14")) == 3  # 10-K 공시 전


def test_split_adjustment_uses_filed_date_and_ignores_future_splits():
    ents = [{"val": 4.0, "start": "2021-01-01", "end": "2021-03-31", "filed": "2021-05-01", "form": "10-Q"},
            {"val": 1.1, "start": "2021-04-01", "end": "2021-06-30", "filed": "2021-08-20", "form": "10-Q"}]
    splits = pd.Series({pd.Timestamp("2021-07-20"): 4.0})
    q_before_split = e1.quarterly_values(ents, "2021-07-19", splits)
    assert q_before_split.iloc[0] == 4.0  # 판단일 뒤 분할은 안 씀
    q_after = e1.quarterly_values(ents, "2021-09-01", splits)
    assert q_after.iloc[0] == pytest.approx(1.0)  # 분할 전 공시 → ÷4
    assert q_after.iloc[1] == pytest.approx(1.1)  # 분할 뒤 공시 → 그대로


def test_growth_rule_ranking_and_ties():
    rising = pd.Series(np.linspace(1.0, 2.5, 16), index=_q_ends(16))
    g = e1.growth_info(rising)
    assert g.eligible and g.pool_ok and g.growth == pytest.approx(g.ttm[0] / g.ttm[1] - 1)
    a = e1.growth_info(rising)
    b = e1.growth_info(rising)
    infos = {"BBB": a, "AAA": b, "CCC": e1.growth_info(rising * 1.0 + np.r_[np.zeros(12), np.ones(4)])}
    assert e1.rank_growth(infos, ["AAA", "BBB", "CCC"]) == ["CCC", "AAA", "BBB"]  # 동점은 티커순


def test_base_ttm_nonpositive_excluded_and_sold():
    vals = [-1.0] * 8 + [-0.5] * 4 + [0.5] * 4  # TTM(t−4) < 0
    g = e1.growth_info(pd.Series(vals, index=_q_ends(16)))
    assert g.growth is None and not g.eligible and g.pool_ok
    assert e1.should_sell(g)
    assert e1.rank_growth({"X": g}, ["X"]) == []
    falling = e1.growth_info(pd.Series(np.r_[np.ones(12), np.full(4, 0.9)], index=_q_ends(16)))
    assert falling.growth < 0 and e1.should_sell(falling)
    flat = e1.growth_info(pd.Series(np.ones(16), index=_q_ends(16)))
    assert flat.growth == 0 and not e1.should_sell(flat) and not flat.eligible  # 같으면 증가 아님, 줄지도 않음
    assert e1.should_sell(None)


def test_consecutive_gap_breaks():
    ends = _q_ends(16)
    q = pd.Series(1.0, index=ends[:12] + ends[13:])  # 최근 8개 안에서 한 분기 빠짐
    assert e1.consecutive_tail(q, 8) is None
    assert e1.growth_info(q).growth is None


def test_revenue_tag_merge_priority():
    a = [{"val": 10.0, "start": "2018-01-01", "end": "2018-03-31", "filed": "2018-05-01", "form": "10-Q"}]
    b = [{"val": 99.0, "start": "2018-01-01", "end": "2018-03-31", "filed": "2018-05-01", "form": "10-Q"},
         {"val": 11.0, "start": "2017-10-01", "end": "2017-12-31", "filed": "2018-02-01", "form": "10-Q"}]
    q = e1.merge_revenue([a, b], "2018-06-01")
    assert list(q) == [11.0, 10.0]


# ── 장부 ─────────────────────────────────────────────────────────────────────


def _market(n_days=300, tickers=("A", "B"), prices=None, fx=1000.0):
    days = pd.bdate_range("2020-01-01", periods=n_days)
    close = pd.DataFrame({t: (prices or {}).get(t, pd.Series(100.0, index=days)) for t in tickers}, index=days)
    div = pd.DataFrame(0.0, index=days, columns=list(tickers))
    last = {t: close[t].last_valid_index() for t in tickers}
    return e1.Market(days=days, close=close, div_cum=div.cumsum(), last_date=last, fx=pd.Series(fx, index=days), buy_days=[])


def _buy_days(m, n):
    per = m.days.to_period("M")
    return [m.days[per == p][0] for p in per.unique()[:n]]


def test_contribution_total_and_no_negative_cash():
    m = _market(n_days=2700)
    m.buy_days = _buy_days(m, 120)
    picks = [["A"] if i % 2 else ["A", "B"] for i in range(120)]
    flags = [{"A": i % 7 == 0} for i in range(120)]
    r = e1.simulate(m, picks, flags, m.days[-1])
    assert r["contributed"] == 120 * 300_000
    assert r["min_cash"] >= -1e-6


def test_sell_first_then_buy_same_day():
    m = _market(n_days=80)
    m.buy_days = _buy_days(m, 3)
    r = e1.simulate(m, [["A"], ["B"], ["B"]], [{}, {"A": True}, {}], m.days[-1], apply_tax=False)
    held_day2 = r["log"][1]["held"]
    assert held_day2 == ["B"]  # A를 팔고 그 돈까지 B를 샀다
    c = 0.0007
    usd1 = 300 * 0.999
    a_sh = usd1 / (1 + c) / 100
    cash2 = a_sh * 100 * (1 - c) + usd1
    b_sh = cash2 / (1 + c) / 100 + usd1 / (1 + c) / 100
    final = b_sh * 100 * (1 - c) * 1000 * 0.999
    assert r["final"] == pytest.approx(final)


def test_tax_deduction_no_carryforward_paid_next_year():
    days = pd.bdate_range("2020-01-01", "2022-03-31")
    px = pd.Series(100.0, index=days)
    px.loc["2020-06-01":] = 300.0  # 2020년 크게 오른 뒤 매도
    px.loc["2021-01-01":] = 100.0
    m = e1.Market(days=days, close=pd.DataFrame({"A": px, "B": pd.Series(100.0, index=days)}),
                  div_cum=pd.DataFrame(0.0, index=days, columns=["A", "B"]), last_date={"A": days[-1], "B": days[-1]},
                  fx=pd.Series(1000.0, index=days), buy_days=[])
    per = days.to_period("M")
    m.buy_days = [days[per == p][0] for p in per.unique()[:24]]
    picks = [["A"]] * 5 + [["B"]] * 19
    flags = [{}] * 5 + [{"A": True}] + [{}] * 18
    r = e1.simulate(m, picks, flags, pd.Timestamp("2021-12-31"))
    a_usd = 5 * 300 * 0.999
    sale = a_usd / 1.0007 / 100 * 300 * 0.9993
    gain_krw = (sale - a_usd) * 1000
    assert r["taxes"][2020] == pytest.approx(0.22 * (gain_krw - 2_500_000))
    assert r["taxes"].get(2021, 0.0) == 0.0  # 2021 손실 없음·이월 없음
    no_tax = e1.simulate(m, picks, flags, pd.Timestamp("2021-12-31"), apply_tax=False)
    assert no_tax["final"] - r["final"] == pytest.approx(r["taxes"][2020] / 0.999 * 0.999, rel=1e-6)


def test_price_gap_liquidates_at_last_close():
    m = _market(n_days=80)
    m.close.loc[m.days[30]:, "A"] = np.nan
    m.ffill = m.close.ffill()
    m.last_date = {"A": m.days[29], "B": m.days[-1]}
    m.buy_days = _buy_days(m, 3)
    r = e1.simulate(m, [["A"], ["B"], ["B"]], [{}, {}, {}], m.days[-1], apply_tax=False)
    assert "A" not in r["log"][-1]["held"] and r["sells"] >= 1


def test_unit_value_first_is_one_and_mdd():
    m = _market(n_days=80)
    m.buy_days = _buy_days(m, 3)
    r = e1.simulate(m, [["A"]] * 3, [{}] * 3, m.days[-1])
    uv = r["unit_value"]
    assert uv.iloc[0] == pytest.approx(300 * 0.999 / 1.0007 * 100 / 100 * 1000 / 300_000)  # 첫날 비용만큼
    assert r["mdd_pct"] <= 0


def test_random_seed_reproducible():
    pool = [f"T{i:02d}" for i in range(30)]

    def draw():
        rng = random.Random(20261007)
        return [mb.draw_random_portfolio(pool, 5, rng) for _ in range(10)]

    assert draw() == draw()
    assert mb.percentile_rank(5.0, [1, 2, 5, 9]) == 75.0


def test_no_negative_shift_and_seal_constant():
    from scripts import e1_experiments

    for mod in (e1, e1_experiments):
        assert not re.search(r"shift\(\s*-", inspect.getsource(mod)), mod.__name__
    assert e1_experiments.SEAL == pd.Timestamp("2021-12-31")


def test_load_prices_drops_rows_after_seal(tmp_path, monkeypatch):
    from scripts import e1_experiments as ex

    monkeypatch.setattr(ex, "PRICE_CACHE", tmp_path)
    idx = pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03"])
    pd.DataFrame({"close": [1.0, 2.0, 3.0], "div": 0.0, "split": 0.0}, index=idx).to_csv(tmp_path / "ZZZ.csv")
    prices, missing = ex.load_prices(["ZZZ"])
    assert prices["ZZZ"].index.max() == pd.Timestamp("2021-12-31") and missing == []
