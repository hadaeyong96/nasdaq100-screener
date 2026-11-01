"""core/winter_2x.py·봉인 검사 테스트 (합성 데이터, 네트워크 없음)."""

import numpy as np
import pandas as pd
import pytest

from core import leverage_timing as lt
from core import winter_2x as w2


def _days(start="1990-01-02", end="2009-12-31"):
    return pd.bdate_range(start, end)


def _close(days, seed=0):
    r = np.random.default_rng(seed).normal(0.0004, 0.01, len(days))
    return pd.Series(100 * np.cumprod(1 + r), index=days)


def test_season_months():
    assert w2.WINTER == {11, 12, 1, 2, 3, 4}
    assert w2.ROBUST["10~4월"] == {10, 11, 12, 1, 2, 3, 4}
    assert w2.ROBUST["12~5월"] == {12, 1, 2, 3, 4, 5}


def test_h1_switches_only_day_after_oct_apr_month_end():
    d = _days()
    w0 = int(np.searchsorted(d, pd.Timestamp("1990-06-01")))  # 여름에 시작 → 첫 판단은 1배라 이벤트 없음
    pos = w2.decision_positions(d, w0)
    ev = w2.events(pos, w2.h1_targets(d, pos))
    assert len(ev) == 2 * 19 + 1  # 1990-10 ~ 2009-10, 연 2회
    for p, w in ev:
        prev = d[p - 1]
        assert prev.month != d[p].month  # 전날이 그달 마지막 거래일
        assert prev.month == (10 if w == w2.W_2X else 4)


def test_h1_starts_levered_if_window_opens_in_winter():
    d = _days()
    pos = w2.decision_positions(d, 0)
    ev = w2.events(pos, w2.h1_targets(d, pos))
    assert ev[0] == (1, w2.W_2X)  # 첫날 판단 → 다음 날 체결


def test_h2_never_levered_below_sma_and_subset_of_h1():
    d = _days()
    close = _close(d)
    pos = w2.decision_positions(d, 300)
    h1, h2 = w2.h1_targets(d, pos), w2.h2_targets(close, pos)
    sma = close.rolling(200).mean()
    for p, a, b in zip(pos, h1, h2):
        if b == w2.W_2X:
            assert a == w2.W_2X and close.iloc[p] >= sma.iloc[p]
        if a == w2.W_2X and close.iloc[p] >= sma.iloc[p]:
            assert b == w2.W_2X


def test_h2_no_future_data():
    d = _days()
    close = _close(d)
    pos = w2.decision_positions(d, 300)
    full = w2.h2_targets(close, pos)
    cut = 3000
    pos_cut = [p for p in pos if p + 1 < cut]
    part = w2.h2_targets(close.iloc[:cut], pos_cut)
    assert part == full[:len(pos_cut)]


def test_random_path_average_exposure_is_1_5():
    d = _days()
    pos = w2.decision_positions(d, 0)
    rng = np.random.default_rng(1)
    day_means = []
    for _ in range(50):
        tg = w2.random_targets(d, pos, rng)
        # 달 기준: 해마다 2배 달이 정확히 6개
        by_year = {}
        for p, w in zip(pos, tg):
            y, m = w2.held_month(d, p)
            by_year.setdefault(y, []).append(w2.leverage_of(w))
        full = [v for y, v in by_year.items() if len(v) == 12]
        assert full and all(np.isclose(np.mean(v), 1.5) for v in full)
        expo = lt.exposure(w2.events(pos, tg), 0, len(d))[1:]
        day_means.append(w2.leverage_of(expo).mean())
    assert abs(np.mean(day_means) - 1.5) < 0.01  # 일 기준(달마다 거래일 수가 달라 근사)


def test_random_with_all_starts_11_equals_h1():
    d = _days()
    pos = w2.decision_positions(d, 0)

    class Fixed:
        def integers(self, lo, hi, size):
            return np.full(size, 11)

    assert w2.random_targets(d, pos, Fixed()) == w2.h1_targets(d, pos)


def test_half_year_returns_and_spells():
    d = _days("2000-01-03", "2002-12-31")
    price = pd.Series(np.arange(len(d), dtype=float) + 100, index=d)
    hy = w2.half_year_returns(price)
    me = price.groupby([d.year, d.month]).tail(1)
    at = {(t.year, t.month): v for t, v in me.items()}
    row = hy[hy["연도"] == 2001].iloc[0]
    assert np.isclose(row["겨울"], at[(2001, 4)] / at[(2000, 10)] - 1)
    assert np.isclose(row["여름"], at[(2001, 10)] / at[(2001, 4)] - 1)
    nav = np.array([1.0, 1.0, 1.2, 0.9, 1.1, 1.1])
    wts = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    sp = w2.lever_spells(nav, wts, d[:6])
    assert len(sp) == 1
    assert np.isclose(sp.iloc[0]["구간 수익률"], 0.1) and np.isclose(sp.iloc[0]["구간 안 최대 낙폭"], 0.9 / 1.2 - 1)


def test_sealed_guard_cuts_2022_and_later(monkeypatch):
    from scripts import sealed_guard as sg

    idx = pd.date_range("2021-12-28", "2022-01-05")
    s = pd.Series(range(len(idx)), index=idx, dtype=float)
    assert sg.cut(s).index.max() <= pd.Timestamp("2021-12-31")
    wrapped = sg._wrap(lambda *_: s)
    assert (wrapped().index < pd.Timestamp("2022-01-01")).all()


def test_runner_loaders_are_sealed():
    import scripts.winter_2x_run  # noqa: F401  (import 시 sealed_guard.install())
    from scripts import shock_dip_gate01 as g1
    from scripts import shock_dip_gate2 as g2
    from scripts import shock_dip_gate3 as g3

    for m in (g1, g2, g3):
        for fn in ("load_yf", "load_fred"):
            if hasattr(m, fn):
                assert hasattr(getattr(m, fn), "__wrapped__"), f"{m.__name__}.{fn} 봉인 안 됨"


def test_next_version_dir_never_overwrites(tmp_path):
    from scripts.winter_2x_run import next_version_dir

    assert next_version_dir(tmp_path).name == "v1"
    (tmp_path / "v1").mkdir()
    (tmp_path / "v2").mkdir()
    assert next_version_dir(tmp_path).name == "v3"


@pytest.mark.parametrize("v", ["H1", "H2"])
def test_judge_requires_all_criteria(v):
    from scripts.winter_2x_run import judge

    us = ["S&P500", "나스닥 종합", "QQQ(보정)"]
    res = pd.DataFrame([{"변형": vv, "구간": g, "연복리": 0.12, "항상 1배 연복리": 0.10, "항상 1.5배 연복리": 0.11,
                         "무작위 95백분위": 0.115, "무작위 백분위": 97.0, "최대 낙폭": -0.4, "항상 1.5배 최대 낙폭": -0.45}
                        for vv in ("H1", "H2") for g in us])
    rob = pd.DataFrame([{"변형": vv, "겨울": lab, "구간": g, "연복리": 0.12, "항상 1배 연복리": 0.10, "항상 1.5배 연복리": 0.11}
                        for vv in ("H1", "H2") for lab in w2.ROBUST for g in us])
    fr = pd.DataFrame([{"변형": vv, "지수": f"F{i}", "세전 연복리": 0.1 if i < 3 else 0.0, "항상 1.5배 연복리": 0.05}
                       for vv in ("H1", "H2") for i in range(5)])
    assert set(judge(res, rob, fr)["종합"]) == {"PASS"}
    res.loc[(res["변형"] == v) & (res["구간"] == "QQQ(보정)"), "최대 낙폭"] = -0.5  # 3번만 깨짐
    out = judge(res, rob, fr).set_index("변형")["종합"]
    assert out[v] == "FAIL" and out[{"H1": "H2", "H2": "H1"}[v]] == "PASS"
