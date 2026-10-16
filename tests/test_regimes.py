"""factory/regimes.py 테스트 (네트워크 없음 — 합성 데이터 + 기존 백테스트 캐시가 있으면 실제 QQQ)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factory import regimes as rg

BT_QQQ = Path(__file__).resolve().parents[1] / "data" / "cache" / "backtest" / "QQQ.parquet"


def _series(values, start="2000-01-03"):
    return pd.Series(np.asarray(values, dtype=float), index=pd.bdate_range(start, periods=len(values)))


def test_no_lookahead_changing_future_does_not_change_past():
    rng = np.random.default_rng(7)
    s = _series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, 900))))
    base = rg.classify(s)
    for t in (250, 500, 800):
        alt = s.copy()
        alt.iloc[t + 1:] = alt.iloc[t + 1:] * rng.uniform(0.3, 3.0, len(alt) - t - 1)
        pd.testing.assert_series_equal(rg.classify(alt).iloc[: t + 1], base.iloc[: t + 1])
        pd.testing.assert_series_equal(rg.classify(s.iloc[: t + 1]), base.iloc[: t + 1])


def test_unknown_before_200_days():
    r = rg.classify(_series(np.linspace(100, 150, 260)))
    assert (r.iloc[:199] == rg.UNKNOWN).all() and r.iloc[199] == rg.UP


def test_up_and_down_by_sma():
    up = rg.classify(_series(np.linspace(100, 120, 260)))
    assert up.iloc[-1] == rg.UP
    # 천천히 내려 200일선 아래지만 고점 대비 −20%·20일 −10%는 아님 → 하락장
    v = np.concatenate([np.full(220, 100.0), np.linspace(100, 93, 40)])
    r = rg.classify(_series(v))
    assert r.iloc[-1] == rg.DOWN


def test_crash_has_priority_and_exact_boundaries():
    # 252일 최고 100 → 정확히 80(−20%)이면 폭락장, 80.01이면 (200일선 아래) 하락장. 20일 하락은 −10%보다 완만
    base = [100.0] * 160  # 100일에 걸쳐 내려 20거래일 하락은 약 −4% (빠른 하락 조건은 안 걸림)
    r = rg.classify(_series(base + list(np.linspace(100, 80, 101))[1:]))
    assert r.iloc[-1] == rg.CRASH
    r2 = rg.classify(_series(base + list(np.linspace(100, 80.01, 101))[1:]))
    assert r2.iloc[-1] == rg.DOWN
    # 20거래일 정확히 −10% → 폭락장 (200일선 위여도 폭락장이 우선)
    v = list(np.linspace(50, 100, 240)) + [100.0] * 20
    v[-1] = 90.0  # 21거래일 전(=20거래일 전 종가) 100 → 90
    s = _series(v)
    assert s.iloc[-21] == 100.0 and s.iloc[-1] > s.rolling(200).mean().iloc[-1]
    assert rg.classify(s).iloc[-1] == rg.CRASH
    s.iloc[-1] = 90.01
    assert rg.classify(s).iloc[-1] == rg.UP


def test_segments():
    reg = pd.Series(["판단불가", "상승장", "상승장", "하락장"], index=pd.bdate_range("2020-01-01", periods=4))
    seg = rg.segments(reg)
    assert list(seg["regime"]) == ["판단불가", "상승장", "하락장"] and list(seg["days"]) == [1, 2, 1]


@pytest.mark.skipif(not BT_QQQ.exists(), reason="기존 QQQ 캐시 없음")
def test_known_history_on_real_qqq():
    q = pd.read_parquet(BT_QQQ)["close"]
    r = rg.classify(q)
    oct08 = r["2008-10-01":"2008-10-31"]
    mar20 = r["2020-03-01":"2020-03-31"]
    assert (oct08 == rg.CRASH).mean() > 0.9
    # 규칙대로면 2020-03 중 03-02~06(고점 대비 −20% 전)·03-26·03-31(반등으로 −20%·20일 −10% 둘 다 벗어남)은 폭락장이 아니다
    assert (mar20 == rg.CRASH).mean() > 0.6
    assert (mar20["2020-03-09":"2020-03-25"] == rg.CRASH).all()
    assert (r["2017"] == rg.UP).mean() > 0.9
