"""core/mu_cycle.py·data/mu_cycle.py 테스트 (합성 데이터, 네트워크 없음)."""

import numpy as np
import pandas as pd
import pytest

from core import mu_cycle as mc
from data import mu_cycle as dm


def _monthly(values, start="1990-01-31"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="ME"), dtype=float)


def test_zigzag_exact_points():
    # 100 → 150(꼭대기) → 80(−46.7%, 바닥) → 140(+75%, 꼭대기) → 70(−50%) → 75(미확정 끝)
    p = _monthly([100, 120, 150, 130, 89, 80, 100, 130, 140, 120, 84, 70, 75])
    z = mc.zigzag(p)
    assert list(z["kind"]) == ["peak", "trough", "peak"]
    assert list(z["price"]) == [150, 80, 140]
    assert list(z["date"]) == [p.index[2], p.index[5], p.index[8]]
    assert list(z["confirmed"]) == [p.index[4], p.index[7], p.index[10]]  # 문턱을 넘은 달


def test_zigzag_ignores_small_moves():
    p = _monthly([100, 130, 90, 125, 80, 110, 70])  # −40%·+60%를 한 번도 못 넘음 … 130→70은 −46%
    z = mc.zigzag(p)
    assert list(z["kind"]) == ["peak"] and z["price"].iloc[0] == 130
    assert mc.zigzag(_monthly([100, 110, 95, 105, 90])).empty


def test_zigzag_starts_with_trough_when_rise_first():
    p = _monthly([50, 40, 70, 41])  # 40 → 70 = +75% 먼저 → 바닥, 그다음 70 → 41 = −41% → 꼭대기
    z = mc.zigzag(p)
    assert list(z["kind"]) == ["trough", "peak"] and list(z["price"]) == [40, 70]


def test_cycle_table_values():
    p = _monthly([100, 120, 150, 130, 89, 80, 100, 130, 160, 120, 84, 70, 75])
    bench = _monthly(np.linspace(100, 112, 13))
    t = mc.cycle_table(mc.zigzag(p), p, bench)
    r = t.iloc[0]
    assert np.isclose(r["하락폭"], 80 / 150 - 1) and r["하락 기간(개월)"] == 3
    assert r["회복 기간(개월)"] == 3  # 80(5) → 160(8)에서 150 회복
    assert np.isclose(r["상승폭"], 160 / 80 - 1) and r["사이클 길이(개월)"] == 6


def test_monthly_lag_two_months():
    obs = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2000-01-01", "2000-02-01", "2000-03-01"]))
    me = pd.date_range("2000-01-31", "2000-07-31", freq="ME")
    k = mc.known_monthly(obs, "monthly", me)
    assert np.isnan(k["2000-02-29"])  # 1월 값은 3월 1일부터
    assert k["2000-03-31"] == 1.0 and k["2000-04-30"] == 2.0 and k["2000-05-31"] == 3.0
    assert k["2000-07-31"] == 3.0


def test_weekly_and_daily_lags():
    me = pd.DatetimeIndex([pd.Timestamp("2000-01-31")])
    w = pd.Series([5.0, 6.0], index=pd.to_datetime(["2000-01-22", "2000-01-29"]))
    assert mc.known_monthly(w, "weekly", me).iloc[0] == 5.0  # 1-29 값은 2-05부터
    d = pd.Series([1.0, 2.0], index=pd.to_datetime(["2000-01-28", "2000-01-31"]))
    assert mc.known_monthly(d, "daily", me).iloc[0] == 1.0  # 1-31 값은 다음 날부터


def test_known_monthly_no_future_data():
    obs = pd.Series(np.arange(60, dtype=float), index=pd.date_range("2000-01-01", periods=60, freq="MS"))
    me = pd.date_range("2000-01-31", periods=60, freq="ME")
    full = mc.known_monthly(obs, "monthly", me)
    part = mc.known_monthly(obs[obs.index <= "2002-06-01"], "monthly", me[me <= "2002-06-30"])
    pd.testing.assert_series_equal(part, full[full.index <= "2002-06-30"])


def test_transforms_rate_vs_level():
    m = _monthly(np.arange(1, 25, dtype=float))
    lvl = mc.transforms(m, False)
    assert np.isclose(lvl["전년비"].iloc[12], (13 / 1 - 1) * 100) and np.isclose(lvl["6개월"].iloc[6], (7 / 1 - 1) * 100)
    rate = mc.transforms(m, True)
    assert rate["전년비"].iloc[12] == 12 and rate["6개월"].iloc[6] == 6


def test_lead_months_sign():
    ind = _monthly(np.r_[np.full(20, 5.0), [1.0], np.full(20, 5.0)])  # 지표 최저 = 20번째 달
    assert mc.lead_months(ind, [ind.index[23]], "trough") == [3.0]  # 3개월 먼저
    assert mc.lead_months(ind, [ind.index[18]], "trough") == [-2.0]
    assert mc.lead_months(-ind, [ind.index[23]], "peak") == [3.0]
    assert np.isnan(mc.lead_months(ind, [ind.index[35]], "trough")[0])  # 창이 끝을 넘음


def test_signal_fires_on_rebound_crossing_only():
    ind = _monthly([0] * 11 + [-5, -4, -2.5, -1, -3.5, 0] + [0] * 14)
    sig = mc.signal_dates(ind)
    # 12개월 창이 다 차는 11번째 달부터 판단(최저 −5 대비): 11 +0 거짓, 12 +1 거짓, 13 +2.5 참 → 신호,
    # 14 +4 참(신호 아님), 15 +1.5 거짓, 16 +5 참 → 두 번째 신호
    assert sig == [ind.index[13], ind.index[16]]


def test_signal_stats_hit_miss_false():
    idx = pd.date_range("1990-01-31", periods=60, freq="ME")
    ind = pd.Series(np.r_[np.full(12, 0.0), [-5, -5, 0], np.full(45, 0.0)], index=idx)  # 신호 = 14번째 달
    price = pd.Series(100.0, index=idx)
    price.iloc[15:20] = 70.0  # 신호 뒤 −30% → 헛신호
    st = mc.signal_stats(ind, price, [idx[24], idx[50]], start=idx[0])
    assert st["신호 수"] == 1 and st["적중"] == 1 and st["헛신호"] == 1
    assert st["평가 바닥"] == 2 and st["놓침"] == 1  # 50번째 달 바닥 앞 12개월엔 신호 없음


def test_quarterly_eps_and_ttm():
    f = pd.DataFrame({
        "start": pd.to_datetime(["2009-09-04", "2009-12-04", "2010-03-05", "2009-09-04", "2010-09-03"]),
        "end": pd.to_datetime(["2009-12-03", "2010-03-04", "2010-06-03", "2010-09-02", "2010-12-02"]),
        "filed": pd.to_datetime(["2010-01-10", "2010-04-10", "2010-07-10", "2010-10-25", "2011-01-10"]),
        "val": [0.2, 0.3, 0.4, 1.5, 0.1]})
    q = mc.quarterly_eps(f)
    assert np.allclose(q["eps"], [0.2, 0.3, 0.4, 0.6, 0.1])
    assert np.isnan(mc.ttm_eps(q, pd.Timestamp("2010-10-01")))  # 4분기 공시 전
    assert np.isclose(mc.ttm_eps(q, pd.Timestamp("2010-11-01")), 1.5)
    assert np.isclose(mc.ttm_eps(q, pd.Timestamp("2011-02-01")), 0.3 + 0.4 + 0.6 + 0.1)


def test_base_hit_rate():
    idx = pd.date_range("1990-01-31", periods=30, freq="ME")
    # 평가 달 0~17(뒤 12개월 필요). 바닥 20 → t = 8~17이 적중 → 10/18
    assert np.isclose(mc.base_hit_rate(idx, [idx[20]], idx[0]), 10 / 18)


def test_quarterly_eps_known_date_capped_at_end_plus_60():
    f = pd.DataFrame({"start": pd.to_datetime(["2009-12-04"]), "end": pd.to_datetime(["2010-03-04"]),
                      "filed": pd.to_datetime(["2011-04-12"]), "val": [0.39]})  # 1년 뒤 비교 값으로만 공시
    q = mc.quarterly_eps(f)
    assert q["filed"].iloc[0] == pd.Timestamp("2010-05-03")


def test_forward_return_has_no_wraparound():
    p = _monthly([100, 110, 121, 133.1])
    fr = mc.forward_return(p, 2)
    assert np.isclose(fr.iloc[0], 0.21) and fr.iloc[2:].isna().all()


# ── 날짜 차단 ──


def test_guard_blocks_2013_and_seal():
    ok = pd.Series([1.0], index=pd.to_datetime(["2012-12-31"]))
    assert dm.guard(ok) is ok
    with pytest.raises(dm.FutureDataError):
        dm.guard(pd.Series([1.0, 2.0], index=pd.to_datetime(["2012-12-31", "2013-01-02"])))
    with pytest.raises(dm.FutureDataError, match="봉인"):
        dm.guard(pd.Series([1.0], index=pd.to_datetime(["2022-01-03"])))


def test_loaders_refuse_cached_2013_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "CACHE", tmp_path)
    pd.DataFrame({"Close": [1.0, 2.0], "Adj Close": [1.0, 2.0], "Volume": [10, 10]},
                 index=pd.to_datetime(["2012-12-31", "2013-01-02"])).to_csv(tmp_path / "yf_MU.csv")
    with pytest.raises(dm.FutureDataError):
        dm.load_yf("MU")
    pd.Series([1.0, 2.0], index=pd.to_datetime(["2012-12-01", "2013-01-01"]), name="value").to_csv(
        tmp_path / "fred_INDPRO.csv", index_label="date")
    with pytest.raises(dm.FutureDataError):
        dm.load_fred("INDPRO")
    pd.DataFrame({"start": ["2012-09-01"], "end": ["2012-11-30"], "filed": ["2013-01-05"], "val": [0.1]}).to_csv(
        tmp_path / "edgar_MU_eps_diluted.csv", index=False)
    with pytest.raises(dm.FutureDataError):
        dm.load_mu_eps()


def test_download_requests_stop_at_2012():
    assert dm.CUTOFF == pd.Timestamp("2012-12-31")
    src = open(dm.__file__, encoding="utf-8").read()
    assert 'end="2013-01-01"' in src and "coed={CUTOFF.date().isoformat()}" in src
