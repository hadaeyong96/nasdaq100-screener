"""사이클 실험실 1단계 — 2013년 이후 차단·지연·누수·재현 테스트 (docs/prereg/cycle_lab_discovery.md 9절).

네트워크 없이 합성 데이터만 쓴다.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.cycle_lab import data as dm
from scripts.cycle_lab import lab
from scripts.cycle_lab.gate import CUTOFF, CutoffError, assert_cutoff, check_request, gate


def months(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="ME")


def french_text(last_yyyymm: int) -> str:
    """French CSV 모양 합성 문자열(첫 표 + 빈 줄 + 연간 표)."""
    lines = ["This file was created by test", "", "  Average Value Weighted Returns -- Monthly", ",Agric,Chips "]
    for m in months("2012-10-31", "2013-02-28"):
        ym = m.year * 100 + m.month
        if ym <= last_yyyymm:
            lines.append(f"{ym},   1.00,  -99.99")
    lines += ["", "  Average Value Weighted Returns -- Annual", ",Agric,Chips ", "2012,  1.0, 2.0", "2013,  1.0, 2.0"]
    return "\n".join(lines)


def rows_published_next_month(values: pd.Series) -> pd.DataFrame:
    """관측 m이 m+1 달 15일에 처음 나오고 고쳐지지 않는 ALFRED 행."""
    return pd.DataFrame({"obs": values.index,
                         "realtime_start": [lab.add_months(m, 1) - pd.offsets.MonthEnd(1) + pd.Timedelta(days=15) for m in values.index],
                         "realtime_end": CUTOFF, "value": values.to_numpy()})


# 1) 2013-01 행은 저장·계산 전에 잘린다 ─────────────────────────────────────────


def test_gate_truncates_2013_row_before_store(tmp_path, monkeypatch):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]}, index=months("2012-11-30", "2013-01-31"))
    out = gate(df)
    assert out.index.max() == pd.Timestamp("2012-12-31")
    assert len(out) == 2

    # French 경로: zip을 메모리에서 풀어 첫 표 → gate → 저장. 저장 파일에 2013이 없어야 한다.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("49_Industry_Portfolios.csv", french_text(201302))

    class Resp:
        content = buf.getvalue()

    monkeypatch.setattr(dm, "CACHE", tmp_path)
    monkeypatch.setattr(dm, "_get", lambda url, params=None: Resp())
    meta = dm.fetch_french("ind49")
    saved = pd.read_csv(tmp_path / "french_ind49.csv", index_col=0, parse_dates=True)
    assert saved.index.max() == pd.Timestamp("2012-12-31")
    assert meta["last"] == "2012-12-31"
    assert not list(tmp_path.glob("*.zip"))
    loaded = dm.load_french("ind49")
    assert loaded["Chips"].isna().all()  # −99.99 → 결측
    assert loaded["Agric"].iloc[0] == pytest.approx(0.01)


def test_gate_applies_to_alfred_rows_before_store(tmp_path, monkeypatch):
    obs = [{"date": "2012-11-01", "realtime_start": "2012-12-14", "realtime_end": "2012-12-31", "value": "1"},
           {"date": "2013-01-01", "realtime_start": "2012-12-20", "realtime_end": "2012-12-31", "value": "2"}]

    class Resp:
        def json(self):
            return {"observations": obs, "count": len(obs)}

    monkeypatch.setattr(dm, "CACHE", tmp_path)
    monkeypatch.setattr(dm, "_get", lambda url, params=None: Resp())
    monkeypatch.setattr(dm, "api_key", lambda: "x" * 32)
    dm.fetch_alfred("TEST")
    saved = dm.load_alfred("TEST")
    assert saved["obs"].max() == pd.Timestamp("2012-11-30")


# 2) 경계: 2012-12 행은 남고 2013-01-01은 제거 ──────────────────────────────────


def test_boundary_daily():
    idx = pd.DatetimeIndex(["2012-12-28", "2012-12-31", "2013-01-01", "2013-01-02"])
    out = gate(pd.Series([1.0, 2.0, 3.0, 4.0], index=idx))
    assert list(out.index) == [pd.Timestamp("2012-12-28"), pd.Timestamp("2012-12-31")]


def test_boundary_monthly_parse():
    df = gate(dm.parse_french_first_table(french_text(201301)))
    assert pd.Timestamp("2012-12-31") in df.index
    assert df.index.max() == pd.Timestamp("2012-12-31")


def test_gate_rejects_non_date_index():
    with pytest.raises(CutoffError):
        gate(pd.Series([1.0], index=[0]))


# 3) 계산 함수에 2013 이후 날짜 → 예외 ─────────────────────────────────────────


@pytest.mark.parametrize("fn", [lab.m1_signal, lab.m2_signal, lab.m3_signal, lab.m5_signal])
def test_compute_rejects_2013(fn):
    s = pd.Series(np.linspace(1, 2, 30) / 100, index=months("2010-08-31", "2013-01-31"))
    with pytest.raises(CutoffError):
        fn(s)


def test_compute_rejects_2013_other_functions():
    idx = months("2011-01-31", "2013-01-31")
    s = pd.Series(0.01, index=idx)
    with pytest.raises(CutoffError):
        lab.event_weights(s > 0, idx)
    with pytest.raises(CutoffError):
        lab.evaluate_cell(pd.Series(0.0, index=idx), [], s, s, n_random=10)
    with pytest.raises(CutoffError):
        lab.realtime_values(None, s, idx)
    with pytest.raises(CutoffError):
        lab.select_cycle(pd.DataFrame({"a": s}), s, s, s, start=idx[0], end=idx[-1])
    with pytest.raises(CutoffError):
        lab.m6_state(pd.DataFrame({"a": s, "b": s, "c": s}))
    with pytest.raises(CutoffError):
        assert_cutoff(pd.DatetimeIndex(["2013-01-01"]))


# 4) ALFRED vintage가 2012-12-31을 넘으면 예외 ─────────────────────────────────


def test_alfred_vintage_after_cutoff_raises():
    with pytest.raises(CutoffError):
        check_request({"observation_end": "2012-12-31", "realtime_end": "2013-01-01"}, alfred=True)
    with pytest.raises(CutoffError):
        check_request({"observation_end": "2012-12-31"}, alfred=True)  # vintage 끝 없음 = 오늘
    with pytest.raises(CutoffError):
        check_request({"observation_end": "2013-01-31", "realtime_end": "2012-12-31"}, alfred=True)
    with pytest.raises(CutoffError):
        check_request({"observation_end": "2012-12-31", "realtime_end": "2012-12-31"}, alfred=False)
    p = dm.alfred_params("A34SNO")
    assert p["realtime_end"] == "2012-12-31" and p["observation_end"] == "2012-12-31"
    assert "api_key" not in p
    assert dm.current_params("USREC")["observation_end"] == "2012-12-31"


def test_alfred_response_with_later_vintage_raises(tmp_path, monkeypatch):
    obs = [{"date": "2012-11-01", "realtime_start": "2013-01-15", "realtime_end": "9999-12-31", "value": "1"}]

    class Resp:
        def json(self):
            return {"observations": obs, "count": 1}

    monkeypatch.setattr(dm, "CACHE", tmp_path)
    monkeypatch.setattr(dm, "_get", lambda url, params=None: Resp())
    monkeypatch.setattr(dm, "api_key", lambda: "x" * 32)
    with pytest.raises(RuntimeError):
        dm.fetch_alfred("TEST")
    assert not (tmp_path / "alfred_TEST.csv").exists()


# 5) 지연: 달 t의 지표는 t+2 전 신호에 쓰이지 않는다 ─────────────────────────────


def test_release_lag_two_months():
    obs_idx = months("1990-01-31", "1999-12-31")
    rng = np.random.default_rng(1)
    vals = pd.Series(100 + rng.normal(0, 3, len(obs_idx)).cumsum(), index=obs_idx)
    judge = months("1990-01-31", "1999-12-31")
    base_rt = lab.realtime_values(rows_published_next_month(vals), None, judge, (2, 8))
    m0 = pd.Timestamp("1995-06-30")
    bumped = vals.copy()
    bumped[m0] += 50.0
    new_rt = lab.realtime_values(rows_published_next_month(bumped), None, judge, (2, 8))
    changed = judge[(base_rt["v2"] != new_rt["v2"]).to_numpy() & base_rt["v2"].notna().to_numpy()]
    assert changed.min() == lab.add_months(m0, 2)
    x0, x1 = lab.pct_change_rt(base_rt), lab.pct_change_rt(new_rt)
    before = judge < lab.add_months(m0, 2)
    pd.testing.assert_series_equal(x0[before], x1[before])
    s0, s1 = lab.m1_signal(x0), lab.m1_signal(x1)
    pd.testing.assert_frame_equal(s0[before], s1[before])
    # 관측 m0는 m0+1 달 15일에 나왔으므로 1순위(당시 발표본)로 읽힌다
    assert base_rt.loc[lab.add_months(m0, 2), "level"] == 1.0


def test_realtime_uses_vintage_then_fallbacks():
    judge = months("2000-01-31", "2000-12-31")
    obs = months("1999-01-31", "2000-12-31")
    vals = pd.Series(np.arange(len(obs), dtype=float) + 1, index=obs)
    rows = rows_published_next_month(vals)
    # 2000-06 관측 첫 발표 10, 2000-09-15에 11로 고침 → 2000-08 판정은 10, 2000-12 판정(관측 10월)은 무관
    o = pd.Timestamp("2000-06-30")
    rows.loc[rows["obs"] == o, "realtime_end"] = pd.Timestamp("2000-09-14")
    rows.loc[rows["obs"] == o, "value"] = 10.0
    rows = pd.concat([rows, pd.DataFrame({"obs": [o], "realtime_start": [pd.Timestamp("2000-09-15")],
                                          "realtime_end": [CUTOFF], "value": [11.0]})])
    rt = lab.realtime_values(rows, None, judge, (2,))
    assert rt.loc[pd.Timestamp("2000-08-31"), "v2"] == 10.0
    # 발표본이 없는 옛 관측 → 2순위(2012-12-31 vintage)
    rows2 = rows[rows["obs"] >= pd.Timestamp("2000-01-31")].copy()
    rows2.loc[rows2["obs"] == pd.Timestamp("2000-01-31"), "realtime_start"] = pd.Timestamp("2005-01-01")
    rt2 = lab.realtime_values(rows2, None, judge, (2,))
    assert rt2.loc[pd.Timestamp("2000-03-31"), "level"] == 2.0
    # ALFRED에 아예 없는 관측 → 3순위(현재값)
    cur = pd.Series(99.0, index=months("1999-01-31", "1999-12-31"))
    rt3 = lab.realtime_values(rows2, cur, judge, (2,))
    assert rt3.loc[pd.Timestamp("2000-02-29"), "level"] == 3.0
    assert rt3.loc[pd.Timestamp("2000-02-29"), "v2"] == 99.0


# 6) 미래 누수: 달 t 뒤 값을 바꿔도 t까지 신호는 같다 ────────────────────────────


def _perturb_after(s: pd.Series, t: pd.Timestamp, seed: int) -> pd.Series:
    out = s.copy()
    rng = np.random.default_rng(seed)
    mask = out.index > t
    out[mask] = out[mask] + rng.normal(0, 5 * float(out.std()), mask.sum())
    return out


@pytest.mark.parametrize("fn,kind", [(lab.m1_signal, "x"), (lab.m2_signal, "s"), (lab.m3_signal, "c"), (lab.m5_signal, "r")])
def test_no_future_leak_signals(fn, kind):
    idx = months("1960-01-31", "2000-12-31")
    rng = np.random.default_rng(7)
    if kind == "r":
        s = pd.Series(rng.normal(0.005, 0.09, len(idx)), index=idx)
    elif kind == "c":  # 금리 곡선: 0 위아래를 오가게
        s = pd.Series(2 * np.sin(np.arange(len(idx)) * 2 * np.pi / 70) + rng.normal(0, 0.3, len(idx)), index=idx)
    else:
        s = pd.Series(rng.normal(0, 1, len(idx)).cumsum(), index=idx)
    full = fn(s)
    for t in [pd.Timestamp("1975-06-30"), pd.Timestamp("1990-12-31")]:
        alt = s.copy()
        if kind == "r":
            alt[alt.index > t] = rng.normal(0.0, 0.2, int((alt.index > t).sum()))
        else:
            alt = _perturb_after(s, t, 3)
        pd.testing.assert_frame_equal(fn(alt).loc[:t], full.loc[:t])
        pd.testing.assert_frame_equal(fn(s.loc[:t]), full.loc[:t])
    assert full["sig"].any()


def test_no_future_leak_m4_m6_and_weights():
    idx = months("1960-01-31", "2000-12-31")
    rng = np.random.default_rng(11)
    ret = pd.DataFrame(rng.normal(0.01, 0.06, (len(idx), 4)), index=idx, columns=list("abcd"))
    t = pd.Timestamp("1985-03-31")
    alt = ret.copy()
    alt.loc[alt.index > t] = rng.normal(0, 0.3, alt.loc[alt.index > t].shape)
    pd.testing.assert_frame_equal(lab.m6_state(alt).loc[:t], lab.m6_state(ret).loc[:t])

    a = pd.Series(100 + rng.normal(0, 1, len(idx)).cumsum(), index=idx)
    b, c, d = a * 1.01, pd.Series(rng.normal(1, 1, len(idx)), index=idx), pd.Series(rng.normal(1, 1, len(idx)), index=idx)
    ph = lab.m4_phase(a, b, c, d)
    ph2 = lab.m4_phase(_perturb_after(a, t, 1), _perturb_after(b, t, 2), _perturb_after(c, t, 3), _perturb_after(d, t, 4))
    pd.testing.assert_series_equal(ph.loc[:t], ph2.loc[:t])

    sig = pd.Series(rng.random(len(idx)) < 0.05, index=idx)
    sig2 = sig.copy()
    sig2[sig2.index > t] = ~sig2[sig2.index > t].to_numpy()
    win = months("1963-07-31", "2000-12-31")
    for fn in (lab.event_weights, lab.state_weights):
        w1, _ = fn(sig, win)
        w2, _ = fn(sig2, win)
        # 판정 달 t까지의 신호는 t+1 달 비중까지만 정한다
        upto = lab.add_months(t, 1)
        pd.testing.assert_series_equal(w1.loc[:upto], w2.loc[:upto])


# 7) 같은 seed면 무작위 대조 결과가 같다 ───────────────────────────────────────


def test_random_control_reproducible():
    W1 = lab.random_weights([24, 24, 7, 3], 300, n=200, seed=lab.SEED)
    W2 = lab.random_weights([24, 24, 7, 3], 300, n=200, seed=lab.SEED)
    W3 = lab.random_weights([24, 24, 7, 3], 300, n=200, seed=lab.SEED + 1)
    np.testing.assert_array_equal(W1, W2)
    assert not np.array_equal(W1, W3)
    # 모든 행: 같은 보유 달 수·같은 비중 합, 겹침 없음(비중 ≤ 20%)
    held = (W1 > 0).sum(axis=1)
    assert (held == 58).all()
    assert W1.max() <= 0.20 + 1e-12
    assert np.allclose(W1.sum(axis=1), 2 * lab.episode_path(24).sum() + lab.episode_path(7).sum() + lab.episode_path(3).sum())

    idx = months("1980-01-31", "1999-12-31")
    rng = np.random.default_rng(5)
    core = pd.Series(rng.normal(0.008, 0.04, len(idx)), index=idx)
    ind = pd.Series(rng.normal(0.01, 0.08, len(idx)), index=idx)
    sig = pd.Series(False, index=idx)
    sig.iloc[[10, 60, 120, 200]] = True
    w, eps = lab.event_weights(sig, idx)
    r1 = lab.evaluate_cell(w, eps, core, ind, n_random=300)
    r2 = lab.evaluate_cell(w, eps, core, ind, n_random=300)
    assert r1 == r2


# 전략 규칙 점검 ────────────────────────────────────────────────────────────


def test_event_weights_ramp_hold_and_ignore():
    idx = months("1990-01-31", "1995-12-31")
    sig = pd.Series(False, index=idx)
    sig[pd.Timestamp("1990-03-31")] = True
    sig[pd.Timestamp("1990-10-31")] = True  # 보유 중 → 무시
    sig[pd.Timestamp("1992-03-31")] = True  # b+23 말(1992-03) 신호 → 1992-04부터 다시 5%
    w, eps = lab.event_weights(sig, idx)
    assert list(w.loc["1990-04-30":"1990-08-31"]) == [0.05, 0.10, 0.15, 0.20, 0.20]
    assert w.loc["1990-03-31"] == 0.0
    assert w.loc["1992-03-31"] == 0.20 and w.loc["1992-04-30"] == 0.05
    assert [e["months"] for e in eps] == [24, 24]


def test_state_weights_exit_and_reentry():
    idx = months("1990-01-31", "1994-12-31")
    st = pd.Series(False, index=idx)
    st.loc["1990-01-31":"1990-02-28"] = True  # 2달 켜짐 → 1990-02, 03 보유, 1990-03 판정 꺼짐 → 04부터 0
    st.loc["1990-06-30":"1993-12-31"] = True  # 길게 켜짐 → 24개월째 말 복귀, 꺼질 때까지 재진입 없음
    w, eps = lab.state_weights(st, idx)
    assert list(w.loc["1990-02-28":"1990-04-30"]) == [0.05, 0.10, 0.0]
    assert eps[1]["start"] == pd.Timestamp("1990-07-31") and eps[1]["months"] == 24
    assert w.loc["1992-07-31"] == 0.0 and w.loc["1993-12-31"] == 0.0
    assert len(eps) == 2


def test_cost_and_metrics():
    w = np.array([0.0, 0.05, 0.10, 0.0])
    core = np.zeros(4)
    ind = np.zeros(4)
    r = lab.strat_returns(w, core, ind)
    np.testing.assert_allclose(r, [0.0, -0.0001, -0.0001, -0.0002])
    assert lab.cagr(np.full(12, 0.01)) == pytest.approx(1.01 ** 12 - 1)
    assert lab.mdd(np.array([0.1, -0.5, 0.2])) == pytest.approx(-0.5)


def test_m1_arming_rules():
    x = pd.Series([5.0] * 12 + [4.0, 6.5, 7.0, 3.0, 5.5, 5.0], index=months("1990-01-31", "1991-06-30"))
    s = lab.m1_signal(x)
    # 12번째(4.0)에서 새 최저 → 무장, 13번째 6.5−4.0 ≥ 2 → 신호, 15번째 3.0 새 최저 → 재무장, 16번째 5.5−3.0 ≥ 2 → 신호
    assert list(np.flatnonzero(s["sig"].to_numpy())) == [13, 16]


def test_m5_rules():
    # 60달 1.0 → 0.5(−50%, 무장)로 10달 → 0.55(10개월 평균 0.505 위) = 신호 한 번
    lv = [1.0] * 60 + [0.5] * 10 + [0.55]
    r = pd.Series(np.diff([1.0] + lv) / np.array([1.0] + lv[:-1]), index=months("1960-01-31", "1965-11-30"))
    s = lab.m5_signal(r)
    assert s["sig"].sum() == 1
    assert s.index[s["sig"].to_numpy()][0] == pd.Timestamp("1965-11-30")


def test_select_cycle_and_pick_top():
    idx = months("1963-07-31", "2012-12-31")
    rng = np.random.default_rng(3)
    mkt = pd.Series(rng.normal(0.005, 0.045, len(idx)), index=idx)
    rf = pd.Series(0.003, index=idx)
    rec = pd.Series((rng.random(len(idx)) < 0.15).astype(float), index=idx)
    cols = {f"i{k}": rf + b * mkt + rng.normal(0, 0.02 + 0.01 * k, len(idx)) for k, b in enumerate(np.linspace(0.6, 1.6, 9))}
    ind = pd.DataFrame(cols)
    sel = lab.select_cycle(ind, mkt, rf, rec)
    assert sel["B"].sum() == 3 and sel["C"].sum() == 4
    assert (sel["cycle"] == ((sel["A"] | sel["B"]) & sel["C"])).all()

    cells = pd.DataFrame([
        {"industry": "a", "method": "M1", "pctile": 95.0, "excess_cagr": 0.01, "entries": 9, "ind_order": 0, "method_order": 0},
        {"industry": "b", "method": "M1", "pctile": 99.0, "excess_cagr": 0.01, "entries": 9, "ind_order": 1, "method_order": 0},
        {"industry": "a", "method": "M2", "pctile": 99.0, "excess_cagr": 0.02, "entries": 9, "ind_order": 0, "method_order": 1},
        {"industry": "c", "method": "M3", "pctile": 89.9, "excess_cagr": 0.05, "entries": 9, "ind_order": 2, "method_order": 2},
        {"industry": "c", "method": "M5", "pctile": 97.0, "excess_cagr": 0.05, "entries": 7, "ind_order": 2, "method_order": 4},
    ])
    top = lab.pick_top(cells)
    assert list(zip(top["industry"], top["method"])) == [("a", "M2"), ("b", "M1")]
    top_x = lab.pick_top(cells, exclude=[("b", "M1")])
    assert list(zip(top_x["industry"], top_x["method"])) == [("a", "M2"), ("a", "M1")]


def test_bh_and_zigzag():
    q = lab.bh_q(np.array([0.01, 0.04, 0.03, 0.5]))
    np.testing.assert_allclose(q, [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.5], rtol=1e-9)
    lv = pd.Series([1.0, 1.2, 0.8, 0.7, 1.0, 1.1, 0.75, 1.2], index=months("1990-01-31", "1990-08-31"))
    tr = lab.zigzag_troughs(lv)
    assert tr == [pd.Timestamp("1990-04-30"), pd.Timestamp("1990-07-31")]
