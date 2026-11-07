"""사이클 실험실 2단계 — 차단·누수·판정·세금·전체 실행 테스트 (docs/prereg/cycle_lab_confirm.md 9절).

네트워크 없이 합성 데이터만 쓴다.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from core.tax import capital_gains_tax
from scripts.cycle_lab import confirm_data as cd
from scripts.cycle_lab import confirm_lab as cl
from scripts.cycle_lab import confirm_run as cr
from scripts.cycle_lab import data as dm
from scripts.cycle_lab import lab
from scripts.cycle_lab.confirm_gate import CUTOFF, CutoffError, assert_cutoff, check_request, gate


def months(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="ME")


def synthetic_ind(end: str = "2021-12-31", seed: int = 7) -> pd.DataFrame:
    idx = months("1990-01-31", end)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.01, 0.07, (len(idx), len(cl.CYCLE_17))), index=idx, columns=cl.CYCLE_17)


def synthetic_spread(end: str = "2021-12-31", seed: int = 8) -> pd.Series:
    idx = months("1990-01-31", end)
    rng = np.random.default_rng(seed)
    return pd.Series(2.0 + np.cumsum(rng.normal(0.0, 0.25, len(idx))), index=idx)


def synthetic_daily(end: str = "2021-12-31", seed: int = 9) -> pd.DataFrame:
    idx = pd.bdate_range("2012-12-03", end)
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.01, len(idx))
    close = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame({"Close": close, "Dividends": 0.0}, index=idx)


def french_text() -> str:
    lines = ["This file was created by test", "", "  Average Value Weighted Returns -- Monthly", "," + ",".join(["Agric"] + cl.CYCLE_17)]
    for m in months("2021-10-31", "2022-03-31"):
        lines.append(f"{m.year * 100 + m.month}," + ",".join(["1.00"] * (len(cl.CYCLE_17) + 1)))
    return "\n".join(lines + ["", "  Annual", "2021,1.0"])


# ── 1. 2022-01 행 자르기 ──────────────────────────────────────────────────────


def test_gate_drops_2022_keeps_2021_12():
    s = pd.Series(1.0, index=months("2021-11-30", "2022-02-28"))
    out = gate(s)
    assert out.index.max() == pd.Timestamp("2021-12-31")
    assert pd.Timestamp("2022-01-31") not in out.index


def test_french_fetch_cuts_before_saving(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("49.csv", french_text())

    class R:
        content = buf.getvalue()

    monkeypatch.setattr(dm, "_get", lambda *a, **k: R())
    meta = cd.fetch_french(tmp_path)
    assert meta["last"] == "2021-12-31"
    saved = (tmp_path / "french_ind49.csv").read_text(encoding="utf-8")
    assert "2021-12-31" in saved and "2022" not in saved
    assert "Agric" not in saved  # 사이클 17개 열만
    assert list(cd.load_french(tmp_path).columns) == cl.CYCLE_17


def test_drop_rows_after_removes_before_parse():
    assert "202201" not in cd.drop_rows_after(french_text(), 202112)
    assert "202112" in cd.drop_rows_after(french_text(), 202112)


# ── 2. 계산 함수 2022 이후 날짜 예외 ──────────────────────────────────────────


def test_compute_functions_raise_on_2022():
    idx = months("2019-01-31", "2022-01-31")
    s = pd.Series(1.0, index=idx)
    for fn, arg in [(cl.m2_signal, s), (cl.m5_signal, s * 0.01), (cl.m6_state, pd.DataFrame({"A": s})),
                    (lambda x: cl.event_weights(x, cl.EVAL_MONTHS), s.astype(bool)),
                    (lambda x: cl.state_weights(x, cl.EVAL_MONTHS), s.astype(bool)),
                    (assert_cutoff, s)]:
        with pytest.raises(CutoffError):
            fn(arg)


def test_compute_raises_on_2022_inputs():
    late = synthetic_ind("2022-02-28")
    with pytest.raises(CutoffError):
        cl.compute(late, synthetic_spread(), pd.Series(0.01, index=cl.EVAL_MONTHS), pd.Series(1100.0, index=cl.BOOK_MONTHS), 5)


# ── 3. ALFRED vintage ─────────────────────────────────────────────────────────


def test_alfred_request_rejects_late_vintage():
    with pytest.raises(CutoffError):
        check_request({"observation_end": "2021-12-31", "realtime_end": "2022-01-01"}, alfred=True)
    with pytest.raises(CutoffError):
        check_request({"observation_end": "2022-01-31", "realtime_end": "2021-12-31"}, alfred=True)
    with pytest.raises(CutoffError):
        cd.alfred_params("BAA", end="2022-01-31")
    assert cd.alfred_params("BAA")["realtime_end"] == "2021-12-31"


def test_alfred_response_with_late_vintage_raises():
    obs = [{"date": "2021-10-01", "realtime_start": "2022-01-03", "realtime_end": "2022-01-03", "value": "3.0"}]
    with pytest.raises(CutoffError):
        cd.alfred_frame(obs, "BAA")


def test_realtime_values_rejects_late_rows():
    rows = pd.DataFrame({"obs": [pd.Timestamp("2021-10-31")], "realtime_start": [pd.Timestamp("2021-11-01")],
                         "realtime_end": [pd.Timestamp("2022-01-05")], "value": [3.0]})
    with pytest.raises(CutoffError):
        cl.realtime_values(rows, None, months("2021-11-30", "2021-12-31"))


# ── 4. 3칸 외 거부 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("cell", [("Fin", "M2"), ("RlEst", "M2"), ("Autos", "M6"), ("PerSv", "M6"), ("Chips", "M1")])
def test_other_cells_rejected(cell):
    with pytest.raises(cl.CellRejected):
        cl.cell_signal(cell[0], cell[1], synthetic_ind(), synthetic_spread())


def test_only_three_cells_registered():
    assert cl.CELLS == {1: ("Autos", "M2"), 2: ("RlEst", "M6"), 3: ("PerSv", "M5")}


# ── 5. 2012년 이전 신호 미승계 ────────────────────────────────────────────────


def test_event_signal_before_2013_not_inherited():
    idx = months("2012-01-31", "2021-12-31")
    sig = pd.Series(False, index=idx)
    sig[pd.Timestamp("2012-06-30")] = True
    sig[pd.Timestamp("2012-12-31")] = True
    w, eps = cl.cell_weights(sig, "M2")
    assert eps == [] and w.iloc[0] == 0.0 and (w == 0.0).all()


def test_state_on_before_2013_starts_new_in_2013_02():
    idx = months("2012-01-31", "2021-12-31")
    st = pd.Series(False, index=idx)
    st[pd.Timestamp("2012-06-30"):pd.Timestamp("2013-03-31")] = True
    w, eps = cl.cell_weights(st, "M6")
    assert w.iloc[0] == 0.0
    assert len(eps) == 1 and eps[0]["start"] == pd.Timestamp("2013-02-28")
    # 마지막으로 켜진 판정 달 2013-03 → 2013-04까지 보유, 꺼진 판정 달 2013-04 → 2013-05부터 복귀
    assert w[pd.Timestamp("2013-02-28")] == 0.05 and w[pd.Timestamp("2013-04-30")] == 0.15
    assert w[pd.Timestamp("2013-05-31")] == 0.0


def test_first_judge_2013_01_can_buy_2013_02():
    sig = pd.Series(False, index=months("2012-01-31", "2021-12-31"))
    sig[pd.Timestamp("2013-01-31")] = True
    w, eps = cl.cell_weights(sig, "M5")
    assert eps[0]["start"] == pd.Timestamp("2013-02-28") and eps[0]["months"] == 24
    assert list(w.iloc[1:6]) == [0.05, 0.10, 0.15, 0.20, 0.20]
    assert w[pd.Timestamp("2015-01-31")] == 0.20 and w[pd.Timestamp("2015-02-28")] == 0.0


# ── 6. 달 t 뒤 값 변경 시 신호 불변 ───────────────────────────────────────────


@pytest.mark.parametrize("no", [1, 2, 3])
def test_no_lookahead(no):
    t = pd.Timestamp("2017-06-30")
    ind, spread = synthetic_ind(), synthetic_spread()
    industry, method = cl.CELLS[no]
    full = cl.cell_signal(industry, method, ind, spread)
    ind2, spread2 = ind.copy(), spread.copy()
    ind2.loc[ind2.index > t] = ind2.loc[ind2.index > t] * -3.0
    spread2.loc[spread2.index > t] = 99.0
    cut = cl.cell_signal(industry, method, ind2, spread2)
    pd.testing.assert_series_equal(full.loc[:t], cut.loc[:t])
    trunc = cl.cell_signal(industry, method, ind.loc[:t], spread.loc[:t])
    pd.testing.assert_series_equal(full.loc[:t], trunc)


def test_realtime_values_uses_only_vintage_known_at_month_end():
    rows = pd.DataFrame({"obs": [pd.Timestamp("2020-01-31")] * 2,
                         "realtime_start": [pd.Timestamp("2020-02-03"), pd.Timestamp("2020-06-01")],
                         "realtime_end": [pd.Timestamp("2020-05-31"), CUTOFF], "value": [1.0, 2.0]})
    out = cl.realtime_values(rows, None, months("2020-03-31", "2020-03-31"))
    assert out["v2"].iloc[0] == 1.0 and out["level"].iloc[0] == 1.0
    assert cl.has_gap(pd.Series([np.nan, 1.0, 1.0])) is False
    assert cl.has_gap(pd.Series([1.0, np.nan, 1.0])) is True


# ── 7. Holm·부호 ──────────────────────────────────────────────────────────────


def test_holm_all_pass_and_order():
    h = cl.holm({1: 0.02, 2: 0.01, 3: 0.04})
    assert [h[2]["rank"], h[1]["rank"], h[3]["rank"]] == [1, 2, 3]
    assert all(h[c]["passed"] for c in h)
    assert h[2]["threshold"] == pytest.approx(0.05 / 3) and h[3]["threshold"] == pytest.approx(0.05)


def test_holm_stops_at_first_failure():
    h = cl.holm({1: 0.03, 2: 0.01, 3: 0.04})
    assert h[2]["passed"] and not h[1]["passed"] and not h[3]["passed"]  # 0.04 ≤ 0.05여도 불합격


def test_holm_first_fail_fails_all():
    h = cl.holm({1: 0.02, 2: 0.03, 3: 0.04})
    assert not any(h[c]["passed"] for c in h)


def test_mdd_sign():
    assert cl.mdd_ok(-0.30, -0.28)  # 2%p 깊음 → 통과
    assert cl.mdd_ok(-0.20, -0.28)  # 얕음 → 통과
    assert not cl.mdd_ok(-0.31, -0.28)  # 정확히 3%p 깊음 → 불합격
    assert not cl.mdd_ok(-0.40, -0.28)


def test_verdict_rules():
    assert cl.verdict({1: True, 2: False, 3: False})["result"] == "PASS"
    assert cl.verdict({1: False, 2: True, 3: True})["result"] == "PASS"
    only2 = cl.verdict({1: False, 2: True, 3: False})
    assert only2["result"] == "FAIL" and not only2["catalog"]
    none = cl.verdict({1: False, 2: False, 3: False})
    assert none["result"] == "FAIL" and none["catalog"]


def test_zero_entries_fails_cell():
    idx = cl.EVAL_MONTHS
    w = pd.Series(0.0, index=idx)
    r = cl.evaluate_cell(w, [], pd.Series(0.01, index=idx), pd.Series(0.02, index=idx), pd.Series(1000.0, index=cl.BOOK_MONTHS), 5)
    assert r["entries"] == 0 and r["p"] == 1.0
    assert not r["c1_beats_qqq"]  # 전략 = QQQ


# ── 8. 같은 seed 재현 ─────────────────────────────────────────────────────────


def full_inputs():
    daily = synthetic_daily()
    fixed, _ = cl.qqq_monthly_total(daily, daily)
    fx = pd.Series(1100.0 + 100.0 * np.sin(np.arange(len(cl.BOOK_MONTHS)) / 6.0), index=cl.BOOK_MONTHS)
    return synthetic_ind(), synthetic_spread(), fixed, fx


def test_same_seed_same_result():
    ind, spread, q, fx = full_inputs()
    a, b = cl.compute(ind, spread, q, fx, 30), cl.compute(ind, spread, q, fx, 30)
    for no in cl.CELLS:
        assert a["cells"][no]["p"] == b["cells"][no]["p"]
        assert a["cells"][no]["cagr"] == b["cells"][no]["cagr"]
    w1 = lab.random_weights([24, 24, 5], 108, 20, cl.SEED)
    w2 = lab.random_weights([24, 24, 5], 108, 20, cl.SEED)
    assert (w1 == w2).all()
    assert cl.SEED == 20261103


# ── 9. 세금 ───────────────────────────────────────────────────────────────────


def book_idx(end: str) -> pd.DatetimeIndex:
    return months("2012-12-31", end)


def test_tax_hold_only_liquidation():
    bm = book_idx("2013-12-31")
    T = len(bm) - 1
    q = np.zeros(T)
    q[0] = 0.10
    r = cl.run_book(np.zeros(T), q, np.zeros(T), np.full(T + 1, 1000.0), bm)
    krw = 110_000_000.0
    cost = krw * 0.001
    tax = capital_gains_tax(10_000_000.0 - cost)
    assert tax == pytest.approx((10_000_000.0 - cost - 2_500_000.0) * 0.22)
    assert r["end_after_tax_krw"] == pytest.approx(krw - cost - tax)
    assert r["costs_krw"] == 0.0 and r["taxes_paid_krw"] == 0.0


def test_tax_realized_paid_next_may_with_costs():
    bm = book_idx("2014-06-30")
    T = len(bm) - 1
    w = np.zeros(T)
    w[0] = 0.2
    s = np.zeros(T)
    s[0] = 0.5
    r = cl.run_book(w, np.zeros(T), s, np.full(T + 1, 1000.0), bm)
    nav0 = 100_000.0 - 0.002 * 0.2 * 100_000.0
    sat0 = 0.2 * nav0
    nav1 = 0.8 * nav0 + sat0 * 1.5
    cost1 = 0.002 * 0.2 * nav1
    gain2013 = sat0 * 0.5 * 1000.0 - cost1 * 1000.0
    tax = (gain2013 - 2_500_000.0) * 0.22
    nav_end = nav1 - cost1 - tax / 1000.0
    assert r["taxes_paid_krw"] == pytest.approx(tax)
    assert r["end_after_tax_krw"] == pytest.approx(nav_end * 1000.0 * 0.999)
    # 납부 전 달에는 곡선에서 밀린 세금을 뺀다
    i_apr = list(bm).index(pd.Timestamp("2014-04-30"))
    assert r["curve"][i_apr] == pytest.approx(nav1 * 1000.0 - cost1 * 1000.0 - tax - (nav1 - cost1) * 1000.0 * 0.001)


def test_tax_uses_krw_gain_from_fx():
    bm = book_idx("2013-02-28")
    fx = np.array([1000.0, 1000.0, 1200.0])
    r = cl.run_book(np.zeros(2), np.zeros(2), np.zeros(2), fx, bm)
    krw = 120_000_000.0
    cost = krw * 0.001
    assert r["end_after_tax_krw"] == pytest.approx(krw - cost - (20_000_000.0 - cost - 2_500_000.0) * 0.22)


def test_monthly_rebalance_gain_is_taxed():
    # 2013-11 시작, 위성 2배 → 2013-12 비중 유지 매도로 일부 실현(2013년분), 나머지는 2014-01 끝날 실현(2014년분)
    bm = months("2013-11-30", "2014-01-31")
    w = np.array([0.5, 0.5])
    r = cl.run_book(w, np.zeros(2), np.array([1.0, 0.0]), np.full(3, 1000.0), bm)
    tax_2013 = (12_487_500.0 - 100_000.0 - 2_500_000.0) * 0.22  # 비중 유지 실현 − 진입 비용
    sell_cost = 149_850_000.0 * 0.001
    tax_2014 = (37_462_500.0 - sell_cost - 2_500_000.0) * 0.22
    assert r["end_after_tax_krw"] == pytest.approx(149_850_000.0 - sell_cost - tax_2013 - tax_2014)


# ── 1단계 함수와 일치 ─────────────────────────────────────────────────────────


def test_same_as_stage1_on_pre2013():
    ind = synthetic_ind("2012-12-31")
    s = synthetic_spread("2012-12-31")
    pd.testing.assert_frame_equal(cl.m2_signal(s), lab.m2_signal(s))
    pd.testing.assert_frame_equal(cl.m5_signal(ind["PerSv"]), lab.m5_signal(ind["PerSv"]))
    pd.testing.assert_frame_equal(cl.m6_state(ind), lab.m6_state(ind))
    mo = months("2000-01-31", "2012-12-31")
    a, ae = cl.event_weights(cl.m2_signal(s)["sig"], mo)
    b, be = lab.event_weights(lab.m2_signal(s)["sig"], mo)
    pd.testing.assert_series_equal(a, b)
    assert ae == be
    st = lab.m6_state(ind)["RlEst"]
    c, ce = cl.state_weights(st, mo)
    d, de = lab.state_weights(st, mo)
    pd.testing.assert_series_equal(c, d)
    assert ce == de


def test_qqq_monthly_same_as_stage1():
    idx = pd.bdate_range("2012-01-02", "2012-12-31")
    rng = np.random.default_rng(3)
    ndx = pd.DataFrame({"Close": 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, len(idx))), "Dividends": 0.0}, index=idx)
    qq = ndx.copy()
    qq.iloc[50:, 0] *= 1.05  # 이후 가격이 모두 어긋나는 하루 오류 — 월 수익에서 상쇄되지 않는다
    qq.iloc[100:103, 1] = 0.3
    a, table = cl.qqq_monthly_total(qq, ndx)
    b, n = dm.qqq_monthly_total(qq, ndx)
    pd.testing.assert_series_equal(a, b)
    assert len(table) == n and n >= 1
    raw, table_raw = cl.qqq_monthly_total(qq, ndx, max_diff=None)
    assert len(table_raw) == n and not np.allclose(raw, a)


def test_month_end_fx_asof():
    d = pd.Series([1000.0, 1100.0], index=pd.to_datetime(["2012-12-28", "2013-01-15"]))
    out = cl.month_end_fx(d, months("2012-12-31", "2013-02-28"))
    assert list(out) == [1000.0, 1100.0, 1100.0]


# ── 합성 전체 실행 (태그 뒤 코드 오류 방지) ───────────────────────────────────


def alfred_rows_for(cache, series, base):
    obs = months("1990-01-31", "2021-10-31")
    rows = pd.DataFrame({"realtime_start": pd.Timestamp("1996-12-03"), "realtime_end": CUTOFF,
                         "value": base + np.sin(np.arange(len(obs)) / 5.0)}, index=pd.DatetimeIndex(obs, name="obs"))
    cd._write(rows, f"alfred_{series}", "합성", cache)


@pytest.fixture
def synthetic_cache(tmp_path):
    cache = tmp_path / "cache"
    raw = synthetic_ind() * 100.0
    cd._write(raw, "french_ind49", "합성", cache)
    alfred_rows_for(cache, "BAA", 6.0)
    alfred_rows_for(cache, "GS10", 4.0)
    fx_idx = pd.bdate_range("2012-12-03", "2021-12-31")
    cd._write(pd.Series(1100.0, index=pd.DatetimeIndex(fx_idx, name="date"), name="value").to_frame(), "fred_DEXKOUS", "합성", cache)
    daily = synthetic_daily()
    bumped = daily.copy()
    bumped.loc[pd.Timestamp("2016-03-15"), "Close"] *= 1.05  # 보정일 생김
    cd._write(bumped, "yf_QQQ", "합성", cache)
    cd._write(daily, "yf_NDX", "합성", cache)
    return cache


def test_end_to_end_synthetic(tmp_path, synthetic_cache):
    inputs = cr.load_inputs(synthetic_cache, allow_current_fetch=False)
    assert all(pd.Timestamp(v) <= CUTOFF for v in inputs["maxdates"].values())
    res = cr.run_all(inputs, n_random=20)
    assert len(res["corrections"]) >= 1
    out = tmp_path / "v2"
    order = {"tag_commit": "abcdef0123", "commit_time": "t", "pushed_at": "t", "first_received": "t"}
    cr.write_outputs(out, res, inputs, cd.manifest_rows(synthetic_cache), order, cd.compare_v1(inputs["ind"], tmp_path / "없음.csv"))
    for name in ("report.md", "cells.csv", "equity_curves.csv", "entries.csv", "data_manifest.csv"):
        assert (out / name).stat().st_size > 0
    cells = pd.read_csv(out / "cells.csv")
    assert len(cells) == 6 and set(cells["basis"]) == {"보정(판정)", "무보정(참고)"}
    assert set(zip(cells["industry"], cells["method"])) == set(cl.CELLS.values())
    eq = pd.read_csv(out / "equity_curves.csv")
    assert eq["month"].iloc[0] == "2012-12" and eq["month"].iloc[-1] == "2021-12" and len(eq) == 109
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "최종 판정" in report and "Holm" in report and "무보정" in report
    with pytest.raises(FileExistsError):
        cr.write_outputs(out, res, inputs, [], order, {})


def test_end_to_end_zero_entries_writes(tmp_path, synthetic_cache):
    inputs = cr.load_inputs(synthetic_cache, allow_current_fetch=False)
    inputs["spread"] = pd.Series(2.0, index=inputs["spread"].index)  # M2 신호 없음
    inputs["ind"]["PerSv"] = 0.001  # M5 신호 없음
    res = cr.run_all(inputs, n_random=5)
    assert res["main"]["cells"][1]["entries"] == 0 and not res["main"]["cells"][1]["passed"]
    assert res["main"]["cells"][3]["entries"] == 0 and not res["main"]["cells"][3]["passed"]
    out = tmp_path / "v2z"
    order = {"tag_commit": "abcdef0123", "commit_time": "t", "pushed_at": "t", "first_received": "t"}
    cr.write_outputs(out, res, inputs, cd.manifest_rows(synthetic_cache), order, {"compared": 0, "different": 0, "max_abs_pct": float("nan")})
    assert (out / "report.md").exists()
