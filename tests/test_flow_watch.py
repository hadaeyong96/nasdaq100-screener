"""F2 자금 흐름 관찰 테스트 (네트워크 없음, 합성 데이터). docs/design/flow_watch.md"""

from __future__ import annotations

import inspect
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from core import flow_watch as fw
from engine import daily
from notify import report_html
from scripts import f2_review
from store import db

ROOT = Path(__file__).resolve().parents[1]
S = fw.settings(None)


def _cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _frame(close, volume, end="2026-10-02"):
    idx = pd.bdate_range(end=end, periods=len(close))
    return pd.DataFrame({"close": np.asarray(close, dtype=float), "volume": np.asarray(volume, dtype=float)}, index=idx)


def _random_prices(n_tickers=8, n=120, seed=0, end="2026-10-02"):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n_tickers):
        c = 50 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
        v = rng.integers(1_000, 50_000, n) * (1 + (i % 3))
        out[f"T{i:02d}"] = _frame(c, v, end)
    return out


def _pair(recent_vol_a: float, rising=True, n=70):
    """A·B가 같은 가격 경로 → 점유율 = 거래량 비율. 최근 5일 A 거래량만 바꾼다."""
    price = [100.0] * (n - 5) + ([101.0, 102.0, 103.0, 104.0, 105.0] if rising else [99.0, 98.0, 97.0, 96.0, 95.0])
    va = [100.0] * (n - 5) + [recent_vol_a] * 5
    vb = [100.0] * n
    return {"A": _frame(price, va), "B": _frame(price, vb)}


# ── 상태 경계값 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("ratio,weekly,up,expected", [
    (1.49, 0.05, 0.9, fw.NORMAL), (1.50, 0.05, 0.9, fw.INFLOW),
    (1.50, 0.05, 0.5, fw.NORMAL), (1.50, 0.05, 0.5001, fw.INFLOW),
    (1.50, 0.0, 0.9, fw.NORMAL), (1.50, -0.0001, 0.9, fw.DUMP), (1.49, -0.10, 0.1, fw.NORMAL),
    (float("nan"), 0.05, 0.9, fw.NORMAL), (2.0, 0.05, float("nan"), fw.NORMAL),
])
def test_classify_boundaries(ratio, weekly, up, expected):
    assert fw.classify(ratio, weekly, up, S) == expected


def test_share_ratio_exact_boundary_end_to_end():
    on = fw.compute_flow(_pair(300.0), {}, "2026-10-02")  # 최근 점유율 0.75 / 직전 0.5 = 1.50
    assert on["metrics"].loc["A", "share_ratio"] == 1.5
    assert on["metrics"].loc["A", "up_volume_ratio"] == 1.0
    assert on["metrics"].loc["A", "status"] == fw.INFLOW
    assert on["metrics"].loc["A", "volume_ratio"] == 3.0
    assert on["metrics"].loc["A", "weekly_return"] == pytest.approx(0.05)
    off = fw.compute_flow(_pair(298.0), {}, "2026-10-02")
    assert off["metrics"].loc["A", "share_ratio"] < 1.5 and off["metrics"].loc["A", "status"] == fw.NORMAL
    dump = fw.compute_flow(_pair(300.0, rising=False), {}, "2026-10-02")
    assert dump["metrics"].loc["A", "status"] == fw.DUMP
    flat = _pair(300.0)
    for t in flat:
        flat[t].iloc[-6:, 0] = 100.0  # 주간 수익률 0
    assert fw.compute_flow(flat, {}, "2026-10-02")["metrics"].loc["A", "status"] == fw.NORMAL


def test_up_volume_ratio_and_52w_high():
    n = 70
    close = [100.0] * (n - 6) + [120.0, 121.0, 120.0, 122.0, 121.0, 123.0]  # 최근 5일: 오름·내림·오름·내림·오름
    vol = [100.0] * (n - 5) + [10.0, 20.0, 30.0, 40.0, 50.0]
    p = {"A": _frame(close, vol), "B": _frame([100.0] * n, [100.0] * n)}
    m = fw.compute_flow(p, {}, "2026-10-02")["metrics"]
    assert m.loc["A", "up_volume_ratio"] == pytest.approx((10 + 30 + 50) / 150)
    assert m.loc["A", "weekly_return"] == pytest.approx(123 / 120 - 1)
    assert m.loc["A", "high_52w"] == 0.0
    # 같은 날 종가(변화 0)는 오른 날이 아니다 → 상승일 비중 0.5 = 유입 아님
    close2 = [100.0] * (n - 6) + [100.0, 101.0, 101.0, 102.0, 102.0, 103.0]
    m2 = fw.compute_flow({"A": _frame(close2, [100.0] * (n - 5) + [50.0, 50.0, 50.0, 50.0, 50.0] ), "B": p["B"]}, {}, "2026-10-02")["metrics"]
    assert m2.loc["A", "up_volume_ratio"] == pytest.approx(0.6)


# ── 미래 데이터 방지 ───────────────────────────────────────────────────────


@pytest.mark.parametrize("cut", [70, 90, 110])
def test_truncated_equals_full(cut):
    prices = _random_prices()
    t = next(iter(prices.values())).index[cut]
    full = fw.compute_flow(prices, {"T00": "Technology"}, t)
    trunc = fw.compute_flow({k: v.loc[:t] for k, v in prices.items()}, {"T00": "Technology"}, t)
    pd.testing.assert_frame_equal(full["metrics"], trunc["metrics"])
    assert full["top"] == trunc["top"] and full["sectors"] == trunc["sectors"]


def test_no_negative_shift_in_sources():
    for mod in (fw, f2_review):
        assert not re.search(r"shift\(\s*-", inspect.getsource(mod)), mod.__name__


# ── 크기와 무관 ─────────────────────────────────────────────────────────


def test_share_ratio_independent_of_size():
    n = 70
    big = _frame(np.full(n, 1000.0), np.full(n, 1_000_000.0))  # 거래대금이 압도적으로 큼, 변화 없음
    small_vol = [1_000.0] * (n - 5) + [3_000.0] * 5
    small = _frame(np.linspace(10, 11, n), small_vol)
    mid = _frame(np.full(n, 100.0), np.full(n, 50_000.0))
    out = fw.compute_flow({"BIG": big, "SMALL": small, "MID": mid}, {}, "2026-10-02")
    assert out["top"][0]["ticker"] == "SMALL"
    assert out["metrics"].loc["BIG", "share_ratio"] < 1.0
    # 큰 종목(BIG)도 거래가 늘면 배율이 오른다 — 크기가 아니라 "평소 대비 변화"를 본다
    big2 = big.assign(volume=np.r_[np.full(n - 5, 1_000_000.0), np.full(5, 4_000_000.0)])
    out2 = fw.compute_flow({"BIG": big2, "SMALL": small, "MID": mid}, {}, "2026-10-02")
    assert out2["metrics"].loc["BIG", "share_ratio"] > 1.0
    prices = _random_prices()
    base = fw.compute_flow(prices, {}, "2026-10-02")["metrics"]
    # 같은 배수로 모든 종목 가격을 바꾸면 모든 배율이 정확히 같다
    allx = {k: v.assign(close=v["close"] * 7) for k, v in prices.items()}
    pd.testing.assert_series_equal(fw.compute_flow(allx, {}, "2026-10-02")["metrics"]["share_ratio"], base["share_ratio"])


# ── 제외·섹터 ────────────────────────────────────────────────────────────


def test_exclusions_and_etf_skip():
    prices = _random_prices(4)
    prices["SHORT"] = prices["T00"].iloc[-50:]  # 65거래일 미만
    gap = prices["T01"].copy()
    gap.iloc[-10, 1] = np.nan  # 거래량 결측
    prices["GAP"] = gap
    prices["QQQM"] = prices["T02"]
    out = fw.compute_flow(prices, {}, "2026-10-02")
    assert out["excluded"] == ["GAP", "SHORT"]
    assert "QQQM" not in out["metrics"].index and "QQQM" not in out["excluded"]
    assert len(out["metrics"]) == 4


def test_sector_table():
    prices = _pair(300.0)
    prices["C"] = _frame([100.0] * 70, [100.0] * 70)
    out = fw.compute_flow(prices, {"A": "Technology", "B": "Utilities"}, "2026-10-02")
    by = {r["sector"]: r for r in out["sectors"]}
    assert by["Technology"]["base_share_pct"] == pytest.approx(100 / 3)
    assert sum(r["recent_share_pct"] for r in out["sectors"]) == pytest.approx(100)
    assert by[fw.UNCLASSIFIED]["count"] == 1
    assert out["sectors"][0]["sector"] == "Technology"  # 변화 내림차순
    assert [r["change_pp"] for r in out["sectors"]] == sorted((r["change_pp"] for r in out["sectors"]), reverse=True)


def test_config_values_match_preregistration():
    cfg = _cfg()["flow_watch"]
    pre = yaml.safe_load((ROOT / "configs" / "f2_preregistration.yaml").read_text(encoding="utf-8"))
    for k, v in pre["config_yaml_keys"].items():
        assert cfg[k] == v
    assert {k: fw.DEFAULTS[k] for k in pre["config_yaml_keys"]} == pre["config_yaml_keys"]


# ── 주간 기록 ─────────────────────────────────────────────────────────────


def _week_prices(end):
    prices = _random_prices(6, n=150, end=end)
    prices["T00"] = prices["T00"].assign(volume=np.r_[np.full(145, 1000.0), np.full(5, 9000.0)],
                                         close=np.r_[np.full(145, 50.0), [51, 52, 53, 54, 55]])
    return prices


def _qqq(index):
    return pd.Series(np.linspace(400, 460, len(index)), index=index)


def test_week_end_detection_with_calendar():
    from data.market_calendar import trading_days_between

    def last(d):
        d = pd.Timestamp(d)
        nxt = trading_days_between((d + pd.Timedelta(days=1)).date(), (d + pd.Timedelta(days=15)).date())[0]
        return fw.is_last_trading_day_of_week(d, nxt)

    assert last("2026-10-02")          # 금요일
    assert not last("2026-10-01")      # 목요일
    assert last("2027-03-25")          # 성금요일(휴장) 전날 목요일
    assert last("2026-11-27")          # 추수감사절 다음 금요일(조기 폐장)
    assert not last("2026-11-25")      # 추수감사절 전날 수요일 → 금요일 거래 있음


def test_weekly_record_only_on_week_end_and_idempotent(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    cfg = _cfg()
    for as_of, expect in (("2026-10-01", 0), ("2026-10-02", None)):
        prices = _week_prices(as_of)
        flow = fw.compute_flow(prices, {}, as_of)
        qqq = _qqq(next(iter(prices.values())).index)
        res = daily.record_flow_watch(conn, flow, prices, cfg, as_of, [], qqq_fetch=lambda: qqq)
        if expect == 0:
            assert res["recorded"] == 0 and not res["is_week_end"]
        else:
            assert res["is_week_end"] and res["recorded"] > 0
    rows = db.load_flow_watch(conn)
    assert set(rows["base_date"]) == {"2026-10-02"}
    assert set(rows["list"]) >= {fw.INFLOW, fw.UNIVERSE_LIST}
    assert (rows[rows["list"] == fw.UNIVERSE_LIST]["rank"].isna()).all()
    inflow = rows[rows["list"] == fw.INFLOW]
    assert inflow.iloc[0]["ticker"] == "T00" and inflow.iloc[0]["rank"] == 1
    assert len(inflow) <= 10
    # 같은 기준일 재실행 → 중복 없음
    n_before = len(rows)
    res = daily.record_flow_watch(conn, flow, prices, cfg, "2026-10-02", [], qqq_fetch=lambda: qqq)
    assert res["recorded"] == 0 and len(db.load_flow_watch(conn)) == n_before


def test_qqq_failure_is_reported_not_silent(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    prices = _week_prices("2026-10-02")
    flow = fw.compute_flow(prices, {}, "2026-10-02")
    warnings: list[str] = []

    def boom():
        raise RuntimeError("network down")

    res = daily.record_flow_watch(conn, flow, prices, _cfg(), "2026-10-02", warnings, qqq_fetch=boom)
    assert res["recorded"] == 0 and len(warnings) == 1 and "QQQ" in warnings[0]


def test_fill_5d_then_20d(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    full_end = "2026-11-06"
    prices_full = _week_prices(full_end)
    idx = next(iter(prices_full.values())).index
    base = pd.Timestamp("2026-10-02")
    qqq_full = _qqq(idx)
    cut = {t: df.loc[:base] for t, df in prices_full.items()}
    flow = fw.compute_flow(cut, {}, base)
    daily.record_flow_watch(conn, flow, cut, _cfg(), base, [], qqq_fetch=lambda: qqq_full.loc[:base])
    bpos = idx.get_loc(base)
    # 4거래일 뒤: 아무것도 못 채움
    d4 = idx[bpos + 4]
    r = daily.record_flow_watch(conn, flow, {t: df.loc[:d4] for t, df in prices_full.items()}, _cfg(), d4, [], qqq_fetch=lambda: qqq_full.loc[:d4])
    assert r["filled"] == 0
    # 5거래일 뒤: 5일만
    d5 = idx[bpos + 5]
    daily.record_flow_watch(conn, flow, {t: df.loc[:d5] for t, df in prices_full.items()}, _cfg(), d5, [], qqq_fetch=lambda: qqq_full.loc[:d5])
    rows = db.load_flow_watch(conn)
    assert rows["filled_5d_at"].notna().all() and rows["filled_20d_at"].isna().all()
    t00 = rows[(rows["ticker"] == "T00") & (rows["list"] == fw.UNIVERSE_LIST)].iloc[0]
    c = prices_full["T00"]["close"]
    exp_r = c.iloc[bpos + 5] / c.iloc[bpos] - 1
    exp_q = qqq_full.iloc[bpos + 5] / qqq_full.iloc[bpos] - 1
    assert t00["ret_5d"] == pytest.approx(exp_r) and t00["qqq_ret_5d"] == pytest.approx(exp_q)
    assert t00["excess_5d"] == pytest.approx(exp_r - exp_q)
    # 20거래일 뒤: 20일도 채움, 5일 값은 그대로
    d20 = idx[bpos + 20]
    daily.record_flow_watch(conn, flow, {t: df.loc[:d20] for t, df in prices_full.items()}, _cfg(), d20, [], qqq_fetch=lambda: qqq_full.loc[:d20])
    rows2 = db.load_flow_watch(conn)
    assert rows2["filled_20d_at"].notna().all()
    pd.testing.assert_series_equal(rows["ret_5d"], rows2["ret_5d"])
    t00b = rows2[(rows2["ticker"] == "T00") & (rows2["list"] == fw.UNIVERSE_LIST)].iloc[0]
    assert t00b["ret_20d"] == pytest.approx(c.iloc[bpos + 20] / c.iloc[bpos] - 1)


# ── 보고서 표시 ─────────────────────────────────────────────────────────


def _flow_summary(mode="paper"):
    prices = _week_prices("2026-10-02")
    states = {"T00": {"state": "정찰"}}
    buy_groups = {"b1": [], "b2": [{"ticker": "T00"}], "b3": [], "b9": []}
    view = daily.build_flow_watch_view(prices, pd.Timestamp("2026-10-02"), _cfg(), {"T00": "가나다"}, buy_groups, states, {"T00": "Technology"})
    summary = daily._empty_run_summary(mode, pd.Timestamp("2026-10-02").date(), [])
    summary["as_of"] = pd.Timestamp("2026-10-02")
    summary["flow_watch"] = view
    return summary, view


def test_our_signal_text():
    _, view = _flow_summary()
    t00 = next(r for r in view["top"] if r["ticker"] == "T00")
    assert t00["our_signal"] == "오늘 2차 신호 · 1차 보유"
    assert any(r["our_signal"] == "없음" for r in view["top"])


@pytest.mark.parametrize("mode", ["live", "paper"])
def test_report_renders_flow_tables_below_screening(tmp_path, mode):
    summary, _ = _flow_summary(mode)
    html = report_html.render_report(summary, _cfg(), tmp_path).read_text(encoding="utf-8")
    assert "돈이 몰리는 종목 TOP 10" in html and "섹터별 자금 흐름" in html
    assert "참고용 · 매수 신호 아님 · 검증 중(2026-10 시작, 판정 2027-10)" in html
    assert html.index('aria-label="오늘의 스크리닝"') < html.index('aria-label="돈이 몰리는 종목"')
    assert "기술" in html and "제외 0종목" in html


def test_public_report_has_no_flow_table(tmp_path):
    summary, _ = _flow_summary("live")
    html = report_html.render_public_report(summary, _cfg(), tmp_path).read_text(encoding="utf-8")
    assert "돈이 몰리는 종목" not in html


# ── 판정 스크립트 ─────────────────────────────────────────────────────────


def test_review_hash_check_passes_and_detects_change(tmp_path):
    f2_review.verify_preregistration()
    for rel in f2_review.PREREG_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, tmp_path / rel)
    f2_review.verify_preregistration(tmp_path)
    p = tmp_path / "configs" / "f2_preregistration.yaml"
    p.write_text(p.read_text(encoding="utf-8") + "\n# 바꿈\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        f2_review.verify_preregistration(tmp_path)


def _records(inflow_excess, dump_excess, univ_excess, dates=("2026-10-09", "2026-10-16")):
    rows = []
    for d in dates:
        for i, x in enumerate(inflow_excess):
            rows.append({"base_date": d, "list": "유입", "ticker": f"I{i}", "excess_20d": x, "filled_20d_at": "2026-11-20", "filled_5d_at": "x"})
        for i, x in enumerate(dump_excess):
            rows.append({"base_date": d, "list": "투매", "ticker": f"D{i}", "excess_20d": x, "filled_20d_at": "2026-11-20", "filled_5d_at": "x"})
        for i, x in enumerate(univ_excess):
            rows.append({"base_date": d, "list": "대상", "ticker": f"U{i:02d}", "excess_20d": x, "filled_20d_at": "2026-11-20", "filled_5d_at": "x"})
    return pd.DataFrame(rows)


def test_review_judge_and_seed_reproducible():
    pre = f2_review.verify_preregistration()
    univ = list(np.linspace(-0.05, 0.05, 30))
    good = f2_review.judge(_records([0.06, 0.04], [-0.03], univ), pre)
    assert good["pass"] and good["inflow_n"] == 4 and good["universe_n"] == 60
    again = f2_review.judge(_records([0.06, 0.04], [-0.03], univ), pre)
    assert again["control_p75"] == good["control_p75"]
    bad = f2_review.judge(_records([0.001], [0.01], univ), pre)
    assert not bad["pass"]
    assert not f2_review.judge(_records([], [-0.01], univ), pre)["pass"]
    out_of_range = _records([0.06], [-0.03], univ, dates=("2027-10-08",))
    assert f2_review.judge(out_of_range, pre)["inflow_n"] == 0


def test_review_before_verdict_date_shows_no_returns(tmp_path, capsys):
    conn = db.connect(tmp_path / "state.db")
    conn.close()
    res = f2_review.main(["--db", str(tmp_path / "state.db"), "--today", "2027-09-30"])
    assert res["mode"] == "status"
    assert "초과수익" not in capsys.readouterr().out
