"""core/screening_view.py + 매수 "왜?" 4단 설명 테스트 (네트워크 없음).

- 깔때기 숫자 = engine.daily._compute_funnel 숫자 (같은 입력)
- 2차 표의 종목 ⊂ 정찰 상태 종목, 안 산 1차 종목은 2차 표에 없음
- 차수별 ✓/✗가 값에 따라 바뀌는지, A3 "종가" 자리에 지정가가 아닌 종가가 나오는지(회귀)
- 미래 데이터 방지: t일까지 자른 지표로 만든 결과 = 전체 지표로 만든 t일 결과
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest
import yaml

from core import explain as expl
from core import screening_view as sv
from core import state as st
from engine.daily import _compute_funnel, simulate_since
from tests import screening_sample as sample
from tests._synthetic_universe import synthetic_indicator_map
from tests.test_equivalence_main import ROOT


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def universe(cfg):
    return synthetic_indicator_map(cfg)


@pytest.fixture(scope="module")
def sample_summary():
    return sample.sample_summary()


def _run_until(cfg, im, upto: int):
    any_df = next(iter(im.values()))
    days = list(any_df.index[-150:])[: upto + 1]
    states = {t: st.init_state(t, t) for t in im}
    fx = {d.date().isoformat(): 1400.0 for d in any_df.index}
    sim = simulate_since(
        im, {t: days for t in im}, states, cfg, {t: None for t in im}, {t: set() for t in im}, pd.DataFrame(),
        virtual_fill=True, max_concurrent=cfg["risk"]["max_concurrent_positions"], fx_rate_by_date=fx,
    )
    return sim, days[-1]


def _assert_matches_funnel(view: dict, funnel: dict) -> None:
    check = view["funnel_check"]
    lanes = view["lanes"]
    assert lanes["a1"]["steps"][1]["count"] == funnel["1차 RSI 30 돌파"]
    assert check["gc_total"] == funnel["2차 MACD 골든크로스"]
    assert check["gc_a2"] + check["gc_b"] + check["gc_other"] == funnel["2차 MACD 골든크로스"]
    assert lanes["a3"]["steps"][1]["count"] == funnel["3차 일목구름 4요소"]
    assert check["ban_total"] == funnel["4차 매매금지 필터"]
    assert check["limit_total"] == funnel["5차 보유한도"]


# ── 깔때기 숫자 = _compute_funnel ──────────────────────────────────────────


def test_sample_funnel_numbers_match_compute_funnel(sample_summary):
    sim = sample_summary["_sim"]
    funnel = _compute_funnel(sim["today_events"], sample.sample_inputs()["im"], sim["as_of_by_ticker"])
    assert funnel == sample_summary["funnel"]
    _assert_matches_funnel(sample_summary["screening"], funnel)
    # 견본은 네 레인 모두 추천·제외(또는 대기)가 섞여 있어야 한다
    for key in ("a1", "a2", "a3", "b"):
        results = {r["result"] for r in sample_summary["screening"]["lanes"][key]["rows"]}
        assert "추천" in results and (results - {"추천"}), key


@pytest.mark.parametrize("upto", [20, 45, 70, 95, 120, 149])
def test_synthetic_funnel_numbers_match_compute_funnel(cfg, universe, upto):
    sim, _ = _run_until(cfg, universe, upto)
    view = sv.build_screening_view(universe, sim["as_of_by_ticker"], sim["states"], sim["today_events"], cfg)
    _assert_matches_funnel(view, _compute_funnel(sim["today_events"], universe, sim["as_of_by_ticker"]))
    # 추천 줄 수 = 오늘 매수 이벤트 수 (차수별)
    kinds = [e["kind"] for e in sim["today_events"]]
    for key, kind in (("a1", "A1"), ("a2", "A2"), ("a3", "A3"), ("b", "B")):
        assert sum(r["result"] == "추천" for r in view["lanes"][key]["rows"]) == kinds.count(kind)


def test_buy_row_funnel_sentence_uses_same_numbers_as_lane(sample_summary):
    lanes = sample_summary["screening"]["lanes"]
    rop = next(r for r in sample_summary["buy_groups"]["b1"] if r["ticker"] == "ROP")
    why = rop["explain"]["sections"][1]["text"]
    s = lanes["a1"]["steps"]
    assert f"오늘 {s[0]['count']}종목 → RSI 30 탈출 {s[1]['count']} → 검사 통과 {s[2]['count']} → 점수 1위" in why


# ── 2차 후보 = 정찰 상태 종목만 ─────────────────────────────────────────────


def test_a2_rows_are_subset_of_scouting_positions(sample_summary):
    sim = sample_summary["_sim"]
    pre_states = sample.sample_inputs()["states"]
    scouting = {t for t, s in pre_states.items() if s["state"] == "정찰"}
    a2_tickers = {r["ticker"] for r in sample_summary["screening"]["lanes"]["a2"]["rows"]}
    assert a2_tickers and a2_tickers <= scouting
    # 오늘 1차 추천(안 산 1차)과 최근 1차 신호가 났지만 사지 않은 종목은 2차 표에 없다
    today_a1 = {e["ticker"] for e in sim["today_events"] if e["kind"] == "A1"}
    not_bought = {x["ticker"] for x in sample_summary["screening"]["lanes"]["a2"]["not_bought"]}
    assert today_a1 and not (a2_tickers & today_a1)
    assert {"ZS", "TEAM"} <= not_bought and not (a2_tickers & not_bought)


def test_a2_rows_subset_of_scouting_on_synthetic_days(cfg, universe):
    for upto in (40, 80, 120, 149):
        sim, as_of = _run_until(cfg, universe, upto)
        view = sv.build_screening_view(universe, sim["as_of_by_ticker"], sim["states"], sim["today_events"], cfg,
                                       recent_events=sim["all_events"])
        for r in view["lanes"]["a2"]["rows"]:
            ev = [e for e in sim["today_events"] if e["ticker"] == r["ticker"]]
            assert sv.entry_state(sim["states"][r["ticker"]], ev, as_of) == "정찰"
        a1_today = {e["ticker"] for e in sim["today_events"] if e["kind"] == "A1"}
        assert not (a1_today & {r["ticker"] for r in view["lanes"]["a2"]["rows"]})


def test_a2_waiting_rows_fold_after_ten(cfg):
    df = sample._scout(100, 95, gc=False, rsi=40, norm=-1.5)
    im, states, as_of = {}, {}, sample.AS_OF
    for i in range(13):
        t = f"W{i:02d}"
        im[t] = df
        states[t] = {**st.init_state(t), "state": "정찰", "units": {"1": 1}, "a1_date": sample._bd(i % 9), "stop": 1.0}
    view = sv.build_screening_view(im, {t: as_of for t in im}, states, [], cfg)
    a2 = view["lanes"]["a2"]
    assert len(a2["rows"]) == 13 and a2["folded_count"] == 3
    assert all(r["result"] == "대기" for r in a2["rows"])


# ── ✓/✗는 값 비교로 정한다 ────────────────────────────────────────────────


def _levels(out, section=0):
    return [c["level"] for c in out["sections"][section]["items"]]


def test_a1_check_flips_with_rsi_values(cfg):
    ok = expl.explain_buy("A1", {"kr": "X", "rsi_prev": 28.0, "rsi_now": 31.0, "limit": 10.0, "stop": 9.0}, cfg)
    bad = expl.explain_buy("A1", {"kr": "X", "rsi_prev": 31.0, "rsi_now": 33.0, "limit": 10.0, "stop": 9.0}, cfg)
    unk = expl.explain_buy("A1", {"kr": "X", "rsi_prev": None, "rsi_now": 33.0, "limit": 10.0, "stop": 9.0}, cfg)
    assert _levels(ok)[0] == "y" and _levels(bad)[0] == "n"
    assert _levels(unk)[0] == "i" and "확인 불가" in unk["sections"][0]["items"][0]["text"]


def _facts(**kw):
    base = {"close": 100.0, "cloud_top": 95.0, "span_a": 98.0, "span_b": 96.0, "high_d_ago": 97.0,
            "macd_prev": -0.2, "signal_prev": -0.1, "macd": 0.1, "signal": 0.0, "macd_norm": 0.1,
            "rsi_now": 55.0, "cross_count_20d": 1, "earnings_within": False, "earnings_date": pd.Timestamp("2026-11-01")}
    base.update(kw)
    return base


@pytest.mark.parametrize(
    "override, index",
    [
        ({"close": 90.0}, 0),  # 종가 < 구름 상단
        ({"span_a": 90.0}, 1),  # 앞구름 음운
        ({"high_d_ago": 105.0}, 2),  # 후행스팬 미돌파
        ({"macd": -0.15}, 3),  # 골든크로스 아님
        ({"macd_norm": -0.9}, 4),  # 정규화 MACD 미달
        ({"rsi_now": 72.0}, 5),  # RSI 50~70 밖
        ({"cross_count_20d": 5}, 6),  # 휩소
        ({"earnings_within": True}, 7),  # 실적 임박
    ],
)
def test_b_checks_flip_with_values(cfg, override, index):
    ok = expl.explain_buy("B", {"kr": "X", "facts": _facts(), "limit": 101.0, "stop": 95.0}, cfg)
    assert _levels(ok)[:8] == ["y"] * 8
    bad = expl.explain_buy("B", {"kr": "X", "facts": _facts(**override), "limit": 101.0, "stop": 95.0}, cfg)
    assert _levels(bad)[index] == "n"
    # 예전처럼 고정 "y"가 아니라 checks(요약 칩)도 같은 결과를 쓴다
    assert bad["checks"][index]["level"] == "n"


def test_b_checks_unknown_without_values(cfg):
    out = expl.explain_buy("B", {"kr": "X", "limit": 101.0, "stop": 95.0}, cfg)
    assert all(lv == "i" for lv in _levels(out)[:7])
    assert all("확인 불가" in c["text"] for c in out["sections"][0]["items"][:7])


def test_a2_golden_cross_and_rsi_checks_use_values(cfg):
    ok = expl.explain_buy("A2", {"kr": "X", "facts": _facts(rsi_now=45.0, elapsed=4, a1_date=pd.Timestamp("2026-09-28")),
                                  "limit": 101.0, "stop": 95.0}, cfg)
    lv = _levels(ok)
    assert lv[:6] == ["y"] * 6
    no_gc = expl.explain_buy("A2", {"kr": "X", "facts": _facts(macd=-0.15, rsi_now=45.0, elapsed=4), "limit": 101.0, "stop": 95.0}, cfg)
    assert _levels(no_gc)[1] == "n"
    hot = expl.explain_buy("A2", {"kr": "X", "facts": _facts(rsi_now=71.0, elapsed=4), "limit": 101.0, "stop": 95.0}, cfg)
    assert _levels(hot)[2] == "n" and _levels(hot)[3] == "n"
    late = expl.explain_buy("A2", {"kr": "X", "facts": _facts(rsi_now=45.0, elapsed=11), "limit": 101.0, "stop": 95.0}, cfg)
    assert _levels(late)[0] == "n"


def test_a3_shows_close_not_limit_price(cfg):
    """회귀: A3 "종가" 칸에 지정가(ctx["limit"])가 아니라 실제 종가가 나온다."""
    facts = _facts(pre_state="확인", units={"1": 1, "2": 2}, gap_pct=1.0, cloud_bot=90.0)
    out = expl.explain_buy("A3", {"kr": "X", "facts": facts, "close": 100.0, "limit": 101.0, "stop": 95.0}, cfg)
    cloud_text = out["sections"][0]["items"][1]["text"]
    assert "종가 $100.00" in cloud_text and "$101.00" not in cloud_text
    assert "종가 $100.00" in out["checks"][1]["text"]
    gap_bad = expl.explain_buy("A3", {"kr": "X", "facts": {**facts, "gap_pct": 4.5}, "limit": 101.0, "stop": 95.0}, cfg)
    assert _levels(gap_bad)[5] == "n"


def test_score_breakdown_matches_score_for_every_sample_buy(sample_summary):
    import re

    for rows in sample_summary["buy_groups"].values():
        for r in rows:
            text = r["explain"]["sections"][1]["items"][0]["text"]
            parts = [int(x) for x in re.findall(r"\(\+(\d+)", text)]
            assert sum(parts) == r["score"], (r["ticker"], text)


def test_buy_explain_has_four_sections_and_source(sample_summary):
    for rows in sample_summary["buy_groups"].values():
        for r in rows:
            e = r["explain"]
            assert [s["title"] for s in e["sections"]] == ["① 우리 규칙", "② 왜 이 종목인가", "③ 어떻게 사나", "④ 다음 단계"]
            assert e["source"].startswith("근거: 전략 v3 4장")
            assert {"title", "badge", "checks", "body", "next"} <= set(e)


# ── 미래 데이터 방지 ───────────────────────────────────────────────────────


def _truncate(im: dict, t) -> dict:
    return {k: df.loc[:t] for k, df in im.items()}


@pytest.mark.parametrize("upto", [60, 110])
def test_screening_view_no_lookahead(cfg, universe, upto):
    sim, t = _run_until(cfg, universe, upto)
    any_df = next(iter(universe.values()))
    future = [d for d in any_df.index if d > t][:30]
    args = (sim["as_of_by_ticker"], sim["states"], sim["today_events"], cfg)
    kw = {"recent_events": sim["all_events"], "future_trading_days": future}
    full = sv.build_screening_view(universe, *args, **kw)
    cut = sv.build_screening_view(_truncate(universe, t), *args, **kw)
    assert full == cut


@pytest.mark.parametrize("stage", ["A1", "A2", "A3", "B"])
def test_buy_facts_no_lookahead(cfg, universe, stage):
    df = universe["T05"]
    t = df.index[200]
    future = list(df.index[201:240])
    state_ = {**st.init_state("T05"), "state": "정찰", "a1_date": df.index[195], "units": {"1": 3}}
    full = sv.buy_facts(df, t, stage, state_, cfg, future_trading_days=future)
    cut = sv.buy_facts(df.loc[:t], t, stage, state_, cfg, future_trading_days=future)
    assert full == cut


def test_a2_deadline_is_last_valid_day(cfg):
    """2차 기한 = A1 당일 포함 10번째 거래일 — core/state.py가 A2를 판정하는 마지막 날."""
    idx = pd.bdate_range("2026-09-01", periods=30)
    a1 = idx[3]
    deadline = sv.a2_deadline(idx, a1, cfg, idx[-1])
    assert idx.get_loc(deadline) - idx.get_loc(a1) == cfg["assumptions"]["a1_to_a2_expiry_days"] - 1


def test_screening_section_renders_in_report(sample_summary):
    from notify import report_html

    cfg_ = sample.sample_inputs()["cfg"]
    html = report_html._env.get_template("report.html.j2").render(**report_html.build_context(sample_summary, cfg_))
    assert "오늘의 스크리닝" in html and "바닥 반전 3단 확인 매수법" in html
    for title in ("1차 정찰 · RSI 30 탈출", "2차 확인", "3차 확정", "재진입"):
        assert title in html
    assert "참고: 2차 대상 아님" in html and "① 우리 규칙" in html


def test_report_without_screening_still_renders(cfg):
    """데이터 지연 등으로 screening이 없으면 전략 카드만 나오고 깔때기는 생략된다."""
    from engine.daily import _empty_run_summary
    from notify import report_html

    summary = _empty_run_summary("live", pd.Timestamp("2026-10-05"), [])
    html = report_html._env.get_template("report.html.j2").render(**report_html.build_context(summary, copy.deepcopy(cfg)))
    assert "바닥 반전 3단 확인 매수법" in html and "오늘의 깔때기" not in html
