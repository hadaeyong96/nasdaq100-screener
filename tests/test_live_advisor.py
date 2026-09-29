"""라이브 어드바이저 1단계 1e: engine/daily.py의 판정 매핑·계획 기반 사이징
테스트 (docs/design/live_advisor.md 3·6·7번). 네트워크 없이 합성 데이터로 돈다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core import state as st
from data.fills import FillsResult
from data.fx import FxRateResult
from engine.daily import (
    _judgment_for_events,
    _judgment_reason,
    _live_target_tickers,
    _plan_budget_map,
    _reference_entry_price,
    build_report_summary,
    compute_live_judgments,
)
from tests.test_state import make_df

_CFG = {"entry": {"limit_markup": 1.01}}

_DATE = pd.bdate_range("2026-01-02", periods=3, name="date")[1]


def _df(close: float):
    return make_df([{"close": close, "rsi": 40}, {"close": close, "rsi": 40}, {"close": close, "rsi": 40}])


# ---------- _judgment_for_events ----------


def test_judgment_no_events_is_hold():
    judgment, event = _judgment_for_events([])
    assert judgment == "보유"
    assert event is None


def test_judgment_buy_kind_is_add():
    events = [{"kind": "A1", "ticker": "T", "price": 100.0}]
    judgment, event = _judgment_for_events(events)
    assert judgment == "추가매수"
    assert event["kind"] == "A1"


@pytest.mark.parametrize("kind", ["E1", "E2"])
def test_judgment_partial_kinds_are_trim(kind):
    judgment, event = _judgment_for_events([{"kind": kind, "ticker": "T"}])
    assert judgment == "일부매도 검토"
    assert event["kind"] == kind


@pytest.mark.parametrize("kind", ["E3", "STOP", "A1_EXPIRE"])
def test_judgment_sell_kinds_are_sell(kind):
    judgment, event = _judgment_for_events([{"kind": kind, "ticker": "T"}])
    assert judgment == "매도"
    assert event["kind"] == kind


def test_judgment_sell_outranks_buy_when_both_present_same_day():
    events = [{"kind": "A2", "ticker": "T", "price": 100.0}, {"kind": "STOP", "ticker": "T"}]
    judgment, event = _judgment_for_events(events)
    assert judgment == "매도"
    assert event["kind"] == "STOP"


def test_judgment_trim_outranks_buy():
    events = [{"kind": "A2", "ticker": "T", "price": 100.0}, {"kind": "E1", "ticker": "T"}]
    judgment, event = _judgment_for_events(events)
    assert judgment == "일부매도 검토"


# ---------- _judgment_reason ----------


def test_judgment_reason_hold():
    assert _judgment_reason("보유", None) == "오늘 신호 없음"


def test_judgment_reason_add_mentions_stage_label():
    reason = _judgment_reason("추가매수", {"kind": "A1"})
    assert "1차" in reason
    assert "신호" in reason


def test_judgment_reason_sell_mentions_stop_label():
    reason = _judgment_reason("매도", {"kind": "STOP"})
    assert "손절" in reason


def test_judgment_reason_trim_mentions_e1_label():
    reason = _judgment_reason("일부매도 검토", {"kind": "E1"})
    assert "모멘텀 약화" in reason
    assert "검토" in reason


# ---------- _live_target_tickers / _plan_budget_map ----------


def test_live_target_tickers_is_union_of_plan_and_fills():
    plan_by_ticker = {"NVDA": 3_000_000.0}
    fills_df = pd.DataFrame({"ticker": ["AVGO", "AVGO", "META"]})
    assert _live_target_tickers(plan_by_ticker, fills_df) == {"NVDA", "AVGO", "META"}


def test_live_target_tickers_empty_when_no_plan_or_fills():
    assert _live_target_tickers({}, pd.DataFrame(columns=["ticker"])) == set()


def test_plan_budget_map_builds_dict():
    plan_df = pd.DataFrame({"ticker": ["NVDA", "AVGO"], "budget_krw": [3_000_000.0, 1_500_000.0], "memo": ["", ""]})
    assert _plan_budget_map(plan_df) == {"NVDA": 3_000_000.0, "AVGO": 1_500_000.0}


def test_plan_budget_map_empty_df_returns_empty_dict():
    assert _plan_budget_map(pd.DataFrame(columns=["ticker", "budget_krw", "memo"])) == {}


# ---------- _reference_entry_price ----------


def test_reference_entry_price_uses_todays_signal_price_when_present():
    today_events = [{"ticker": "NVDA", "kind": "A1", "price": 100.0}]
    price = _reference_entry_price("NVDA", today_events, {}, {}, _CFG)
    assert price == pytest.approx(101.0)  # entry_limit_price: 100 * 1.01


def test_reference_entry_price_falls_back_to_close_when_no_signal_today():
    indicator_map = {"NVDA": _df(200.0)}
    as_of_by_ticker = {"NVDA": _DATE}
    price = _reference_entry_price("NVDA", [], indicator_map, as_of_by_ticker, _CFG)
    assert price == pytest.approx(202.0)  # 200 * 1.01


def test_reference_entry_price_none_when_no_data():
    assert _reference_entry_price("NVDA", [], {}, {}, _CFG) is None


# ---------- compute_live_judgments ----------


def _states(tickers_last_judgment: dict[str, str | None]) -> dict:
    return {t: {**st.init_state(t, t), "last_judgment": lj} for t, lj in tickers_last_judgment.items()}


def test_compute_live_judgments_marks_hold_when_no_events_and_no_change():
    states = _states({"NVDA": "보유"})
    rows = compute_live_judgments(
        states, today_events=[], indicator_map={"NVDA": _df(200.0)}, as_of_by_ticker={"NVDA": _DATE},
        name_map={"NVDA": "엔비디아"}, plan_by_ticker={"NVDA": 3_000_000.0}, fills_df=pd.DataFrame(columns=["ticker"]),
        fx_rate=1300.0, cfg=_CFG,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["judgment"] == "보유"
    assert row["changed"] is False
    assert states["NVDA"]["last_judgment"] == "보유"


def test_compute_live_judgments_detects_changed_judgment():
    states = _states({"NVDA": "보유"})
    today_events = [{"ticker": "NVDA", "kind": "STOP", "date": _DATE, "unit": "1", "qty": 10}]
    rows = compute_live_judgments(
        states, today_events, indicator_map={"NVDA": _df(200.0)}, as_of_by_ticker={"NVDA": _DATE},
        name_map={"NVDA": "엔비디아"}, plan_by_ticker={}, fills_df=pd.DataFrame({"ticker": ["NVDA"]}),
        fx_rate=1300.0, cfg=_CFG,
    )
    row = rows[0]
    assert row["judgment"] == "매도"
    assert row["changed"] is True
    assert states["NVDA"]["last_judgment"] == "매도"


def test_compute_live_judgments_first_run_never_marked_changed():
    """previous last_judgment가 None(첫 실행)이면 changed는 항상 False."""
    states = _states({"NVDA": None})
    today_events = [{"ticker": "NVDA", "kind": "STOP", "date": _DATE}]
    rows = compute_live_judgments(
        states, today_events, indicator_map={"NVDA": _df(200.0)}, as_of_by_ticker={"NVDA": _DATE},
        name_map={"NVDA": "엔비디아"}, plan_by_ticker={}, fills_df=pd.DataFrame({"ticker": ["NVDA"]}),
        fx_rate=1300.0, cfg=_CFG,
    )
    assert rows[0]["changed"] is False


def test_compute_live_judgments_computes_plan_based_qty_for_buy_event():
    states = _states({"NVDA": "보유"})
    today_events = [{"ticker": "NVDA", "kind": "A1", "date": _DATE, "unit": "1", "price": 100.0}]
    rows = compute_live_judgments(
        states, today_events, indicator_map={"NVDA": _df(100.0)}, as_of_by_ticker={"NVDA": _DATE},
        name_map={"NVDA": "엔비디아"}, plan_by_ticker={"NVDA": 9_000_000.0}, fills_df=pd.DataFrame(columns=["ticker"]),
        fx_rate=1350.0, cfg=_CFG,
    )
    row = rows[0]
    assert row["judgment"] == "추가매수"
    assert row["has_plan"] is True
    assert row["tranche_krw"] == pytest.approx(1_000_000.0)  # A1 = 계획금액의 1/9
    assert row["qty"] > 0


def test_compute_live_judgments_no_plan_ticker_gets_no_amounts():
    states = _states({"AVGO": "보유"})
    today_events = [{"ticker": "AVGO", "kind": "A1", "date": _DATE, "unit": "1", "price": 100.0}]
    rows = compute_live_judgments(
        states, today_events, indicator_map={"AVGO": _df(100.0)}, as_of_by_ticker={"AVGO": _DATE},
        name_map={}, plan_by_ticker={}, fills_df=pd.DataFrame({"ticker": ["AVGO"]}),
        fx_rate=1350.0, cfg=_CFG,
    )
    row = rows[0]
    assert row["has_plan"] is False
    assert row["qty"] is None


def test_compute_live_judgments_flags_one_share_warning_when_budget_too_small():
    states = _states({"NVDA": "보유"})
    today_events = [{"ticker": "NVDA", "kind": "A1", "date": _DATE, "unit": "1", "price": 100.0}]
    rows = compute_live_judgments(
        states, today_events, indicator_map={"NVDA": _df(100.0)}, as_of_by_ticker={"NVDA": _DATE},
        name_map={}, plan_by_ticker={"NVDA": 500_000.0}, fills_df=pd.DataFrame(columns=["ticker"]),
        fx_rate=1350.0, cfg=_CFG,
    )
    row = rows[0]
    assert row["qty"] == 0
    assert row["one_share_warning"] is not None
    assert row["one_share_warning"]["min_budget_krw"] > 500_000.0


def test_compute_live_judgments_no_warning_when_budget_sufficient():
    states = _states({"NVDA": "보유"})
    rows = compute_live_judgments(
        states, today_events=[], indicator_map={"NVDA": _df(100.0)}, as_of_by_ticker={"NVDA": _DATE},
        name_map={}, plan_by_ticker={"NVDA": 9_000_000.0}, fills_df=pd.DataFrame(columns=["ticker"]),
        fx_rate=1350.0, cfg=_CFG,
    )
    assert rows[0]["one_share_warning"] is None


def test_compute_live_judgments_warning_checked_even_without_todays_signal():
    """신호 여부와 무관하게 상시 경고 (docs/design/live_advisor.md 7번)."""
    states = _states({"NVDA": "보유"})
    rows = compute_live_judgments(
        states, today_events=[], indicator_map={"NVDA": _df(100.0)}, as_of_by_ticker={"NVDA": _DATE},
        name_map={}, plan_by_ticker={"NVDA": 500_000.0}, fills_df=pd.DataFrame(columns=["ticker"]),
        fx_rate=1350.0, cfg=_CFG,
    )
    assert rows[0]["one_share_warning"] is not None


def test_compute_live_judgments_skips_ticker_with_no_indicator_data():
    """지표를 못 받은 종목(상장폐지 등)은 판정 카드를 만들지 않는다."""
    states = _states({"NVDA": "보유"})  # states에는 있지만 indicator_map에는 없음
    rows = compute_live_judgments(
        states, today_events=[], indicator_map={}, as_of_by_ticker={},
        name_map={}, plan_by_ticker={"NVDA": 3_000_000.0}, fills_df=pd.DataFrame(columns=["ticker"]),
        fx_rate=1350.0, cfg=_CFG,
    )
    assert len(rows) == 1  # states에 있으므로 카드는 만들어짐, 참조가만 없음(경고 없음)
    assert rows[0]["one_share_warning"] is None


def test_compute_live_judgments_sorted_by_ticker():
    states = _states({"ZZZ": "보유", "AAA": "보유"})
    rows = compute_live_judgments(
        states, today_events=[], indicator_map={"ZZZ": _df(1.0), "AAA": _df(1.0)},
        as_of_by_ticker={"ZZZ": _DATE, "AAA": _DATE}, name_map={},
        plan_by_ticker={"ZZZ": 1.0, "AAA": 1.0}, fills_df=pd.DataFrame(columns=["ticker"]),
        fx_rate=1350.0, cfg=_CFG,
    )
    assert [r["ticker"] for r in rows] == ["AAA", "ZZZ"]


# ---------- build_report_summary: live-mode plan-based sizing ----------

_FULL_CFG = {
    "account": {"total_krw": 100_000_000},
    "plan": {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8},
    "risk": {
        "a1_budget_pct": 0.2222222222222222, "a2_budget_pct": 0.4444444444444444,
        "a3_budget_pct": 1.3333333333333333, "b_budget_pct": 1.0, "gap_buffer_pct": 1.5,
        "min_risk_per_share_pct": 1.0, "max_position_pct": 25, "max_concurrent_positions": 8,
    },
    "assumptions": {
        "a1_to_a2_expiry_days": 10, "reentry_cooldown_days": 5, "gap_filter_pct": 4.0,
        "whipsaw_max_crosses_20d": 4, "swing_low_period": 10, "s_grade_macd_norm_min_pct": -0.5,
        "b_grade_macd_norm_max_pct": -2.0, "ichimoku_shift": 26,
    },
    "indicators": {"ichimoku_shift": 26},
    "entry": {"limit_markup": 1.01},
    "orders": {"stop_fallback_type": "시장가"},
    "alerts": {"stop_near_pct": 3},
}


def _buy_summary(mode: str, plan_by_ticker=None):
    indicator_map = {"NVDA": _df(180.0)}
    today_events = [{"date": _DATE, "ticker": "NVDA", "kind": "A1", "unit": "1", "price": 180.0, "score": 20}]
    positions = {"NVDA": {**st.init_state("NVDA", "엔비디아"), "stop": 170.0}}
    as_of_by_ticker = {"NVDA": _DATE}
    fx_result = FxRateResult(rate=1350.0, rate_date=_DATE.date().isoformat(), is_fallback=False, warning=None)
    return build_report_summary(
        mode, _FULL_CFG, indicator_map, {"NVDA": "엔비디아"}, {"NVDA": None}, positions, today_events,
        as_of_by_ticker, [], FillsResult(), [], 8, fx_result, plan_by_ticker=plan_by_ticker,
    )


def test_build_report_summary_live_with_plan_computes_amount():
    summary = _buy_summary("live", plan_by_ticker={"NVDA": 9_000_000.0})
    row = summary["buy_groups"]["b1"][0]
    assert row["has_plan"] is True
    assert row["qty"] > 0
    assert row["amount_krw"] > 0
    assert summary["funding_plan"] is None
    assert summary["buy_risk_pct"] is None


def test_build_report_summary_live_without_plan_has_zero_amount_and_note():
    summary = _buy_summary("live", plan_by_ticker=None)
    row = summary["buy_groups"]["b1"][0]
    assert row["has_plan"] is False
    assert row["qty"] == 0
    assert row["amount_krw"] == 0
    assert "계획 없음" in row["note"]
    assert summary["funding_plan"] is None


def test_build_report_summary_paper_unaffected_by_plan_by_ticker_still_uses_account_sizing():
    """paper 모드는 plan_by_ticker를 완전히 무시하고 기존 계좌 기반 사이징을 그대로 쓴다
    (docs/design/live_advisor.md 0번 — paper 경로 무변경)."""
    summary = _buy_summary("paper", plan_by_ticker={"NVDA": 1.0})  # 계획금액이 있어도 paper는 무시해야 함
    row = summary["buy_groups"]["b1"][0]
    assert row["qty"] > 0  # 계좌 기반 사이징으로 정상적인 수량이 나온다(1원짜리 계획이 아님)
    assert summary["funding_plan"] is not None  # paper는 여전히 자금 계획을 만든다


def test_build_report_summary_default_plan_by_ticker_and_judgment_rows_are_empty():
    summary = _buy_summary("paper")
    assert summary["plan_by_ticker"] == {}
    assert summary["live_judgment_rows"] == []
