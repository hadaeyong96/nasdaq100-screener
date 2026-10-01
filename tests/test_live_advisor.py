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
from tests._offline import block_engine_network
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


# ── 계획 탭 개편(2026-09-30): 기준가 기반 사이징 + 남은 주수(계획 한도) 제한 ───


def _buy_summary_with_ref_price(budget, ref_price, held_qty=0, fx_rate=1350.0):
    indicator_map = {"NVDA": _df(180.0)}
    today_events = [{"date": _DATE, "ticker": "NVDA", "kind": "A1", "unit": "1", "price": 180.0, "score": 20}]
    # held_qty>0은 실제로는 A1이 아니라 A2/A3에서나 나오는 조합이지만, 여기서는
    # positions["units"] 합만으로 정해지는 "남은 주수" 계산만 순수하게 검증한다.
    units = {"1": held_qty} if held_qty else {}
    positions = {"NVDA": {**st.init_state("NVDA", "엔비디아"), "stop": 170.0, "units": units}}
    as_of_by_ticker = {"NVDA": _DATE}
    fx_result = FxRateResult(rate=fx_rate, rate_date=_DATE.date().isoformat(), is_fallback=False, warning=None)
    return build_report_summary(
        "live", _FULL_CFG, indicator_map, {"NVDA": "엔비디아"}, {"NVDA": None}, positions, today_events,
        as_of_by_ticker, [], FillsResult(), [], 8, fx_result,
        plan_by_ticker={"NVDA": budget}, plan_ref_price_by_ticker={"NVDA": ref_price},
    )


def test_build_report_summary_live_ref_price_uses_fixed_total_shares():
    # 총 주수 = int(9,000,000 / (180.0 * 1350.0)) = int(37.03..) = 37 -> 1차 = MAX(1, ROUND(37/9)) = 4
    summary = _buy_summary_with_ref_price(budget=9_000_000, ref_price=180.0)
    row = summary["buy_groups"]["b1"][0]
    assert row["qty"] == 4


def test_build_report_summary_live_ref_price_caps_qty_to_remaining_shares():
    # 총 37주, 보유 36주 -> 남은 1주뿐. 1차 계산값(4)보다 작게 잘려야 한다.
    summary = _buy_summary_with_ref_price(budget=9_000_000, ref_price=180.0, held_qty=36)
    row = summary["buy_groups"]["b1"][0]
    assert row["qty"] == 1
    assert "계획 한도 초과" not in row["note"]


def test_build_report_summary_live_ref_price_shows_over_limit_note_when_already_at_total():
    summary = _buy_summary_with_ref_price(budget=9_000_000, ref_price=180.0, held_qty=37)
    row = summary["buy_groups"]["b1"][0]
    assert row["qty"] == 0
    assert "계획 한도 초과" in row["note"]


def test_build_report_summary_without_ref_price_is_unaffected_by_held_qty():
    """기준가가 없는 종목(기존 방식)은 보유 수량과 무관하게 지금처럼 계산된다 —
    남은 주수 개념 자체가 없다."""
    indicator_map = {"NVDA": _df(180.0)}
    today_events = [{"date": _DATE, "ticker": "NVDA", "kind": "A1", "unit": "1", "price": 180.0, "score": 20}]
    positions = {"NVDA": {**st.init_state("NVDA", "엔비디아"), "stop": 170.0, "units": {"1": 999}}}
    as_of_by_ticker = {"NVDA": _DATE}
    fx_result = FxRateResult(rate=1350.0, rate_date=_DATE.date().isoformat(), is_fallback=False, warning=None)
    summary = build_report_summary(
        "live", _FULL_CFG, indicator_map, {"NVDA": "엔비디아"}, {"NVDA": None}, positions, today_events,
        as_of_by_ticker, [], FillsResult(), [], 8, fx_result, plan_by_ticker={"NVDA": 9_000_000.0},
    )
    row = summary["buy_groups"]["b1"][0]
    assert row["qty"] == 4  # test_build_report_summary_live_with_plan_computes_amount와 같은 값
    assert "계획 한도 초과" not in row["note"]


# ── compute_live_judgments도 같은 기준가 방식·한도 제한을 쓴다 ───────────────


def _add_judgment_row(ref_price, held_qty, budget=9_000_000.0, fx_rate=1350.0):
    state_ = {**st.init_state("NVDA", "엔비디아"), "stop": 170.0, "units": {"1": held_qty} if held_qty else {}}
    states = {"NVDA": state_}
    today_events = [{"date": _DATE, "ticker": "NVDA", "kind": "A2", "unit": "2", "price": 180.0, "score": 20}]
    indicator_map = {"NVDA": _df(180.0)}
    as_of_by_ticker = {"NVDA": _DATE}
    rows = compute_live_judgments(
        states, today_events, indicator_map, as_of_by_ticker, {"NVDA": "엔비디아"},
        {"NVDA": budget}, pd.DataFrame(columns=["ticker"]), fx_rate, _CFG,
        plan_ref_price_by_ticker={"NVDA": ref_price},
    )
    return rows[0]


def test_compute_live_judgments_ref_price_caps_qty_and_flags_over_limit():
    # 총 주수 = int(9,000,000/(180*1350)) = 37. held=37 -> 이미 한도 도달.
    row = _add_judgment_row(ref_price=180.0, held_qty=37)
    assert row["qty"] == 0
    assert row["plan_limit_exceeded"] is True


def test_compute_live_judgments_ref_price_not_over_limit_when_remaining_shares_left():
    row = _add_judgment_row(ref_price=180.0, held_qty=4)  # 남은 33주 > 2차 추천량
    assert row["plan_limit_exceeded"] is False
    assert row["qty"] > 0


def test_compute_live_judgments_plan_limit_exceeded_defaults_false_without_ref_price():
    state_ = {**st.init_state("NVDA", "엔비디아"), "stop": 170.0}
    states = {"NVDA": state_}
    today_events = [{"date": _DATE, "ticker": "NVDA", "kind": "A1", "unit": "1", "price": 180.0, "score": 20}]
    rows = compute_live_judgments(
        states, today_events, {"NVDA": _df(180.0)}, {"NVDA": _DATE}, {"NVDA": "엔비디아"},
        {"NVDA": 9_000_000.0}, pd.DataFrame(columns=["ticker"]), 1350.0, _CFG,
    )
    assert rows[0]["plan_limit_exceeded"] is False


# ── run(): 구글 시트 우선, 인증 정보 없으면 fills.xlsx로 폴백 (1g) ─────────────


def _patch_run_io(monkeypatch, tmp_path):
    """run()의 네트워크·DB·현재 시각 접근을 모두 가짜로 바꾼다 (tests/test_stale_guard.py와 같은 패턴)."""
    import numpy as np
    from datetime import datetime

    from data.prices import US_EASTERN, PriceFetchResult
    from engine import daily as engine_daily
    from store import db

    last_date = "2026-09-25"
    rng = np.random.default_rng(0)
    periods = 60
    close = np.clip(100 + np.cumsum(rng.normal(0.05, 1.2, periods)), 5, None)
    df = pd.DataFrame(
        {"open": close - 0.05, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": np.full(periods, 2_000_000)},
        index=pd.bdate_range(end=last_date, periods=periods, name="date"),
    )
    df["close_source"] = "yahoo"
    df.attrs["warnings"] = []
    df.attrs["close_meta_time"] = None

    monkeypatch.setattr(engine_daily, "get_universe", lambda: pd.DataFrame({"ticker": ["TEST"], "name_kr": ["테스트"]}))
    monkeypatch.setattr(engine_daily, "fetch_universe_prices", lambda tickers, cfg: PriceFetchResult(prices={"TEST": df}))
    monkeypatch.setattr(engine_daily, "get_earnings_dates", lambda tickers: {t: None for t in tickers})
    monkeypatch.setattr(engine_daily, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "state.db")

    now_et = datetime.fromisoformat(f"{last_date}T20:00:00").replace(tzinfo=US_EASTERN)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_et if tz is not None else now_et.replace(tzinfo=None)

    block_engine_network(monkeypatch)  # 시장 온도·환율 네트워크 차단
    monkeypatch.setattr(engine_daily, "datetime", _FrozenDateTime)
    return engine_daily


def test_run_live_falls_back_to_local_fills_xlsx_without_sheets_credentials(monkeypatch, tmp_path, cfg):
    """GOOGLE_SERVICE_ACCOUNT_JSON·GOOGLE_SHEETS_ID가 없으면(로컬 개발) 조용히
    fills.xlsx/load_plan으로 폴백해야 한다 — 실제 네트워크를 타면 안 된다."""
    engine_daily = _patch_run_io(monkeypatch, tmp_path)
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_SHEETS_ID", raising=False)
    monkeypatch.setattr(engine_daily, "load_fills", lambda: FillsResult())
    monkeypatch.setattr(
        engine_daily, "load_plan", lambda: (pd.DataFrame({"ticker": ["TEST"], "budget_krw": [3_000_000.0], "memo": [""]}), [])
    )

    summary = engine_daily.run(cfg, "live", do_replay=False, dry_run=True)

    assert summary["plan_by_ticker"] == {"TEST": 3_000_000.0}


def test_run_live_uses_injected_sheets_client_when_sheet_id_configured(monkeypatch, tmp_path, cfg):
    """GOOGLE_SHEETS_ID가 있고 sheets_client가 주입되면 그 시트의 계획을 쓴다
    (fills.xlsx의 load_plan은 전혀 안 불려야 한다)."""
    engine_daily = _patch_run_io(monkeypatch, tmp_path)
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")

    def _load_plan_should_not_be_called():
        raise AssertionError("구글 시트가 설정돼 있으면 load_plan()을 부르면 안 된다")

    monkeypatch.setattr(engine_daily, "load_plan", _load_plan_should_not_be_called)

    class _FakeWorksheet:
        def __init__(self, records):
            self._records = records

        def get_all_records(self):
            return self._records

    class _FakeSpreadsheet:
        def worksheet(self, name):
            if name == "계획":
                return _FakeWorksheet([{"티커": "TEST", "계획금액": "5000000", "메모": ""}])
            return _FakeWorksheet([])

    class _FakeClient:
        def open_by_key(self, sheet_id):
            assert sheet_id == "sheet-123"
            return _FakeSpreadsheet()

    summary = engine_daily.run(cfg, "live", do_replay=False, dry_run=True, sheets_client=_FakeClient())

    assert summary["plan_by_ticker"] == {"TEST": 5_000_000.0}


def test_run_paper_never_touches_sheets_even_when_credentials_configured(monkeypatch, tmp_path, cfg):
    """paper 모드는 시트 인증이 있어도 절대 시도하지 않는다 — fills.xlsx만 쓴다
    (docs/design/live_advisor.md 0번 — paper 경로 무변경)."""
    engine_daily = _patch_run_io(monkeypatch, tmp_path)
    from store import db as db_module

    monkeypatch.setattr(db_module, "PAPER_DB_PATH", tmp_path / "paper_state.db")
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", '{"type": "service_account"}')
    monkeypatch.setattr(engine_daily, "load_fills", lambda: FillsResult())

    def _read_sheets_should_not_be_called(client=None):
        raise AssertionError("paper 모드는 read_sheets를 절대 부르면 안 된다")

    monkeypatch.setattr(engine_daily.sheets, "read_sheets", _read_sheets_should_not_be_called)

    summary = engine_daily.run(cfg, "paper", do_replay=False, dry_run=True)

    assert summary["plan_by_ticker"] == {}


def _real_header_sheet_client(fills_records):
    """실제 구글 시트 헤더(계획금액(원)·구분(매수/매도) 등)를 돌려주는 가짜 클라이언트."""

    class _FakeWorksheet:
        def __init__(self, records):
            self._records = records

        def get_all_records(self):
            return self._records

    class _FakeSpreadsheet:
        def worksheet(self, name):
            if name == "계획":
                return _FakeWorksheet(
                    [
                        {"티커": "TEST", "계획금액(원)": "3,000,000", "등록일": "2026-09-20", "메모": ""},
                        {"티커": "", "계획금액(원)": "", "등록일": "", "메모": ""},
                    ]
                )
            return _FakeWorksheet(fills_records)

    class _FakeClient:
        def open_by_key(self, sheet_id):
            return _FakeSpreadsheet()

    return _FakeClient()


def test_run_live_rebuilds_when_late_fill_added_on_already_processed_day(monkeypatch, tmp_path, cfg):
    """2026-09-30 Actions 사고 재현: 아침 실행이 체결 0건으로 기준일을 처리해 버린 뒤,
    시트 헤더를 고쳐 같은 기준일에 다시 돌리면 — 이미 처리된 날이라도 체결 기록이
    바뀌었으므로 가장 이른 체결일부터 다시 계산해 보유가 보고서에 나와야 한다.
    체결일이 일요일(한국 날짜)이어도 다음 거래일에 반영돼야 한다."""
    from data import fx as fx_module
    from data import sheets as sheets_module

    engine_daily = _patch_run_io(monkeypatch, tmp_path)
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")
    monkeypatch.setattr(fx_module, "_load_cache", lambda: {})
    monkeypatch.setattr(sheets_module, "_default_fx_provider", lambda start, end: {"2026-09-18": 1390.0})

    first = engine_daily.run(cfg, "live", do_replay=False, dry_run=False, sheets_client=_real_header_sheet_client([]))
    assert first["input_errors"] == []
    assert not any(r["티커"] == "TEST" for r in first["hold_rows"])

    fills = [
        {"날짜": "2026-09-20", "티커": "TEST", "구분(매수/매도)": "매수", "수량": "3", "체결가($)": "100",
         "환율(원/$)": "", "수수료($)": "", "메모": ""},
        {"날짜": "", "티커": "", "구분(매수/매도)": "", "수량": "", "체결가($)": "", "환율(원/$)": "", "수수료($)": "", "메모": ""},
    ]
    second = engine_daily.run(cfg, "live", do_replay=False, dry_run=False, sheets_client=_real_header_sheet_client(fills))

    assert not second.get("skipped")
    assert second["input_errors"] == []
    assert second["rebuilt_from"] == "2026-09-20"
    held = [r for r in second["hold_rows"] if r["티커"] == "TEST"]
    assert held, "늦게 적은 체결이 보유 현황에 나와야 한다"

    third = engine_daily.run(cfg, "live", do_replay=False, dry_run=False, sheets_client=_real_header_sheet_client(fills))
    assert third.get("skipped")  # 체결 기록이 그대로면 같은 기준일 재실행은 예전처럼 건너뛴다
