"""시장 온도(P3.8) 지표가 신호·수량 계산에 영향을 주지 않는지 확인 (지시문 5번).

cfg["macro"] 값을 극단적으로 바꿔도(또는 통째로 없애도) build_report_summary의
buy_groups·buy_count·hold_rows가 완전히 같아야 한다 — engine.daily._build_macro_rows는
run()에서 build_report_summary와 별도로 호출되고 그 결과(summary["macro_rows"])는
build_report_summary 바깥에서 덧붙여질 뿐, build_report_summary 내부 로직은
cfg["macro"]를 전혀 읽지 않기 때문이다.
"""

from __future__ import annotations

import copy

import pandas as pd

from data import fx
from data.fills import FillsResult
from engine.daily import build_report_summary, simulate_since
from tests.test_engine_modes import _setup
from tests.test_state import cfg  # noqa: F401 (pytest fixture)

ADVERSARIAL_MACRO = {
    "enabled": True,
    "fear_greed": {"source": "cnn", "fallback": "VIXCLS", "fallback_after_days": 3},
    "series": ["DGS10", "T10Y2Y", "BAMLH0A0HYM2", "DFEDTARU", "DEXKOUS"],
    "thresholds": {
        "FEAR_GREED": {"extreme_fear": 99, "fear": 100, "neutral_high": 100, "greed": 100},
        "VIXCLS": {"caution": -1, "alert": -1},
        "DGS10": {"rise_3m_caution": -999},
        "T10Y2Y": {"normal": 999, "alert": 999},
        "BAMLH0A0HYM2": {"caution": -1, "alert": -1},
        "DEXKOUS": {"pct_high": 0, "pct_low": 100},
    },
    "stale_days": 0,
}


_SIZING_CFG = {
    "account": {"total_krw": 130_000_000},
    "plan": {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8},
    "risk": {
        "a1_budget_pct": 0.2222222222222222, "a2_budget_pct": 0.4444444444444444,
        "a3_budget_pct": 1.3333333333333333, "b_budget_pct": 1.0,
        "gap_buffer_pct": 1.5, "min_risk_per_share_pct": 1.0, "max_position_pct": 25,
    },
    "alerts": {"stop_near_pct": 3},
    "orders": {"stop_fallback_type": "시장가"},
}


def _run_build_report_summary(cfg_variant):
    cfg_variant = {**cfg_variant, **_SIZING_CFG}
    indicator_map, per_ticker_dates, gap_dates_by_ticker, earnings_map = _setup(cfg_variant)
    states = {"TEST": __import__("core.state", fromlist=["init_state"]).init_state("TEST")}
    empty_fills = pd.DataFrame(columns=["date", "ticker", "unit", "side", "price", "qty"])

    sim = simulate_since(
        indicator_map, per_ticker_dates, states, cfg_variant, earnings_map, gap_dates_by_ticker,
        empty_fills, virtual_fill=False, max_concurrent=8,
    )
    fx_result = fx.FxRateResult(rate=1300.0, rate_date="2026-01-01", is_fallback=False)
    return build_report_summary(
        "live", cfg_variant, indicator_map, {}, earnings_map, sim["states"], sim["today_events"],
        sim["as_of_by_ticker"], sim["data_gap_tickers"], FillsResult(), list(sim["warnings"]), 8,
        fx_result, replay_needed=False,
    )


def test_buy_groups_unchanged_when_macro_thresholds_are_adversarial(cfg):
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg["macro"] = {"enabled": True, "thresholds": {}, "fear_greed": {}, "series": []}
    adversarial_cfg = copy.deepcopy(cfg)
    adversarial_cfg["macro"] = ADVERSARIAL_MACRO

    baseline_summary = _run_build_report_summary(baseline_cfg)
    adversarial_summary = _run_build_report_summary(adversarial_cfg)

    assert baseline_summary["buy_groups"] == adversarial_summary["buy_groups"]
    assert baseline_summary["buy_count"] == adversarial_summary["buy_count"]
    assert baseline_summary["hold_rows"] == adversarial_summary["hold_rows"]


def test_buy_groups_unchanged_when_macro_key_missing_entirely(cfg):
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg["macro"] = {"enabled": True, "thresholds": {}, "fear_greed": {}, "series": []}
    no_macro_cfg = copy.deepcopy(cfg)
    no_macro_cfg.pop("macro", None)

    baseline_summary = _run_build_report_summary(baseline_cfg)
    no_macro_summary = _run_build_report_summary(no_macro_cfg)

    assert baseline_summary["buy_groups"] == no_macro_summary["buy_groups"]
