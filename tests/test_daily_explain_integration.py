"""engine/daily.py의 build_report_summary가 core/explain.py를 실제로 불러
각 행에 실제 지표 값이 담긴 "explain"을 붙이는지 확인한다 (P3.7 통합 테스트).

네트워크 없이 합성 DataFrame(tests/test_state.make_df)으로 돈다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core import state as st
from data.fills import FillsResult
from data.fx import FxRateResult
from engine.daily import build_report_summary
from tests.test_state import make_df

_CFG = {
    "account": {"total_krw": 100_000_000},
    "plan": {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8},
    "risk": {
        "a1_budget_pct": 0.2222222222222222,
        "a2_budget_pct": 0.4444444444444444,
        "a3_budget_pct": 1.3333333333333333,
        "b_budget_pct": 1.0,
        "gap_buffer_pct": 1.5,
        "min_risk_per_share_pct": 1.0,
        "max_position_pct": 25,
        "max_concurrent_positions": 8,
    },
    "assumptions": {
        "a1_to_a2_expiry_days": 10,
        "reentry_cooldown_days": 5,
        "gap_filter_pct": 4.0,
        "whipsaw_max_crosses_20d": 4,
        "swing_low_period": 10,
        "s_grade_macd_norm_min_pct": -0.5,
        "b_grade_macd_norm_max_pct": -2.0,
        "ichimoku_shift": 26,
    },
    "indicators": {"ichimoku_shift": 26},
    "entry": {"limit_markup": 1.01},
    "orders": {"stop_fallback_type": "시장가"},
    "alerts": {"stop_near_pct": 3},
}

_DATE = pd.bdate_range("2026-01-02", periods=8, name="date")[5]  # "오늘"


def _pep_df():
    rows = [{"close": 100, "rsi": 40} for _ in range(8)]
    rows[4] = {"close": 100, "rsi": 25.1}  # 어제
    rows[5] = {"close": 100, "rsi": 31.2, "vol_ratio": 1.3, "swing_low": 94}  # 오늘 A1
    return make_df(rows)


def _amzn_df():
    rows = [{"close": 220, "rsi": 50} for _ in range(8)]
    rows[5] = {"close": 217.90, "rsi": 50}
    return make_df(rows)


def _mu_df():
    rows = [{"close": 80, "rsi": 50} for _ in range(8)]
    return make_df(rows)


def _qcom_df():
    rows = [{"close": 150, "rsi": 55} for _ in range(8)]
    rows[5] = {"close": 125.0, "rsi": 55}
    return make_df(rows)


def _tsla_df():
    rows = [{"close": 250, "rsi": 55, "macd": -0.10, "signal": -0.10} for _ in range(8)]
    rows[5] = {"close": 250, "rsi": 47.2, "macd": -0.10, "signal": 0.32}
    return make_df(rows)


@pytest.fixture
def indicator_map():
    return {"PEP": _pep_df(), "AMZN": _amzn_df(), "MU": _mu_df(), "QCOM": _qcom_df(), "TSLA": _tsla_df()}


@pytest.fixture
def name_map():
    return {"PEP": "펩시코", "AMZN": "아마존", "MU": "마이크론", "QCOM": "퀄컴", "TSLA": "테슬라"}


@pytest.fixture
def summary(indicator_map, name_map):
    today_events = [
        {"date": _DATE, "ticker": "PEP", "kind": "A1", "unit": "1", "price": 100.0, "score": 20},
        {"date": _DATE, "ticker": "AMZN", "kind": "STOP", "unit": "1", "qty": 12, "entry_price": 221.40, "stop_price": 218.70},
        {
            "date": _DATE, "ticker": "MU", "kind": "BLOCKED", "stage": "B", "unit": "9",
            "reasons": ["실적 발표 3거래일 이내"], "blocked_type": "ban", "score": 30,
        },
    ]
    positions = {
        "PEP": {**st.init_state("PEP", "펩시코"), "stop": 90.0},
        "AMZN": st.init_state("AMZN", "아마존"),
        "MU": st.init_state("MU", "마이크론"),
        "QCOM": {
            **st.init_state("QCOM", "퀄컴"),
            "state": "확정",
            "units": {"1": 10, "2": 20, "6": 60},
            "entries": {"1": 100.0, "2": 100.0, "6": 100.0},
            "stop": 90.0,
        },
        "TSLA": {
            **st.init_state("TSLA", "테슬라"),
            "state": "정찰",
            "units": {"1": 41},
            "entries": {"1": 250.0},
            "a1_date": pd.bdate_range("2026-01-02", periods=8, name="date")[1],
            "stop": 230.0,
        },
    }
    as_of_by_ticker = {t: _DATE for t in indicator_map}
    earnings_map = {"PEP": None, "AMZN": None, "MU": (_DATE + pd.Timedelta(days=2)).date(), "QCOM": None, "TSLA": None}
    fx_result = FxRateResult(rate=1_300.0, rate_date=_DATE.date().isoformat(), is_fallback=False, warning=None)

    return build_report_summary(
        "live", _CFG, indicator_map, name_map, earnings_map, positions, today_events,
        as_of_by_ticker, [], FillsResult(), [], 8, fx_result,
    )


def test_buy_row_explain_carries_real_rsi_values(summary):
    row = summary["buy_groups"]["b1"][0]
    assert row["ticker"] == "PEP"
    assert row["explain"] is not None
    text = " ".join(c["text"] for c in row["explain"]["checks"])
    assert "25.1" in text and "31.2" in text


def test_sell_row_explain_carries_real_close_and_stop(summary):
    row = next(r for r in summary["sell_rows"] if r["티커"] == "AMZN")
    assert row["explain"] is not None
    text = " ".join(c["text"] for c in row["explain"]["checks"])
    assert "$217.90" in text and "$218.70" in text


def test_filtered_row_explain_carries_earnings_reason(summary):
    row = next(r for r in summary["filtered_rows"] if r["티커"] == "MU")
    assert row["explain"] is not None
    assert "실적 발표가 가까워 제외" in row["explain"]["title"]


def test_warn_row_explain_carries_real_avg_entry(summary):
    row = next(r for r in summary["warn_rows"] if r["티커"] == "QCOM")
    assert row["explain"] is not None
    assert "목표 도달" in row["explain"]["title"]
    text = " ".join(c["text"] for c in row["explain"]["checks"])
    assert "$100.00" in text  # 평균단가


def test_watch_row_explain_carries_real_macd_diff(summary):
    row = next(r for r in summary["watch_rows"] if r["티커"] == "TSLA")
    assert row["explain"] is not None
    text = " ".join(c["text"] for c in row["explain"]["checks"])
    assert "0.42" in text
