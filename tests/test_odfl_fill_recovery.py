"""ODFL 2026-09-29 상태 오류에 대한 회귀 테스트. 네트워크 없이 합성 데이터로 돈다.

배경: 2026-09-28 ODFL A1(1차 정찰) 신호가 나 `주문대기`로 들어갔다. 9/29 첫 실행
시점에는 아직 체결 기록이 없어 `core.state._resolve_pending`이 UNFILLED로 판정해
상태를 "대기"·손절가 None으로 되돌렸다. 그 뒤 실제 체결(1차 4주, $178.41)이
data/fills.xlsx(구글 시트)에 기록됐지만, 이미 "대기"로 되돌아간 뒤라
`engine.daily.simulate_since`의 일반 체결 반영 경로(_apply_live_fills, "주문대기"
사전 반영이 아닌 사후 catch-up)는 `core.state.apply_fill`을 거쳐 units·entries만
채우고 state·stop은 건드리지 않는다 — 그 결과 실제로는 4주를 보유한 종목인데
state.db에는 state="대기"·stop=None으로 남아, 9/29 보고서에 "미체결 1 · ODFL"과
"손절 예약 · ODFL(신규)"이 동시에(모순되게) 나타났다.

이 상태로 다음 날(9/30)이 처리되면 core/state.py의 손절 체크(`state != "대기"`
조건)가 이 포지션을 건너뛰어 손절이 전혀 감시되지 않는 심각한 문제였다. 지금은
data/state.db의 ODFL positions 행을 원래 A1 신호 때 계산됐던 값(state="정찰",
stop=169.64999389648438, a1_date=2026-09-28)으로 직접 복구했고, 9/29 이벤트
로그에서 수정 전 실행이 남긴 UNFILLED 기록도 지웠다(FILL 기록의 state_after·
stop_after도 바로잡았다).

아래 테스트는 이 복구된 상태를 engine.daily.build_report_summary에 그대로 넣어
9/30 보고서가 ODFL을 "미체결"이 아니라 "보유(정찰)"로, 올바른 손절가와 함께
보여주는지 고정한다 — 같은 상태로 회귀하면 이 테스트가 잡는다.
"""

from __future__ import annotations

import pandas as pd

from core import state as st
from data.fills import FillsResult
from data.fx import FxRateResult
from engine.daily import build_report_summary
from tests.test_state import make_df

_CFG = {
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

_NEXT_DATE = pd.Timestamp("2026-09-30")  # 9/30(수) 마감 — ODFL 실제 체결 다음 거래일

# data/state.db에서 복구한 실제 ODFL 상태 그대로 (2026-09-28 A1 신호 · 2026-09-29 체결 4주).
_RECOVERED_ODFL_STATE = {
    **st.init_state("ODFL", "올드도미니언"),
    "state": "정찰",
    "units": {"1": 4},
    "entries": {"1": 178.41},
    "stop": 169.64999389648438,
    "a1_date": pd.Timestamp("2026-09-28"),
}


def _summary_for_odfl_only(state: dict, today_events: list[dict] | None = None) -> dict:
    indicator_map = {"ODFL": make_df([{"close": 182.50, "rsi": 55}], start="2026-09-30")}
    positions = {"ODFL": state}
    as_of_by_ticker = {"ODFL": _NEXT_DATE}
    fx_result = FxRateResult(rate=1350.0, rate_date=_NEXT_DATE.date().isoformat(), is_fallback=False, warning=None)
    return build_report_summary(
        "live", _CFG, indicator_map, {"ODFL": "올드도미니언"}, {"ODFL": None}, positions,
        today_events or [], as_of_by_ticker, [], FillsResult(), [], 8, fx_result,
    )


def test_recovered_odfl_state_shows_as_held_scouting_not_unfilled():
    summary = _summary_for_odfl_only(_RECOVERED_ODFL_STATE)

    hold_row = next((r for r in summary["hold_rows"] if r["티커"] == "ODFL"), None)
    assert hold_row is not None, "9/30 보고서 보유 현황에 ODFL이 나와야 한다"
    assert hold_row["단계"] == "정찰"
    assert hold_row["수량"] == 4
    assert hold_row["손절가"] == 169.65  # round(169.64999389648438, 2)

    assert not any(r["티커"] == "ODFL" for r in summary["unfilled_rows"]), "확정 체결된 종목이 미체결로 나오면 안 된다"


def test_corrupted_odfl_state_before_fix_would_have_hidden_stop_and_stage():
    """수정 전 상태(state="대기", stop=None)였다면 정찰 단계·손절가가 안 보였다는
    걸 기록해 둔다 — 복구가 실제로 무엇을 고쳤는지 보여주는 대조군."""
    broken_state = {**_RECOVERED_ODFL_STATE, "state": "대기", "stop": None, "a1_date": None}
    summary = _summary_for_odfl_only(broken_state)

    hold_row = next((r for r in summary["hold_rows"] if r["티커"] == "ODFL"), None)
    assert hold_row is not None  # 수량 기준으로는 여전히 나온다 — 문제는 표시 내용이었다
    assert hold_row["단계"] == "대기"  # "정찰"이 아니라 잘못된 라벨
    assert hold_row["손절가"] is None  # 손절가가 아예 없었다 — core.state의 손절 체크도 건너뛴다
