"""P3: live/paper 운용 모드 정리 테스트. 네트워크 없이 합성 DataFrame으로 돈다.

- live 모드: 신호 당일은 `주문대기`, 체결 기록이 없으면 다음 거래일에 "미체결"을
  한 번 표시하고 신호 이전 상태로 되돌린다(보유 아님, 재진입 대기 미적용) — P3.1 보완 1번.
- live 모드: 체결 기록을 며칠 늦게 넣어도, 처음부터 다시 계산하면 그 날짜부터
  올바르게 반영된다(core.state.process_day는 결정적이라 같은 입력이면 언제
  다시 계산해도 같은 결과가 나온다).
- paper 모드: 추천대로 체결됐다고 가정하는 가상 체결이 자동으로 들어간다.
- live/paper DB 경로가 분리돼 있다.
"""

from __future__ import annotations

import pandas as pd

from core import state as st
from engine.daily import _compute_funnel, _pending_order_rows, _unfilled_rows_from_events, simulate_since
from store import db
from tests.test_state import cfg, make_df  # noqa: F401 (cfg는 pytest fixture로 재사용)


def _setup(cfg):
    """A1이 발생하고 그 뒤로 이틀 더 흐르는 3일짜리 합성 데이터를 만든다."""
    rows = [
        {"close": 100, "rsi": 20},  # day-1 기준 (전일 rsi<30)
        {"close": 100, "rsi": 32, "swing_low": 94},  # day0: A1
        {"close": 100, "rsi": 40},  # day1
        {"close": 100, "rsi": 40},  # day2
    ]
    df = make_df(rows)
    indicator_map = {"TEST": df}
    per_ticker_dates = {"TEST": list(df.index)}
    gap_dates_by_ticker = {"TEST": set()}
    earnings_map = {"TEST": None}
    return indicator_map, per_ticker_dates, gap_dates_by_ticker, earnings_map


def test_live_mode_without_fills_has_no_holdings(cfg):
    indicator_map, per_ticker_dates, gap_dates_by_ticker, earnings_map = _setup(cfg)
    states = {"TEST": st.init_state("TEST")}
    empty_fills = pd.DataFrame(columns=["date", "ticker", "unit", "side", "price", "qty"])

    result = simulate_since(
        indicator_map, per_ticker_dates, states, cfg, earnings_map, gap_dates_by_ticker,
        empty_fills, virtual_fill=False, max_concurrent=8,
    )

    final_state = result["states"]["TEST"]
    assert final_state["state"] == "대기"  # 체결 기록이 없어 다음 거래일에 신호 전 상태로 되돌아간다
    assert final_state["units"].get("1", 0) == 0  # 보유는 0
    assert any(e["kind"] == "A1" for e in result["all_events"])
    assert any(e["kind"] == "UNFILLED" for e in result["all_events"])  # 미체결 알림이 한 번 남는다
    assert not any(e["kind"] == "VIRTUAL_FILL" for e in result["all_events"])  # 가상 체결 없음


def test_paper_mode_auto_fills_at_recommended_qty(cfg):
    # 가상 체결은 core.sizing.size_buy_signals(자금 계획)로 수량을 계산해야 해서
    # account·plan·risk 설정과 그날의 환율이 모두 필요하다 (P5-1 0번).
    cfg = {
        **cfg,
        "account": {"total_krw": 130_000_000},
        "plan": {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8},
        "risk": {
            "a1_budget_pct": 0.2222222222222222,
            "a2_budget_pct": 0.4444444444444444,
            "a3_budget_pct": 1.3333333333333333,
            "b_budget_pct": 1.0,
            "gap_buffer_pct": 1.5,
            "min_risk_per_share_pct": 1.0,
            "max_position_pct": 25,
        },
    }
    indicator_map, per_ticker_dates, gap_dates_by_ticker, earnings_map = _setup(cfg)
    states = {"TEST": st.init_state("TEST")}
    empty_fills = pd.DataFrame(columns=["date", "ticker", "unit", "side", "price", "qty"])
    fx_rate_by_date = {d.date().isoformat(): 1_300.0 for d in indicator_map["TEST"].index}

    result = simulate_since(
        indicator_map, per_ticker_dates, states, cfg, earnings_map, gap_dates_by_ticker,
        empty_fills, virtual_fill=True, max_concurrent=8, fx_rate_by_date=fx_rate_by_date,
    )

    final_state = result["states"]["TEST"]
    assert final_state["units"].get("1", 0) > 0  # paper는 추천대로 체결됐다고 가정
    assert any(e["kind"] == "VIRTUAL_FILL" for e in result["all_events"])


def test_live_mode_late_fill_is_reflected_on_recompute(cfg):
    """체결 기록을 며칠 늦게 넣어도, 처음부터 다시 계산하면 A1 당일 체결로 반영된다."""
    indicator_map, per_ticker_dates, gap_dates_by_ticker, earnings_map = _setup(cfg)
    a1_date = indicator_map["TEST"].index[1]

    # 1차 실행: 체결 기록 없음 -> 미체결
    states1 = {"TEST": st.init_state("TEST")}
    empty_fills = pd.DataFrame(columns=["date", "ticker", "unit", "side", "price", "qty"])
    result1 = simulate_since(
        indicator_map, per_ticker_dates, states1, cfg, earnings_map, gap_dates_by_ticker,
        empty_fills, virtual_fill=False, max_concurrent=8,
    )
    assert result1["states"]["TEST"]["units"].get("1", 0) == 0

    # 2차 실행(며칠 뒤): 사용자가 A1 당일 날짜로 체결 기록을 늦게 추가 -> 처음부터 다시 계산
    late_fills = pd.DataFrame(
        [{"date": a1_date, "ticker": "TEST", "unit": "1", "side": "buy", "price": 101.0, "qty": 55}]
    )
    states2 = {"TEST": st.init_state("TEST")}  # 항상 처음부터 다시 계산한다 (P3 핵심 규칙)
    result2 = simulate_since(
        indicator_map, per_ticker_dates, states2, cfg, earnings_map, gap_dates_by_ticker,
        late_fills, virtual_fill=False, max_concurrent=8,
    )
    final_state = result2["states"]["TEST"]
    assert final_state["units"]["1"] == 55
    assert final_state["entries"]["1"] == 101.0


def test_db_path_for_mode_separates_live_and_paper():
    live_path = db.db_path_for_mode("live")
    paper_path = db.db_path_for_mode("paper")
    assert live_path != paper_path
    assert live_path.name == "state.db"
    assert paper_path.name == "paper_state.db"


def test_live_and_paper_positions_do_not_cross_contaminate(tmp_path):
    live_conn = db.connect(tmp_path / "live.db")
    paper_conn = db.connect(tmp_path / "paper.db")

    db.save_position(live_conn, st.init_state("AAPL", "애플"))
    assert db.load_all_positions(paper_conn) == {}
    assert set(db.load_all_positions(live_conn)) == {"AAPL"}


def test_meta_start_date_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    assert db.get_meta(conn, "start_date") is None
    db.set_meta(conn, "start_date", "2026-09-23")
    assert db.get_meta(conn, "start_date") == "2026-09-23"


def test_notification_dedup_by_as_of_date(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    assert db.has_notified(conn, "2026-09-23") is False
    db.record_notified(conn, "2026-09-23", "2026-09-23T07:30:00")
    assert db.has_notified(conn, "2026-09-23") is True
    assert db.has_notified(conn, "2026-09-24") is False


def test_pending_order_rows_lists_today_signals_in_live_mode_only():
    state = st.init_state("NVDA", "엔비디아")
    state["state"] = "주문대기"
    state["pending"] = {"kind": "A1"}
    states = {"NVDA": state}
    name_map = {"NVDA": "엔비디아"}

    live_rows = _pending_order_rows(states, name_map, "live")
    assert len(live_rows) == 1
    assert live_rows[0]["티커"] == "NVDA"

    assert _pending_order_rows(states, name_map, "paper") == []


def test_unfilled_rows_from_events_flags_today_unfilled_in_live_mode_only():
    today_events = [{"kind": "UNFILLED", "ticker": "NVDA", "stage": "A1", "unit": "1", "date": pd.Timestamp("2026-09-23")}]
    name_map = {"NVDA": "엔비디아"}

    live_rows = _unfilled_rows_from_events(today_events, name_map, "live")
    assert len(live_rows) == 1
    assert live_rows[0]["티커"] == "NVDA"
    assert live_rows[0]["내용"] == "미체결 (기록 없음)"

    assert _unfilled_rows_from_events(today_events, name_map, "paper") == []  # paper는 항상 즉시 체결


def test_unfilled_rows_from_events_ignores_other_kinds():
    today_events = [{"kind": "A1", "ticker": "NVDA", "unit": "1", "date": pd.Timestamp("2026-09-23")}]
    assert _unfilled_rows_from_events(today_events, {}, "live") == []


def test_live_mode_unfilled_shows_pending_day_then_unfilled_day_then_late_fill_is_retroactive(cfg):
    """P3.1 보완 1번 전체 시나리오: 신호 당일 주문대기 -> 다음 날 기록 없음(미체결, 대기로 복귀)
    -> 사흘 뒤 신호일 날짜로 체결 기록을 입력하면 처음부터 다시 계산해 소급 반영된다."""
    rows = [
        {"close": 100, "rsi": 20},  # day-1
        {"close": 100, "rsi": 32, "swing_low": 94},  # day0: A1 신호
        {"close": 100, "rsi": 40},  # day1: 다음 거래일(체결 기록 확인)
        {"close": 100, "rsi": 40},  # day2
        {"close": 100, "rsi": 40},  # day3 (오늘, "사흘 뒤")
    ]
    df = make_df(rows)
    indicator_map = {"TEST": df}
    per_ticker_dates = {"TEST": list(df.index)}
    gap_dates_by_ticker = {"TEST": set()}
    earnings_map = {"TEST": None}
    a1_date = df.index[1]
    empty_fills = pd.DataFrame(columns=["date", "ticker", "unit", "side", "price", "qty"])

    # day0(신호 당일)만 처리 -> 주문대기
    day0_dates = {"TEST": [df.index[1]]}
    result_day0 = simulate_since(
        indicator_map, day0_dates, {"TEST": st.init_state("TEST")}, cfg, earnings_map, gap_dates_by_ticker,
        empty_fills, virtual_fill=False, max_concurrent=8,
    )
    assert result_day0["states"]["TEST"]["state"] == "주문대기"

    # day0~day1(다음 거래일)까지, 체결 기록 없음 -> 미체결로 확정, 대기로 복귀
    day1_dates = {"TEST": list(df.index[1:3])}
    result_day1 = simulate_since(
        indicator_map, day1_dates, {"TEST": st.init_state("TEST")}, cfg, earnings_map, gap_dates_by_ticker,
        empty_fills, virtual_fill=False, max_concurrent=8,
    )
    assert result_day1["states"]["TEST"]["state"] == "대기"
    assert any(e["kind"] == "UNFILLED" for e in result_day1["today_events"])

    # 사흘 뒤(day3, "오늘"): 신호일(day0) 날짜로 체결 기록을 늦게 입력 -> 처음부터 다시 계산
    late_fills = pd.DataFrame(
        [{"date": a1_date, "ticker": "TEST", "unit": "1", "side": "buy", "price": 101.0, "qty": 55}]
    )
    result_day3 = simulate_since(
        indicator_map, per_ticker_dates, {"TEST": st.init_state("TEST")}, cfg, earnings_map, gap_dates_by_ticker,
        late_fills, virtual_fill=False, max_concurrent=8,
    )
    final_state = result_day3["states"]["TEST"]
    assert final_state["state"] == "정찰"
    assert final_state["units"]["1"] == 55
    assert final_state["entries"]["1"] == 101.0
    assert final_state["a1_date"] == a1_date  # 진입일은 신호일 기준
    assert not any(e["kind"] == "UNFILLED" for e in result_day3["today_events"])  # 소급 반영되어 오늘은 미체결이 아니다


def test_compute_funnel_counts_stages():
    today_events = [
        {"kind": "A3", "ticker": "AAA"},
        {"kind": "BLOCKED", "stage": "A3", "ticker": "BBB", "blocked_type": "ban"},
        {"kind": "BLOCKED", "stage": "A2", "ticker": "CCC", "blocked_type": "ban"},
        {"kind": "BLOCKED", "stage": "A1", "ticker": "DDD", "blocked_type": "limit"},
    ]
    rows = [{"close": 100, "rsi": 32}]  # day0: RSI 30 돌파(전일 <30)
    df = pd.concat(
        [pd.DataFrame([{"close": 100, "rsi": 20, "gc": False}], index=[pd.Timestamp("2026-01-01")]),
         pd.DataFrame([{"close": 100, "rsi": 32, "gc": True}], index=[pd.Timestamp("2026-01-02")])]
    )
    indicator_map = {"AAA": df}
    as_of_by_ticker = {"AAA": df.index[-1]}

    funnel = _compute_funnel(today_events, indicator_map, as_of_by_ticker)
    assert funnel["1차 RSI 30 돌파"] == 1
    assert funnel["2차 MACD 골든크로스"] == 1
    assert funnel["3차 일목구름 4요소"] == 2  # A3 이벤트 1 + A3 blocked 1
    assert funnel["4차 매매금지 필터"] == 2  # A3, A2 blocked(ban)
    assert funnel["5차 보유한도"] == 1  # A1 blocked(limit)
