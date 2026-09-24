"""P3.5: 손절 예약 알림(근접·변경·신규)과 손절일 주문 안내 문구. 네트워크 없음."""

from __future__ import annotations

from engine.daily import (
    _need_stop_order_tickers,
    _order_guidance,
    _stop_alert_candidates,
    _stop_changed_by_ticker,
    _stop_event_tickers,
)

_CFG = {"orders": {"stop_fallback_type": "시장가"}}


# ── 근접·변경·신규 후보 우선순위 (P3.5 1번) ──────────────────────────────


def test_stop_alert_candidates_empty_when_nothing_applies():
    assert _stop_alert_candidates(close=110.0, stop=100.0, stop_near_pct=3, changed=None, needs_order=False) == []


def test_stop_alert_candidates_near_only():
    candidates = _stop_alert_candidates(close=102.0, stop=100.0, stop_near_pct=3, changed=None, needs_order=False)
    assert [c[0] for c in candidates] == ["근접"]
    assert candidates[0][1] == "b-warn"
    assert "100.00" in candidates[0][2]


def test_stop_alert_candidates_priority_order_is_near_then_changed_then_new():
    """근접 > 변경 > 신규 순서로 후보 목록을 만든다 (배지는 호출부가 첫 번째를 쓴다)."""
    candidates = _stop_alert_candidates(
        close=102.0,
        stop=100.0,
        stop_near_pct=3,
        changed={"old_stop": 90.0, "new_stop": 100.0},
        needs_order=True,
    )
    assert [c[0] for c in candidates] == ["근접", "변경", "신규"]
    assert candidates[1][1] == "b-info"
    assert candidates[2][1] == "b-info"


def test_stop_alert_candidates_changed_text_shows_old_and_new():
    candidates = _stop_alert_candidates(
        close=200.0, stop=100.0, stop_near_pct=3, changed={"old_stop": 90.0, "new_stop": 100.0}, needs_order=False
    )
    assert candidates[0][0] == "변경"
    assert "90.00" in candidates[0][2] and "100.00" in candidates[0][2]


def test_stop_alert_candidates_new_fill_needs_order():
    candidates = _stop_alert_candidates(close=200.0, stop=100.0, stop_near_pct=3, changed=None, needs_order=True)
    assert candidates[0][0] == "신규"


# ── 오늘 이벤트에서 종목 집합을 뽑는 헬퍼들 ──────────────────────────────


def test_stop_event_tickers_picks_only_stop_kind():
    today_events = [
        {"kind": "STOP", "ticker": "AAA"},
        {"kind": "E1", "ticker": "BBB"},
    ]
    assert _stop_event_tickers(today_events) == {"AAA"}


def test_stop_changed_by_ticker_maps_event():
    today_events = [
        {"kind": "stop_changed", "ticker": "AAA", "old_stop": 90.0, "new_stop": 100.0},
        {"kind": "A3", "ticker": "BBB"},
    ]
    result = _stop_changed_by_ticker(today_events)
    assert set(result) == {"AAA"}
    assert result["AAA"]["new_stop"] == 100.0


def test_need_stop_order_tickers_from_fill_and_virtual_fill():
    """새로 체결된 매수가 있는 날 "손절 예약 필요" 대상이 된다 (1차·2차·3차·재진입 매수 모두)."""
    today_events = [
        {"kind": "FILL", "ticker": "AAA", "side": "buy", "qty": 10},
        {"kind": "VIRTUAL_FILL", "ticker": "BBB", "side": "buy", "qty": 5},
        {"kind": "FILL", "ticker": "CCC", "side": "sell", "qty": 3},  # 매도는 대상 아님
        {"kind": "FILL", "ticker": "DDD", "side": "buy", "qty": 0},  # 미체결(0주)은 대상 아님
    ]
    assert _need_stop_order_tickers(today_events) == {"AAA", "BBB"}


# ── 매도·손절 탭 "주문 안내" 문구 (P3.5 5번) ────────────────────────────


def test_order_guidance_for_stop_includes_stop_price_and_fallback_type():
    text = _order_guidance("STOP", qty=14, stop_price=405.0, cfg=_CFG)
    assert "손절 예약 확인" in text
    assert "$405.00" in text
    assert "14주" in text
    assert "시장가" in text


def test_order_guidance_for_stop_without_stop_price_says_unconfirmed():
    text = _order_guidance("STOP", qty=14, stop_price=None, cfg=_CFG)
    assert "미확인" in text


def test_order_guidance_for_e1_e2_e3_and_expire_is_generic():
    for kind in ("E1", "E2", "E3", "A1_EXPIRE"):
        text = _order_guidance(kind, qty=7, stop_price=100.0, cfg=_CFG)
        assert text == "다음 거래일 장 시작 시 7주 매도 주문 예약"
