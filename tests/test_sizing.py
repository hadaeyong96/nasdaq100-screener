"""6장 수량 계산 테스트. "계산 예시" 표(갭 여유 0 설정 시 A1 37주, A2 49주, A3 148주)를
정확히 재현하는지 확인한다. 네트워크 없이, config.yaml을 그대로 읽어 돈다.
"""

from __future__ import annotations

import copy

import pytest

from core import sizing


def _cfg_no_gap_buffer(cfg: dict) -> dict:
    """계산 예시 표는 갭 여유를 0으로 두고 계산한 값이다."""
    out = copy.deepcopy(cfg)
    out["risk"]["gap_buffer_pct"] = 0.0
    return out


def test_a1_example_37_shares(cfg):
    c = _cfg_no_gap_buffer(cfg)
    qty = sizing.position_size("A1", entry_price=100.0, stop_price=94.0, equity_usd=100_000, cfg=c)
    assert qty == 37


def test_a2_example_49_shares(cfg):
    c = _cfg_no_gap_buffer(cfg)
    qty = sizing.position_size("A2", entry_price=103.0, stop_price=94.0, equity_usd=100_000, cfg=c)
    assert qty == 49


def test_a3_example_148_shares(cfg):
    c = _cfg_no_gap_buffer(cfg)
    qty = sizing.position_size("A3", entry_price=115.0, stop_price=106.0, equity_usd=100_000, cfg=c)
    assert qty == 148


def test_min_risk_per_share_floor_applies(cfg):
    """주당 위험이 매수가의 min_risk_per_share_pct 미만이면 그 값으로 올린다."""
    c = _cfg_no_gap_buffer(cfg)
    # 손절이 매수가의 0.1%만 아래 -> 주당 위험이 최소값(1%)보다 작아 올려야 한다.
    risk = sizing.per_share_risk(entry_price=100.0, stop_price=99.9, cfg=c)
    assert risk == 1.0  # 100 * 1% 최소값


def test_cap_qty_by_position_limit(cfg):
    c = _cfg_no_gap_buffer(cfg)
    # 25% 한도 = $25,000. entry=100이면 최대 250주.
    qty = sizing.cap_qty_by_position_limit(qty=1000, entry_price=100.0, equity_usd=100_000, cfg=c)
    assert qty == 250


def test_gap_buffer_increases_risk_and_shrinks_qty(cfg):
    """기본 갭 여유(1.5%)를 쓰면 계산 예시보다 수량이 줄어야 한다."""
    qty = sizing.position_size("A1", entry_price=100.0, stop_price=94.0, equity_usd=100_000, cfg=cfg)
    assert qty < 37


# ── 자금 계획 (P3.6 6-6번) ────────────────────────────────────────────────

def _plan_cfg(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out["account"] = {"total_krw": 100_000_000}  # 1억
    out["plan"] = {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8}
    return out


def test_slot_krw_is_total_times_limit_over_slots(cfg):
    c = _plan_cfg(cfg)
    # 1억 * 60% / 8 = 750만
    assert sizing.slot_krw(c) == 7_500_000


def test_stage_target_krw_splits_slot_1_2_6_9(cfg):
    c = _plan_cfg(cfg)
    slot = sizing.slot_krw(c)
    assert sizing.stage_target_krw("A1", c) == pytest.approx(slot * 1 / 9)
    assert sizing.stage_target_krw("A2", c) == pytest.approx(slot * 2 / 9)
    assert sizing.stage_target_krw("A3", c) == pytest.approx(slot * 6 / 9)
    assert sizing.stage_target_krw("B", c) == pytest.approx(slot)


def test_reserved_fraction_by_state(cfg):
    assert sizing.reserved_fraction_for_state("정찰") == pytest.approx(8 / 9)
    assert sizing.reserved_fraction_for_state("확인") == pytest.approx(6 / 9)
    assert sizing.reserved_fraction_for_state("확정") == 0
    assert sizing.reserved_fraction_for_state("대기") == 0


def test_funding_qty_uses_target_amount_when_risk_allows(cfg):
    c = _plan_cfg(cfg)
    out = sizing.funding_qty("A1", entry_price=100.0, stop_price=94.0, fx_rate=1_300, cfg=c)
    target_usd = sizing.stage_target_krw("A1", c) / 1_300
    assert out["target_qty"] == int(target_usd // 100.0)
    assert out["qty"] == out["target_qty"]
    assert out["risk_capped"] is False


def test_funding_qty_capped_by_risk_when_stop_is_far(cfg):
    """손절이 멀면(주당 위험이 크면) 위험 상한 수량이 목표 수량보다 작아져 그쪽을 따른다."""
    c = _plan_cfg(cfg)
    out = sizing.funding_qty("A1", entry_price=100.0, stop_price=50.0, fx_rate=1_300, cfg=c)
    assert out["risk_capped"] is True
    assert out["qty"] == out["risk_cap_qty"]
    assert out["qty"] < out["target_qty"]


def test_funding_qty_zero_without_stop_or_fx(cfg):
    c = _plan_cfg(cfg)
    assert sizing.funding_qty("A1", 100.0, None, 1_300, c)["qty"] == 0
    assert sizing.funding_qty("A1", 100.0, 94.0, None, c)["qty"] == 0


def test_allocate_remaining_limit_full_when_enough_room(cfg):
    c = _plan_cfg(cfg)
    slot = sizing.slot_krw(c)
    candidates = [{"key": "AAA", "entry_price": 100.0, "fx_rate": 1_300, "target_qty": 50, "risk_cap_qty": 100}]
    out = sizing.allocate_remaining_limit(candidates, remaining_krw=slot * 2, cfg=c)
    assert out[0]["qty"] == 50
    assert out[0]["limited"] is False


def test_allocate_remaining_limit_zero_when_no_room(cfg):
    c = _plan_cfg(cfg)
    candidates = [{"key": "AAA", "entry_price": 100.0, "fx_rate": 1_300, "target_qty": 50, "risk_cap_qty": 100}]
    out = sizing.allocate_remaining_limit(candidates, remaining_krw=0, cfg=c)
    assert out[0]["qty"] == 0
    assert out[0]["limited"] is True


def test_allocate_remaining_limit_shrinks_and_prioritizes_by_order(cfg):
    """점수 내림차순으로 이미 정렬된 순서를 그대로 따른다 — 앞쪽이 우선 배분된다."""
    c = _plan_cfg(cfg)
    slot = sizing.slot_krw(c)
    candidates = [
        {"key": "HIGH", "entry_price": 100.0, "fx_rate": 1_300, "target_qty": 50, "risk_cap_qty": 100},
        {"key": "LOW", "entry_price": 100.0, "fx_rate": 1_300, "target_qty": 50, "risk_cap_qty": 100},
    ]
    out = sizing.allocate_remaining_limit(candidates, remaining_krw=slot * 1.5, cfg=c)
    assert out[0]["key"] == "HIGH" and out[0]["limited"] is False
    assert out[1]["key"] == "LOW" and out[1]["limited"] is True
    assert 0 < out[1]["qty"] < 50


def test_size_buy_signals_matches_funding_qty_and_allocation(cfg):
    """P5-1 0번: size_buy_signals는 funding_qty + allocate_remaining_limit을 합친 것과 같은 결과를 내야 한다
    (라이브 보고서·paper 가상 체결·백테스트가 이 함수 하나로 항상 같은 수량을 내게 하는 핵심 계약)."""
    c = _plan_cfg(cfg)
    fx = 1_300
    signals = [
        {"key": "HIGH-A1", "stage": "A1", "entry_price": 100.0, "stop_price": 94.0, "score": 50, "is_new_position": True},
        {"key": "LOW-A1", "stage": "A1", "entry_price": 100.0, "stop_price": 94.0, "score": 5, "is_new_position": True},
    ]
    out = sizing.size_buy_signals(signals, held=[], cfg=c, fx_rate=fx)
    assert out["funding_plan"] is not None
    high_expected = sizing.funding_qty("A1", 100.0, 94.0, fx, c)
    assert out["rows"]["HIGH-A1"]["qty"] == high_expected["qty"]
    assert out["rows"]["HIGH-A1"]["limited"] is False
    # 두 종목의 슬롯을 합치면 전략 한도(6,000만)를 넘지 않는지에 따라 LOW가 줄어들 수 있다.
    slot = sizing.slot_krw(c)
    strategy_limit = sizing.strategy_limit_krw(c)
    if strategy_limit >= slot * 2:
        assert out["rows"]["LOW-A1"]["qty"] == high_expected["qty"]
        assert out["rows"]["LOW-A1"]["limited"] is False
    else:
        assert out["rows"]["LOW-A1"]["limited"] is True


def test_size_buy_signals_accounts_for_held_and_reserved(cfg):
    """이미 보유 중인 종목(정찰 — 8/9 예약)이 있으면 신규 포지션에 쓸 남은 한도가 줄어든다."""
    c = _plan_cfg(cfg)
    c["account"]["total_krw"] = 10_000_000  # 작게 잡아 한도 압박을 쉽게 만든다
    fx = 1_300
    held = [{"ticker": "OLD", "qty": 10, "close": 50.0, "state_label": "정찰"}]
    signals = [{"key": "NEW-A1", "stage": "A1", "entry_price": 100.0, "stop_price": 94.0, "score": 10, "is_new_position": True}]
    out = sizing.size_buy_signals(signals, held, c, fx)
    fp = out["funding_plan"]
    assert fp["held_krw"] == pytest.approx(10 * 50.0 * fx)
    assert fp["reserved_krw"] == pytest.approx(sizing.slot_krw(c) * 8 / 9)


def test_size_buy_signals_existing_position_not_limited(cfg):
    """A2·A3(기존 포지션 추가 매수)는 남은 한도 배분을 거치지 않는다 — 처음 포지션이 열릴 때 이미 슬롯이 한도에 잡혀 있다."""
    c = _plan_cfg(cfg)
    c["account"]["total_krw"] = 1_000_000  # 남은 한도가 0에 가깝도록 아주 작게
    fx = 1_300
    held = [{"ticker": "OLD", "qty": 10, "close": 1000.0, "state_label": "정찰"}]  # 예약이 한도를 이미 넘김
    signals = [{"key": "OLD-A2", "stage": "A2", "entry_price": 100.0, "stop_price": 94.0, "score": 10, "is_new_position": False}]
    out = sizing.size_buy_signals(signals, held, c, fx)
    assert out["rows"]["OLD-A2"]["limited"] is False


def test_size_buy_signals_no_fx_returns_zero_qty_and_no_plan(cfg):
    c = _plan_cfg(cfg)
    signals = [{"key": "A-A1", "stage": "A1", "entry_price": 100.0, "stop_price": 94.0, "score": 10, "is_new_position": True}]
    out = sizing.size_buy_signals(signals, held=[], cfg=c, fx_rate=None)
    assert out["rows"]["A-A1"]["qty"] == 0
    assert out["funding_plan"] is None


def test_format_krw_examples():
    assert sizing.format_krw(730_000) == "73만"
    assert sizing.format_krw(123_000_000) == "1억 2,300만"
    assert sizing.format_krw(100_000_000) == "1억"
    assert sizing.format_krw(0) == "0원"
    assert sizing.format_krw(None) == "-"
    assert sizing.format_krw(-730_000) == "-73만"
