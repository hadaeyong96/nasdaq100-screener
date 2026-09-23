"""6장 수량 계산 테스트. "계산 예시" 표(갭 여유 0 설정 시 A1 37주, A2 49주, A3 148주)를
정확히 재현하는지 확인한다. 네트워크 없이, config.yaml을 그대로 읽어 돈다.
"""

from __future__ import annotations

import copy

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
