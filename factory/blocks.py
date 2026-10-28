"""전략공장 블록 목록 (2026-10-28 처음 만듦 — 지금은 조건 블록 market_temperature 하나).

블록은 조립 탐색·사전 등록 시험용 부품이다. 지금 돌아가는 가상 장부(puddle_v32·puddle_rebound_v2 등)에는
연결하지 않는다 — 매매 조건으로 쓰려면 사전 등록 후 새 장부로.
"""

from __future__ import annotations


def market_temperature(date, temp_min: float, temp_max: float, prices: dict | None = None, cfg: dict | None = None) -> bool:
    """조건 블록: 그날 시장 온도(thermometer.core.get_temperature)가 temp_min 이상 temp_max 이하면 True.
    온도를 구할 수 없으면(자료 없음) False."""
    from thermometer.core import get_temperature

    t = get_temperature(date, prices=prices, cfg=cfg)["temp"]
    return t is not None and float(temp_min) <= t <= float(temp_max)


# id → {kind(condition|…), params(매개변수 이름), fn, desc}
BLOCKS: dict[str, dict] = {
    "market_temperature": {
        "kind": "condition",
        "params": ["temp_min", "temp_max"],
        "fn": market_temperature,
        "desc": "시장 온도계(0~100도)가 [temp_min, temp_max] 안이면 참",
    },
}


def get_block(block_id: str) -> dict:
    """블록 id → 블록 정보. 없으면 KeyError."""
    return BLOCKS[block_id]
