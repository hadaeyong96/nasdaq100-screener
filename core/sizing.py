"""정리본 6장(포지션 크기: 2% 룰의 단계별 배분)의 수량 계산 순수 함수 모음.

수량 = 내림( 계좌 × 단계 위험 예산 ÷ 주당 위험 )
주당 위험 = (매수가 − 손절가) + 매수가 × 갭 여유
"""

from __future__ import annotations

import math

import pandas as pd

_BUDGET_KEY = {
    "A1": "a1_budget_pct",
    "A2": "a2_budget_pct",
    "A3": "a3_budget_pct",
    "B": "b_budget_pct",
}


def per_share_risk(entry_price: float, stop_price: float, cfg: dict) -> float:
    """주당 위험 = (매수가 − 손절가) + 매수가 × 갭 여유. 매수가의 최소 비율 미만이면 올린다."""
    risk_cfg = cfg["risk"]
    raw = (entry_price - stop_price) + entry_price * risk_cfg["gap_buffer_pct"] / 100
    min_risk = entry_price * risk_cfg["min_risk_per_share_pct"] / 100
    return max(raw, min_risk)

def budget_usd(stage: str, equity_usd: float, cfg: dict) -> float:
    """단계별 위험 예산(달러) = 계좌 × 단계 위험 예산 비율."""
    pct = cfg["risk"][_BUDGET_KEY[stage]]
    return equity_usd * pct / 100


def position_size(stage: str, entry_price: float, stop_price: float, equity_usd: float, cfg: dict) -> int:
    """단계별 추천 수량 = 내림(위험 예산 ÷ 주당 위험).

    입력값에 NaN이 있거나 주당 위험이 0 이하이면 0주.
    """
    if pd.isna(entry_price) or pd.isna(stop_price) or entry_price <= 0:
        return 0
    risk_per_share = per_share_risk(entry_price, stop_price, cfg)
    if risk_per_share <= 0:
        return 0
    budget = budget_usd(stage, equity_usd, cfg)
    return max(int(math.floor(budget / risk_per_share)), 0)


def cap_qty_by_position_limit(qty: int, entry_price: float, equity_usd: float, cfg: dict) -> int:
    """종목당 투입 금액 한도(계좌 대비 max_position_pct)를 넘지 않게 수량을 줄인다."""
    if qty <= 0 or entry_price <= 0:
        return 0
    max_value = equity_usd * cfg["risk"]["max_position_pct"] / 100
    max_qty_by_value = int(math.floor(max_value / entry_price))
    return max(min(qty, max_qty_by_value), 0)


def stop_price_a1_a2(df: pd.DataFrame, a1_date) -> float:
    """A1·A2 손절가: A1 발생일 기준 swing_low."""
    return float(df.loc[a1_date, "swing_low"])


def stop_price_a3(df: pd.DataFrame, a1_date, a3_date) -> float:
    """A3 손절가: max(A1 발생일 기준 swing_low, A3 확정일 구름 하단)."""
    a1_stop = stop_price_a1_a2(df, a1_date)
    cloud_bot = df.loc[a3_date, "cloud_bot"]
    if pd.isna(cloud_bot):
        return a1_stop
    return max(a1_stop, float(cloud_bot))


def stop_price_b(df: pd.DataFrame, entry_date) -> float:
    """B형 손절가: 진입일 기준 swing_low."""
    return float(df.loc[entry_date, "swing_low"])


# ── 자금 계획 (P3.6 6-2번) ────────────────────────────────────────────────
# 종목별 금액 입력칸 대신, 총자금(원)·전략 한도·슬롯에서 차수별 목표 금액을
# 자동으로 정한다. 위 position_size류(위험 예산 기준)는 그대로 두고, 여기서는
# "목표 금액 ÷ 환율 ÷ 지정가"로 구한 수량과 위험 상한 수량 중 작은 값을 쓴다.

STAGE_SLOT_FRACTION = {"A1": 1 / 9, "A2": 2 / 9, "A3": 6 / 9, "B": 1.0}
# 이 상태면 아직 안 산 남은 차수를 위해 슬롯의 이만큼을 예약해 둔다 (그 외 상태는 0).
_RESERVE_BY_STATE = {"정찰": 8 / 9, "확인": 6 / 9}


def slot_krw(cfg: dict) -> float:
    """종목당 슬롯(원) = 총자금 × 전략 한도 ÷ 최대 보유 종목 수."""
    plan = cfg["plan"]
    total_krw = cfg["account"]["total_krw"]
    return total_krw * plan["strategy_limit_pct"] / 100 / plan["max_slots"]


def strategy_limit_krw(cfg: dict) -> float:
    """전략 한도(원) = 총자금 × 전략 한도 비율."""
    return cfg["account"]["total_krw"] * cfg["plan"]["strategy_limit_pct"] / 100


def stage_target_krw(stage: str, cfg: dict) -> float:
    """차수별 목표 금액(원) = 슬롯 × 차수 비중(1차 1/9, 2차 2/9, 3차 6/9, 재진입 1)."""
    return slot_krw(cfg) * STAGE_SLOT_FRACTION[stage]


def reserved_fraction_for_state(state_label: str) -> float:
    """이 상태(정찰·확인)면 아직 안 산 차수를 위해 슬롯의 얼마를 예약해 둬야 하는지.

    1차만 보유 중(정찰)이면 8/9, 2차까지 보유 중(확인)이면 6/9. 그 외(확정·추세보유·
    청산중·대기)는 더 살 차수가 없거나 아직 아무것도 안 샀으므로 0.
    """
    return _RESERVE_BY_STATE.get(state_label, 0.0)


def funding_qty(stage: str, entry_price: float, stop_price: float | None, fx_rate: float | None, cfg: dict) -> dict:
    """자금 계획 기준 추천 수량. 목표 금액(슬롯 기준) 수량과 위험 상한 수량 중 작은 값.

    입력: stage(A1|A2|A3|B), entry_price(지정가, 달러), stop_price(손절가, 달러 또는
         None), fx_rate(원/달러 환율, 구하지 못했으면 None), cfg
    출력: {"qty", "target_qty"(목표 금액 기준 수량), "risk_cap_qty"(위험 상한 수량),
          "risk_capped"(위험 상한 때문에 줄었는지), "target_krw"(차수별 목표 금액, 원)}
    """
    target_krw = stage_target_krw(stage, cfg)
    empty = {"qty": 0, "target_qty": 0, "risk_cap_qty": 0, "risk_capped": False, "target_krw": target_krw}
    if stop_price is None or entry_price is None or entry_price <= 0 or not fx_rate or fx_rate <= 0:
        return empty
    total_usd = cfg["account"]["total_krw"] / fx_rate
    target_usd = target_krw / fx_rate
    target_qty = max(int(math.floor(target_usd / entry_price)), 0)
    risk_cap_qty = position_size(stage, entry_price, stop_price, total_usd, cfg)
    qty = min(target_qty, risk_cap_qty)
    return {
        "qty": qty,
        "target_qty": target_qty,
        "risk_cap_qty": risk_cap_qty,
        "risk_capped": risk_cap_qty < target_qty,
        "target_krw": target_krw,
    }


def allocate_remaining_limit(candidates: list[dict], remaining_krw: float, cfg: dict) -> list[dict]:
    """오늘 새로 생기는 포지션(A1·B, 신규 종목만) 후보를 점수 순으로 남은 한도에서 차감한다.

    이미 보유 중인 종목의 2·3차 매수(A2·A3)는 처음 포지션이 열릴 때 이미 슬롯
    전체(보유+예약)가 한도에 잡혀 있어 이 함수를 거치지 않는다 — engine이 신규
    종목(A1·B)만 candidates로 넘긴다.

    입력: candidates([{"key", "entry_price", "fx_rate", "target_qty", "risk_cap_qty"}, ...],
         점수 내림차순으로 이미 정렬됨), remaining_krw(남은 한도, 원), cfg
    출력: [{"key", "qty", "limited"(한도 부족으로 줄었는지), "consumed_krw"(이 종목이
          오늘 한도에서 새로 차지한 금액, 원 — 남은 한도 부족이면 남은 만큼만)}, ...]
          candidates와 같은 순서
    """
    slot = slot_krw(cfg)
    remaining = max(remaining_krw, 0.0)
    out = []
    for c in candidates:
        base_qty = min(c["target_qty"], c["risk_cap_qty"])
        if remaining >= slot:
            qty = base_qty
            limited = False
            consumed = slot
            remaining -= slot
        elif remaining <= 0:
            qty = 0
            limited = base_qty > 0
            consumed = 0.0
        else:
            fx_rate = c["fx_rate"]
            entry_price = c["entry_price"]
            reduced_qty = int(remaining / fx_rate // entry_price) if (fx_rate and entry_price) else 0
            qty = min(base_qty, max(reduced_qty, 0))
            limited = qty < base_qty
            consumed = remaining
            remaining = 0.0
        out.append({"key": c["key"], "qty": qty, "limited": limited, "consumed_krw": consumed})
    return out


def size_buy_signals(signals: list[dict], held: list[dict], cfg: dict, fx_rate: float | None) -> dict:
    """자금 계획(슬롯+위험상한+남은한도+예약) 기준으로 오늘 매수 신호들의 수량을 한 번에 정한다.

    라이브 보고서(engine.daily.build_report_summary)와 paper 가상 체결·백테스트
    (engine.daily.simulate_since, engine.backtest)가 모두 이 함수 하나로 항상 같은
    수량을 낸다 (P5-1 0번 — 라이브 보고서와 paper 가상 체결 수량이 갈리지 않게 한다).

    입력: signals([{"key"(고유 식별자), "stage"(A1|A2|A3|B), "entry_price", "stop_price"
         (없으면 그 신호는 수량 0), "score", "is_new_position"(신규 종목 진입이면 True —
         core.state는 대기 상태에서만 A1·B를 내므로 그 둘만 True)}, ...]),
         held([{"qty", "close"(없으면 시장가치 0으로 봄), "state_label"(core.state.STATES
         값 — 정찰·확인이면 예약을 잡는다)}, ...] — 오늘 신호가 나기 전, 현재 보유 중인
         신호 종목만. QQQM 등 신호 종목이 아닌 보유는 넣지 않는다), cfg, fx_rate(원/달러 —
         없으면 모든 신호 수량이 0이고 funding_plan은 None)
    출력: {
        "rows": {key: {"qty", "target_qty", "risk_cap_qty", "risk_capped"(위험 상한 때문에
               목표보다 줄었는지), "limited"(남은 한도 부족으로 더 줄었는지), "amount_krw",
               "max_loss_krw"}},
        "funding_plan": {strategy_limit_krw, slot_krw, held_krw, reserved_krw, new_krw
               (오늘 신규 포지션이 한도에서 새로 차지한 금액), remaining_krw} 또는
               fx_rate가 없으면 None,
    }
    """
    if not fx_rate:
        rows = {s["key"]: {"qty": 0, "target_qty": 0, "risk_cap_qty": 0, "risk_capped": False, "limited": False, "amount_krw": 0, "max_loss_krw": 0} for s in signals}
        return {"rows": rows, "funding_plan": None}

    slot = slot_krw(cfg)
    strategy_limit = strategy_limit_krw(cfg)

    held_krw = 0.0
    reserved_krw = 0.0
    for h in held:
        close = h.get("close")
        if close is not None:
            held_krw += h["qty"] * close * fx_rate
        reserved_krw += reserved_fraction_for_state(h.get("state_label", "")) * slot

    remaining_baseline = strategy_limit - held_krw - reserved_krw

    base_by_key = {}
    for s in signals:
        funding = funding_qty(s["stage"], s["entry_price"], s.get("stop_price"), fx_rate, cfg)
        base_by_key[s["key"]] = {
            **funding,
            "entry_price": s["entry_price"],
            "stop_price": s.get("stop_price"),
            "is_new_position": bool(s.get("is_new_position")),
            "score": s.get("score", 0),
        }

    new_candidates = sorted(
        (
            {"key": k, "entry_price": v["entry_price"], "fx_rate": fx_rate, "target_qty": v["target_qty"], "risk_cap_qty": v["risk_cap_qty"]}
            for k, v in base_by_key.items()
            if v["is_new_position"] and v["stop_price"] is not None
        ),
        key=lambda c: base_by_key[c["key"]]["score"],
        reverse=True,
    )
    allocations = {a["key"]: a for a in allocate_remaining_limit(new_candidates, remaining_baseline, cfg)}
    new_commit_krw = sum(a["consumed_krw"] for a in allocations.values())

    rows = {}
    for key, base in base_by_key.items():
        alloc = allocations.get(key)
        qty = alloc["qty"] if alloc is not None else base["qty"]
        limited = alloc["limited"] if alloc is not None else False
        risk_per_share = per_share_risk(base["entry_price"], base["stop_price"], cfg) if base["stop_price"] is not None else None
        amount_krw = round(qty * base["entry_price"] * fx_rate) if qty else 0
        max_loss_krw = round(qty * risk_per_share * fx_rate) if (qty and risk_per_share is not None) else 0
        rows[key] = {
            "qty": qty,
            "target_qty": base["target_qty"],
            "risk_cap_qty": base["risk_cap_qty"],
            "risk_capped": base["risk_capped"],
            "limited": limited,
            "amount_krw": amount_krw,
            "max_loss_krw": max_loss_krw,
        }

    remaining_final = max(strategy_limit - held_krw - reserved_krw - new_commit_krw, 0.0)
    funding_plan = {
        "strategy_limit_krw": strategy_limit,
        "slot_krw": slot,
        "held_krw": held_krw,
        "reserved_krw": reserved_krw,
        "new_krw": new_commit_krw,
        "remaining_krw": remaining_final,
    }
    return {"rows": rows, "funding_plan": funding_plan}


# ── 계획금액 기반 차수·수량 (라이브 어드바이저 1단계, docs/design/live_advisor.md 2번) ──
# 계좌 총액(cfg["account"]["total_krw"])과 완전히 분리된 계산이다. 위 slot_krw·
# funding_qty류(계좌 총액 기반, paper·P5·KJB가 계속 씀)는 그대로 두고 새 함수만
# 추가한다. 위험 예산(2% 룰) 기반 상한은 "자금 배분 로직"의 일부로 보고 여기서는
# 뺀다 — 계획금액 자체가 사용자가 이미 정한 위험 한도라고 본다.


def plan_tranche_krw(budget_krw: float, stage: str) -> float:
    """계획금액 기준 차수별 목표 금액(원) = 계획금액 × 차수 비중(1차 1/9, 2차 2/9,
    3차 6/9, 재진입 1)."""
    return budget_krw * STAGE_SLOT_FRACTION[stage]


def _plan_stage_qty_from_total(total_shares: int, stage: str) -> int:
    """기준가 기반 계획 총 주수를 1:2:6 비중으로 나눈다 (구글 시트 계획 탭 개편,
    2026-09-30 — 시트의 1차~남은(주) 수식과 같은 결과를 낸다).

    1차 = MAX(1, ROUND(총 주수/9)), 2차 = ROUND(총 주수×2/9), 3차 = 총 주수 − 1차 − 2차
    (나머지로 계산해 반올림 오차를 흡수, 셋을 더하면 항상 총 주수). 재진입(B)은
    한 번에 총 주수 전체(재진입 비중 1.0과 같다).
    """
    a1 = max(1, round(total_shares / 9))
    if stage == "A1":
        return a1
    a2 = round(total_shares * 2 / 9)
    if stage == "A2":
        return a2
    if stage == "A3":
        return total_shares - a1 - a2
    return total_shares  # B(재진입)


def plan_tranche_qty(
    budget_krw: float,
    stage: str,
    entry_price: float | None,
    fx_rate: float | None,
    ref_price: float | None = None,
) -> dict:
    """계획금액 기준 차수별 추천 수량.

    ref_price(계획 시트 "기준가($)")가 있으면 고정 총 주수 = INT(계획금액 ÷ (기준가 ×
    환율))를 _plan_stage_qty_from_total로 1:2:6 비중 나눈 값을 쓴다(구글 시트 수식과
    맞추기 위한 방식, 2026-09-30). 없으면 기존 방식 그대로 위험 상한 없이
    수량 = 내림(차수별 목표 금액 ÷ 환율 ÷ 오늘 지정가)를 쓴다.

    입력: budget_krw(계획 시트의 계획금액, 원), stage(A1|A2|A3|B), entry_price(오늘
         지정가, 달러 — ref_price 없을 때만 쓴다), fx_rate(원/달러 환율 — 없거나
         0 이하면 qty=0), ref_price(계획 시트 기준가, 달러 — 없으면 기존 방식)
    출력: {"tranche_krw"(차수별 목표 금액, 원), "tranche_usd"(달러, 계산 불가면 None),
          "qty"(정수 주식 수), "total_shares"(기준가 방식일 때 계획 총 주수, 아니면 None)}
    """
    tranche_krw = plan_tranche_krw(budget_krw, stage)
    if not fx_rate or fx_rate <= 0:
        return {"tranche_krw": tranche_krw, "tranche_usd": None, "qty": 0, "total_shares": None}
    tranche_usd = tranche_krw / fx_rate

    if ref_price is not None and not pd.isna(ref_price) and ref_price > 0:
        total_shares = int(budget_krw / (ref_price * fx_rate))
        qty = max(_plan_stage_qty_from_total(total_shares, stage), 0)
        return {"tranche_krw": tranche_krw, "tranche_usd": tranche_usd, "qty": qty, "total_shares": total_shares}

    if entry_price is None or pd.isna(entry_price) or entry_price <= 0:
        return {"tranche_krw": tranche_krw, "tranche_usd": tranche_usd, "qty": 0, "total_shares": None}
    qty = max(int(math.floor(tranche_usd / entry_price)), 0)
    return {"tranche_krw": tranche_krw, "tranche_usd": tranche_usd, "qty": qty, "total_shares": None}


def clamp_plan_qty_to_remaining(qty: int, total_shares: int | None, held_qty: int) -> dict:
    """기준가 기반 계획(total_shares가 있을 때만)에서, 추천 수량이 남은 주수
    (총 주수 − 현재 보유 수량)를 넘지 않게 자른다. 기준가가 없으면(total_shares가
    None — 지금 방식) 그대로 둔다.

    입력: qty(plan_tranche_qty가 낸 원래 추천 수량), total_shares(plan_tranche_qty
         결과의 "total_shares"), held_qty(이 종목의 현재 보유 수량 합)
    출력: {"qty"(잘린 수량), "over_limit"(이미 한도를 넘었으면 True — 남은 주수 <= 0)}
    """
    if total_shares is None:
        return {"qty": qty, "over_limit": False}
    remaining = total_shares - held_qty
    if remaining <= 0:
        return {"qty": 0, "over_limit": True}
    return {"qty": min(qty, remaining), "over_limit": False}


def min_budget_for_one_share_krw(entry_price: float, fx_rate: float, stage: str = "A1") -> float:
    """이 차수에서 1주라도 사려면 필요한 최소 계획금액(원) — 계획 경고용
    (docs/design/live_advisor.md 7번: "1차 금액으로 1주도 못 사면 경고").

    입력: entry_price(지정가, 달러), fx_rate(원/달러 환율), stage(기본 A1)
    출력: 최소 계획금액(원) = 1주 금액(원) ÷ 그 차수 비중
    """
    one_share_krw = entry_price * fx_rate
    return one_share_krw / STAGE_SLOT_FRACTION[stage]


_KRW_UNIT = 10_000


def format_krw(amount: float | None) -> str:
    """금액(원)을 "73만"·"1억 2,300만" 형식으로 바꾼다 (만 단위로 반올림, P3.6 6-3번).

    입력: amount(원) 또는 None
    출력: 문자열. None이면 "-". 반올림 결과가 0이면 "0원".
    """
    if amount is None:
        return "-"
    sign = "-" if amount < 0 else ""
    man_total = round(abs(amount) / _KRW_UNIT)
    if man_total == 0:
        return "0원"
    eok, man = divmod(man_total, 10_000)
    if eok and man:
        return f"{sign}{eok}억 {man:,}만"
    if eok:
        return f"{sign}{eok}억"
    return f"{sign}{man:,}만"
