"""정리본 8장(상태 전이)을 그대로 구현한다.

- 상태는 dict로 받고 새 dict를 반환한다 (원본 변경 금지).
- 하루 처리 순서: ① 손절 ② A1 만료 ③ E3 ④ E1·E2 ⑤ 진입 신호.
  손절은 모든 신호보다 우선하고, 같은 날 손절이 나면 나머지는 무시한다.
- 이미 판 묶음의 E 신호는 무시하고, 청산중에는 추가 매수를 하지 않으며,
  대기로 돌아온 종목은 재진입 대기(cooldown)가 지나야 새 A1을 받는다.
- 중복 알림 방지 키(sent_alerts, 예: "A1:2026-09-10")로 같은 날 두 번 실행해도
  이벤트가 한 번만 기록되게 한다.

P2.1 보완: 매매 금지 구간(core/filters.py의 ban_reasons)에 걸리면 **상태 전이가
일어나지 않는다** — "BLOCKED" 이벤트만 남기고 이전 상태를 유지한다. 동시 보유
8개 한도는 종목 간 비교가 필요해 이 모듈 혼자 판단할 수 없으므로, engine이
`preview_new_entry`로 오늘 새 진입(A1·B) 후보를 모두 모아 점수 순으로 추려낸
뒤 `process_day(..., new_entry_allowed=...)`로 허용 여부를 넘겨준다. 종목당
투입 25% 한도는 수량 계산(core/sizing.py)에서 처리하므로 상태 전이를 막지
않는다.
"""

from __future__ import annotations

import copy

import pandas as pd

from core import filters
from core import signals as sig

STATES = ("대기", "정찰", "확인", "확정", "추세보유", "청산중")

# A형 묶음 라벨(1:2:6). B형은 전체를 "9"로 간주해 같은 순서(1→2→6)로 나눠 판다.
_UNIT_1, _UNIT_2, _UNIT_6, _UNIT_B = "1", "2", "6", "9"
_B_WEIGHTS = {"1": 1, "2": 2, "6": 6}  # B형 잔량을 9분의 1/2/6 비율로 나눠 팔 때 쓴다


def init_state(ticker: str, name_kr: str = "") -> dict:
    """새 종목의 초기 상태(대기)를 만든다."""
    return {
        "ticker": ticker,
        "name_kr": name_kr,
        "state": "대기",
        "units": {},
        "entries": {},
        "stop": None,
        "a1_date": None,
        "cooldown_until": None,
        "sent_alerts": [],
        "updated_at": None,
        # 8장 표에는 없지만 B형 손절가·비례 청산 계산, A2·A3 등급 표시에 필요한 내부 보조 필드.
        "b_entry_date": None,
        "b_total_qty": None,
        "grade": None,
    }


def _trading_days_between(df: pd.DataFrame, start_date, end_date) -> int:
    """df 안에서 start_date부터 end_date까지의 거래일 오프셋(end - start)."""
    return df.index.get_loc(end_date) - df.index.get_loc(start_date)


def _add_cooldown(date, cooldown_days: int) -> pd.Timestamp:
    """대기 복귀 후 재진입 대기 종료일. 영업일(주말 제외) 기준 근사치."""
    return (pd.Timestamp(date) + pd.tseries.offsets.BDay(cooldown_days)).normalize()


def _in_cooldown(state: dict, date) -> bool:
    if not state.get("cooldown_until"):
        return False
    return pd.Timestamp(date) < pd.Timestamp(state["cooldown_until"])


def _sent(state: dict, key: str) -> bool:
    return key in state["sent_alerts"]


def _mark_sent(state: dict, key: str) -> None:
    state["sent_alerts"].append(key)


def _held_units(state: dict) -> dict:
    return {u: q for u, q in state["units"].items() if q and q > 0}


def _prev_close(df: pd.DataFrame, idx: int) -> float:
    return float(df.iloc[idx - 1]["close"]) if idx > 0 else float("nan")


def _ban_reasons(df: pd.DataFrame, date, row: pd.Series, stage: str, cfg: dict, earnings_date) -> list[str]:
    """core/filters.py의 매매 금지 구간(7장)을 이 종목·날짜·단계에 대해 확인한다."""
    idx = df.index.get_loc(date)
    gc_count = filters.macd_cross_count(df, date) if stage in ("A2", "B") else 0
    return filters.ban_reasons(
        stage=stage, row=row, gc_count_20d=gc_count, prev_close=_prev_close(df, idx), cfg=cfg, earnings_date=earnings_date
    )


def _entry_score(df: pd.DataFrame, date, row: pd.Series, kind: str, cfg: dict) -> tuple[str | None, int]:
    """진입 이벤트의 등급(A2·B형만)과 우선순위 점수(7장)를 계산한다."""
    grade_letter = None
    if kind in ("A2", "B"):
        gc_count = filters.macd_cross_count(df, date)
        grade_letter = filters.grade(row.get("macd_norm"), gc_count, cfg)
    thickness = filters.cloud_thickness_pct(row.get("cloud_top"), row.get("cloud_bot"), row.get("close"))
    score = filters.priority_score(grade_letter, row.get("vol_ratio"), thickness, row.get("bb_width_pct"), cfg)
    return grade_letter, score


def _detect_new_entry(df: pd.DataFrame, date, state: dict, cfg: dict) -> tuple[str, str] | tuple[None, None]:
    """대기 상태에서 오늘 A1 또는 B 조건이 성립하는지만 본다 (금지·한도는 보지 않는다)."""
    if state.get("state") != "대기":
        return None, None
    idx = df.index.get_loc(date)
    row = df.loc[date]
    prev_row = df.iloc[idx - 1] if idx > 0 else None
    prev_rsi = prev_row["rsi"] if prev_row is not None else float("nan")

    if not _in_cooldown(state, date) and sig.check_a1(prev_rsi, row["rsi"]):
        return "A1", _UNIT_1
    if sig.check_b(row, cfg):
        return "B", _UNIT_B
    return None, None


def preview_new_entry(df: pd.DataFrame, date, state: dict, cfg: dict, earnings_date=None) -> dict | None:
    """오늘 새 진입(A1·B) 후보가 있는지 미리 본다 (상태를 바꾸지 않는다).

    동시 보유 8개 한도는 종목을 가로질러 점수를 비교해야 해서 이 함수 하나로는
    판단할 수 없다. engine이 모든 종목의 이 결과를 모아 점수 순으로 추려
    admitted 종목만 `process_day(..., new_entry_allowed=True)`로 넘긴다.
    매매 금지(ban)에 걸리는 신호는 한도 계산에서도 제외한다(어차피 발생하지 않음).

    입력: df, date, state, cfg, earnings_date(있으면 실적 필터도 확인)
    출력: {"kind": "A1"|"B", "unit":..., "price":..., "score":...} 또는 None
    """
    if date not in df.index:
        return None
    kind, unit = _detect_new_entry(df, date, state, cfg)
    if kind is None:
        return None
    row = df.loc[date]
    if _ban_reasons(df, date, row, kind, cfg, earnings_date):
        return None
    _, score = _entry_score(df, date, row, kind, cfg)
    return {"kind": kind, "unit": unit, "price": float(row["close"]), "score": score}


def _liquidate_all(state: dict, date, reason: str, events: list, cooldown_days: int) -> None:
    """전량 매도(손절·E3)로 대기 상태로 되돌린다."""
    for unit, qty in _held_units(state).items():
        events.append(
            {
                "date": date,
                "kind": reason,
                "unit": unit,
                "qty": qty,
                "entry_price": state["entries"].get(unit),
            }
        )
    state["units"] = {}
    state["entries"] = {}
    state["stop"] = None
    state["a1_date"] = None
    state["b_entry_date"] = None
    state["b_total_qty"] = None
    state["grade"] = None
    state["state"] = "대기"
    state["cooldown_until"] = _add_cooldown(date, cooldown_days)


def _record_blocked(new_state: dict, date, kind: str, unit: str, reasons: list[str], blocked_type: str, score: int, events: list) -> None:
    """매매 금지·한도 초과로 막힌 진입을 기록한다. 상태는 건드리지 않는다."""
    key = f"{kind}:{pd.Timestamp(date).date()}"
    if _sent(new_state, key):
        return
    events.append(
        {
            "date": date,
            "kind": "BLOCKED",
            "stage": kind,
            "unit": unit,
            "reasons": reasons,
            "blocked_type": blocked_type,  # "ban" | "limit"
            "score": score,
        }
    )
    _mark_sent(new_state, key)


def process_day(
    df: pd.DataFrame,
    date,
    state: dict,
    cfg: dict,
    *,
    earnings_date=None,
    new_entry_allowed: bool = True,
) -> tuple[list[dict], dict]:
    """하루치 상태 전이를 처리한다.

    입력: df(지표가 계산된 DataFrame), date(df.index의 값), state(종목 상태 dict),
         cfg(config.yaml 로드값), earnings_date(다음 실적 발표일, 모르면 None),
         new_entry_allowed(오늘 이 종목이 새 포지션(A1·B)을 열어도 되는지 —
         동시 보유 8개 한도를 engine이 점수 순으로 정해 넘겨준다. A2·A3 추가
         매수는 이미 보유 중인 종목이라 이 값의 영향을 받지 않는다)
    출력: (그날 발생한 이벤트 목록, 새 상태 dict). state는 변경하지 않는다.
    이벤트 dict: {date, kind, unit, qty 또는 price, ...}
      kind: "A1"|"A2"|"A3"|"B"(진입, price 포함) | "STOP"|"A1_EXPIRE"|"E3"|"E1"|"E2"(매도, qty 포함)
           | "BLOCKED"(매매 금지·한도 초과로 막힘, stage·reasons·blocked_type 포함)
    """
    if date not in df.index:
        raise ValueError(f"{date}가 df 인덱스에 없습니다")

    new_state = copy.deepcopy(state)
    events: list[dict] = []
    row = df.loc[date]
    idx = df.index.get_loc(date)
    prev_row = df.iloc[idx - 1] if idx > 0 else None
    cooldown_days = cfg["assumptions"]["reentry_cooldown_days"]

    # ① 손절 (최우선 — 발생하면 나머지 신호는 모두 무시)
    if new_state["state"] != "대기" and sig.check_stop(row["close"], new_state.get("stop")):
        key = f"STOP:{pd.Timestamp(date).date()}"
        if not _sent(new_state, key):
            _liquidate_all(new_state, date, "STOP", events, cooldown_days)
            _mark_sent(new_state, key)
        new_state["updated_at"] = date
        return events, new_state

    # ② A1 만료 (정찰 단계에서만 의미가 있다)
    if new_state["state"] == "정찰" and new_state.get("a1_date") is not None:
        bars_since = _trading_days_between(df, new_state["a1_date"], date)
        expiry_days = cfg["assumptions"]["a1_to_a2_expiry_days"]
        if sig.check_a1_expired(bars_since, expiry_days):
            key = f"A1_EXPIRE:{pd.Timestamp(date).date()}"
            if not _sent(new_state, key):
                qty1 = new_state["units"].get(_UNIT_1, 0)
                if qty1:
                    events.append(
                        {
                            "date": date,
                            "kind": "A1_EXPIRE",
                            "unit": _UNIT_1,
                            "qty": qty1,
                            "entry_price": new_state["entries"].get(_UNIT_1),
                        }
                    )
                new_state["units"] = {}
                new_state["entries"] = {}
                new_state["stop"] = None
                new_state["a1_date"] = None
                new_state["state"] = "대기"
                new_state["cooldown_until"] = _add_cooldown(date, cooldown_days)
                _mark_sent(new_state, key)

    # ③ E3 구조 붕괴 (확정·추세보유·청산중에서만 의미가 있다) — 전량 매도
    if new_state["state"] in ("확정", "추세보유", "청산중") and sig.check_e3(row):
        key = f"E3:{pd.Timestamp(date).date()}"
        if not _sent(new_state, key):
            _liquidate_all(new_state, date, "E3", events, cooldown_days)
            _mark_sent(new_state, key)

    # ④ E1·E2 (확인·확정·추세보유·청산중) — 묶음별 부분 매도, 청산중으로 전환
    if new_state["state"] in ("확인", "확정", "추세보유", "청산중"):
        sold_any = False

        if sig.check_e1(row):
            key = f"E1:{pd.Timestamp(date).date()}"
            if not _sent(new_state, key):
                sold_any = _sell_bundle(new_state, date, "1", "E1", events) or sold_any
                _mark_sent(new_state, key)

        prev_rsi = prev_row["rsi"] if prev_row is not None else float("nan")
        if sig.check_e2(prev_rsi, row["rsi"]):
            key = f"E2:{pd.Timestamp(date).date()}"
            if not _sent(new_state, key):
                sold_any = _sell_bundle(new_state, date, "2", "E2", events) or sold_any
                _mark_sent(new_state, key)

        if sold_any and new_state["state"] != "대기":
            new_state["state"] = "청산중"

        # 잔량이 0이 되면(모든 묶음을 다 팔았으면) 대기로 복귀
        if new_state["state"] == "청산중" and not _held_units(new_state):
            new_state["units"] = {}
            new_state["entries"] = {}
            new_state["stop"] = None
            new_state["a1_date"] = None
            new_state["b_entry_date"] = None
            new_state["b_total_qty"] = None
            new_state["grade"] = None
            new_state["state"] = "대기"
            new_state["cooldown_until"] = _add_cooldown(date, cooldown_days)

    # ⑤ 진입 신호 (대기·정찰·확인 단계에서만 새 묶음을 받는다. 청산중은 추가 매수 없음)
    st = new_state["state"]

    if st == "대기":
        kind, unit = _detect_new_entry(df, date, new_state, cfg)
        if kind is not None:
            key = f"{kind}:{pd.Timestamp(date).date()}"
            if not _sent(new_state, key):
                grade_letter, score = _entry_score(df, date, row, kind, cfg)
                reasons = _ban_reasons(df, date, row, kind, cfg, earnings_date)
                if reasons:
                    _record_blocked(new_state, date, kind, unit, reasons, "ban", score, events)
                elif not new_entry_allowed:
                    _record_blocked(new_state, date, kind, unit, ["동시 보유 종목 수 한도 초과"], "limit", score, events)
                else:
                    events.append({"date": date, "kind": kind, "unit": unit, "price": float(row["close"]), "score": score})
                    new_state["state"] = "정찰" if kind == "A1" else "추세보유"
                    new_state["entries"][unit] = None  # engine이 지정가·수량을 채운다
                    new_state["units"].setdefault(unit, 0)
                    if not pd.isna(row.get("swing_low")):  # 진입일 기준 swing_low (6장)
                        new_state["stop"] = float(row["swing_low"])
                    if kind == "A1":
                        new_state["a1_date"] = date
                    else:
                        new_state["b_entry_date"] = date
                    _mark_sent(new_state, key)

    elif st == "정찰":
        # A2 판정은 당일 RSI 기준이다: A1 이후 중간에 RSI가 30 밑으로 다시 내려간
        # 날이 있어도 A1을 무효로 하지 않는다(손절 규칙이 관리) — P2.1 보완 2번.
        if sig.check_a2(row):
            key = f"A2:{pd.Timestamp(date).date()}"
            if not _sent(new_state, key):
                grade_letter, score = _entry_score(df, date, row, "A2", cfg)
                reasons = _ban_reasons(df, date, row, "A2", cfg, earnings_date)
                if reasons:
                    # 정찰 상태를 유지한다 — A1 유효기간 안에 다음 골든크로스가 오면 다시 판정.
                    _record_blocked(new_state, date, "A2", _UNIT_2, reasons, "ban", score, events)
                else:
                    events.append(
                        {"date": date, "kind": "A2", "unit": _UNIT_2, "price": float(row["close"]), "score": score, "grade": grade_letter}
                    )
                    new_state["state"] = "확인"
                    new_state["entries"][_UNIT_2] = None
                    new_state["units"].setdefault(_UNIT_2, 0)
                    new_state["grade"] = grade_letter
                    _mark_sent(new_state, key)

    elif st == "확인":
        if sig.check_a3(row, cfg):
            key = f"A3:{pd.Timestamp(date).date()}"
            if not _sent(new_state, key):
                grade_letter = new_state.get("grade")
                _, score = _entry_score(df, date, row, "A3", cfg)
                reasons = _ban_reasons(df, date, row, "A3", cfg, earnings_date)
                if reasons:
                    # 확인 상태를 유지한다 — 다음 날 다시 판정한다.
                    _record_blocked(new_state, date, "A3", _UNIT_6, reasons, "ban", score, events)
                else:
                    events.append(
                        {"date": date, "kind": "A3", "unit": _UNIT_6, "price": float(row["close"]), "score": score, "grade": grade_letter}
                    )
                    new_state["state"] = "확정"
                    new_state["entries"][_UNIT_6] = None
                    new_state["units"].setdefault(_UNIT_6, 0)
                    # A3 확정 시 손절선을 max(A1 기준 swing_low, 오늘 구름 하단)으로 올린다 (6장).
                    if not pd.isna(row.get("cloud_bot")) and new_state.get("stop") is not None:
                        new_state["stop"] = max(new_state["stop"], float(row["cloud_bot"]))
                    _mark_sent(new_state, key)

    new_state["updated_at"] = date
    return events, new_state


def _sell_bundle(state: dict, date, unit: str, kind: str, events: list) -> bool:
    """묶음 unit을 판다. B형("9")이면 1/2/6 비율로 잔량을 나눠 판다.

    이미 판(수량 0) 묶음의 신호는 무시한다. 실제로 팔았으면 True.
    """
    if _UNIT_B in state["units"]:
        return _sell_b_bundle(state, date, unit, kind, events)

    qty = state["units"].get(unit, 0)
    if not qty:
        return False
    events.append({"date": date, "kind": kind, "unit": unit, "qty": qty, "entry_price": state["entries"].get(unit)})
    state["units"][unit] = 0
    return True


def _sell_b_bundle(state: dict, date, unit: str, kind: str, events: list) -> bool:
    """B형(단일 묶음 "9")을 E1/E2 비율(1/9, 2/9)만큼 잘라 판다 (5장 "B형은 9단위로 간주")."""
    remaining = state["units"].get(_UNIT_B, 0)
    total = state.get("b_total_qty") or remaining
    if not remaining:
        return False
    sold_key = f"_b_sold_{unit}"
    if state.get(sold_key):  # 이미 이 비율만큼 판 적이 있으면 무시
        return False
    target = min(int(total * _B_WEIGHTS[unit] / 9), remaining)
    if target <= 0:
        return False
    events.append({"date": date, "kind": kind, "unit": unit, "qty": target, "entry_price": state["entries"].get(_UNIT_B)})
    state["units"][_UNIT_B] = remaining - target
    state[sold_key] = True
    return True


def apply_fill(state: dict, fill: dict, cfg: dict) -> dict:
    """체결 기록(data/fills.csv 한 행, 또는 되돌려 보기의 가상 체결)을 상태에 반영한다.

    입력: state, fill({date, ticker, unit, side, price, qty}), cfg
    출력: 새 상태 dict.
    - side="buy"이고 qty>0: 그 묶음의 실제 체결가·수량으로 덮어쓴다.
    - side="buy"이고 qty=0: 체결 안 됨. 그 묶음을 보유하지 않은 것으로 되돌린다.
      정찰 단계에서 A1이 체결 안 됨이면 쿨다운 없이 대기로 돌린다.
    - side="sell": 그 묶음의 보유 수량을 체결 수량만큼 줄인다.
    """
    new_state = copy.deepcopy(state)
    unit = str(fill["unit"])

    if fill["side"] == "buy":
        if fill["qty"] and fill["qty"] > 0:
            new_state["units"][unit] = int(fill["qty"])
            new_state["entries"][unit] = float(fill["price"])
            if unit == _UNIT_B:
                new_state["b_total_qty"] = int(fill["qty"])
        else:
            new_state["units"].pop(unit, None)
            new_state["entries"].pop(unit, None)
            if unit == _UNIT_1 and new_state["state"] == "정찰":
                new_state["state"] = "대기"
                new_state["a1_date"] = None
                new_state["stop"] = None
                new_state["cooldown_until"] = None  # 체결 안 됨은 쿨다운 없이 즉시 재도전 가능
            elif unit == _UNIT_B and new_state["state"] == "추세보유":
                new_state["state"] = "대기"
                new_state["b_entry_date"] = None
                new_state["stop"] = None
                new_state["cooldown_until"] = None
    elif fill["side"] == "sell":
        held = new_state["units"].get(unit, 0)
        new_state["units"][unit] = max(held - int(fill["qty"]), 0)

    return new_state
