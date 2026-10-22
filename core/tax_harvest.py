"""12월 공제 안내 — 올해 세금 없는 매도 한도 (QQQM, 해외주식 양도소득 기본공제 250만 원).

순수 함수만 둔다(네트워크·파일·시트·현재 시각 접근 없음). 입력은 「체결」 탭을 data.fills로 파싱한 행
(date, ticker, side=buy/sell, qty, price=체결가($), fx_rate=환율(원/$))과 원/달러 종가 Series.

계산(이동평균법, 원화):
- 매수: 수량 += q, 원화 취득가 += q × 체결가 × 환율
- 매도: 주당 이동평균 취득가 = 원화 취득가 ÷ 수량. 실현 이익 = q × 매도가 × 그날 환율 − q × 주당 취득가,
  원화 취득가 −= q × 주당 취득가, 수량 −= q
- 환율 칸이 비면 그날(없으면 그 전 가장 가까운 날) 원/달러 종가를 쓴다. 수수료는 넣지 않는다.
- 남은 한도 = 250만 원 − 올해 실현 이익(0 미만이면 0), 한 주당 원화 이익 = 전날 QQQM 종가 × 전날 원/달러 − 주당 취득가,
  팔 수량 = 내림(남은 한도 ÷ 한 주당 이익). 한 주당 이익 ≤ 0이면 손실 상태(팔 필요 없음).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

DEDUCTION_KRW = 2_500_000
TICKER = "QQQM"


@dataclass
class Position:
    qty: float = 0.0
    cost_krw: float = 0.0
    realized_by_year: dict[int, float] = field(default_factory=dict)
    rows: int = 0  # 계산에 쓴 QQQM 행 수
    warnings: list[str] = field(default_factory=list)

    @property
    def avg_cost_krw(self) -> float:
        """주당 이동평균 취득가(원화). 보유 0이면 0."""
        return self.cost_krw / self.qty if self.qty > 0 else 0.0


def _fx_on(fx_close: pd.Series | None, day: pd.Timestamp) -> float | None:
    """그날(없으면 그 전 가장 가까운 날) 원/달러 종가. 없으면 None."""
    if fx_close is None or fx_close.empty:
        return None
    s = fx_close.dropna().sort_index()
    s = s[s.index <= day]
    return float(s.iloc[-1]) if len(s) else None


def moving_average_position(fills: pd.DataFrame, fx_close: pd.Series | None = None, ticker: str = TICKER) -> Position:
    """체결 행에서 ticker만 날짜 순(같은 날은 시트 순서)으로 이동평균 보유·연도별 실현 이익을 만든다.

    입력: fills(date, ticker, side, qty, price, fx_rate), fx_close(날짜 인덱스 원/달러 종가 — 빈 환율 칸 채우기용)
    출력: Position. 환율을 끝내 못 구한 행·보유보다 많이 판 행은 건너뛰고 warnings에 남긴다.
    """
    pos = Position()
    if fills is None or len(fills) == 0:
        return pos
    df = fills[fills["ticker"].astype(str).str.upper() == ticker].copy()
    if df.empty:
        return pos
    df["_d"] = pd.to_datetime(df["date"])
    df["_o"] = range(len(df))
    df = df.sort_values(["_d", "_o"])
    for _, r in df.iterrows():
        d = r["_d"]
        q, price = float(r["qty"]), float(r["price"])
        fx = pd.to_numeric(r.get("fx_rate"), errors="coerce")
        if pd.isna(fx) or fx <= 0:
            fx = _fx_on(fx_close, d)
        if fx is None:
            pos.warnings.append(f"{d.date()} {r['side']} 행의 환율을 구하지 못해 건너뜀")
            continue
        side = str(r["side"]).lower()
        if side in ("buy", "매수"):
            pos.qty += q
            pos.cost_krw += q * price * fx
        elif side in ("sell", "매도"):
            if q > pos.qty + 1e-9:
                pos.warnings.append(f"{d.date()} 매도 {q:g}주가 보유 {pos.qty:g}주보다 많아 건너뜀")
                continue
            avg = pos.avg_cost_krw
            gain = q * price * fx - q * avg
            pos.realized_by_year[d.year] = pos.realized_by_year.get(d.year, 0.0) + gain
            pos.cost_krw -= q * avg
            pos.qty -= q
            if pos.qty <= 1e-9:
                pos.qty, pos.cost_krw = 0.0, 0.0
        else:
            pos.warnings.append(f"{d.date()} 구분 {r['side']!r}을 몰라 건너뜀")
            continue
        pos.rows += 1
    return pos


@dataclass
class HarvestPlan:
    status: str  # "ok" | "loss" | "no_record" | "no_room"
    remaining_krw: float = 0.0
    realized_this_year_krw: float = 0.0
    per_share_gain_krw: float = 0.0
    sell_qty: int = 0
    expected_gain_krw: float = 0.0
    holding_qty: float = 0.0
    avg_cost_krw: float = 0.0


def harvest_plan(pos: Position, year: int, prev_close_usd: float, prev_fx: float,
                 deduction_krw: float = DEDUCTION_KRW) -> HarvestPlan:
    """올해 세금 없는 매도 한도 계산. 입력: pos(moving_average_position), year(올해), 전날 QQQM 종가($)·원/달러.

    출력: HarvestPlan — no_record(QQQM 보유 기록 없음), loss(한 주당 이익 ≤ 0), no_room(남은 한도 0 또는 1주도 못 팖), ok.
    """
    realized = pos.realized_by_year.get(year, 0.0)
    remaining = max(deduction_krw - realized, 0.0)
    if pos.rows == 0 or pos.qty <= 0:
        return HarvestPlan("no_record", remaining, realized)
    per_share = prev_close_usd * prev_fx - pos.avg_cost_krw
    base = dict(remaining_krw=remaining, realized_this_year_krw=realized, per_share_gain_krw=per_share,
                holding_qty=pos.qty, avg_cost_krw=pos.avg_cost_krw)
    if per_share <= 0:
        return HarvestPlan("loss", **base)
    n = min(math.floor(remaining / per_share), math.floor(pos.qty))
    if n <= 0:
        return HarvestPlan("no_room", **base)
    return HarvestPlan("ok", sell_qty=n, expected_gain_krw=n * per_share, **base)


def show_days(trading_days: pd.DatetimeIndex, year: int) -> list[pd.Timestamp]:
    """안내를 보이는 날: 그해 12월 첫 NYSE 거래일, 12월 15일 이후(15일 포함) 첫 거래일. 입력: NYSE 거래일 목록."""
    dec = pd.DatetimeIndex(trading_days)
    dec = dec[(dec.year == year) & (dec.month == 12)]
    out = []
    if len(dec):
        out.append(dec[0])
        mid = dec[dec.day >= 15]
        if len(mid) and mid[0] != dec[0]:
            out.append(mid[0])
    return out


def deadline(trading_days: pd.DatetimeIndex, year: int) -> pd.Timestamp:
    """체결 마감 = 그해 마지막 NYSE 거래일의 2거래일 전 날짜 (결제일이 그해 안에 들어오게)."""
    days = pd.DatetimeIndex(trading_days)
    days = days[days.year == year].sort_values()
    return days[-3]


def _won(x: float) -> str:
    return f"{x:,.0f}원"


def build_lines(plan: HarvestPlan, deadline_day: pd.Timestamp) -> list[str]:
    """접기 「▶ 올해 세금 없는 매도 한도」 본문 줄들(이모지 없음)."""
    dl = f"{deadline_day.month}/{deadline_day.day}"
    how = "팔고 같은 날 같은 수량 다시 사기, 달러로 매매(환전 안 함)"
    check = "증권사 앱의 예상 양도소득으로 250만 원 이하인지 꼭 확인. 다른 해외주식 매매가 있으면 그만큼 줄이기"
    if plan.status == "no_record":
        return [f"QQQM 일부를 팔았다 다시 사서 올해 250만 원 공제를 쓰는 방법: {how}",
                f"마감: {dl}까지 체결", check,
                "QQQM 모으기 탭에 보유 주식 수·평균단가를 넣으면 수량을 계산해 드려요"]  # 2026-10-22 새 출발: 체결 탭 → 보유현황
    lines = [f"남은 한도 {_won(plan.remaining_krw)} (올해 실현 이익 {_won(plan.realized_this_year_krw)})"]
    if plan.status == "loss":
        lines.append("올해는 팔 필요 없음(손실 상태)")
        return lines
    if plan.status == "no_room":
        lines.append("올해 한도를 이미 다 썼거나 1주 이익이 남은 한도보다 큼 — 팔 필요 없음")
        return lines
    lines += [f"팔 수량 약 {plan.sell_qty}주, 예상 실현 이익 {_won(plan.expected_gain_krw)} (주당 {_won(plan.per_share_gain_krw)})",
              how, f"마감: {dl}까지 체결", check]
    return lines
