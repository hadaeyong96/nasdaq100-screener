"""계좌·납입 구조 연구(P6-3, docs/design/p6_3_plan.md) 시뮬레이션 (순수 함수).

네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). pandas를 쓰지 않는다 — 입력은
전부 날짜순 리스트·딕셔너리(스크립트 쪽에서 engine.portfolio.prepare_data의 DataFrame을
이 형태로 변환해 넘긴다). 세법·비용·규칙 수치는 전부 cfg(dict, configs/
p6_3_preregistration.yaml 로드값)에서 읽는다 — 하드코딩하지 않는다.

**핵심 단순화**: ISA·연금저축 보유분은 종목별 취득가 원장이 필요 없다(ISA는 해지 때
"순이익(평가액-총납입액)"만, 연금은 "잔액"만 과세 대상이라 부분매도 손익을 따질 일이
없다) — 그래서 이 둘은 core.account_tax의 원장(Ledger) 없이 그냥 KRW 평가액(float)로
추적한다. 해외계좌(QQQM)만 실제 취득가 원장(MovingAverageLedger 또는 FifoLedger)을 쓴다
(연말 공제소진 매도·부분 스윕이 실현손익에 의존하므로).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from core import account_tax as at

MONTHS_PER_YEAR = 12


def net_of_commission(gross_amount: float, commission_pct: float) -> float:
    """고정 예산(gross_amount, 수수료 포함 총 지출)으로 실제 투자되는 금액 (순수 함수).

    예: 100만 원으로 매수하는데 수수료 0.07%면, 실제 주식에 들어가는 돈은
    100만 / 1.0007 ≈ 999,300원이고 나머지가 수수료로 나간다. 매도 비용은 반대로
    core.account_tax.MovingAverageLedger.sell/FifoLedger.sell이 받는 돈에서 직접 뗀다.
    """
    if gross_amount <= 0:
        return 0.0
    return gross_amount / (1 + commission_pct / 100)


class SealViolationError(RuntimeError):
    """요청한 구간이 사전 등록된 봉인 경계(seal_end)를 넘을 때 낸다."""


def enforce_window_not_sealed(window_end: date, seal_end: date) -> None:
    """window_end가 seal_end를 넘으면 SealViolationError (순수 함수, P6-3 전용 봉인 확인).

    core/seal.py(과거 검증용, 다른 설계)와는 별개 — P6-3은 자체적으로 이 가드를 쓴다.
    """
    if window_end > seal_end:
        raise SealViolationError(
            f"요청한 구간 끝({window_end.isoformat()})이 봉인 기준일({seal_end.isoformat()})을 넘습니다."
        )


# ── 월별 가격·배당·환율 시계열 (일별 데이터 -> 월별 요약, 스크립트가 호출) ─────────


@dataclass
class MonthStep:
    """시뮬레이션의 한 달(= 한 번의 납입 시점) 요약 자료."""

    month_index: int  # 0부터, 0이 시작월(첫 납입)
    contribution_date: date
    qqq_price: float
    core_price: float
    fx: float
    qqq_div_yield_since_prev: float  # 직전 시점 대비 QQQ 배당수익률(국내 ETF 재투자용)
    core_div_per_share_since_prev: float  # 직전 시점 대비 QQQM 주당 배당(해외계좌용)
    domestic_fee_factor_since_prev: float  # 국내 ETF 연보수를 그 기간만큼 반영한 배수(<=1)
    reserve_factor_since_prev: float  # 그 기간 동안의 DTB3 복리 배수(>=1)
    is_year_end: bool  # 그 해의 마지막 달(해외계좌 연말 공제소진 매도 시점, K1)
    is_year_start: bool  # 그 해의 첫 달(해외->ISA 스윕 시점, K2b)
    calendar_year: int


def build_month_steps(
    trading_days: list[date],
    qqq_close: dict[date, float],
    core_close: dict[date, float],
    qqq_dividends: dict[date, float],
    core_dividends: dict[date, float],
    fx_by_date: dict[date, float],
    reserve_daily_rate: dict[date, float],
    window_start: date,
    window_end: date,
    domestic_fee_pct: float,
) -> list[MonthStep]:
    """일별 시세·배당·환율·DTB3 일율을 월별(매달 첫 거래일 기준) 요약으로 바꾼다 (순수 함수).

    입력: trading_days(window_start~window_end 사이 거래일, 오름차순), 각종 {날짜: 값}
         딕셔너리(스크립트가 DataFrame에서 변환), domestic_fee_pct(국내 ETF 연보수, %)
    출력: MonthStep 리스트, month_index 0(시작월)부터 horizon 끝 달 "하나 더"(평가 전용,
         contribution_date는 있지만 그 달엔 납입하지 않고 평가만 함 — 마지막 원소가 그것).
         각 원소의 *_since_prev 필드는 "직전 원소의 contribution_date(이전 달)"부터
         "이 원소의 contribution_date"까지(그 날 제외, 전날까지) 누적한 값이다.
    """
    days_in_range = [d for d in trading_days if window_start <= d <= window_end]
    if not days_in_range:
        return []

    # 월별 첫 거래일 찾기
    seen_months: set[tuple[int, int]] = set()
    month_firsts: list[date] = []
    for d in days_in_range:
        key = (d.year, d.month)
        if key not in seen_months:
            seen_months.add(key)
            month_firsts.append(d)

    steps: list[MonthStep] = []
    daily_fee_factor = (1 - domestic_fee_pct / 100) ** (1 / 365.25)
    prev_date: date | None = None
    for i, d in enumerate(month_firsts):
        if prev_date is None:
            qqq_div_yield = 0.0
            core_div = 0.0
            fee_factor = 1.0
            reserve_factor = 1.0
        else:
            between = [x for x in days_in_range if prev_date <= x < d]
            qqq_div_sum = sum(qqq_dividends.get(x, 0.0) for x in between)
            prev_qqq_price = qqq_close.get(prev_date)
            qqq_div_yield = (qqq_div_sum / prev_qqq_price) if prev_qqq_price else 0.0
            core_div = sum(core_dividends.get(x, 0.0) for x in between)
            n_days = max((d - prev_date).days, 0)
            fee_factor = daily_fee_factor ** n_days
            reserve_factor = 1.0
            for x in between:
                reserve_factor *= 1 + reserve_daily_rate.get(x, 0.0)
        steps.append(
            MonthStep(
                month_index=i,
                contribution_date=d,
                qqq_price=qqq_close.get(d, float("nan")),
                core_price=core_close.get(d, float("nan")),
                fx=fx_by_date.get(d, float("nan")),
                qqq_div_yield_since_prev=qqq_div_yield,
                core_div_per_share_since_prev=core_div,
                domestic_fee_factor_since_prev=fee_factor,
                reserve_factor_since_prev=reserve_factor,
                is_year_end=(i + 1 < len(month_firsts) and month_firsts[i + 1].year != d.year) or (i == len(month_firsts) - 1),
                is_year_start=(i == 0 or month_firsts[i - 1].year != d.year),
                calendar_year=d.year,
            )
        )
        prev_date = d
    return steps


# ── 납입 배분 ────────────────────────────────────────────────────────────────


def allocate_isa_then_overseas_krw(saving_krw: float, isa_annual_room_krw: float, isa_lifetime_room_krw: float) -> tuple[float, float]:
    """ISA 한도 안까지 먼저, 넘으면 해외계좌로 (순수 함수, K2·K2b·K3·K4 공통 waterfall).

    출력: (isa_amount_krw, overseas_amount_krw)
    """
    isa_amount = max(min(saving_krw, isa_annual_room_krw, isa_lifetime_room_krw), 0.0)
    return isa_amount, saving_krw - isa_amount


def split_equity_and_reserve(amount: float, skim_pct: float, skimming_active: bool) -> tuple[float, float]:
    """금액을 (주식 매수분, 대기자금 적립분)으로 가른다 (순수 함수, K4).

    skimming_active가 False(고점 회복 대기 중)면 전액 주식으로.
    """
    if not skimming_active or amount <= 0:
        return amount, 0.0
    reserve = amount * skim_pct / 100
    return amount - reserve, reserve


# ── K4 급락 신호 캘린더 ────────────────────────────────────────────────────────


@dataclass
class DrawdownEvent:
    date: date
    kind: str  # "trigger" | "recovered"


def compute_drawdown_events(trading_days: list[date], qqq_close: dict[date, float], trigger_pct: float) -> list[DrawdownEvent]:
    """구간 시작 이후 직전 고점 대비 trigger_pct(%, 음수)에 처음 닿는 날(trigger)과, 그
    뒤 고점을 다시 회복하는 날(recovered)을 번갈아 찾는다 (순수 함수, K4 "급락 대비").

    상태기계: 평소(적립 중) -> 고점 대비 하락폭이 trigger_pct 이하로 처음 떨어지는 날
    "trigger"(그날 적립금 전부 투입, 적립 멈춤) -> 그 뒤 종가가 "그 시점까지의" 최고가를
    다시 넘는 날 "recovered"(적립 재개) -> 다시 평소로. trigger 이후 recovered 전까지는
    추가 trigger를 내지 않는다(이미 투입한 상태라 또 투입할 대기자금이 없음).
    """
    events: list[DrawdownEvent] = []
    if not trading_days:
        return events
    peak = qqq_close.get(trading_days[0], float("nan"))
    paused = False
    for d in trading_days:
        price = qqq_close.get(d)
        if price is None:
            continue
        if not paused:
            if price > peak:
                peak = price
            elif peak and (price / peak - 1) * 100 <= trigger_pct:
                events.append(DrawdownEvent(date=d, kind="trigger"))
                paused = True
        else:
            if price >= peak:
                peak = price
                events.append(DrawdownEvent(date=d, kind="recovered"))
                paused = False
    return events


def skimming_phase_by_month(month_dates: list[date], events: list[DrawdownEvent]) -> list[bool]:
    """각 월(contribution_date) 시점에 적립이 활성 상태인지(True=적립, False=멈춤 — 직전에
    trigger가 나고 아직 recovered 전) 표시한다 (순수 함수).
    """
    out = []
    active = True
    event_i = 0
    for d in month_dates:
        while event_i < len(events) and events[event_i].date <= d:
            active = events[event_i].kind != "trigger"
            event_i += 1
        out.append(active)
    return out


def deployable_reserve_events(month_dates: list[date], events: list[DrawdownEvent]) -> dict[date, bool]:
    """month_dates 중 "이 달 안에 trigger가 발생했다"에 해당하는 달을 표시한다 (순수 함수).

    실제 트리거는 거래일(일) 단위로 나지만, 이 시뮬레이션은 월별 평가라 "그 달 안에 trigger가
    있었으면 그 달 말(=다음 달 납입 전)에 전액 투입"으로 근사한다.
    """
    out: dict[date, bool] = {}
    for i in range(len(month_dates) - 1):
        start, end = month_dates[i], month_dates[i + 1]
        out[start] = any(start <= e.date < end and e.kind == "trigger" for e in events)
    if month_dates:
        out[month_dates[-1]] = any(e.date >= month_dates[-1] and e.kind == "trigger" for e in events)
    return out


# ── 가정용 평가(마크투마켓, "1억 원 도달" 측정용) ────────────────────────────────


def hypothetical_posttax_ex_pension_krw(
    overseas_ledger, overseas_price_usd: float, overseas_reserve_usd: float, fx: float,
    overseas_realized_gain_this_year_krw: float,
    isa_value_krw: float, isa_reserve_krw: float, isa_contributed_lifetime_krw: float,
    cfg: dict,
) -> float:
    """지금 전부(해외계좌+ISA) 정리하면 남을 세후 금액(연금 제외) — "1억 원 도달" 측정용
    (순수 함수, 매달 마크투마켓).

    해외계좌: 미실현손익(현재 취득가 기준)에 그 해 이미 쓴 공제를 뺀 남은 공제로 양도세.
    ISA: 지금 해지한다고 보고 순이익(평가액-누적납입액)에 일반형 비과세+9.9%.
    """
    tax_cfg = cfg["tax"]
    oa_cfg = tax_cfg["overseas_account"]
    remaining_deduction = max(oa_cfg["capital_gains_deduction_krw"] - overseas_realized_gain_this_year_krw, 0.0)
    unrealized_gain_krw = overseas_ledger.unrealized_gain(overseas_price_usd) * fx
    tax_overseas = at.capital_gains_tax(unrealized_gain_krw, remaining_deduction, oa_cfg["capital_gains_rate_pct"] / 100)
    overseas_value_krw = (overseas_ledger.value(overseas_price_usd) + overseas_reserve_usd) * fx
    overseas_posttax = overseas_value_krw - tax_overseas

    isa_total = isa_value_krw + isa_reserve_krw
    isa_net_profit = isa_total - isa_contributed_lifetime_krw
    isa_tax = at.isa_exit_tax(isa_net_profit, tax_cfg["isa"]["general_type_exemption_krw"], tax_cfg["isa"]["rate_pct"])
    isa_posttax = isa_total - isa_tax

    return overseas_posttax + isa_posttax


# ── 한 창(window) 한 후보 전체 시뮬레이션 ─────────────────────────────────────


@dataclass
class CandidateResult:
    candidate: str
    cost_basis: str
    pension_credit_rate_pct: float | None
    final_posttax_ex_pension_krw: float
    final_pension_value_pretax_krw: float | None
    final_pension_posttax_pension_rate_krw: float | None  # (a) 연금수령 5.5%
    final_pension_posttax_lump_sum_krw: float | None      # (b) 중도인출 16.5%
    months_to_100m_krw: int | None
    total_tax_krw: float
    total_tax_credit_krw: float
    trade_count: int


def run_candidate(
    candidate: str,
    steps: list[MonthStep],
    start_capital_krw: float,
    saving_krw: float,
    cfg: dict,
    cost_basis: str = "moving_average",
    pension_credit_rate_pct: float | None = None,
    skimming_active_by_month: list[bool] | None = None,
) -> CandidateResult:
    """한 창(10년, steps)에서 candidate 규칙대로 시뮬레이션하고 결과를 낸다 (순수 함수).

    입력: candidate("K0"|"K1"|"K2"|"K2b"|"K3"|"K4"), steps(build_month_steps 결과,
         마지막 원소는 납입 없이 평가만), start_capital_krw, saving_krw(매달 M),
         cfg(preregistration yaml 로드값), cost_basis("moving_average"|"fifo" — K1·K2b만
         fifo로도 불러볼 수 있음), pension_credit_rate_pct(K3 전용, 16.5 또는 13.2),
         skimming_active_by_month(K4 전용, len(steps)-1 길이 — compute_drawdown_events +
         skimming_phase_by_month으로 미리 구해 넘긴다)
    """
    if len(steps) < 2:
        raise ValueError("steps가 너무 짧습니다(최소 2개: 시작월 + 평가월)")

    tax_cfg = cfg["tax"]
    oa_cfg = tax_cfg["overseas_account"]
    isa_cfg = tax_cfg["isa"]
    pension_cfg = tax_cfg["pension"]
    costs_cfg = cfg["costs"]
    commission_pct = costs_cfg["commission_pct"]
    fx_spread_pct = costs_cfg["fx_spread_pct"]
    skim_pct = cfg["reserve"]["skim_pct_of_monthly_allocation"]
    haircut = tax_cfg["domestic_wrapper_dividend_haircut_pct"] / 100
    reserve_interest_tax_pct = tax_cfg["reserve_interest_tax_pct"]
    deduction_krw = oa_cfg["capital_gains_deduction_krw"]
    div_wh = oa_cfg["dividend_withholding_pct"] / 100
    reopen_cycle_months = isa_cfg["reopen_cycle_years"] * 12
    uses_isa = candidate in ("K2", "K2b", "K3", "K4")
    # K3는 ISA 연 한도 규칙만 K2에서 가져올 뿐 3년 재가입 주기는 쓰지 않는다
    # (계획서 4장 K3 — "K2와 같은 연 2,000만 원 한도 규칙"만 명시, 재가입은 명시 안 됨).
    uses_isa_reopen = candidate in ("K2", "K2b", "K4")
    uses_reserve = candidate == "K4"

    Ledger = at.FifoLedger if cost_basis == "fifo" else at.MovingAverageLedger
    overseas = Ledger()
    overseas_reserve_usd = 0.0
    overseas_realized_gain_by_year: dict[int, float] = {}
    reserve_interest_tax_krw_total = 0.0

    isa_value_krw = 0.0
    isa_reserve_krw = 0.0
    isa_contributed_this_year_krw = 0.0
    isa_contributed_lifetime_krw = 0.0
    isa_cycle_open_month = 0
    isa_exit_tax_paid_total = 0.0

    pension_value_krw = 0.0
    pension_contributed_this_year_krw = 0.0
    pension_contributed_lifetime_krw = 0.0
    pension_credit_total_krw = 0.0

    trade_count = 0
    month_index_by_ym = {(s.calendar_year, s.contribution_date.month): i for i, s in enumerate(steps)}
    pending_pension_refund: dict[int, float] = {}

    months_to_100m: int | None = None

    first = steps[0]
    start_usd = (start_capital_krw / first.fx) * (1 - fx_spread_pct / 100)
    overseas.buy(net_of_commission(start_usd, commission_pct), first.core_price, commission_pct)
    trade_count += 1

    for i, step in enumerate(steps[:-1]):
        year = step.calendar_year

        # 1) 성장 (첫 달은 방금 시작이라 성장 없음)
        if i > 0:
            prev = steps[i - 1]
            growth = (
                (step.qqq_price / prev.qqq_price) * (step.fx / prev.fx)
                * (1 + step.qqq_div_yield_since_prev * (1 - haircut))
                * step.domestic_fee_factor_since_prev
            )
            isa_value_krw *= growth
            pension_value_krw *= growth
            if uses_reserve:
                isa_reserve_krw *= step.reserve_factor_since_prev

            if step.core_div_per_share_since_prev > 0 and overseas.shares > 0:
                gross = overseas.shares * step.core_div_per_share_since_prev
                overseas.buy(gross * (1 - div_wh), step.core_price, 0.0)

            if uses_reserve and overseas_reserve_usd > 0:
                grown = overseas_reserve_usd * step.reserve_factor_since_prev
                interest = grown - overseas_reserve_usd
                interest_tax = interest * reserve_interest_tax_pct / 100
                overseas_reserve_usd = grown - interest_tax
                reserve_interest_tax_krw_total += interest_tax * step.fx

        if step.is_year_start:
            isa_contributed_this_year_krw = 0.0
            pension_contributed_this_year_krw = 0.0
        overseas_realized_gain_by_year.setdefault(year, 0.0)

        # 2) K3: 5월 세액공제 환급 ISA(우선)/해외 납입
        if candidate == "K3" and i in pending_pension_refund:
            refund = pending_pension_refund.pop(i)
            annual_room = at.isa_annual_room(isa_contributed_this_year_krw, isa_cfg["annual_limit_krw"])
            lifetime_room = at.isa_lifetime_room(isa_contributed_lifetime_krw, isa_cfg["lifetime_limit_krw"])
            isa_amt, overseas_amt = allocate_isa_then_overseas_krw(refund, annual_room, lifetime_room)
            isa_value_krw += net_of_commission(isa_amt, commission_pct)
            isa_contributed_this_year_krw += isa_amt
            isa_contributed_lifetime_krw += isa_amt
            if isa_amt > 0:
                trade_count += 1
            if overseas_amt > 0:
                usd = (overseas_amt / step.fx) * (1 - fx_spread_pct / 100)
                overseas.buy(net_of_commission(usd, commission_pct), step.core_price, commission_pct)
                trade_count += 1

        # 3) 이 달 납입 배분
        if candidate in ("K0", "K1"):
            usd = (saving_krw / step.fx) * (1 - fx_spread_pct / 100)
            overseas.buy(net_of_commission(usd, commission_pct), step.core_price, commission_pct)
            trade_count += 1
        elif candidate in ("K2", "K2b", "K4"):
            annual_room = at.isa_annual_room(isa_contributed_this_year_krw, isa_cfg["annual_limit_krw"])
            lifetime_room = at.isa_lifetime_room(isa_contributed_lifetime_krw, isa_cfg["lifetime_limit_krw"])
            isa_amt, overseas_amt = allocate_isa_then_overseas_krw(saving_krw, annual_room, lifetime_room)
            active = skimming_active_by_month[i] if (uses_reserve and skimming_active_by_month) else True
            isa_equity, isa_reserve_add = split_equity_and_reserve(isa_amt, skim_pct, active) if uses_reserve else (isa_amt, 0.0)
            isa_value_krw += net_of_commission(isa_equity, commission_pct)
            isa_reserve_krw += isa_reserve_add
            isa_contributed_this_year_krw += isa_amt
            isa_contributed_lifetime_krw += isa_amt
            if isa_amt > 0:
                trade_count += 1

            over_equity, over_reserve_add = split_equity_and_reserve(overseas_amt, skim_pct, active) if uses_reserve else (overseas_amt, 0.0)
            if over_equity > 0:
                usd = (over_equity / step.fx) * (1 - fx_spread_pct / 100)
                overseas.buy(net_of_commission(usd, commission_pct), step.core_price, commission_pct)
                trade_count += 1
            if over_reserve_add > 0:
                overseas_reserve_usd += (over_reserve_add / step.fx) * (1 - fx_spread_pct / 100)
        elif candidate == "K3":
            pension_amt = min(saving_krw, pension_cfg["fixed_monthly_contribution_krw"])
            pension_value_krw += net_of_commission(pension_amt, commission_pct)
            pension_contributed_this_year_krw += pension_amt
            pension_contributed_lifetime_krw += pension_amt
            if pension_amt > 0:
                trade_count += 1
            remaining = saving_krw - pension_amt
            annual_room = at.isa_annual_room(isa_contributed_this_year_krw, isa_cfg["annual_limit_krw"])
            lifetime_room = at.isa_lifetime_room(isa_contributed_lifetime_krw, isa_cfg["lifetime_limit_krw"])
            isa_amt, overseas_amt = allocate_isa_then_overseas_krw(remaining, annual_room, lifetime_room)
            isa_value_krw += net_of_commission(isa_amt, commission_pct)
            isa_contributed_this_year_krw += isa_amt
            isa_contributed_lifetime_krw += isa_amt
            if isa_amt > 0:
                trade_count += 1
            if overseas_amt > 0:
                usd = (overseas_amt / step.fx) * (1 - fx_spread_pct / 100)
                overseas.buy(net_of_commission(usd, commission_pct), step.core_price, commission_pct)
                trade_count += 1

        # 4) K1: 연말 공제소진 매도+재매수
        if candidate == "K1" and step.is_year_end:
            used = overseas_realized_gain_by_year[year]
            target_krw = max(deduction_krw - used, 0.0)
            target_usd = target_krw / step.fx
            shares = overseas.shares_for_target_gain(step.core_price, target_usd)
            if shares > 0:
                amount = shares * step.core_price
                proceeds, gain_usd = overseas.sell(amount, step.core_price, commission_pct)
                overseas_realized_gain_by_year[year] += gain_usd * step.fx
                overseas.buy(net_of_commission(proceeds, commission_pct), step.core_price, commission_pct)
                trade_count += 2

        # 4b) K2b: 1월 해외->ISA 스윕(그 해 남은 공제 안에서만, 세금 0)
        if candidate == "K2b" and step.is_year_start and i > 0:
            used = overseas_realized_gain_by_year[year]
            remaining_deduction_krw = max(deduction_krw - used, 0.0)
            if remaining_deduction_krw > 0 and overseas.shares > 0:
                annual_room = at.isa_annual_room(isa_contributed_this_year_krw, isa_cfg["annual_limit_krw"])
                lifetime_room = at.isa_lifetime_room(isa_contributed_lifetime_krw, isa_cfg["lifetime_limit_krw"])
                room_krw = min(annual_room, lifetime_room)
                if overseas.unrealized_gain(step.core_price) <= 0:
                    max_amount_krw = overseas.shares * step.core_price * step.fx
                else:
                    target_gain_usd = remaining_deduction_krw / step.fx
                    max_shares = overseas.shares_for_target_gain(step.core_price, target_gain_usd)
                    max_amount_krw = max_shares * step.core_price * step.fx
                move_krw = min(room_krw, max_amount_krw)
                if move_krw > 0:
                    amount_usd = min(move_krw / step.fx, overseas.shares * step.core_price)
                    proceeds_usd, gain_usd = overseas.sell(amount_usd, step.core_price, commission_pct)
                    overseas_realized_gain_by_year[year] += gain_usd * step.fx
                    proceeds_krw = proceeds_usd * step.fx
                    isa_value_krw += net_of_commission(proceeds_krw, commission_pct)
                    isa_contributed_this_year_krw += proceeds_krw
                    isa_contributed_lifetime_krw += proceeds_krw
                    trade_count += 1

        # 5) ISA 3년 재가입
        if uses_isa_reopen and (i - isa_cycle_open_month) == reopen_cycle_months:
            total_isa = isa_value_krw + isa_reserve_krw
            net_profit = total_isa - isa_contributed_lifetime_krw
            tax = at.isa_exit_tax(net_profit, isa_cfg["general_type_exemption_krw"], isa_cfg["rate_pct"])
            isa_exit_tax_paid_total += tax
            isa_value_krw = total_isa - tax
            isa_reserve_krw = 0.0
            isa_contributed_this_year_krw = 0.0
            isa_contributed_lifetime_krw = 0.0
            isa_cycle_open_month = i
            trade_count += 2  # 해지(매도) + 재가입(매수)

        # 6) K3: 그 해(year) 마지막 달이면 세액공제 계산, 다음 해 5월에 환급 예약
        if candidate == "K3" and step.is_year_end and pension_credit_rate_pct is not None:
            credit = at.pension_tax_credit(pension_contributed_this_year_krw, pension_cfg["deduction_cap_krw"], pension_credit_rate_pct)
            pension_credit_total_krw += credit
            refund_key = (year + 1, pension_cfg["refund_paid_month_offset"])
            refund_month_index = month_index_by_ym.get(refund_key)
            if refund_month_index is not None and refund_month_index < len(steps) - 1:
                pending_pension_refund[refund_month_index] = pending_pension_refund.get(refund_month_index, 0.0) + credit

        # 7) 월말 마크투마켓(1억원 도달 측정)
        hv = hypothetical_posttax_ex_pension_krw(
            overseas, step.core_price, overseas_reserve_usd, step.fx,
            overseas_realized_gain_by_year[year],
            isa_value_krw, isa_reserve_krw, isa_contributed_lifetime_krw,
            cfg,
        )
        if months_to_100m is None and hv >= 100_000_000:
            months_to_100m = i + 1

    # ── 끝: 전량 정리 ────────────────────────────────────────────────────────
    last = steps[-1]
    final_year = last.calendar_year
    overseas_realized_gain_by_year.setdefault(final_year, 0.0)
    if last.month_index > 0:
        prev = steps[-2]
        growth = (
            (last.qqq_price / prev.qqq_price) * (last.fx / prev.fx)
            * (1 + last.qqq_div_yield_since_prev * (1 - haircut))
            * last.domestic_fee_factor_since_prev
        )
        isa_value_krw *= growth
        pension_value_krw *= growth
        if uses_reserve:
            isa_reserve_krw *= last.reserve_factor_since_prev
        if last.core_div_per_share_since_prev > 0 and overseas.shares > 0:
            gross = overseas.shares * last.core_div_per_share_since_prev
            overseas.buy(gross * (1 - div_wh), last.core_price, 0.0)
        if uses_reserve and overseas_reserve_usd > 0:
            grown = overseas_reserve_usd * last.reserve_factor_since_prev
            interest = grown - overseas_reserve_usd
            interest_tax = interest * reserve_interest_tax_pct / 100
            overseas_reserve_usd = grown - interest_tax
            reserve_interest_tax_krw_total += interest_tax * last.fx

    overseas_amount_usd = overseas.shares * last.core_price
    _, final_gain_usd = overseas.sell(overseas_amount_usd, last.core_price, commission_pct)
    trade_count += 1
    final_gain_krw = final_gain_usd * last.fx + overseas_realized_gain_by_year[final_year]
    overseas_cg_tax = at.capital_gains_tax(final_gain_krw, deduction_krw, oa_cfg["capital_gains_rate_pct"] / 100)
    overseas_final_value_krw = (overseas_amount_usd + overseas_reserve_usd) * last.fx
    overseas_posttax_krw = overseas_final_value_krw - overseas_cg_tax

    isa_posttax_krw = 0.0
    if uses_isa:
        total_isa = isa_value_krw + isa_reserve_krw
        net_profit = total_isa - isa_contributed_lifetime_krw
        final_isa_tax = at.isa_exit_tax(net_profit, isa_cfg["general_type_exemption_krw"], isa_cfg["rate_pct"])
        isa_exit_tax_paid_total += final_isa_tax
        isa_posttax_krw = total_isa - final_isa_tax

    final_posttax_ex_pension = overseas_posttax_krw + isa_posttax_krw

    pension_pretax = None
    pension_posttax_a = None
    pension_posttax_b = None
    if candidate == "K3":
        pension_pretax = pension_value_krw
        pension_posttax_a = pension_value_krw - at.pension_withdrawal_tax(pension_value_krw, pension_cfg["withdrawal_tax_pct"]["pension"])
        pension_posttax_b = pension_value_krw - at.pension_withdrawal_tax(pension_value_krw, pension_cfg["withdrawal_tax_pct"]["lump_sum"])

    total_tax_krw = overseas_cg_tax + isa_exit_tax_paid_total + reserve_interest_tax_krw_total

    return CandidateResult(
        candidate=candidate, cost_basis=cost_basis, pension_credit_rate_pct=pension_credit_rate_pct,
        final_posttax_ex_pension_krw=final_posttax_ex_pension,
        final_pension_value_pretax_krw=pension_pretax,
        final_pension_posttax_pension_rate_krw=pension_posttax_a,
        final_pension_posttax_lump_sum_krw=pension_posttax_b,
        months_to_100m_krw=months_to_100m,
        total_tax_krw=total_tax_krw,
        total_tax_credit_krw=pension_credit_total_krw,
        trade_count=trade_count,
    )
