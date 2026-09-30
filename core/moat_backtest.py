"""해자 과거 검증(H4) 포트폴리오 구성·수익 계산 (순수 함수, docs/design/moat_plan.md 4장).

네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). 연 1회 교체(전량 매도 후 재매수)
구조라 실현손익이 매 기간 100% 발생한다 — 그래서 기간별 세전 성장 배수와 원화 실현손익만
알면 core.tax.capital_gains_tax로 그해 양도세를 매길 수 있다.

**단순화(보고서에 명시)**: 환전 스프레드(0.1%)는 최초 원화->달러 전환 1회에만 적용한다.
이후 종목<->QQQM 재배분은 전부 달러 자산 간 이동이라 환전이 없다(기존 engine/backtest.py
Broker의 관례와 동일). 실현손익(원화)은 청산 시점 환율로 환산한다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from core.tax import capital_gains_tax


def allocate_equal_weight_with_sector_cap(tickers: list[str], sectors: dict[str, str], cap_pct: float) -> dict[str, float]:
    """동일 비중을 주되 업종 비중이 cap_pct(%)를 넘으면 초과분을 다른 업종에 비례
    배분한다 (순수 함수, docs/design/moat_plan.md 4장 "비중" 규칙).

    입력: tickers(이번 포트폴리오 종목), sectors({ticker: 업종}, 없으면 "확인 필요"로 봄),
         cap_pct(0~100, 예: 30)
    출력: {ticker: weight} 합 1.0(부동소수 오차 제외). 한 업종이 전체를 차지해 캡을
         지킬 수 없으면(재배분할 곳이 없음) 최대한 캡에 가깝게 맞추고 멈춘다.
    """
    n = len(tickers)
    if n == 0:
        return {}
    weights = {t: 1.0 / n for t in tickers}
    cap = cap_pct / 100
    all_sectors = {sectors.get(t, "확인 필요") for t in tickers}
    capped_sectors: set[str] = set()  # 한 번 캡에 고정된 업종은 다시 건드리지 않는다(물채우기 방식 — 왔다갔다 안 하게)

    for _ in range(len(all_sectors) + 1):
        sector_totals: dict[str, float] = {}
        for t in tickers:
            s = sectors.get(t, "확인 필요")
            sector_totals[s] = sector_totals.get(s, 0.0) + weights[t]
        over = {s for s in all_sectors if s not in capped_sectors and sector_totals.get(s, 0.0) > cap + 1e-9}
        if not over:
            break
        free_sectors = all_sectors - capped_sectors - over
        free_total = sum(weights[t] for t in tickers if sectors.get(t, "확인 필요") in free_sectors)
        if free_total <= 0:
            # 아직 캡에 안 걸린 업종이 남아 있지 않다 — 더는 캡을 지킬 수 없으니 여기서
            # 멈춘다(지금까지 캡에 고정한 업종은 그대로 두고, 나머지는 원래 비중 유지).
            break
        excess = 0.0
        for t in tickers:
            s = sectors.get(t, "확인 필요")
            if s in over:
                new_w = weights[t] * (cap / sector_totals[s])
                excess += weights[t] - new_w
                weights[t] = new_w
        for t in tickers:
            s = sectors.get(t, "확인 필요")
            if s in free_sectors:
                weights[t] += excess * (weights[t] / free_total)
        capped_sectors |= over
    return weights


def draw_random_portfolio(universe: list[str], n: int, rng: random.Random) -> list[str]:
    """유니버스에서 무작위로 n개를 고른다 (순수 함수 — rng를 주입받아 결정적).

    입력: universe(그 시점 나스닥100 전체), n(고를 개수), rng(random.Random 인스턴스,
         호출부가 시드를 고정해 재현 가능하게 함)
    출력: n개 티커 목록(n > len(universe)면 universe 전체)
    """
    n = min(n, len(universe))
    return rng.sample(universe, n)


def compute_stock_period_return(
    entry_price: float,
    exit_price: float,
    dividends_per_share: float,
    costs_cfg: dict,
    dividend_withholding_rate: float,
    apply_sell_costs: bool = True,
) -> tuple[float, float]:
    """한 종목의 한 교체 기간(진입~청산) 세전 가격 성장 배수와 세후 배당 수익률을
    계산한다 (순수 함수, 달러 기준 — 환율은 포트폴리오 단계에서 한 번만 적용한다).

    입력: entry_price(진입가, 슬리피지 반영 전 시가 — 진입은 언제나 실제 체결로 본다),
         exit_price(청산가, 슬리피지 반영 전), dividends_per_share(보유 기간 중 받은
         주당 배당 합, 세전), costs_cfg(cfg["backtest"]["costs"] — slippage_pct·
         commission_buy_pct·commission_sell_pct), dividend_withholding_rate(보통 0.15),
         apply_sell_costs(기본 True — False면 청산가에 슬리피지·매도수수료를 안 붙인다.
         계획서 4장 "봉인": 2021년 4월 편입분을 2021-12-31 가격으로 "평가"만 하고 끝낼 때
         쓴다 — 실제 매도가 아니라 평가라 매도 비용이 없다)
    출력: (price_return_factor, dividend_aftertax_yield) — 둘 다 "매수에 실제로 쓴 금액"
         기준의 배수·수익률. entry_price<=0이면 (1.0, 0.0)(계산 불가, 중립으로 처리).
    """
    slippage = costs_cfg["slippage_pct"] / 100
    buy_price = entry_price * (1 + slippage) * (1 + costs_cfg["commission_buy_pct"] / 100)
    if buy_price <= 0:
        return 1.0, 0.0
    if apply_sell_costs:
        sell_price = exit_price * (1 - slippage) * (1 - costs_cfg["commission_sell_pct"] / 100)
    else:
        sell_price = exit_price
    price_return_factor = sell_price / buy_price
    div_yield_pretax = dividends_per_share / buy_price
    div_yield_aftertax = div_yield_pretax * (1 - dividend_withholding_rate)
    return price_return_factor, div_yield_aftertax


@dataclass
class PeriodResult:
    """한 포트폴리오의 한 교체 기간 결과 — chain_periods_with_tax의 입력.

    entry_date·exit_date를 둘 다 남긴다: 기간이 이어질 때(연 1회 교체) 한 기간의
    exit_date는 보통 다음 기간의 entry_date와 같은 날짜지만(같은 교체일에 전량 매도
    후 재매수), 마지막 기간(봉인 평가)은 다음 기간이 없어 exit_date를 따로 알아야
    equity_rows에 날짜를 제대로 남길 수 있다.
    """

    entry_date: str  # ISO 날짜
    exit_date: str  # ISO 날짜 — chain_periods_with_tax가 결과 행에 이 날짜를 쓴다
    fx_entry: float
    fx_exit: float
    price_return_factor: float  # 종목별 (price_return_factor, weight)를 이미 가중합한 포트폴리오 값
    dividend_aftertax_yield: float  # 마찬가지로 가중합
    note: str = ""


def portfolio_period_return(
    stock_returns: dict[str, tuple[float, float]], weights: dict[str, float]
) -> tuple[float, float]:
    """종목별 (가격 성장 배수, 세후 배당 수익률)을 비중으로 가중합해 포트폴리오
    기간 수익을 낸다 (순수 함수).

    입력: stock_returns({ticker: (price_return_factor, dividend_aftertax_yield)}),
         weights({ticker: 비중}, allocate_equal_weight_with_sector_cap 결과)
    출력: (포트폴리오 price_return_factor, 포트폴리오 dividend_aftertax_yield)
    """
    if not weights:
        return 1.0, 0.0
    price_factor = 0.0
    div_yield = 0.0
    for t, w in weights.items():
        pr, dy = stock_returns.get(t, (1.0, 0.0))
        price_factor += w * pr
        div_yield += w * dy
    return price_factor, div_yield


def chain_periods_with_tax(periods: list[PeriodResult], starting_capital_usd: float, tax_cfg: dict) -> list[dict]:
    """여러 교체 기간을 이어 붙이면서, 매 기간 전량 매도로 발생하는 실현손익에 그해
    (원화) 양도세를 매긴다 (순수 함수, docs/design/moat_plan.md 4장 "비용" 규칙).

    입력: periods(시간순, 각 기간 진입 시점의 환율·기간수익), starting_capital_usd
         (첫 진입 시점 달러 자본 — 환전 스프레드는 이 자본을 만들 때 이미 반영했다고
         가정), tax_cfg(cfg["backtest"]["tax"])
    출력: [{"date": 각 기간의 exit_date(맨 앞은 periods[0].entry_date), "total_krw":
         그 시점 평가액(세후, 원화)}, ...] — engine.backtest.compute_equity_metrics에
         바로 쓸 수 있다.
    """
    if not periods:
        return []
    rows = [{"date": periods[0].entry_date, "total_krw": starting_capital_usd * periods[0].fx_entry}]
    capital_usd = starting_capital_usd
    for p in periods:
        gross_usd = capital_usd * (p.price_return_factor + p.dividend_aftertax_yield)
        realized_gain_krw = capital_usd * (p.price_return_factor - 1) * p.fx_exit
        tax_krw = capital_gains_tax(
            realized_gain_krw, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100
        )
        gross_krw = gross_usd * p.fx_exit
        net_krw = gross_krw - tax_krw
        capital_usd = net_krw / p.fx_exit if p.fx_exit else 0.0
        rows.append({"date": p.exit_date, "total_krw": net_krw, "note": p.note, "gain_krw": realized_gain_krw, "tax_krw": tax_krw})
    return rows


def percentile_rank(value: float, distribution: list[float]) -> float:
    """value가 distribution에서 몇 번째 백분위인지 구한다 (순수 함수).

    출력: 0~100 사이 값. distribution의 value 이하인 원소 비율 × 100(선형 보간 없음,
         "이 값보다 같거나 작은 게 몇 %인가").
    """
    if not distribution:
        return 50.0
    below_or_equal = sum(1 for v in distribution if v <= value)
    return round(below_or_equal / len(distribution) * 100, 1)


def annualized_return(start_value: float, end_value: float, years: float) -> float | None:
    """세후 연복리 수익률(%) — (끝값/시작값)^(1/년수) - 1 (순수 함수)."""
    if years <= 0 or start_value <= 0 or end_value <= 0:
        return None
    return round(((end_value / start_value) ** (1 / years) - 1) * 100, 2)
