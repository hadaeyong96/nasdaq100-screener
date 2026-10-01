"""계좌·납입 구조 연구(P6-3, docs/design/p6_3_plan.md) 세금 계산 (순수 함수).

네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). 세법 수치(공제 한도·세율 등)는
전부 호출부가 `configs/p6_3_preregistration.yaml`에서 읽어 인자로 넘긴다 — 이 파일에
하드코딩하지 않는다.

해외계좌 양도세 자체(공제 후 세율)는 core/tax.py의 capital_gains_tax를 그대로 재사용한다
(재구현하지 않음). 이 파일은 그 위에 필요한 것들을 더한다:
- 취득가 원장 두 가지(이동평균법 MovingAverageLedger, 선입선출법 FifoLedger)
- 공제 한도를 정확히 채우는 매도 수량 계산(K1·K2b)
- ISA 해지 세금, 연금저축 세액공제·인출 세금
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from core.tax import capital_gains_tax  # noqa: F401  (재export — 호출부 편의, 재구현 안 함)


# ── 취득가 원장 ──────────────────────────────────────────────────────────────


@dataclass
class MovingAverageLedger:
    """평균원가법 취득가 원장 (engine/portfolio.py의 PortfolioAsset과 같은 방식,
    core/는 engine/에 의존하면 안 되므로 독립적으로 둔다).

    shares: 보유 수량, cost: 총 매입원가(통화 단위는 호출부가 일관되게 — 보통 USD)
    """

    shares: float = 0.0
    cost: float = 0.0

    @property
    def avg_cost_per_share(self) -> float:
        return self.cost / self.shares if self.shares > 0 else 0.0

    def value(self, price: float) -> float:
        return self.shares * price

    def unrealized_gain(self, price: float) -> float:
        """지금 전량 매도하면 생길 실현손익(수수료 전, 미리보기 — 상태를 바꾸지 않음)."""
        return self.value(price) - self.cost

    def buy(self, amount: float, price: float, commission_pct: float = 0.0) -> float:
        """amount(수수료 전 매수금액)를 매수한다. 실제 지출(수수료 포함)을 반환."""
        if amount <= 0 or price <= 0:
            return 0.0
        shares = amount / price
        self.shares += shares
        self.cost += amount
        return amount * (1 + commission_pct / 100)

    def sell(self, amount: float, price: float, commission_pct: float = 0.0) -> tuple[float, float]:
        """amount(수수료 전 평가액)만큼 판다. (순수령액, 실현손익)을 반환."""
        if amount <= 0 or self.shares <= 0 or price <= 0:
            return 0.0, 0.0
        amount = min(amount, self.shares * price)
        shares_sold = amount / price
        avg_cost = self.avg_cost_per_share
        cost_removed = avg_cost * shares_sold
        proceeds = amount * (1 - commission_pct / 100)
        gain = proceeds - cost_removed  # 매도수수료는 과세소득에서 공제(engine/portfolio.py와 같은 관례)
        self.shares -= shares_sold
        self.cost -= cost_removed
        return proceeds, gain

    def shares_for_target_gain(self, price: float, target_gain: float) -> float:
        """실현손익이 정확히 target_gain이 되는 매도 수량 (순수 함수, K1·K2b "공제 소진 매도").

        단가당 손익 = price - avg_cost_per_share. 0 이하(손실 중이거나 본전)면 아무리 팔아도
        target_gain(통상 양수)에 못 닿으므로 0을 반환 — 호출부가 "그 해는 매도 안 함"으로 처리.
        target_gain이 음수거나 0이면 0(매도 불필요).
        """
        if target_gain <= 0 or self.shares <= 0 or price <= 0:
            return 0.0
        gain_per_share = price - self.avg_cost_per_share
        if gain_per_share <= 0:
            return 0.0
        shares = target_gain / gain_per_share
        return min(shares, self.shares)


@dataclass
class _Lot:
    shares: float
    cost_per_share: float


@dataclass
class FifoLedger:
    """선입선출법 취득가 원장 — 매수 순서대로 lot을 쌓고, 매도는 가장 오래된 lot부터 소진한다."""

    lots: deque = field(default_factory=deque)

    @property
    def shares(self) -> float:
        return sum(lot.shares for lot in self.lots)

    @property
    def cost(self) -> float:
        return sum(lot.shares * lot.cost_per_share for lot in self.lots)

    def value(self, price: float) -> float:
        return self.shares * price

    def unrealized_gain(self, price: float) -> float:
        """지금 전량 매도하면 생길 실현손익(미리보기) — lot 구조와 무관하게
        총매도대금-총매입원가로 FIFO·이동평균이 전량 매도에선 항상 같다(계획서 3장)."""
        return self.value(price) - self.cost

    def buy(self, amount: float, price: float, commission_pct: float = 0.0) -> float:
        if amount <= 0 or price <= 0:
            return 0.0
        shares = amount / price
        self.lots.append(_Lot(shares=shares, cost_per_share=price))
        return amount * (1 + commission_pct / 100)

    def sell(self, amount: float, price: float, commission_pct: float = 0.0) -> tuple[float, float]:
        """amount(수수료 전 평가액)만큼, 오래된 lot부터 소진하며 판다. (순수령액, 실현손익)."""
        if amount <= 0 or price <= 0 or self.shares <= 0:
            return 0.0, 0.0
        amount = min(amount, self.shares * price)
        shares_to_sell = amount / price
        gross_gain = 0.0
        remaining = shares_to_sell
        while remaining > 1e-12 and self.lots:
            lot = self.lots[0]
            take = min(lot.shares, remaining)
            gross_gain += take * (price - lot.cost_per_share)
            lot.shares -= take
            remaining -= take
            if lot.shares <= 1e-12:
                self.lots.popleft()
        proceeds = amount * (1 - commission_pct / 100)
        gain = gross_gain - amount * commission_pct / 100  # 매도수수료는 과세소득에서 공제(이동평균법과 같은 관례)
        return proceeds, gain

    def shares_for_target_gain(self, price: float, target_gain: float) -> float:
        """FIFO로 lot을 순서대로 소진했을 때 누적 실현손익이 정확히 target_gain이 되는
        매도 수량 (순수 함수, lots를 바꾸지 않음 — 미리보기).

        각 lot의 손익률이 달라 lot 경계에서 부분 매도가 필요할 수 있다(그 lot 안에서는
        단가당 손익이 일정하므로 선형 보간). 끝까지 가도 target_gain에 못 닿으면
        보유 전량(모든 lot 손익 합)과 target_gain 중 작은 쪽에 해당하는 수량 — 보유 전량을
        반환한다(target_gain이 그보다 크면 "최대한"의 의미로 전량 매도).
        """
        if target_gain <= 0 or not self.lots or price <= 0:
            return 0.0
        cumulative_gain = 0.0
        cumulative_shares = 0.0
        for lot in self.lots:
            gain_per_share = price - lot.cost_per_share
            lot_total_gain = gain_per_share * lot.shares
            if gain_per_share <= 0:
                # 이 lot은 손실(또는 본전) — 팔아도 누적 손익이 줄거나 그대로다.
                # target_gain(양수)에 못 보태므로 건너뛴다(그 lot은 안 파는 것으로 간주).
                continue
            if cumulative_gain + lot_total_gain >= target_gain:
                remaining_gain_needed = target_gain - cumulative_gain
                shares_needed = remaining_gain_needed / gain_per_share
                return cumulative_shares + min(shares_needed, lot.shares)
            cumulative_gain += lot_total_gain
            cumulative_shares += lot.shares
        return cumulative_shares


# ── ISA ──────────────────────────────────────────────────────────────────────


def isa_exit_tax(net_profit_krw: float, exemption_krw: float, rate_pct: float) -> float:
    """ISA 해지 세금 = max(순이익-비과세한도, 0) × 세율 (순수 함수).

    입력: net_profit_krw(해지 시점 평가액 − 총 납입액, 음수 가능), exemption_krw
         (일반형 200만/서민형 400만 — 호출부가 선택), rate_pct(9.9)
    출력: 세액(원). 순이익이 한도 이하(손실 포함)면 0.
    """
    taxable = max(net_profit_krw - exemption_krw, 0.0)
    return taxable * rate_pct / 100


def isa_annual_room(contributed_this_year_krw: float, annual_limit_krw: float) -> float:
    """ISA의 그 해 남은 연 납입 한도 (순수 함수)."""
    return max(annual_limit_krw - contributed_this_year_krw, 0.0)


def isa_lifetime_room(total_contributed_krw: float, lifetime_limit_krw: float) -> float:
    """ISA 계좌 하나의 남은 총 납입 한도 (순수 함수, 재가입하면 새 계좌라 다시 꽉 참)."""
    return max(lifetime_limit_krw - total_contributed_krw, 0.0)


# ── 연금저축 ──────────────────────────────────────────────────────────────────


def pension_tax_credit(contribution_krw: float, deduction_cap_krw: float, credit_rate_pct: float) -> float:
    """연금저축 세액공제 환급액 = min(납입액, 한도) × 공제율 (순수 함수)."""
    base = min(max(contribution_krw, 0.0), deduction_cap_krw)
    return base * credit_rate_pct / 100


def pension_withdrawal_tax(balance_krw: float, rate_pct: float) -> float:
    """연금저축 인출 세금 = 잔액 × 세율 (순수 함수).

    rate_pct에 연금수령(5.5)·중도인출(16.5) 둘 중 호출부가 원하는 값을 넘긴다. 전액
    세액공제를 받은 납입(K3는 매달 정확히 공제 한도만큼만 납입)이라는 전제라 잔액
    전체(원금+운용수익)가 과세 대상이다 — 실제 연금저축 규정과 같은 단순화.
    """
    return max(balance_krw, 0.0) * rate_pct / 100
