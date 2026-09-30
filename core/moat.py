"""해자(경쟁 우위) 지표 계산 (해자 분석 H2, docs/design/moat_plan.md 3장).

data/edgar.py가 공급한 원본 XBRL 사실에서 M1~M5 지표와 해자 등급을 계산한다. 순수 함수
모음이다 — 네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). 같은 company_facts·
cfg·as_of 입력이면 항상 같은 출력이다.

**미래 데이터 금지**: 모든 연간 값은 annual_value_series()가 filed(공시 제출일)가 as_of
이하인 것만 골라서 돌려준다. 같은 회계연도가 나중 10-K에 비교연도로 다시 실리면(흔함 —
SEC XBRL 사실은 누적된다) as_of 시점에 이미 제출돼 있던 것 중 가장 최근 제출본을 쓴다
(정정·재작성 반영, 그 뒤에 나온 값은 안 씀).

**XBRL 태그 대체**: 같은 재무 개념도 회사·시대마다 다른 태그를 쓴다(예: 매출은
"Revenues" 또는 ASC606 이후 "RevenueFromContractWithCustomerExcludingAssessedTax").
_TAG_CANDIDATES에 우선순위 목록을 두고, 실제 쓴 태그를 결과에 남긴다.

**해외 기업(20-F, IFRS)**: ANNUAL_FORMS에 "20-F"를 포함하고 _TAG_CANDIDATES마다
ifrs-full 대체 태그를 둔다. 그래도 못 찾으면 해당 지표는 "판단 불가"로 남는다 — 억지로
값을 만들지 않는다.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

from data import edgar

ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")  # 40-F: 캐나다 기업(MJDS) 연차 보고서(예: SHOP)

_MIN_ANNUAL_PERIOD_DAYS = 340
_MAX_ANNUAL_PERIOD_DAYS = 380

# 재무 개념 -> (taxonomy, 태그) 우선순위 목록. us-gaap을 먼저 시도하고 ifrs-full로 대체.
_TAG_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),  # ODFL·CRWD 등 일부는 이 변형을 쓴다(2026-10-01 확인)
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
        ("ifrs-full", "Revenue"),
    ],
    "operating_income": [
        ("us-gaap", "OperatingIncomeLoss"),
        ("ifrs-full", "ProfitLossFromOperatingActivities"),
    ],
    "gross_profit": [
        ("us-gaap", "GrossProfit"),
        ("ifrs-full", "GrossProfit"),
    ],
    "cost_of_revenue": [
        ("us-gaap", "CostOfGoodsAndServicesSold"),
        ("us-gaap", "CostOfRevenue"),
        ("us-gaap", "CostOfGoodsSold"),
        ("ifrs-full", "CostOfSales"),
    ],
    "net_income": [
        ("us-gaap", "NetIncomeLoss"),
        ("us-gaap", "ProfitLoss"),
        ("ifrs-full", "ProfitLoss"),
    ],
    "stockholders_equity": [
        ("us-gaap", "StockholdersEquity"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ("ifrs-full", "Equity"),
    ],
    "cash": [
        ("us-gaap", "CashAndCashEquivalentsAtCarryingValue"),
        ("us-gaap", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"),
        ("ifrs-full", "CashAndCashEquivalents"),
    ],
    "long_term_debt": [
        ("us-gaap", "LongTermDebtNoncurrent"),
        ("us-gaap", "LongTermDebt"),
    ],
    "short_term_debt": [
        ("us-gaap", "LongTermDebtCurrent"),
        ("us-gaap", "ShortTermBorrowings"),
        ("us-gaap", "DebtCurrent"),
    ],
    "operating_cash_flow": [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities"),
        ("ifrs-full", "CashFlowsFromUsedInOperatingActivities"),
    ],
    "capex": [
        ("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment"),
        ("us-gaap", "PaymentsForCapitalImprovements"),
    ],
    "diluted_shares": [
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstandingIncludingParticipatingSecurities"),
    ],
}


def annual_value_series(company_facts: dict, concept: str, as_of: date) -> tuple[list[dict], tuple[str, str] | None]:
    """한 재무 개념의 연간(회계연도) 시계열을 as_of 시점 기준으로 뽑는다 (순수 함수).

    입력: company_facts(data.edgar.fetch_company_facts 결과), concept(_TAG_CANDIDATES의 키),
         as_of(이 날짜까지 제출된 값만 쓴다 — 미래 데이터 금지)
    출력: (series, tag_used) — series는 [{"fy","end","val","filed","form"}, ...] 회계연도
         종료일 오름차순. 못 찾으면 ([], None).

    form·fp만으로는 부족하다: 10-K 안의 보조 공시(분기별 세부 내역 등)가 같은 accession의
    form="10-K"·fp="FY" 메타데이터를 그대로 물고 나오는 경우가 실제로 있다(KLAC 확인,
    2026-10-01 — 회계연도가 6월 말인 회사에서 분기말 날짜들이 fp="FY"로 섞여 나옴). "start"가
    있는 흐름(duration) 개념은 (end-start)가 약 1년(340~380일)인 것만 연간으로 인정해
    이런 오염을 걸러낸다. "start"가 없는 시점(instant, 재무상태표) 개념은 이 검사를 건너뛴다.

    태그 후보는 "연간 필터를 통과하는 사실이 있는" 첫 번째 것을 쓴다 — 그냥 "사실이 있는"
    첫 번째가 아니다. BKNG(Booking Holdings)에서 확인(2026-10-01): 우선순위가 더 높은
    "RevenueFromContractWithCustomerExcludingAssessedTax" 태그는 10-Q 각주에만 쓰이고
    10-K 연간 합계는 여전히 옛 태그 "Revenues"로 공시된다. 후보 순서대로 사실 목록만 보고
    고르면(옛 extract_first_available 방식) 10-Q뿐인 태그에 걸려 연간 값을 영영 못 찾는다.
    """
    as_of_str = as_of.isoformat()

    def _annual_series_for(raw_entries: list[dict]) -> dict[str, dict]:
        by_end: dict[str, dict] = {}
        for e in raw_entries:
            if e.get("form") not in ANNUAL_FORMS or e.get("fp") != "FY":
                continue
            start = e.get("start")
            if start:
                period_days = (date.fromisoformat(e["end"]) - date.fromisoformat(start)).days
                if not (_MIN_ANNUAL_PERIOD_DAYS <= period_days <= _MAX_ANNUAL_PERIOD_DAYS):
                    continue
            filed = e.get("filed")
            if not filed or filed > as_of_str:
                continue
            end = e["end"]
            existing = by_end.get(end)
            if existing is None or filed > existing["filed"]:
                by_end[end] = {"fy": e.get("fy"), "end": end, "val": e["val"], "filed": filed, "form": e["form"]}
        return by_end

    for taxonomy, tag in _TAG_CANDIDATES[concept]:
        raw_entries = edgar.extract_fact_entries(company_facts, taxonomy, tag)
        if not raw_entries:
            continue
        by_end = _annual_series_for(raw_entries)
        if by_end:
            return sorted(by_end.values(), key=lambda r: r["end"]), (taxonomy, tag)
    return [], None


@dataclass
class IndicatorResult:
    """M1~M5 지표 하나의 계산 결과."""

    status: str  # "좋음" | "보통" | "주의" | "판단 불가"
    yearly_values: dict[str, float] = field(default_factory=dict)  # {회계연도 종료일: 값(%, 배수 등 지표별 단위)}
    tags_used: dict[str, tuple[str, str] | None] = field(default_factory=dict)  # 개념 -> 실제 쓴 (taxonomy, 태그)
    detail: str = ""


def compute_m1_roic(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M1 투하자본수익률(ROIC) = 세후 영업이익 ÷ (자기자본 + 차입금 − 현금).

    부채(장단기 차입금)가 아예 공시에 없는 회사는 0으로 본다(무차입 경영으로 흔함) —
    "판단 불가"로 만들지 않는다. 영업이익·자기자본·현금 중 하나라도 없는 연도는 그 연도를
    건너뛴다.
    """
    m1_cfg = cfg["moat"]["m1_roic"]
    tax_rate = cfg["moat"]["roic_tax_rate_pct"] / 100
    lookback = cfg["moat"]["lookback_years"]

    op_income, op_tag = annual_value_series(company_facts, "operating_income", as_of)
    equity, equity_tag = annual_value_series(company_facts, "stockholders_equity", as_of)
    cash, cash_tag = annual_value_series(company_facts, "cash", as_of)
    lt_debt, lt_tag = annual_value_series(company_facts, "long_term_debt", as_of)
    st_debt, st_tag = annual_value_series(company_facts, "short_term_debt", as_of)

    tags = {"operating_income": op_tag, "stockholders_equity": equity_tag, "cash": cash_tag, "long_term_debt": lt_tag, "short_term_debt": st_tag}

    if not op_income or not equity or not cash:
        return IndicatorResult(status="판단 불가", tags_used=tags, detail="영업이익·자기자본·현금 중 공시를 못 찾음")

    lt_by_end = {r["end"]: r["val"] for r in lt_debt}
    st_by_end = {r["end"]: r["val"] for r in st_debt}
    equity_by_end = {r["end"]: r["val"] for r in equity}
    cash_by_end = {r["end"]: r["val"] for r in cash}

    yearly: dict[str, float] = {}
    for row in op_income[-lookback:]:
        end = row["end"]
        if end not in equity_by_end or end not in cash_by_end:
            continue
        invested_capital = equity_by_end[end] + lt_by_end.get(end, 0) + st_by_end.get(end, 0) - cash_by_end[end]
        if invested_capital <= 0:
            continue  # 자기자본이 음수·0 이하면 ROIC%가 의미 없다 — 이상치로 별도 처리(detect_outliers)
        yearly[end] = round(row["val"] * (1 - tax_rate) / invested_capital * 100, 2)

    if len(yearly) < min(3, lookback):
        return IndicatorResult(status="판단 불가", tags_used=tags, yearly_values=yearly, detail="투하자본이 유효한 연도가 너무 적음(자기자본 음수 등)")

    good_years = sum(1 for v in yearly.values() if v >= m1_cfg["good_threshold_pct"])
    if good_years >= m1_cfg["good_min_years"]:
        status = "좋음"
    elif good_years <= m1_cfg["caution_max_good_years"]:
        status = "주의"
    else:
        status = "보통"
    return IndicatorResult(status=status, yearly_values=yearly, tags_used=tags, detail=f"최근 {len(yearly)}년 중 {good_years}년이 {m1_cfg['good_threshold_pct']}% 이상")


def compute_m2_gross_margin(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M2 매출총이익률 안정성 = 최근 5년 매출총이익률의 표준편차·수준.

    많은 회사(소매·서비스업 등, 예: COST·SBUX·KLAC — 2026-10-01 확인, 나스닥100의 약 44%가
    해당)는 손익계산서에 "매출총이익" 줄을 따로 안 두고 "GrossProfit" 태그를 안 쓴다.
    그런 회사는 매출총이익 = 매출 − 매출원가(cost_of_revenue)로 직접 계산한다.
    """
    m2_cfg = cfg["moat"]["m2_gross_margin"]
    lookback = cfg["moat"]["lookback_years"]

    revenue, rev_tag = annual_value_series(company_facts, "revenue", as_of)
    gross_profit, gp_tag = annual_value_series(company_facts, "gross_profit", as_of)
    cost_of_revenue, cost_tag = annual_value_series(company_facts, "cost_of_revenue", as_of)
    tags = {"revenue": rev_tag, "gross_profit": gp_tag, "cost_of_revenue": cost_tag}

    if not revenue or (not gross_profit and not cost_of_revenue):
        return IndicatorResult(status="판단 불가", tags_used=tags, detail="매출·매출총이익(또는 매출원가) 공시를 못 찾음")

    gp_by_end = {r["end"]: r["val"] for r in gross_profit}
    cost_by_end = {r["end"]: r["val"] for r in cost_of_revenue}
    yearly: dict[str, float] = {}
    for row in revenue[-lookback:]:
        end = row["end"]
        if not row["val"]:
            continue
        if end in gp_by_end:
            gp_val = gp_by_end[end]
        elif end in cost_by_end:
            gp_val = row["val"] - cost_by_end[end]  # GrossProfit 태그가 없으면 매출-매출원가로 직접 계산
        else:
            continue
        yearly[end] = round(gp_val / row["val"] * 100, 2)

    if len(yearly) < min(3, lookback):
        return IndicatorResult(status="판단 불가", tags_used=tags, yearly_values=yearly, detail="유효 연도가 너무 적음")

    values = list(yearly.values())
    level = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    if std <= m2_cfg["std_good_max_pp"] and level >= m2_cfg["level_good_min_pct"]:
        status = "좋음"
    elif std >= m2_cfg["std_caution_min_pp"] or level <= m2_cfg["level_caution_max_pct"]:
        status = "주의"
    else:
        status = "보통"
    return IndicatorResult(status=status, yearly_values=yearly, tags_used=tags, detail=f"평균 {level:.1f}%, 표준편차 {std:.1f}%p")


def compute_m3_recession_resilience(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M3 불황 버팀력 = 5년 중 가장 나빴던 해의 영업이익률 − 5년 평균."""
    m3_cfg = cfg["moat"]["m3_recession_resilience"]
    lookback = cfg["moat"]["lookback_years"]

    revenue, rev_tag = annual_value_series(company_facts, "revenue", as_of)
    op_income, op_tag = annual_value_series(company_facts, "operating_income", as_of)
    tags = {"revenue": rev_tag, "operating_income": op_tag}

    if not revenue or not op_income:
        return IndicatorResult(status="판단 불가", tags_used=tags, detail="매출·영업이익 공시를 못 찾음")

    op_by_end = {r["end"]: r["val"] for r in op_income}
    yearly: dict[str, float] = {}
    for row in revenue[-lookback:]:
        end = row["end"]
        if end not in op_by_end or not row["val"]:
            continue
        yearly[end] = round(op_by_end[end] / row["val"] * 100, 2)

    if len(yearly) < min(3, lookback):
        return IndicatorResult(status="판단 불가", tags_used=tags, yearly_values=yearly, detail="유효 연도가 너무 적음")

    values = list(yearly.values())
    avg = statistics.mean(values)
    worst = min(values)
    drop = avg - worst  # 양수면 하락, 음수면 최악의 해가 평균보다도 높다는 뜻(더 좋음)
    if drop <= m3_cfg["drop_good_max_pp"]:
        status = "좋음"
    elif drop >= m3_cfg["drop_caution_min_pp"]:
        status = "주의"
    else:
        status = "보통"
    return IndicatorResult(status=status, yearly_values=yearly, tags_used=tags, detail=f"평균 {avg:.1f}%, 최악의 해 {worst:.1f}% (하락폭 {drop:.1f}%p)")


def compute_m4_cash_conversion(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M4 현금 전환율 = (영업현금흐름 − 설비투자) ÷ 순이익, 연도별 비율의 평균 [가정: 계획서는
    단일 공식만 주고 5년 집계 방식을 안 정해 매년 비율을 낸 뒤 평균하는 쪽을 택함]."""
    m4_cfg = cfg["moat"]["m4_cash_conversion"]
    lookback = cfg["moat"]["lookback_years"]

    ocf, ocf_tag = annual_value_series(company_facts, "operating_cash_flow", as_of)
    capex, capex_tag = annual_value_series(company_facts, "capex", as_of)
    net_income, ni_tag = annual_value_series(company_facts, "net_income", as_of)
    tags = {"operating_cash_flow": ocf_tag, "capex": capex_tag, "net_income": ni_tag}

    if not ocf or not net_income:
        return IndicatorResult(status="판단 불가", tags_used=tags, detail="영업현금흐름·순이익 공시를 못 찾음")

    capex_by_end = {r["end"]: r["val"] for r in capex}
    ni_by_end = {r["end"]: r["val"] for r in net_income}
    yearly: dict[str, float] = {}
    for row in ocf[-lookback:]:
        end = row["end"]
        if end not in ni_by_end or not ni_by_end[end]:
            continue
        capex_val = capex_by_end.get(end, 0)  # 설비투자 공시가 없으면 0으로 본다(서비스업 등 소액)
        yearly[end] = round((row["val"] - capex_val) / ni_by_end[end], 3)

    if len(yearly) < min(3, lookback):
        return IndicatorResult(status="판단 불가", tags_used=tags, yearly_values=yearly, detail="유효 연도가 너무 적음")

    avg_ratio = statistics.mean(yearly.values())
    if avg_ratio >= m4_cfg["good_min_ratio"]:
        status = "좋음"
    elif avg_ratio <= m4_cfg["caution_max_ratio"]:
        status = "주의"
    else:
        status = "보통"
    return IndicatorResult(status=status, yearly_values=yearly, tags_used=tags, detail=f"최근 {len(yearly)}년 평균 {avg_ratio:.2f}")


def compute_m5_dilution(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M5 희석 = 희석 주식 수의 최근 1년 변화율."""
    m5_cfg = cfg["moat"]["m5_dilution"]

    shares, tag = annual_value_series(company_facts, "diluted_shares", as_of)
    tags = {"diluted_shares": tag}
    if len(shares) < 2:
        return IndicatorResult(status="판단 불가", tags_used=tags, detail="희석 주식 수 공시가 2년 미만")

    latest, prior = shares[-1], shares[-2]
    if not prior["val"]:
        return IndicatorResult(status="판단 불가", tags_used=tags, detail="직전 연도 값이 0 — 변화율 계산 불가")

    growth_pct = round((latest["val"] - prior["val"]) / prior["val"] * 100, 2)
    yearly = {prior["end"]: 0.0, latest["end"]: growth_pct}
    if growth_pct < m5_cfg["good_max_pct"]:
        status = "좋음"
    elif growth_pct >= m5_cfg["caution_min_pct"]:
        status = "주의"
    else:
        status = "보통"
    return IndicatorResult(status=status, yearly_values=yearly, tags_used=tags, detail=f"{prior['end']}→{latest['end']} {growth_pct:+.1f}%")


def compute_moat_grade(m1: IndicatorResult, m2: IndicatorResult, m3: IndicatorResult, m4: IndicatorResult, m5: IndicatorResult) -> str:
    """M1~M5로 해자 등급을 정한다 (순수 함수, 계획서 3장 규칙 그대로).

    - 넓음: M1·M2·M4 모두 "좋음" AND M3·M5 중 "주의"가 없음
    - 좁음: M1·M2·M4 중 2개 이상 "좋음"
    - 없음: 그 외
    ("판단 불가" 전체 등급은 core/moat.py 밖(analyze_company)에서 핵심 항목 부족으로 따로 판정한다.)
    """
    core = [m1.status, m2.status, m4.status]
    if all(s == "좋음" for s in core) and m3.status != "주의" and m5.status != "주의":
        return "넓음"
    if sum(1 for s in core if s == "좋음") >= 2:
        return "좁음"
    return "없음"


def detect_outliers(m1: IndicatorResult, cfg: dict, equity_series: list[dict] | None = None) -> list[str]:
    """ROIC 100% 초과, 자기자본 음수 등 이상치를 찾는다 (순수 함수, 사용자 지시 H3).

    입력: m1(compute_m1_roic 결과), cfg, equity_series(annual_value_series(..., "stockholders_equity", ...)
         결과 — 넘기면 음수 자기자본 연도를 따로 찾는다)
    출력: 사람이 읽을 이상치 설명 목록
    """
    out = []
    extreme = cfg["moat"]["outliers"]["roic_extreme_pct"]
    for end, val in m1.yearly_values.items():
        if val > extreme:
            out.append(f"ROIC {end} {val:.1f}% (>{extreme}%, 자기자본이 작아서 과장됐을 수 있음)")
    if equity_series:
        for row in equity_series:
            if row["val"] < 0:
                out.append(f"자기자본 음수: {row['end']} {row['val']:,}")
    return out


@dataclass
class MoatProfile:
    """한 회사의 해자 분석 결과 전체(H3 표 한 줄에 대응)."""

    ticker: str
    grade: str  # "넓음" | "좁음" | "없음" | "판단 불가"
    m1: IndicatorResult
    m2: IndicatorResult
    m3: IndicatorResult
    m4: IndicatorResult
    m5: IndicatorResult
    outliers: list[str] = field(default_factory=list)
    insufficient_data_reason: str | None = None
    data_years: list[str] = field(default_factory=list)  # 사용한 회계연도 종료일 목록(revenue 기준)


def analyze_company(company_facts: dict, ticker: str, cfg: dict, as_of: date) -> MoatProfile:
    """한 회사의 M1~M5 + 해자 등급을 전부 계산한다 (순수 함수, H2·H3의 진입점).

    핵심 항목(매출·영업이익·자기자본)의 연간 데이터가 cfg["moat"]["min_annual_years_required"]
    미만이면 전체를 "판단 불가"로 낸다(계획서 3장 등급 규칙).
    """
    revenue, _ = annual_value_series(company_facts, "revenue", as_of)
    op_income, _ = annual_value_series(company_facts, "operating_income", as_of)
    equity, _ = annual_value_series(company_facts, "stockholders_equity", as_of)
    core_years = len({r["end"] for r in revenue} & {r["end"] for r in op_income} & {r["end"] for r in equity})
    min_required = cfg["moat"]["min_annual_years_required"]

    m1 = compute_m1_roic(company_facts, cfg, as_of)
    m2 = compute_m2_gross_margin(company_facts, cfg, as_of)
    m3 = compute_m3_recession_resilience(company_facts, cfg, as_of)
    m4 = compute_m4_cash_conversion(company_facts, cfg, as_of)
    m5 = compute_m5_dilution(company_facts, cfg, as_of)
    outliers = detect_outliers(m1, cfg, equity)

    if core_years < min_required:
        grade = "판단 불가"
        reason = f"핵심 항목(매출·영업이익·자기자본)이 겹치는 연도가 {core_years}개뿐 (기준 {min_required}개)"
    else:
        grade = compute_moat_grade(m1, m2, m3, m4, m5)
        reason = None

    return MoatProfile(
        ticker=ticker, grade=grade, m1=m1, m2=m2, m3=m3, m4=m4, m5=m5,
        outliers=outliers, insufficient_data_reason=reason,
        data_years=sorted(r["end"] for r in revenue),
    )
