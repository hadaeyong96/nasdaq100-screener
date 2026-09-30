"""해자(경쟁 우위) 지표 계산 (해자 분석 H2, docs/design/moat_plan.md 3장, 2026-10-01 계산 방식 수정).

data/edgar.py가 공급한 원본 XBRL 사실에서 M1~M5 지표와 해자 등급을 계산한다. 순수 함수
모음이다 — 네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). 같은 company_facts·
cfg·as_of 입력이면 항상 같은 출력이다.

**미래 데이터 금지**: 모든 연간 값은 annual_value_series()가 filed(공시 제출일)가 as_of
이하인 것만 골라서 돌려준다. 같은 회계연도가 나중 10-K에 비교연도로 다시 실리면(흔함 —
SEC XBRL 사실은 누적된다) as_of 시점에 이미 제출돼 있던 것 중 가장 최근 제출본을 쓴다
(정정·재작성 반영, 그 뒤에 나온 값은 안 씀).

**XBRL 태그 병합**(2026-10-01 수정): 회사가 시간이 지나며 태그를 바꾸는 경우가 실제로
많다(AVGO는 2019년까지 "StockholdersEquity", 2023년부터
"StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"). 후보 중 하나만
골라 그 회사의 전체 기간에 쓰면 태그를 안 쓰는 연도가 통째로 빠진다. 그래서
annual_value_series는 "연도마다" 우선순위대로 후보 태그를 순회해 값을 채운다 — 어떤 태그를
썼는지 각 행의 "tag" 필드에 연도별로 남는다.

**해외 기업(20-F, 40-F, IFRS)**: ANNUAL_FORMS에 "20-F"(외국 민간 발행인)·"40-F"(캐나다
MJDS)를 포함하고 _TAG_CANDIDATES마다 ifrs-full 대체 태그를 둔다. 그래도 못 찾으면 해당
지표는 "판단 불가"로 남는다 — 억지로 값을 만들지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from datetime import date

from data import edgar

ANNUAL_FORMS = ("10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A")  # 40-F: 캐나다 기업(MJDS) 연차 보고서(예: SHOP)

_MIN_ANNUAL_PERIOD_DAYS = 340
_MAX_ANNUAL_PERIOD_DAYS = 380

# 재무 개념 -> (taxonomy, 태그) 우선순위 목록. us-gaap을 먼저 시도하고 ifrs-full로 대체.
# 후보가 여러 개면 "연도마다" 우선순위가 가장 높은 것부터 값을 채운다(_merge_tag_candidates).
_TAG_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "revenue": [
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax"),  # ODFL·CRWD 등 일부는 이 변형을 쓴다
        ("us-gaap", "Revenues"),
        ("us-gaap", "SalesRevenueNet"),
        ("ifrs-full", "Revenue"),
    ],
    "operating_income": [
        ("us-gaap", "OperatingIncomeLoss"),
        ("ifrs-full", "ProfitLossFromOperatingActivities"),
    ],
    "pretax_income": [
        ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"),
        ("us-gaap", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"),
        ("ifrs-full", "ProfitLossBeforeTax"),
    ],
    "interest_expense": [
        ("us-gaap", "InterestExpense"),
        ("us-gaap", "InterestExpenseDebt"),
        ("us-gaap", "InterestAndDebtExpense"),  # HON은 이 태그만 최근 연도에 씀(2026-10-01 확인)
        ("ifrs-full", "InterestExpense"),
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
    "current_assets": [
        ("us-gaap", "AssetsCurrent"),
        ("ifrs-full", "CurrentAssets"),
    ],
    "current_liabilities": [
        ("us-gaap", "LiabilitiesCurrent"),
        ("ifrs-full", "CurrentLiabilities"),
    ],
    "short_term_investments": [
        ("us-gaap", "ShortTermInvestments"),
        ("us-gaap", "MarketableSecuritiesCurrent"),
        ("ifrs-full", "CurrentFinancialAssetsAtFairValueThroughProfitOrLoss"),
    ],
    "net_ppe": [
        ("us-gaap", "PropertyPlantAndEquipmentNet"),
        # HON은 2023년부터 금융리스 사용권자산을 합친 이 태그로 바꿨다(2026-10-01 확인).
        ("us-gaap", "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"),
        ("ifrs-full", "PropertyPlantAndEquipment"),
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


def _annual_entries_by_end(raw_entries: list[dict], as_of_str: str) -> dict[str, dict]:
    """한 태그의 원본 사실 중 "진짜 연간 값"만 골라 회계연도 종료일로 묶는다 (순수 함수).

    form·fp만으로는 부족하다: 10-K 안의 보조 공시(분기별 세부 내역 등)가 같은 accession의
    form="10-K"·fp="FY" 메타데이터를 그대로 물고 나오는 경우가 실제로 있다(KLAC 확인 —
    회계연도가 6월 말인 회사에서 분기말 날짜들이 fp="FY"로 섞여 나옴). "start"가 있는 흐름
    (duration) 개념은 (end-start)가 약 1년(340~380일)인 것만 인정해 이런 오염을 걸러낸다.
    "start"가 없는 시점(instant) 개념은 이 검사를 건너뛴다. 같은 종료일이 여러 번(정정·
    비교연도 재수록) 나오면 as_of 이전 중 가장 최근 제출본을 쓴다.
    """
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


def annual_value_series(company_facts: dict, concept: str, as_of: date) -> list[dict]:
    """한 재무 개념의 연간(회계연도) 시계열을 as_of 시점 기준으로, 여러 태그를 병합해 뽑는다
    (순수 함수).

    입력: company_facts(data.edgar.fetch_company_facts 결과), concept(_TAG_CANDIDATES의 키),
         as_of(이 날짜까지 제출된 값만 쓴다 — 미래 데이터 금지)
    출력: [{"fy","end","val","filed","form","tag":(taxonomy,태그)}, ...] 회계연도 종료일
         오름차순. 어느 태그로도 못 찾은 연도는 그냥 없다.

    태그 병합(2026-10-01 수정): 후보를 "연도마다" 순서대로 시도한다 — 회사 전체 기간에 한
    태그만 쓰지 않는다. 우선순위가 낮은 태그부터 채우고 높은 태그로 덮어써서, 같은 연도에
    여러 태그가 다 있으면 우선순위가 더 높은 쪽이 이긴다. 예(2026-10-01 실측):
    - AVGO: "StockholdersEquity"는 2019년까지, "...IncludingPortionAttributableTo
      NoncontrollingInterest"는 2023년부터 — 병합해야 둘 다 살아남는다.
    - XEL·EXC·PYPL: 우선순위가 높은 "RevenueFromContractWithCustomer..." 태그가 옛날
      값이나 세그먼트 분할값만 갖고 있고, 진짜 연간 합계는 "Revenues"에 있다 — 그냥
      "사실이 있는 첫 태그"를 쓰면(옛 방식) 이 경우를 못 잡는다.
    """
    as_of_str = as_of.isoformat()
    by_end: dict[str, dict] = {}
    for taxonomy, tag in reversed(_TAG_CANDIDATES[concept]):
        raw_entries = edgar.extract_fact_entries(company_facts, taxonomy, tag)
        if not raw_entries:
            continue
        for end, row in _annual_entries_by_end(raw_entries, as_of_str).items():
            by_end[end] = {**row, "tag": (taxonomy, tag)}
    return sorted(by_end.values(), key=lambda r: r["end"])


def revenue_series(company_facts: dict, cfg: dict, as_of: date) -> tuple[list[dict], list[str]]:
    """매출 연간 시계열 — 같은 연도에 후보 태그가 여러 개 다 있고 값이 서로 다르면 차이
    크기에 따라 다르게 처리한다 (순수 함수, 사용자 지시 2026-10-01 확정·잠금).

    annual_value_series는 연도마다 "우선순위가 가장 높은 태그 하나"만 쓴다 — 같은 연도에
    다른 후보 태그가 훨씬 다른 값을 갖고 있어도 조용히 무시된다. 매출은 지표 5개 중
    4개(M1·M2·M3, analyze_company의 핵심 항목 검사)에 쓰여 영향이 크므로 이 검사를 추가한다.

    차이 크기별 처리(cfg["moat"]의 revenue_tag_mismatch_tolerance_pct·
    revenue_tag_mismatch_review_threshold_pct, 둘 다 [가정]):
    - tolerance_pct(5%) 이하: 무시(정상적인 반올림·재작성 차이로 봄)
    - tolerance_pct 초과 ~ review_threshold_pct(50%) 이하: "태그 불일치"로 보고 더 큰 값 사용
      (총매출과 부분 항목을 착각한 경우, 큰 쪽이 총매출에 더 가까울 때가 많다)
    - review_threshold_pct 초과: 후보 중 앞뒤 연도(병합 전 원래 값 기준) 평균과 가장 가까운
      값을 쓰고 "검토 필요"로 표시 — 차이가 이 정도면 "더 큰 값"이라는 단순 규칙을 못 믿는다
      (예: 세그먼트 매출 하나를 전사 매출로 잘못 태깅한 경우, 오히려 작은 값이 맞을 수도 있음).

    출력: (merged_series, notes) — notes는 "YYYY-MM-DD: 태그 불일치(...) — ..." 형식의
         문자열 목록(불일치 없으면 빈 목록).
    """
    as_of_str = as_of.isoformat()
    tolerance_pct = cfg["moat"]["revenue_tag_mismatch_tolerance_pct"]
    review_threshold_pct = cfg["moat"]["revenue_tag_mismatch_review_threshold_pct"]

    by_end_all_tags: dict[str, dict[tuple[str, str], float]] = {}
    for taxonomy, tag in _TAG_CANDIDATES["revenue"]:
        raw_entries = edgar.extract_fact_entries(company_facts, taxonomy, tag)
        if not raw_entries:
            continue
        for end, row in _annual_entries_by_end(raw_entries, as_of_str).items():
            by_end_all_tags.setdefault(end, {})[(taxonomy, tag)] = row["val"]

    merged = annual_value_series(company_facts, "revenue", as_of)
    merged_by_end = {r["end"]: r["val"] for r in merged}
    ends = [r["end"] for r in merged]

    notes: list[str] = []
    out: list[dict] = []
    for i, row in enumerate(merged):
        end = row["end"]
        values = list(by_end_all_tags.get(end, {}).values())
        if len(set(values)) <= 1:
            out.append(row)
            continue
        max_val, min_val = max(values), min(values)
        if max_val == 0:
            out.append(row)
            continue
        diff_pct = abs(max_val - min_val) / abs(max_val) * 100
        if diff_pct <= tolerance_pct:
            out.append(row)
        elif diff_pct <= review_threshold_pct:
            notes.append(f"{end}: 태그 불일치({min_val:,.0f} vs {max_val:,.0f}, {diff_pct:.0f}%) — 큰 값 사용")
            out.append({**row, "val": max_val, "tag_mismatch": True})
        else:
            prior_val = merged_by_end.get(ends[i - 1]) if i > 0 else None
            next_val = merged_by_end.get(ends[i + 1]) if i + 1 < len(ends) else None
            neighbors = [v for v in (prior_val, next_val) if v is not None]
            best = min(set(values), key=lambda v: abs(v - statistics.mean(neighbors))) if neighbors else max_val
            notes.append(
                f"{end}: 태그 불일치 큼({min_val:,.0f} vs {max_val:,.0f}, {diff_pct:.0f}%) — 검토 필요, 앞뒤 연도와 가장 가까운 값({best:,.0f}) 사용"
            )
            out.append({**row, "val": best, "tag_mismatch": True, "tag_mismatch_review": True})
    return out, notes


def operating_income_series(company_facts: dict, as_of: date) -> list[dict]:
    """영업이익 연간 시계열. 영업이익 줄이 없는 회사는 "세전이익 + 이자비용"으로 대신 계산한다
    (순수 함수, 사용자 지시 2026-10-01).

    KLAC·PCAR·ADP에서 확인: 최근 10-K에 "영업이익"(OperatingIncomeLoss) 줄 자체가 없다
    (대체 태그도 없음 — 손익계산서 구조상 아예 안 나눔). 세전이익(pretax_income)에 이자비용
    (interest_expense)을 다시 더하면 "이자·세금 차감 전 이익"(대략의 영업이익)에 가까운
    근사값을 얻는다. 실제 영업이익 공시가 있는 연도는 그 값을 그대로 쓰고, 없는 연도만
    대체 계산을 채운다. 대체로 채운 연도는 tag=("computed", "pretax_income+interest_expense")
    로 표시한다 — 실제 공시값과 구분해서 볼 수 있게.

    PCAR에서 확인(2026-10-01): 이자비용 태그가 2011년 이후로 아예 없다(대체 태그도 없음).
    이런 경우는 세전이익만으로 영업이익을 대신하고 tag=("computed", "pretax_income_only_no_interest_expense")
    로 "이자비용 없음"을 표시한다(사용자 지시) — 세전이익+이자비용보다는 덜 정확하지만 아예
    판단 불가로 두는 것보다는 낫다고 판단.
    """
    real = {r["end"]: r for r in annual_value_series(company_facts, "operating_income", as_of)}
    pretax = {r["end"]: r["val"] for r in annual_value_series(company_facts, "pretax_income", as_of)}
    interest = {r["end"]: r["val"] for r in annual_value_series(company_facts, "interest_expense", as_of)}

    out = dict(real)
    for end, pretax_val in pretax.items():
        if end in out:
            continue
        if end in interest:
            out[end] = {"fy": None, "end": end, "val": pretax_val + interest[end], "filed": None, "form": None, "tag": ("computed", "pretax_income+interest_expense")}
        else:
            out[end] = {"fy": None, "end": end, "val": pretax_val, "filed": None, "form": None, "tag": ("computed", "pretax_income_only_no_interest_expense")}
    return sorted(out.values(), key=lambda r: r["end"])


@dataclass
class IndicatorResult:
    """M1~M5 지표 하나의 계산 결과."""

    status: str  # "좋음" | "보통" | "주의" | "판단 불가"
    yearly_values: dict[str, float] = field(default_factory=dict)  # {회계연도 종료일: 값(%, 배수 등 지표별 단위)}
    yearly_display: dict[str, str] = field(default_factory=dict)  # {회계연도 종료일: 표시용 문자열} — M1은 100% 이상을 이렇게 뭉뚱그려 보여줌
    tags_used: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)  # 개념 -> {연도: 실제 쓴 (taxonomy, 태그)}
    detail: str = ""
    reference_yearly_values: dict[str, float] = field(default_factory=dict)  # M1: 기존 식(자기자본+차입금-현금) 참고값. M4: 설비투자 안 뺀 참고값(영업현금흐름÷순이익)
    reference_status: str | None = None  # M4 전용: 참고 비율을 reference_good_min_ratio로 판정했을 때의 등급(실제 등급 판정엔 안 씀)
    unmeasurable_years: list[str] = field(default_factory=list)  # M1 전용: 세후영업이익도 0 이하라 "측정 불가"로 뺀 연도
    capital_light_years: list[str] = field(default_factory=list)  # M1 전용: 분모가 매출의 2% 미만인데 세후영업이익>0 — "기준 충족(자본 거의 불필요)"
    notes: list[str] = field(default_factory=list)  # 매출 태그 불일치 등 추가로 남길 메모


def _tags_by_year(series: list[dict]) -> dict[str, tuple[str, str]]:
    return {r["end"]: r["tag"] for r in series}


def compute_m1_roic(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M1 투하자본수익률(ROIC) = 세후 영업이익 ÷ 영업 투하자본 (그린블랫 방식 + 하한, 2026-10-01 확정).

    영업 순운전자본 = (유동자산 − 현금·현금성자산 − 단기투자) − (유동부채 − 단기차입금).
    영업 투하자본 = max(영업 순운전자본, 0) + 순유형자산 — 운전자본이 마이너스면 0으로 본다
    (선수금으로 사업이 돌아가는 구독·소프트웨어 회사는 자본이 덜 드는 것이지 오류가 아니다 —
    ADSK 등에서 이전 정의(단순 유동자산−유동부채)가 "판단 불가"를 내던 문제를 해결).

    분모가 그해 매출의 min_invested_capital_ratio_of_revenue[가정] 미만이면:
    - 세후 영업이익 > 0 → "기준 충족"(자본이 거의 안 드는 사업)으로 보고 good_years에 넣는다.
      yearly_values엔 100.0을 넣고 yearly_display엔 "100% 이상"이라고 표시한다(실제 배수를
      보여주는 게 무의미해서다 — ADBE 201% 같은 값도 전부 이렇게 뭉뚱그린다).
    - 세후 영업이익 ≤ 0 → "측정 불가"로 그 연도를 뺀다.

    ROIC 100% 초과는 (자본 부담이 거의 없다는 뜻이라) 이상치로 안 보고 표시만 "100% 이상"으로
    한다 — detect_outliers에서 뺐다.

    기존 식(자기자본+차입금−현금)은 `reference_yearly_values`에 참고용으로 남긴다.
    """
    m1_cfg = cfg["moat"]["m1_roic"]
    tax_rate = cfg["moat"]["roic_tax_rate_pct"] / 100
    lookback = cfg["moat"]["lookback_years"]
    min_ratio = m1_cfg["min_invested_capital_ratio_of_revenue"]

    op_income = operating_income_series(company_facts, as_of)
    revenue, revenue_notes = revenue_series(company_facts, cfg, as_of)
    current_assets = annual_value_series(company_facts, "current_assets", as_of)
    current_liabilities = annual_value_series(company_facts, "current_liabilities", as_of)
    net_ppe = annual_value_series(company_facts, "net_ppe", as_of)
    cash = annual_value_series(company_facts, "cash", as_of)
    short_term_inv = annual_value_series(company_facts, "short_term_investments", as_of)
    short_term_debt_for_wc = annual_value_series(company_facts, "short_term_debt", as_of)
    equity = annual_value_series(company_facts, "stockholders_equity", as_of)
    lt_debt = annual_value_series(company_facts, "long_term_debt", as_of)
    st_debt = annual_value_series(company_facts, "short_term_debt", as_of)

    tags = {
        "operating_income": _tags_by_year(op_income), "current_assets": _tags_by_year(current_assets),
        "current_liabilities": _tags_by_year(current_liabilities), "net_ppe": _tags_by_year(net_ppe),
        "cash": _tags_by_year(cash), "short_term_investments": _tags_by_year(short_term_inv),
        "stockholders_equity": _tags_by_year(equity),
    }

    if not op_income or not current_assets or not current_liabilities or not net_ppe:
        return IndicatorResult(status="판단 불가", tags_used=tags, notes=revenue_notes, detail="영업이익·유동자산·유동부채·유형자산 중 공시를 못 찾음")

    rev_by_end = {r["end"]: r["val"] for r in revenue}
    ca_by_end = {r["end"]: r["val"] for r in current_assets}
    cl_by_end = {r["end"]: r["val"] for r in current_liabilities}
    ppe_by_end = {r["end"]: r["val"] for r in net_ppe}
    cash_by_end = {r["end"]: r["val"] for r in cash}
    sti_by_end = {r["end"]: r["val"] for r in short_term_inv}
    stdebt_by_end = {r["end"]: r["val"] for r in short_term_debt_for_wc}
    equity_by_end = {r["end"]: r["val"] for r in equity}
    lt_by_end = {r["end"]: r["val"] for r in lt_debt}
    st_by_end = {r["end"]: r["val"] for r in st_debt}

    yearly: dict[str, float] = {}
    yearly_display: dict[str, str] = {}
    reference_yearly: dict[str, float] = {}
    unmeasurable: list[str] = []
    capital_light: list[str] = []
    for row in op_income[-lookback:]:
        end = row["end"]
        if end not in ca_by_end or end not in cl_by_end or end not in ppe_by_end:
            continue
        net_working_capital = (ca_by_end[end] - cash_by_end.get(end, 0) - sti_by_end.get(end, 0)) - (cl_by_end[end] - stdebt_by_end.get(end, 0))
        invested_capital = max(net_working_capital, 0) + ppe_by_end[end]
        after_tax_income = row["val"] * (1 - tax_rate)
        rev = rev_by_end.get(end)

        too_small = rev is not None and rev > 0 and invested_capital < rev * min_ratio
        if too_small:
            if after_tax_income > 0:
                yearly[end] = 100.0
                yearly_display[end] = "100% 이상"
                capital_light.append(end)
            else:
                unmeasurable.append(end)
            continue
        if invested_capital <= 0:
            # 매출 공시가 없어 too_small 판단을 못 했지만 분모 자체가 0 이하인 경우의 안전망.
            unmeasurable.append(end)
            continue

        val = round(after_tax_income / invested_capital * 100, 2)
        yearly[end] = val
        yearly_display[end] = "100% 이상" if val >= 100 else f"{val:.1f}%"

        if end in equity_by_end and end in cash_by_end:
            old_capital = equity_by_end[end] + lt_by_end.get(end, 0) + st_by_end.get(end, 0) - cash_by_end[end]
            if old_capital > 0:
                reference_yearly[end] = round(after_tax_income / old_capital * 100, 2)

    if len(yearly) < min(3, lookback):
        return IndicatorResult(
            status="판단 불가", tags_used=tags, yearly_values=yearly, yearly_display=yearly_display,
            reference_yearly_values=reference_yearly, unmeasurable_years=unmeasurable, capital_light_years=capital_light,
            notes=revenue_notes, detail="영업 투하자본이 유효한 연도가 너무 적음(측정 불가 연도 제외 후 부족)",
        )

    good_years = sum(1 for v in yearly.values() if v >= m1_cfg["good_threshold_pct"])
    if good_years >= m1_cfg["good_min_years"]:
        status = "좋음"
    elif good_years <= m1_cfg["caution_max_good_years"]:
        status = "주의"
    else:
        status = "보통"
    detail = f"최근 {len(yearly)}년 중 {good_years}년이 {m1_cfg['good_threshold_pct']}% 이상"
    if capital_light:
        detail += f" (자본 거의 불필요 {len(capital_light)}개 연도 포함: {', '.join(capital_light)})"
    if unmeasurable:
        detail += f" (측정 불가 {len(unmeasurable)}개 연도 제외: {', '.join(unmeasurable)})"
    return IndicatorResult(
        status=status, yearly_values=yearly, yearly_display=yearly_display, tags_used=tags, detail=detail,
        reference_yearly_values=reference_yearly, unmeasurable_years=unmeasurable, capital_light_years=capital_light,
        notes=revenue_notes,
    )


def compute_m2_gross_margin(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M2 매출총이익률 안정성 = 최근 5년 매출총이익률의 표준편차만 본다.

    **2026-10-01 변경**: 수준 기준(40% 이상)을 뺐다(사용자 지시) — 도매유통(COST)처럼
    사업 구조상 매출총이익률이 원래 낮은 업종이 "주의"로 잘못 찍히는 문제가 있었다. 안정성
    (표준편차)만으로 판단한다. 평균 수준은 참고용으로 detail에 남긴다.

    많은 회사(소매·유틸리티 등)는 손익계산서에 "매출총이익" 줄을 따로 안 두고
    "GrossProfit" 태그를 안 쓴다(2026-10-01 확인, 나스닥100의 약 44%). 그런 회사는
    매출총이익 = 매출 − 매출원가(cost_of_revenue)로 직접 계산한다.
    """
    m2_cfg = cfg["moat"]["m2_gross_margin"]
    lookback = cfg["moat"]["lookback_years"]

    revenue, revenue_notes = revenue_series(company_facts, cfg, as_of)
    gross_profit = annual_value_series(company_facts, "gross_profit", as_of)
    cost_of_revenue = annual_value_series(company_facts, "cost_of_revenue", as_of)
    tags = {"revenue": _tags_by_year(revenue), "gross_profit": _tags_by_year(gross_profit), "cost_of_revenue": _tags_by_year(cost_of_revenue)}

    if not revenue or (not gross_profit and not cost_of_revenue):
        return IndicatorResult(status="판단 불가", tags_used=tags, notes=revenue_notes, detail="매출·매출총이익(또는 매출원가) 공시를 못 찾음")

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
        return IndicatorResult(status="판단 불가", tags_used=tags, yearly_values=yearly, notes=revenue_notes, detail="유효 연도가 너무 적음")

    values = list(yearly.values())
    level = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    if std <= m2_cfg["std_good_max_pp"]:
        status = "좋음"
    elif std >= m2_cfg["std_caution_min_pp"]:
        status = "주의"
    else:
        status = "보통"
    return IndicatorResult(
        status=status, yearly_values=yearly, tags_used=tags, notes=revenue_notes,
        detail=f"표준편차 {std:.1f}%p (참고: 평균 수준 {level:.1f}%, 등급엔 안 씀)",
    )


def compute_m3_recession_resilience(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M3 불황 버팀력 = 5년 중 가장 나빴던 해의 영업이익률 − 5년 평균.

    영업이익은 operating_income_series로 구해 "영업이익 줄이 없는 회사"(세전이익+이자비용
    대체)도 가급적 계산한다(사용자 지시 2026-10-01).
    """
    m3_cfg = cfg["moat"]["m3_recession_resilience"]
    lookback = cfg["moat"]["lookback_years"]

    revenue, revenue_notes = revenue_series(company_facts, cfg, as_of)
    op_income = operating_income_series(company_facts, as_of)
    tags = {"revenue": _tags_by_year(revenue), "operating_income": _tags_by_year(op_income)}

    if not revenue or not op_income:
        return IndicatorResult(status="판단 불가", tags_used=tags, notes=revenue_notes, detail="매출·영업이익 공시를 못 찾음")

    op_by_end = {r["end"]: r["val"] for r in op_income}
    yearly: dict[str, float] = {}
    for row in revenue[-lookback:]:
        end = row["end"]
        if end not in op_by_end or not row["val"]:
            continue
        yearly[end] = round(op_by_end[end] / row["val"] * 100, 2)

    if len(yearly) < min(3, lookback):
        return IndicatorResult(status="판단 불가", tags_used=tags, yearly_values=yearly, notes=revenue_notes, detail="유효 연도가 너무 적음")

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
    return IndicatorResult(
        status=status, yearly_values=yearly, tags_used=tags, notes=revenue_notes,
        detail=f"평균 {avg:.1f}%, 최악의 해 {worst:.1f}% (하락폭 {drop:.1f}%p)",
    )


def compute_m4_cash_conversion(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M4 현금 전환율 = 영업현금흐름 ÷ 순이익, 연도별 비율의 평균 (2026-10-01 확정·잠금).

    **주 지표 변경**(사용자 지시, 결과를 본 뒤 확정 — store/trials.db에 사유 기록,
    scripts/moat_lock.py): 라운드3에서 설비투자를 뺀 기존 식과 안 뺀 참고 비율을
    비교해 봤을 때 8개 종목의 등급이 달라졌다. 설비투자는 연도별로 들쭉날쭉해(대형
    투자 시기 vs 아닌 시기) "이익이 진짜 현금인가"라는 M4의 원래 질문에는 영업현금
    흐름 자체가 더 안정적인 대리 지표라고 판단해 주 지표로 승격했다. 기존 식
    ((영업현금흐름-설비투자)÷순이익)은 reference_yearly_values·reference_status에
    참고용으로 남긴다.
    """
    m4_cfg = cfg["moat"]["m4_cash_conversion"]
    lookback = cfg["moat"]["lookback_years"]

    ocf = annual_value_series(company_facts, "operating_cash_flow", as_of)
    capex = annual_value_series(company_facts, "capex", as_of)
    net_income = annual_value_series(company_facts, "net_income", as_of)
    tags = {"operating_cash_flow": _tags_by_year(ocf), "capex": _tags_by_year(capex), "net_income": _tags_by_year(net_income)}

    if not ocf or not net_income:
        return IndicatorResult(status="판단 불가", tags_used=tags, detail="영업현금흐름·순이익 공시를 못 찾음")

    capex_by_end = {r["end"]: r["val"] for r in capex}
    ni_by_end = {r["end"]: r["val"] for r in net_income}
    yearly: dict[str, float] = {}
    reference_yearly: dict[str, float] = {}
    for row in ocf[-lookback:]:
        end = row["end"]
        if end not in ni_by_end or not ni_by_end[end]:
            continue
        capex_val = capex_by_end.get(end, 0)  # 설비투자 공시가 없으면 0으로 본다(서비스업 등 소액)
        yearly[end] = round(row["val"] / ni_by_end[end], 3)  # 주 지표: 설비투자 안 뺌
        reference_yearly[end] = round((row["val"] - capex_val) / ni_by_end[end], 3)  # 참고: 기존 식(설비투자 뺌)

    if len(yearly) < min(3, lookback):
        return IndicatorResult(
            status="판단 불가", tags_used=tags, yearly_values=yearly, reference_yearly_values=reference_yearly, detail="유효 연도가 너무 적음",
        )

    avg_ratio = statistics.mean(yearly.values())
    if avg_ratio >= m4_cfg["good_min_ratio"]:
        status = "좋음"
    elif avg_ratio <= m4_cfg["caution_max_ratio"]:
        status = "주의"
    else:
        status = "보통"

    reference_status = None
    if reference_yearly:
        ref_avg = statistics.mean(reference_yearly.values())
        if ref_avg >= m4_cfg["reference_good_min_ratio"]:
            reference_status = "좋음"
        elif ref_avg <= m4_cfg["caution_max_ratio"]:
            reference_status = "주의"
        else:
            reference_status = "보통"

    return IndicatorResult(
        status=status, yearly_values=yearly, tags_used=tags, detail=f"최근 {len(yearly)}년 평균 {avg_ratio:.2f}",
        reference_yearly_values=reference_yearly, reference_status=reference_status,
    )


def compute_m5_dilution(company_facts: dict, cfg: dict, as_of: date) -> IndicatorResult:
    """M5 희석 = 희석 주식 수의 최근 1년 변화율."""
    m5_cfg = cfg["moat"]["m5_dilution"]

    shares = annual_value_series(company_facts, "diluted_shares", as_of)
    tags = {"diluted_shares": _tags_by_year(shares)}
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
    """자기자본 음수 등 이상치를 찾는다 (순수 함수, 사용자 지시 H3·2026-10-01 수정).

    ROIC 100% 초과는 더 이상 이상치로 안 본다(M1이 "100% 이상"으로 뭉뚱그려 표시하므로 —
    compute_m1_roic 참고). 자기자본 음수 점검은 최근 lookback_years(기본 5년) 안으로
    좁힌다(사용자 지시) — 오래된 상장 전·초기 성장기 수치까지 매번 이상치로 뜨는 걸 막는다.

    입력: m1(compute_m1_roic 결과, 지금은 안 씀 — 시그니처 유지용), cfg,
         equity_series(annual_value_series(..., "stockholders_equity", ...) 결과)
    출력: 사람이 읽을 이상치 설명 목록
    """
    out = []
    lookback = cfg["moat"]["lookback_years"]
    if equity_series:
        for row in equity_series[-lookback:]:
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
    미만이면 전체를 "판단 불가"로 낸다(계획서 3장 등급 규칙). 영업이익은 대체 계산
    (operating_income_series)을 포함해서 센다 — 그래야 영업이익 줄이 없는 회사도 매출·
    자기자본만 충분하면 판단 불가를 면할 수 있다.
    """
    revenue, _revenue_notes = revenue_series(company_facts, cfg, as_of)
    op_income = operating_income_series(company_facts, as_of)
    equity = annual_value_series(company_facts, "stockholders_equity", as_of)
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


def compute_config_hash(moat_cfg: dict) -> str:
    """config.yaml의 moat: 섹션 해시를 만든다 (순수 함수, 기준 잠금 확인용, 2026-10-01).

    입력: moat_cfg(cfg["moat"] 그대로)
    출력: 16자리 16진 해시 — 키 순서와 무관하게 내용이 같으면 같은 값(store.fund.params_hash와
         같은 방식).
    """
    canon = json.dumps(moat_cfg, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def check_lock(moat_cfg: dict, locked_hash: str) -> str | None:
    """현재 moat 설정이 잠긴 기준과 다르면 경고 문구를 돌려준다 (순수 함수, 2026-10-01).

    과거 검증(H4, 아직 안 만듦)이 이 기준으로 계산했다는 걸 전제로 하므로, 설정이
    바뀐 뒤에도 조용히 다른 결과를 내면 안 된다 — H4 스크립트는 실행 전에 이 함수로
    확인하고 경고가 있으면 멈추거나 사용자에게 알려야 한다.

    입력: moat_cfg(cfg["moat"]), locked_hash(store/trials.db에 기록된 잠금 해시)
    출력: 다르면 경고 문자열, 같으면 None
    """
    current = compute_config_hash(moat_cfg)
    if current != locked_hash:
        return (
            f"⚠️ moat 설정이 잠긴 기준과 다릅니다(잠긴 해시 {locked_hash}, 현재 해시 {current}). "
            "과거 검증 결과는 이 변경 전 기준으로 낸 것일 수 있습니다 — 설정을 잠긴 값으로 되돌리거나, "
            "바꾼 내용을 새 시도로 사전 등록하세요."
        )
    return None
