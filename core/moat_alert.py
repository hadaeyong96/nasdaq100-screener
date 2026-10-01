"""해자 약화 경보 B3 (순수 함수, docs/p6_2a_instructions.md 3장, 사용자 지시 2026-10-01).

보유 종목의 해자가 무너지고 있는지 분기 단위로 조기에 감지한다. 세 가지 경보:
1. 이익률 경보: 분기 매출총이익률 2분기 연속 하락 그리고 전년 같은 분기 대비 2%p 이상 하락
2. 수익성 경보: 최근 연도 M1(ROIC) 15% 미만 (config.yaml moat.m1_roic.good_threshold_pct)
3. 등급 경보: 해자 등급이 "넓음"에서 내려감

분기 값은 data.edgar.extract_duration_fact_entries로 10-Q의 "3개월" 값(기간 80~100일)만
골라낸다. 4분기는 연간(core.moat.annual_value_series) − 9개월 누적(10-Q, 기간 260~290일)으로
계산한다. 필요한 공시를 못 찾으면 "판단 불가"로 남긴다 — 억지로 값을 만들지 않는다.

네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). core/moat.py는 재사용만 하고
수정하지 않는다. data/edgar.py도 extract_duration_fact_entries(2026-10-01 신규 추가분)만
쓰고 기존 함수는 바꾸지 않는다.
"""

from __future__ import annotations

from datetime import date

from core import moat
from data import edgar

_QUARTER_MIN_DAYS = 80
_QUARTER_MAX_DAYS = 100
_CUMULATIVE_9MO_MIN_DAYS = 260
_CUMULATIVE_9MO_MAX_DAYS = 290
# Q4 = 연간 - 9개월 누적: 9개월 누적의 end(3분기말)가 연간 end(기말)보다 이만큼(일) 전이어야
# "같은 회계연도"로 본다 — 분기 길이(약 3개월) 범위에 여유를 둔 60~110일.
_Q4_LOOKBACK_MIN_DAYS = 60
_Q4_LOOKBACK_MAX_DAYS = 110
# 전년 같은 분기 비교: 분기말 날짜가 약 1년(365일) 전이어야 한다. 윤년·영업일 차이 여유.
_YOY_MIN_DAYS = 340
_YOY_MAX_DAYS = 390
_GROSS_MARGIN_YOY_DROP_PP = 2.0  # [고정값] 사용자 지시 2026-10-01 — 전년 같은 분기 대비 하락폭 기준


def quarterly_concept_series(company_facts: dict, concept: str, as_of: date) -> list[dict]:
    """한 재무 개념의 분기(3개월) 시계열을 as_of 시점 기준으로, 여러 태그를 병합해 뽑는다
    (순수 함수, core.moat.annual_value_series와 같은 병합 방식 — 연도 대신 분기말 기준).

    입력: company_facts, concept(core.moat._TAG_CANDIDATES의 키), as_of
    출력: [{"end","val","filed","tag":(taxonomy,태그)}, ...] 분기말 오름차순
    """
    by_end: dict[str, dict] = {}
    for taxonomy, tag in reversed(moat._TAG_CANDIDATES[concept]):
        for e in edgar.extract_duration_fact_entries(
            company_facts, taxonomy, tag, as_of, _QUARTER_MIN_DAYS, _QUARTER_MAX_DAYS, forms=("10-Q",)
        ):
            end = e["end"]
            existing = by_end.get(end)
            if existing is None or e["filed"] > existing["filed"]:
                by_end[end] = {"end": end, "val": e["val"], "filed": e["filed"], "tag": (taxonomy, tag)}
    return sorted(by_end.values(), key=lambda r: r["end"])


def cumulative_9mo_concept_series(company_facts: dict, concept: str, as_of: date) -> list[dict]:
    """한 재무 개념의 "회계연도 시작~3분기말" 9개월 누적 시계열 (순수 함수, Q4 계산용 재료).

    출력: [{"end"(3분기말),"val","filed","tag":(taxonomy,태그)}, ...] 오름차순
    """
    by_end: dict[str, dict] = {}
    for taxonomy, tag in reversed(moat._TAG_CANDIDATES[concept]):
        for e in edgar.extract_duration_fact_entries(
            company_facts, taxonomy, tag, as_of, _CUMULATIVE_9MO_MIN_DAYS, _CUMULATIVE_9MO_MAX_DAYS, forms=("10-Q",)
        ):
            end = e["end"]
            existing = by_end.get(end)
            if existing is None or e["filed"] > existing["filed"]:
                by_end[end] = {"end": end, "val": e["val"], "filed": e["filed"], "tag": (taxonomy, tag)}
    return sorted(by_end.values(), key=lambda r: r["end"])


def quarterly_series_with_q4(company_facts: dict, concept: str, as_of: date, cfg: dict | None = None) -> list[dict]:
    """Q1~Q3는 10-Q 3개월 값, Q4는 연간 − 9개월 누적으로 채운 분기 시계열 (순수 함수).

    9개월 누적 중 연간 종료일보다 60~110일 전에 끝나는 것을 "같은 회계연도 3분기"로 보고
    뺀다. 짝지을 9개월 누적이 없으면 그 연도의 Q4는 건너뛴다(계산 불가 — 억지로 안 만듦).

    concept이 "revenue"이고 cfg가 있으면 연간 값은 core.moat.annual_value_series 대신
    core.moat.revenue_series(태그 불일치 보정 — M1·M2·M3도 전부 이걸 쓴다)를 쓴다. 매출은
    같은 연도에 후보 태그 값이 크게 달라지는 경우가 실제로 있어서(예: MELI 2025년 10-K,
    두 태그가 30% 차이) 보정 없이 annual_value_series를 그대로 쓰면 Q4(연간−9개월누적)가
    터무니없이 작거나 음수로 나올 수 있다(2026-10-01 실측으로 발견).
    """
    quarters = quarterly_concept_series(company_facts, concept, as_of)
    if concept == "revenue" and cfg is not None:
        annual, _notes = moat.revenue_series(company_facts, cfg, as_of)
    else:
        annual = moat.annual_value_series(company_facts, concept, as_of)
    cumulative = cumulative_9mo_concept_series(company_facts, concept, as_of)

    out = list(quarters)
    seen_ends = {r["end"] for r in quarters}
    for a in annual:
        a_end = date.fromisoformat(a["end"])
        candidates = [
            c for c in cumulative
            if _Q4_LOOKBACK_MIN_DAYS <= (a_end - date.fromisoformat(c["end"])).days <= _Q4_LOOKBACK_MAX_DAYS
        ]
        if not candidates or a["end"] in seen_ends:
            continue
        best = max(candidates, key=lambda c: c["end"])
        out.append({"end": a["end"], "val": a["val"] - best["val"], "filed": a["filed"], "tag": ("computed", "annual_minus_9mo_cumulative")})
    return sorted(out, key=lambda r: r["end"])


def quarterly_gross_margin_series(company_facts: dict, cfg: dict, as_of: date) -> list[dict]:
    """분기 매출총이익률 시계열(%) — core.moat.compute_m2_gross_margin과 같은 매출총이익
    확보 방식(매출총이익 태그 우선, 없으면 매출−매출원가)을 분기 단위로 적용한다 (순수 함수).

    출력: [{"end","margin_pct"}, ...] 분기말 오름차순. 매출이 0이거나 매출총이익(또는
         매출원가)을 못 구한 분기는 건너뛴다.
    """
    revenue = quarterly_series_with_q4(company_facts, "revenue", as_of, cfg)
    gross_profit = quarterly_series_with_q4(company_facts, "gross_profit", as_of)
    cost_of_revenue = quarterly_series_with_q4(company_facts, "cost_of_revenue", as_of)
    gp_by_end = {r["end"]: r["val"] for r in gross_profit}
    cost_by_end = {r["end"]: r["val"] for r in cost_of_revenue}

    out = []
    for r in revenue:
        if not r["val"]:
            continue
        end = r["end"]
        if end in gp_by_end:
            gp = gp_by_end[end]
        elif end in cost_by_end:
            gp = r["val"] - cost_by_end[end]
        else:
            continue
        out.append({"end": end, "margin_pct": round(gp / r["val"] * 100, 2)})
    return sorted(out, key=lambda r: r["end"])


def gross_margin_alert(quarterly_margins: list[dict]) -> dict:
    """이익률 경보: 분기 매출총이익률 2분기 연속 하락 **그리고** 전년 같은 분기 대비
    2%p 이상 하락 (순수 함수, 둘 다 만족해야 경보).

    입력: quarterly_margins(quarterly_gross_margin_series 결과, 분기말 오름차순)
    출력: {"status": "경보"|"경보 아님"|"판단 불가", "detail": str, ...}
    """
    series = sorted(quarterly_margins, key=lambda r: r["end"])
    if len(series) < 3:
        return {"status": "판단 불가", "detail": f"유효 분기가 {len(series)}개뿐(3개 이상 필요)"}

    latest, prev1, prev2 = series[-1], series[-2], series[-3]
    consecutive_decline = latest["margin_pct"] < prev1["margin_pct"] < prev2["margin_pct"]

    latest_end = date.fromisoformat(latest["end"])
    yoy_candidates = [r for r in series[:-1] if _YOY_MIN_DAYS <= (latest_end - date.fromisoformat(r["end"])).days <= _YOY_MAX_DAYS]
    if not yoy_candidates:
        return {
            "status": "판단 불가", "detail": "전년 같은 분기 값을 찾지 못함",
            "consecutive_decline": consecutive_decline,
        }
    yoy = max(yoy_candidates, key=lambda r: r["end"])
    yoy_drop_pp = round(yoy["margin_pct"] - latest["margin_pct"], 2)
    yoy_triggered = yoy_drop_pp >= _GROSS_MARGIN_YOY_DROP_PP
    triggered = consecutive_decline and yoy_triggered

    yoy_label = f"{yoy_drop_pp:.1f}%p 하락" if yoy_drop_pp >= 0 else f"{abs(yoy_drop_pp):.1f}%p 상승"
    detail = (
        f"{latest['end']} {latest['margin_pct']:.1f}% (직전 {prev1['end']} {prev1['margin_pct']:.1f}%, "
        f"전전 {prev2['end']} {prev2['margin_pct']:.1f}%, 전년동분기 {yoy['end']} {yoy['margin_pct']:.1f}%, "
        f"전년동분기 대비 {yoy_label})"
    )
    return {"status": "경보" if triggered else "경보 아님", "detail": detail, "consecutive_decline": consecutive_decline, "yoy_drop_pp": yoy_drop_pp}


def profitability_alert(m1: "moat.IndicatorResult", cfg: dict) -> dict:
    """수익성 경보: 최근 연도 M1(ROIC)이 기준(config.yaml moat.m1_roic.good_threshold_pct,
    기본 15%) 미만이면 경보 (순수 함수).

    입력: m1(core.moat.compute_m1_roic 결과), cfg
    출력: {"status": "경보"|"경보 아님"|"판단 불가", "detail": str}
    """
    threshold = cfg["moat"]["m1_roic"]["good_threshold_pct"]
    if not m1.yearly_values:
        return {"status": "판단 불가", "detail": "M1 연간 값 없음"}
    latest_end = max(m1.yearly_values)
    latest_val = m1.yearly_values[latest_end]
    triggered = latest_val < threshold
    return {"status": "경보" if triggered else "경보 아님", "detail": f"{latest_end} ROIC {latest_val:.1f}% (기준 {threshold}%)"}


def grade_alert(previous_grade: str, current_grade: str) -> dict:
    """등급 경보: 해자 등급이 "넓음"에서 내려가면 경보 (순수 함수)."""
    triggered = previous_grade == "넓음" and current_grade != "넓음"
    return {"status": "경보" if triggered else "경보 아님", "detail": f"{previous_grade} → {current_grade}"}


def evaluate_company_alerts(
    ticker: str, company_facts: dict, cfg: dict, as_of: date, previous_grade: str, profile: "moat.MoatProfile"
) -> dict:
    """한 종목의 B3 경보 3종을 모두 평가한다 (순수 함수, core.moat.analyze_company 결과를
    그대로 받아 재사용 — M1을 다시 계산하지 않는다).

    입력: ticker, company_facts, cfg, as_of, previous_grade(직전 기록된 등급),
         profile(core.moat.analyze_company(company_facts, ticker, cfg, as_of) 결과)
    출력: {"ticker", "gross_margin": {...}, "profitability": {...}, "grade": {...}, "any_triggered": bool}
    """
    gm_series = quarterly_gross_margin_series(company_facts, cfg, as_of)
    gm = gross_margin_alert(gm_series)
    prof = profitability_alert(profile.m1, cfg)
    grade = grade_alert(previous_grade, profile.grade)
    return {
        "ticker": ticker, "gross_margin": gm, "profitability": prof, "grade": grade,
        "any_triggered": any(x["status"] == "경보" for x in (gm, prof, grade)),
    }
