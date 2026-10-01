"""해자 실시간 paper 검증 — 시작 포트폴리오 구성·평가 (순수 함수, docs/design/moat_paper.md).

docs/design/moat_paper.md(moat-paper-v1-locked)의 규칙을 그대로 구현한다. core/moat_backtest.py
(H4용, 동일비중+업종캡만 지원)는 재사용만 하고 수정하지 않는다 — P2(시가총액 비중+업종캡+
종목캡)는 시작 비중이 균등하지 않고 캡이 두 종류라 이 파일에 새 함수로 둔다.

네트워크·파일·DB·현재 시각에 접근하지 않는다(core/ 원칙). scripts/moat_paper.py가 EDGAR·
yfinance·가격 조회를 담당하고, 여기는 이미 가져온 값으로 계산만 한다.
"""

from __future__ import annotations

from datetime import date

from core import moat
from data import edgar

# paper 코드가 절대 읽으면 안 되는 가격 하한선(moat_paper.md "봉인 규칙") — 진입일 자체가
# 2026-10-01이라 이보다 이른 가격을 쓸 이유가 없다. core/seal.py의 seal_date(2022-01-01,
# H4 과거 검증 전용)와는 별개의 하한선이다.
PAPER_PRICE_FLOOR = date(2026, 10, 1)


class PaperSealError(RuntimeError):
    """paper 코드가 PAPER_PRICE_FLOOR보다 이른 날짜의 가격을 요청했을 때 낸다."""


def enforce_paper_floor(requested_date: date) -> None:
    """요청 날짜가 PAPER_PRICE_FLOOR보다 이르면 PaperSealError를 낸다 (순수 함수).

    입력: requested_date(가격을 요청하려는 날짜)
    출력: 없음. 통과하면 아무 일도 하지 않는다.
    """
    if requested_date < PAPER_PRICE_FLOOR:
        raise PaperSealError(
            f"paper는 {PAPER_PRICE_FLOOR.isoformat()} 이전 가격을 읽을 수 없습니다(요청: {requested_date.isoformat()})."
        )


def compute_entry_price(open_price: float, slippage_pct: float) -> float:
    """체결가 = 시가 × (1 + 슬리피지) (순수 함수, moat_paper.md "진입" 규칙 — 매수는 불리하게)."""
    return open_price * (1 + slippage_pct / 100)


def merge_price_fills(existing_entries: dict[str, dict], new_fills: dict[str, dict]) -> dict[str, dict]:
    """이미 "체결"된 entry는 손대지 않고, "진입 대기"인 것만 new_fills로 채운다 (순수 함수,
    moat_paper.md "진입" 규칙 — "시작 파일이 이미 있으면 덮어쓰지 마"의 유일한 예외가
    "아직 비어 있는 진입 대기 칸을 채우는 것"이다).

    입력: existing_entries({티커: {"status": "진입 대기"|"체결", ...}}), new_fills(이번 실행에서
         새로 구한 값 — "진입 대기"였던 티커만 넘겨도 되고, 다 넘겨도 "체결"인 건 무시된다)
    출력: 병합된 {티커: {...}}. existing에 없던 티커는 추가하지 않는다(호출부가 미리 전체
         티커로 초기화해 둔다는 전제).
    """
    out = dict(existing_entries)
    for t, entry in existing_entries.items():
        if entry.get("status") == "진입 대기" and t in new_fills:
            out[t] = new_fills[t]
    return out


# ── P1: 동일 비중 (core/moat_backtest.allocate_equal_weight_with_sector_cap 재사용) ────────


# ── P2: 시가총액 비중 + 업종 30% + 종목 10% ────────────────────────────────────────────


def apply_group_cap(weights: dict[str, float], groups: dict[str, str], cap_pct: float) -> dict[str, float]:
    """임의의 시작 비중에 그룹(업종 또는 종목 자신)별 상한을 적용한다 (순수 함수,
    core.moat_backtest.allocate_equal_weight_with_sector_cap과 같은 water-filling 알고리즘을
    "임의의 시작 비중 + 임의의 그룹"으로 일반화한 것 — 그 함수는 동일비중 전용이라 재사용할
    수 없어 새로 둔다).

    groups에 {종목: 종목}을 넘기면 종목별 캡(싱글톤 그룹)이 되고, {종목: 업종}을 넘기면
    업종 캡이 된다 — market_cap_weights_with_caps가 둘을 번갈아 적용한다.

    입력: weights(합 1, 임의 분포), groups({종목: 그룹 이름}, 없으면 종목 자신을 그룹으로 봄),
         cap_pct(0~100)
    출력: 캡을 적용한 {종목: 비중}. 캡을 지킬 수 없으면(재배분할 곳이 없음) 최대한 캡에
         가깝게 맞추고 멈춘다(기존 함수와 같은 안전장치).
    """
    weights = dict(weights)
    if not weights:
        return weights
    cap = cap_pct / 100
    all_groups = {groups.get(t, t) for t in weights}
    capped_groups: set[str] = set()

    for _ in range(len(all_groups) + 1):
        totals: dict[str, float] = {}
        for t, w in weights.items():
            g = groups.get(t, t)
            totals[g] = totals.get(g, 0.0) + w
        over = {g for g in all_groups if g not in capped_groups and totals.get(g, 0.0) > cap + 1e-9}
        if not over:
            break
        free_groups = all_groups - capped_groups - over
        free_total = sum(weights[t] for t in weights if groups.get(t, t) in free_groups)
        if free_total <= 0:
            break
        excess = 0.0
        for t in weights:
            g = groups.get(t, t)
            if g in over:
                new_w = weights[t] * (cap / totals[g])
                excess += weights[t] - new_w
                weights[t] = new_w
        for t in weights:
            g = groups.get(t, t)
            if g in free_groups:
                weights[t] += excess * (weights[t] / free_total)
        capped_groups |= over
    return weights


def market_cap_weights_with_caps(
    tickers: list[str], market_caps: dict[str, float], sectors: dict[str, str],
    sector_cap_pct: float, stock_cap_pct: float, max_passes: int = 20,
) -> dict[str, float]:
    """시가총액 비중을 내고, 업종 상한과 종목 상한을 번갈아 적용해 수렴시킨다 (순수 함수,
    moat_paper.md P2 "비중" 규칙).

    종목 캡을 적용하면 다른 종목 비중이 올라가 업종 캡을 새로 넘을 수 있고, 반대도
    마찬가지라 두 캡을 번갈아 적용하며 더 안 바뀔 때까지(또는 max_passes) 반복한다.

    입력: tickers, market_caps({종목: 시가총액, 없거나 0 이하면 0으로 봄}), sectors,
         sector_cap_pct(기본 30), stock_cap_pct(기본 10)
    출력: {종목: 비중}(합 1). market_caps가 전부 0/누락이면 동일 비중으로 시작한다(안전망).
    """
    n = len(tickers)
    if n == 0:
        return {}
    total = sum(max(market_caps.get(t, 0.0) or 0.0, 0.0) for t in tickers)
    if total <= 0:
        weights = {t: 1.0 / n for t in tickers}
    else:
        weights = {t: max(market_caps.get(t, 0.0) or 0.0, 0.0) / total for t in tickers}

    stock_groups = {t: t for t in tickers}
    for _ in range(max_passes):
        before = dict(weights)
        weights = apply_group_cap(weights, stock_groups, stock_cap_pct)
        weights = apply_group_cap(weights, sectors, sector_cap_pct)
        if all(abs(weights[t] - before[t]) < 1e-9 for t in tickers):
            break
    return weights


# ── 시가총액 재료: 유통주식수 (사용자 지시 2026-10-01, moat_paper.md P2) ──────────────────


def latest_cover_page_shares(company_facts: dict, as_of: date) -> float | None:
    """EDGAR 표지(cover page) 공시 dei:EntityCommonStockSharesOutstanding 중 filed<=as_of인
    가장 최근(end) 값 (순수 함수, 우선순위 1번).

    주식 종류가 여러 개인 회사는 종류마다 별도 공시가 있을 수 있다 — 이 함수는 "그 회사의
    EDGAR 사실 하나"에 대한 값만 뽑는다. 종류 합산은 aggregate_shares_by_cik가 한다.
    """
    as_of_str = as_of.isoformat()
    entries = edgar.extract_fact_entries(company_facts, "dei", "EntityCommonStockSharesOutstanding")
    valid = [e for e in entries if e.get("filed") and e.get("end") and e["filed"] <= as_of_str]
    if not valid:
        return None
    best = max(valid, key=lambda e: (e["end"], e["filed"]))
    return best["val"]


def latest_quarterly_diluted_shares(company_facts: dict, as_of: date) -> float | None:
    """가장 최근 10-Q 3개월 구간의 희석주식수 (순수 함수, 우선순위 2번 — dei 표지 공시가
    없을 때만 쓴다). core.moat._TAG_CANDIDATES["diluted_shares"]와 같은 태그 후보를 쓴다.
    """
    as_of_str = as_of.isoformat()
    best = None
    for taxonomy, tag in moat._TAG_CANDIDATES["diluted_shares"]:
        for e in edgar.extract_duration_fact_entries(company_facts, taxonomy, tag, as_of, 80, 100, forms=("10-Q",)):
            if e["filed"] > as_of_str:
                continue
            if best is None or (e["end"], e["filed"]) > (best["end"], best["filed"]):
                best = e
    return best["val"] if best else None


def shares_outstanding_from_candidates(
    dei_val: float | None, diluted_val: float | None, yfinance_val: float | None
) -> tuple[float | None, str | None]:
    """유통주식수를 우선순위대로 고른다 (순수 함수, moat_paper.md P2 1~3번).

    1. dei 표지 공시 2. 최근 분기 희석주식수 3. yfinance 현재값(시점이 다름 — 표시로 구분)
    출력: (값, 출처 라벨) — 셋 다 없으면 (None, None)
    """
    if dei_val is not None and dei_val > 0:
        return dei_val, "dei_cover_page"
    if diluted_val is not None and diluted_val > 0:
        return diluted_val, "diluted_shares_fallback"
    if yfinance_val is not None and yfinance_val > 0:
        return yfinance_val, "yfinance_current(시점 다름)"
    return None, None


def aggregate_shares_by_cik(ticker_shares: dict[str, float | None], ticker_cik: dict[str, int | None]) -> dict[str, float | None]:
    """같은 회사의 복수 주식 종류(예: GOOGL·GOOG) 유통주식수를 합산한다 (순수 함수,
    moat_paper.md P2 "모든 종류를 합산하고, 회사는 한 번만 셈").

    입력: ticker_shares({티커: 유통주식수 또는 None}), ticker_cik({티커: CIK 또는 None})
    출력: {티커: 그 티커가 속한 회사(CIK)의 전체 유통주식수 합}. CIK를 모르는 티커는 자기
         값 그대로(합산 대상에서 빠짐).
    """
    totals_by_cik: dict[int, float] = {}
    for t, shares in ticker_shares.items():
        cik = ticker_cik.get(t)
        if cik is None or shares is None:
            continue
        totals_by_cik[cik] = totals_by_cik.get(cik, 0.0) + shares
    out: dict[str, float | None] = {}
    for t, shares in ticker_shares.items():
        cik = ticker_cik.get(t)
        out[t] = totals_by_cik.get(cik) if (cik is not None and cik in totals_by_cik) else shares
    return out


# ── 평가(마크 투 마켓, 세전 — 교체 전까지는 매도가 없어 세금이 없다) ───────────────────────


def mark_to_market_factor(weights: dict[str, float], entry_prices: dict[str, float], current_prices: dict[str, float]) -> float | None:
    """진입 시점을 1.0으로 둔 포트폴리오 평가 배수(세전, 배당 미반영) (순수 함수).

    입력: weights(합 1), entry_prices({종목: 체결가}), current_prices({종목: 현재가})
    출력: 가중합 배수. entry_prices에 아예 없는(진입 대기) 종목이 하나라도 있으면 전체를
         계산할 수 없으므로 None을 돌려준다(호출부가 "진입 대기" 상태로 처리).
    """
    if not weights:
        return None
    total = 0.0
    for t, w in weights.items():
        entry = entry_prices.get(t)
        if not entry:
            return None
        cur = current_prices.get(t, entry)
        total += w * (cur / entry if cur else 1.0)
    return total
