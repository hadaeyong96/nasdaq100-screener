"""core/moat_paper.py 테스트 — 네트워크 없이, 합성 데이터로 돈다."""

from __future__ import annotations

from datetime import date

import pytest

from core import moat_paper as mp


def _entry(end, val, filed, start=None, form="10-Q"):
    e = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        e["start"] = start
    return e


def _facts(**tags_by_concept_tag):
    facts: dict = {"facts": {}}
    for key, entries in tags_by_concept_tag.items():
        taxonomy, tag = key.split(":", 1)
        facts["facts"].setdefault(taxonomy, {})[tag] = {"units": {"u": entries}}
    return facts


# ── 봉인 규칙 ────────────────────────────────────────────────────────────────


def test_enforce_paper_floor_rejects_before_2026_10_01():
    with pytest.raises(mp.PaperSealError):
        mp.enforce_paper_floor(date(2026, 9, 30))


def test_enforce_paper_floor_accepts_floor_and_after():
    mp.enforce_paper_floor(date(2026, 10, 1))  # 예외 없이 통과
    mp.enforce_paper_floor(date(2026, 10, 2))


# ── 체결가 ───────────────────────────────────────────────────────────────────


def test_compute_entry_price_applies_slippage_unfavorably():
    assert mp.compute_entry_price(100.0, 0.05) == pytest.approx(100.05)


# ── P1류: 동일 비중 합 1 (core.moat_backtest 재사용 확인용 — 여기선 apply_group_cap만) ──


def test_apply_group_cap_sector_redistributes_and_sums_to_one():
    weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    sectors = {"A": "S1", "B": "S1", "C": "S2", "D": "S3"}
    out = mp.apply_group_cap(weights, sectors, 30)
    totals: dict[str, float] = {}
    for t, w in out.items():
        totals[sectors[t]] = totals.get(sectors[t], 0.0) + w
    assert totals["S1"] <= 0.30 + 1e-6
    assert sum(out.values()) == pytest.approx(1.0)


def test_apply_group_cap_empty():
    assert mp.apply_group_cap({}, {}, 30) == {}


# ── P2: 시가총액 비중 + 업종 30% + 종목 10% ──────────────────────────────────


def test_market_cap_weights_sum_to_one():
    tickers = [f"T{i}" for i in range(10)]
    market_caps = {t: (i + 1) * 100.0 for i, t in enumerate(tickers)}
    sectors = {t: "S1" if i < 5 else "S2" for i, t in enumerate(tickers)}
    weights = mp.market_cap_weights_with_caps(tickers, market_caps, sectors, sector_cap_pct=30, stock_cap_pct=10)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_market_cap_weights_respects_stock_cap():
    # T9 혼자 시가총액의 90%를 차지 -> 종목 캡 10%에 걸려야 함
    tickers = [f"T{i}" for i in range(10)]
    market_caps = {t: 10.0 for t in tickers}
    market_caps["T9"] = 900.0
    sectors = {t: f"S{i}" for i, t in enumerate(tickers)}  # 업종은 전부 달라 업종 캡엔 안 걸림
    weights = mp.market_cap_weights_with_caps(tickers, market_caps, sectors, sector_cap_pct=30, stock_cap_pct=10)
    assert weights["T9"] <= 0.10 + 1e-6
    assert sum(weights.values()) == pytest.approx(1.0)


def test_market_cap_weights_respects_sector_cap():
    # 업종 4개(캡 30%씩, 지킬 수 있는 여지 충분) 중 S1(A,B)만 시가총액이 압도적으로 큼
    tickers = ["A", "B", "C", "D", "E"]
    market_caps = {"A": 500.0, "B": 500.0, "C": 50.0, "D": 50.0, "E": 50.0}
    sectors = {"A": "S1", "B": "S1", "C": "S2", "D": "S3", "E": "S4"}
    weights = mp.market_cap_weights_with_caps(tickers, market_caps, sectors, sector_cap_pct=30, stock_cap_pct=100)
    sector_total_s1 = weights["A"] + weights["B"]
    assert sector_total_s1 <= 0.30 + 1e-6
    assert sum(weights.values()) == pytest.approx(1.0)


def test_market_cap_weights_falls_back_to_equal_when_all_caps_missing():
    tickers = ["A", "B", "C"]
    weights = mp.market_cap_weights_with_caps(tickers, {}, {t: "S1" for t in tickers}, sector_cap_pct=100, stock_cap_pct=100)
    assert weights == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})


# ── 유통주식수 우선순위 ───────────────────────────────────────────────────────


def test_shares_outstanding_prefers_dei_cover_page():
    val, source = mp.shares_outstanding_from_candidates(1000.0, 2000.0, 3000.0)
    assert (val, source) == (1000.0, "dei_cover_page")


def test_shares_outstanding_falls_back_to_diluted():
    val, source = mp.shares_outstanding_from_candidates(None, 2000.0, 3000.0)
    assert (val, source) == (2000.0, "diluted_shares_fallback")


def test_shares_outstanding_falls_back_to_yfinance():
    val, source = mp.shares_outstanding_from_candidates(None, None, 3000.0)
    assert (val, source) == (3000.0, "yfinance_current(시점 다름)")


def test_shares_outstanding_all_missing_is_none():
    assert mp.shares_outstanding_from_candidates(None, None, None) == (None, None)


def test_aggregate_shares_by_cik_sums_multi_class():
    ticker_shares = {"GOOGL": 1000.0, "GOOG": 500.0, "AAPL": 2000.0}
    ticker_cik = {"GOOGL": 1, "GOOG": 1, "AAPL": 2}
    out = mp.aggregate_shares_by_cik(ticker_shares, ticker_cik)
    assert out["GOOGL"] == pytest.approx(1500.0)
    assert out["GOOG"] == pytest.approx(1500.0)
    assert out["AAPL"] == pytest.approx(2000.0)


def test_aggregate_shares_by_cik_unknown_cik_keeps_own_value():
    ticker_shares = {"X": 10.0}
    ticker_cik = {"X": None}
    assert mp.aggregate_shares_by_cik(ticker_shares, ticker_cik) == {"X": 10.0}


def test_latest_cover_page_shares_future_data_excluded():
    facts = _facts(**{
        "dei:EntityCommonStockSharesOutstanding": [
            _entry("2026-07-31", 1000.0, "2026-08-05", form="10-Q"),
            _entry("2026-09-30", 1100.0, "2026-10-15", form="10-Q"),  # as_of(09-30)보다 늦게 filed
        ]
    })
    assert mp.latest_cover_page_shares(facts, date(2026, 9, 30)) == 1000.0


def test_latest_quarterly_diluted_shares_picks_3month_window():
    facts = _facts(**{
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": [
            _entry("2026-06-30", 900.0, "2026-08-01", start="2026-04-01"),  # 3개월(91일)
            _entry("2026-06-30", 1800.0, "2026-08-01", start="2026-01-01"),  # 6개월 누적(180일) - 제외
        ]
    })
    assert mp.latest_quarterly_diluted_shares(facts, date(2026, 9, 30)) == 900.0


# ── 진입 대기 병합(시작 파일 부분 채움, 덮어쓰기 금지 예외) ───────────────────


def test_merge_price_fills_only_updates_pending():
    existing = {
        "A": {"status": "체결", "entry_price": 100.0},
        "B": {"status": "진입 대기"},
    }
    new_fills = {
        "A": {"status": "체결", "entry_price": 999.0},  # 이미 체결된 A는 바뀌면 안 됨
        "B": {"status": "체결", "entry_price": 50.0},
    }
    out = mp.merge_price_fills(existing, new_fills)
    assert out["A"] == {"status": "체결", "entry_price": 100.0}
    assert out["B"] == {"status": "체결", "entry_price": 50.0}


def test_merge_price_fills_leaves_still_pending_untouched():
    existing = {"A": {"status": "진입 대기"}}
    out = mp.merge_price_fills(existing, {})
    assert out == {"A": {"status": "진입 대기"}}


# ── 평가(마크 투 마켓) ────────────────────────────────────────────────────────


def test_mark_to_market_factor_basic():
    weights = {"A": 0.5, "B": 0.5}
    entry = {"A": 100.0, "B": 200.0}
    current = {"A": 110.0, "B": 190.0}
    factor = mp.mark_to_market_factor(weights, entry, current)
    assert factor == pytest.approx(0.5 * 1.10 + 0.5 * 0.95)


def test_mark_to_market_factor_none_when_entry_missing():
    weights = {"A": 0.5, "B": 0.5}
    assert mp.mark_to_market_factor(weights, {"A": 100.0}, {"A": 110.0}) is None
