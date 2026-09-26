"""data/fx.py 환율 조회·캐시 테스트 (P3.6 6-6번). 네트워크 없이, 캐시 파일은
tmp_path로 갈아끼워 돈다."""

from __future__ import annotations

from data import fx


def test_pick_rate_for_date_exact_match():
    history = {"2026-09-22": 1_380.0, "2026-09-23": 1_390.0}
    rate, date = fx.pick_rate_for_date(history, "2026-09-23")
    assert rate == 1_390.0
    assert date == "2026-09-23"


def test_pick_rate_for_date_falls_back_to_most_recent_earlier_date():
    """기준일 값이 없으면(주말·휴일 등) 그 이전 가장 최근 날짜 값을 쓴다."""
    history = {"2026-09-18": 1_370.0, "2026-09-19": 1_375.0}
    rate, date = fx.pick_rate_for_date(history, "2026-09-22")  # 월요일, 주말 지나 환율 없음
    assert rate == 1_375.0
    assert date == "2026-09-19"


def test_pick_rate_for_date_none_when_nothing_available():
    assert fx.pick_rate_for_date({}, "2026-09-23") is None
    assert fx.pick_rate_for_date({"2026-09-24": 1_400.0}, "2026-09-23") is None  # 미래 값뿐


# ── merge_fx_with_fallback (P5-5 3번: yfinance 우선 + FRED로 빈 날짜만 보완) ──────


def test_merge_fx_with_fallback_prefers_primary_on_overlap():
    primary = {"2007-01-02": 950.0, "2007-01-03": 951.0}
    fallback = {"2007-01-02": 949.5, "2007-01-03": 950.5}
    merged, stats = fx.merge_fx_with_fallback(primary, fallback)
    assert merged["2007-01-02"] == 950.0  # primary 값을 그대로 쓴다(덮어써지지 않음)
    assert merged["2007-01-03"] == 951.0
    assert stats["overlap_days"] == 2
    assert stats["filled_from_fallback_days"] == 0


def test_merge_fx_with_fallback_fills_gaps_from_fallback():
    primary = {"2007-01-02": 950.0}  # 2007-01-03이 비어 있음(예: yfinance 결측일)
    fallback = {"2007-01-02": 949.5, "2007-01-03": 950.5}
    merged, stats = fx.merge_fx_with_fallback(primary, fallback)
    assert merged["2007-01-03"] == 950.5  # fallback으로 채워짐
    assert stats["filled_from_fallback_days"] == 1
    assert stats["filled_dates"] == ["2007-01-03"]


def test_merge_fx_with_fallback_reports_overlap_diff_stats():
    primary = {"2007-01-02": 950.0, "2007-01-03": 960.0}
    fallback = {"2007-01-02": 949.0, "2007-01-03": 955.0}  # 차이 1.0, 5.0
    merged, stats = fx.merge_fx_with_fallback(primary, fallback)
    assert stats["mean_abs_diff"] == 3.0
    assert stats["max_abs_diff"] == 5.0
    assert stats["max_abs_diff_date"] == "2007-01-03"


def test_merge_fx_with_fallback_empty_fallback_is_noop():
    primary = {"2007-01-02": 950.0}
    merged, stats = fx.merge_fx_with_fallback(primary, {})
    assert merged == primary
    assert stats["overlap_days"] == 0
    assert stats["filled_from_fallback_days"] == 0


def test_get_usd_krw_rate_uses_fresh_value(tmp_path, monkeypatch):
    monkeypatch.setattr(fx, "CACHE_PATH", tmp_path / "fx_krw.json")
    result = fx.get_usd_krw_rate("2026-09-23", fetch_provider=lambda: {"2026-09-23": 1_385.5})
    assert result.rate == 1_385.5
    assert result.rate_date == "2026-09-23"
    assert result.is_fallback is False
    assert result.warning is None


def test_get_usd_krw_rate_falls_back_to_cache_on_fetch_failure(tmp_path, monkeypatch):
    cache_path = tmp_path / "fx_krw.json"
    monkeypatch.setattr(fx, "CACHE_PATH", cache_path)
    # 먼저 정상 조회로 캐시를 채운다.
    fx.get_usd_krw_rate("2026-09-22", fetch_provider=lambda: {"2026-09-22": 1_380.0})

    def _failing():
        raise RuntimeError("network down")

    result = fx.get_usd_krw_rate("2026-09-23", fetch_provider=_failing)
    assert result.rate == 1_380.0
    assert result.rate_date == "2026-09-22"
    assert result.is_fallback is True
    assert result.warning is not None


def test_get_usd_krw_rate_none_when_no_cache_and_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(fx, "CACHE_PATH", tmp_path / "fx_krw.json")

    def _failing():
        raise RuntimeError("network down")

    result = fx.get_usd_krw_rate("2026-09-23", fetch_provider=_failing)
    assert result.rate is None
    assert result.is_fallback is True
    assert result.warning is not None
