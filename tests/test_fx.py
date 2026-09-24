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
