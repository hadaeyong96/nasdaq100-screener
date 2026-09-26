"""data/macro.py 테스트 (P3.8). 네트워크 없이, 캐시 경로는 tmp_path로 갈아끼운다."""

from __future__ import annotations

from datetime import date

from data import macro


def test_fetch_fred_indicator_merges_cache_and_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FRED_CACHE_DIR", tmp_path)
    macro._save_fred_cache("DGS10", {"2026-09-20": 4.10})

    def provider(series_id, start, end):
        return {"2026-09-22": 4.12}

    out = macro.fetch_fred_indicator("DGS10", date(2026, 9, 22), lookback_days=10, fetch_provider=provider)
    assert out["series"]["2026-09-20"] == 4.10
    assert out["series"]["2026-09-22"] == 4.12
    assert out["warning"] is None


def test_fetch_fred_indicator_network_failure_falls_back_to_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FRED_CACHE_DIR", tmp_path)
    macro._save_fred_cache("DGS10", {"2026-09-20": 4.10})

    def failing_provider(series_id, start, end):
        raise RuntimeError("network down")

    out = macro.fetch_fred_indicator("DGS10", date(2026, 9, 22), lookback_days=10, fetch_provider=failing_provider)
    assert out["series"]["2026-09-20"] == 4.10
    assert "실패" in out["warning"]


def test_fetch_fred_indicator_no_cache_no_fresh_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FRED_CACHE_DIR", tmp_path)

    def failing_provider(series_id, start, end):
        raise RuntimeError("down")

    out = macro.fetch_fred_indicator("DGS10", date(2026, 9, 22), fetch_provider=failing_provider)
    assert out["series"] == {}
    assert out["warning"] is not None


def test_get_fear_greed_fetches_once_per_day(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FEAR_GREED_CSV", tmp_path / "fear_greed.csv")
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return {"2026-09-22": 58.0}

    out1 = macro.get_fear_greed(date(2026, 9, 22), fetch_provider=provider)
    out2 = macro.get_fear_greed(date(2026, 9, 22), fetch_provider=provider)
    assert calls["n"] == 1  # 두 번째 호출은 캐시에 이미 오늘치가 있어 다시 안 부름
    assert out1["value"] == 58.0
    assert out2["value"] == 58.0
    assert out1["is_fallback"] is False


def test_get_fear_greed_network_failure_uses_recent_cache_as_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FEAR_GREED_CSV", tmp_path / "fear_greed.csv")
    macro._append_fear_greed_cache({"2026-09-20": 55.0})

    def failing_provider():
        raise RuntimeError("blocked")

    out = macro.get_fear_greed(date(2026, 9, 22), fetch_provider=failing_provider, stale_fallback_days=3)
    assert out["value"] == 55.0
    assert out["is_fallback"] is True
    assert out["use_vix_fallback"] is False  # 2일 지연 <= 3일


def test_get_fear_greed_stale_beyond_3_days_signals_vix_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FEAR_GREED_CSV", tmp_path / "fear_greed.csv")
    macro._append_fear_greed_cache({"2026-09-15": 55.0})

    def failing_provider():
        raise RuntimeError("blocked")

    out = macro.get_fear_greed(date(2026, 9, 22), fetch_provider=failing_provider, stale_fallback_days=3)
    assert out["use_vix_fallback"] is True


def test_get_fear_greed_nothing_available_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FEAR_GREED_CSV", tmp_path / "fear_greed.csv")

    def failing_provider():
        raise RuntimeError("blocked")

    out = macro.get_fear_greed(date(2026, 9, 22), fetch_provider=failing_provider)
    assert out["value"] is None
    assert out["use_vix_fallback"] is True


def test_get_fear_greed_series_1y_window(tmp_path, monkeypatch):
    monkeypatch.setattr(macro, "FEAR_GREED_CSV", tmp_path / "fear_greed.csv")
    macro._append_fear_greed_cache({"2024-01-01": 10.0, "2026-01-01": 40.0, "2026-09-22": 58.0})

    out = macro.get_fear_greed(date(2026, 9, 22), fetch_provider=lambda: {})
    assert 10.0 not in out["series_1y"]  # 1년보다 오래된 값은 빠짐
    assert 40.0 in out["series_1y"]
    assert 58.0 in out["series_1y"]
