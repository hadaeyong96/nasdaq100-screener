"""data/sectors.py 테스트. 네트워크 없음(yfinance 호출을 몽키패치)."""

from __future__ import annotations

from data import sectors


def test_get_sectors_fetches_and_caches(tmp_path, monkeypatch):
    cache_path = tmp_path / "sectors.parquet"
    calls = []

    def _fake_fetch(ticker):
        calls.append(ticker)
        return {"AAPL": "Technology", "JNJ": "Healthcare"}.get(ticker)

    monkeypatch.setattr(sectors, "_fetch_sector", _fake_fetch)

    result, failed = sectors.get_sectors(["AAPL", "JNJ"], cache_path=cache_path)

    assert result == {"AAPL": "Technology", "JNJ": "Healthcare"}
    assert failed == []
    assert calls == ["AAPL", "JNJ"]

    # 두 번째 호출은 캐시에서만 읽고 다시 네트워크를 부르지 않는다.
    result2, failed2 = sectors.get_sectors(["AAPL", "JNJ"], cache_path=cache_path)
    assert result2 == result
    assert calls == ["AAPL", "JNJ"]  # 늘지 않음


def test_get_sectors_records_failure_and_does_not_retry(tmp_path, monkeypatch):
    cache_path = tmp_path / "sectors.parquet"
    calls = []

    def _fake_fetch(ticker):
        calls.append(ticker)
        return None

    monkeypatch.setattr(sectors, "_fetch_sector", _fake_fetch)

    result, failed = sectors.get_sectors(["XXXX"], cache_path=cache_path)
    assert result == {}
    assert failed == ["XXXX"]

    result2, failed2 = sectors.get_sectors(["XXXX"], cache_path=cache_path)
    assert result2 == {}
    assert failed2 == ["XXXX"]
    assert calls == ["XXXX"]  # 두 번째는 캐시된 실패를 그대로 써서 다시 부르지 않음
