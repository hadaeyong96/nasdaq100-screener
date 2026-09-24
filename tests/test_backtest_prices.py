"""data/backtest_prices.py 테스트. 네트워크 없이 _fetch_raw를 갈아끼워 돈다 (P5-1 6번)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from data import backtest_prices as bp


def _mk(dates: list[str], start_close: float = 100.0) -> pd.DataFrame:
    idx = pd.DatetimeIndex(dates, name="date")
    n = len(dates)
    closes = [start_close + i for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
            "close_source": ["yahoo"] * n,
        },
        index=idx,
    )


def test_fetch_history_uses_fresh_fetch_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bp, "find_mid_series_gaps", lambda df: [])
    calls = []

    def fake_fetch_raw(ticker, start, end):
        calls.append((start, end))
        return _mk(["2015-01-02", "2015-01-05", "2015-01-06"])

    monkeypatch.setattr(bp, "_fetch_raw", fake_fetch_raw)
    df, warnings = bp.fetch_history("AAA", date(2015, 1, 1), date(2015, 1, 10))
    assert len(df) == 3
    assert warnings == []
    assert len(calls) == 1


def test_fetch_history_reuses_cache_when_range_covered(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bp, "find_mid_series_gaps", lambda df: [])
    cached = _mk(["2015-01-02", "2015-01-05", "2015-01-06", "2015-01-07"])
    bp._save_cache("AAA", cached)

    def fail_fetch(*a, **k):
        raise AssertionError("네트워크를 다시 부르면 안 된다 — 캐시가 구간을 이미 덮는다")

    monkeypatch.setattr(bp, "_fetch_raw", fail_fetch)
    df, warnings = bp.fetch_history("AAA", date(2015, 1, 2), date(2015, 1, 6))
    assert len(df) == 3  # 1/2, 1/5, 1/6만 — 1/7은 요청 구간 밖


def test_fetch_history_raises_when_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bp, "_fetch_raw", lambda t, s, e: pd.DataFrame(columns=["open", "high", "low", "close", "volume", "close_source"]))
    try:
        bp.fetch_history("ZZZ", date(2015, 1, 1), date(2015, 1, 10))
        assert False, "빈 데이터면 ValueError가 나야 한다"
    except ValueError:
        pass


def test_fetch_universe_history_collects_failures_and_continues(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(bp, "find_mid_series_gaps", lambda df: [])

    def fake_fetch_raw(ticker, start, end):
        if ticker == "BAD":
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "close_source"])
        return _mk(["2015-01-02", "2015-01-05"])

    monkeypatch.setattr(bp, "_fetch_raw", fake_fetch_raw)
    result = bp.fetch_universe_history(["AAA", "BAD"], date(2015, 1, 1), date(2015, 1, 10))
    assert "AAA" in result.prices
    assert "BAD" in result.failed
