"""data/edgar.py 테스트 — 네트워크 없이 돈다(요청 함수는 monkeypatch로 가짜 응답을 준다)."""

from __future__ import annotations

import json

import pytest

from data import edgar


# ── User-Agent 필수 확인 ────────────────────────────────────────────────────


def test_get_user_agent_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(edgar, "ROOT", tmp_path)  # .env 없는 빈 디렉터리
    with pytest.raises(edgar.EdgarConfigError):
        edgar.get_user_agent()


def test_get_user_agent_raises_when_blank(monkeypatch, tmp_path):
    monkeypatch.setenv("SEC_USER_AGENT", "   ")
    monkeypatch.setattr(edgar, "ROOT", tmp_path)
    with pytest.raises(edgar.EdgarConfigError):
        edgar.get_user_agent()


def test_get_user_agent_returns_value_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("SEC_USER_AGENT", "test-agent you@example.com")
    monkeypatch.setattr(edgar, "ROOT", tmp_path)
    assert edgar.get_user_agent() == "test-agent you@example.com"


# ── 속도 제한 ────────────────────────────────────────────────────────────


def test_rate_limiter_sleeps_when_called_too_soon():
    calls = []
    limiter = edgar._RateLimiter(min_interval=0.5, sleep_fn=calls.append, time_fn=iter([0.0, 0.1, 0.1]).__next__)
    limiter.wait()  # 첫 호출, _last_call=None -> sleep 없음
    limiter.wait()  # now=0.1, last=0.0, elapsed=0.1 < 0.5 -> 0.4초 자야 함
    assert calls == [pytest.approx(0.4)]


def test_rate_limiter_no_sleep_when_interval_already_passed():
    calls = []
    limiter = edgar._RateLimiter(min_interval=0.1, sleep_fn=calls.append, time_fn=iter([0.0, 1.0]).__next__)
    limiter.wait()
    limiter.wait()  # elapsed=1.0 >= 0.1 -> sleep 안 함
    assert calls == []


# ── 사실 추출(순수 함수) ────────────────────────────────────────────────────


def _sample_facts():
    return {
        "facts": {
            "us-gaap": {
                "OperatingIncomeLoss": {
                    "units": {
                        "USD": [
                            {"start": "2019-01-01", "end": "2019-12-31", "val": 100, "filed": "2020-02-01", "fy": 2019, "fp": "FY", "form": "10-K", "accn": "a1"},
                            {"start": "2020-01-01", "end": "2020-12-31", "val": 120, "filed": "2021-02-01", "fy": 2020, "fp": "FY", "form": "10-K", "accn": "a2"},
                        ]
                    }
                }
            },
            "ifrs-full": {
                "Revenue": {"units": {"USD": [{"end": "2019-12-31", "val": 999, "filed": "2020-03-01", "fy": 2019, "fp": "FY", "form": "20-F", "accn": "b1"}]}}
            },
        }
    }


def test_extract_fact_entries_returns_entries_with_unit():
    entries = edgar.extract_fact_entries(_sample_facts(), "us-gaap", "OperatingIncomeLoss")
    assert len(entries) == 2
    assert entries[0]["unit"] == "USD"
    assert entries[0]["val"] == 100


def test_extract_fact_entries_missing_tag_returns_empty():
    assert edgar.extract_fact_entries(_sample_facts(), "us-gaap", "DoesNotExist") == []


def test_extract_fact_entries_missing_taxonomy_returns_empty():
    assert edgar.extract_fact_entries(_sample_facts(), "dei", "EntityCommonStockSharesOutstanding") == []


# ── 캐시 동작 ────────────────────────────────────────────────────────────


def test_fetch_ticker_to_cik_uses_cache_without_network(monkeypatch, tmp_path):
    cache_path = tmp_path / "company_tickers.json"
    cache_path.write_text(json.dumps({"AAPL": 320193}), encoding="utf-8")
    monkeypatch.setattr(edgar, "TICKER_MAP_CACHE_PATH", cache_path)

    def _boom(*a, **k):
        raise AssertionError("캐시가 있으면 네트워크를 타면 안 된다")

    monkeypatch.setattr(edgar, "_get", _boom)
    out = edgar.fetch_ticker_to_cik()
    assert out == {"AAPL": 320193}


def test_fetch_ticker_to_cik_parses_response_and_writes_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "sub" / "company_tickers.json"
    monkeypatch.setattr(edgar, "TICKER_MAP_CACHE_PATH", cache_path)
    monkeypatch.setattr(edgar, "CACHE_DIR", tmp_path / "sub")

    class _Resp:
        def json(self):
            return {"0": {"cik_str": 320193, "ticker": "aapl", "title": "Apple Inc."}}

    monkeypatch.setattr(edgar, "_get", lambda url, ua, timeout=30.0: _Resp())
    out = edgar.fetch_ticker_to_cik(user_agent="test-agent x@y.com")
    assert out == {"AAPL": 320193}
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {"AAPL": 320193}


def test_fetch_company_facts_uses_cache_without_network(monkeypatch, tmp_path):
    cache_path = tmp_path / "CIK0000320193.json"
    cache_path.write_text(json.dumps({"cik": 320193}), encoding="utf-8")
    monkeypatch.setattr(edgar, "FACTS_CACHE_DIR", tmp_path)

    def _boom(*a, **k):
        raise AssertionError("캐시가 있으면 네트워크를 타면 안 된다")

    monkeypatch.setattr(edgar, "_get", _boom)
    out = edgar.fetch_company_facts(320193)
    assert out == {"cik": 320193}


def test_fetch_company_facts_force_refresh_ignores_cache(monkeypatch, tmp_path):
    cache_path = tmp_path / "CIK0000320193.json"
    cache_path.write_text(json.dumps({"cik": "stale"}), encoding="utf-8")
    monkeypatch.setattr(edgar, "FACTS_CACHE_DIR", tmp_path)

    class _Resp:
        def json(self):
            return {"cik": 320193, "entityName": "Apple Inc."}

    monkeypatch.setattr(edgar, "_get", lambda url, ua, timeout=30.0: _Resp())
    out = edgar.fetch_company_facts(320193, user_agent="a b@c.com", force_refresh=True)
    assert out["entityName"] == "Apple Inc."
