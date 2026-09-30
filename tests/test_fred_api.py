"""FRED 공식 API(FRED_API_KEY) + fredgraph.csv 폴백, 수집 실패 경고 (네트워크 없음 — requests.get 대체)."""

from __future__ import annotations

import pytest
import requests

from data import fx
from engine.daily import _fred_missing_codes
from notify import briefing

_SECRET = "abc123SECRETKEY"


class _Resp:
    def __init__(self, status=200, json_data=None, text=""):
        self.status_code = status
        self._json = json_data
        self.text = text

    def json(self):
        return self._json


def _patch_get(monkeypatch, handler):
    calls = []

    def _get(url, params=None, headers=None, timeout=None):
        calls.append((url, params))
        return handler(url, params)

    monkeypatch.setattr(requests, "get", _get)
    return calls


def test_uses_official_api_when_key_set(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", _SECRET)
    calls = _patch_get(monkeypatch, lambda url, p: _Resp(json_data={"observations": [
        {"date": "2026-09-28", "value": "4.12"}, {"date": "2026-09-29", "value": "."}]}))
    out = fx.fetch_fred_series_range("DGS10", "2026-09-01", "2026-09-29")
    assert out == {"2026-09-28": 4.12}
    assert calls[0][0] == "https://api.stlouisfed.org/fred/series/observations"
    assert calls[0][1]["series_id"] == "DGS10" and calls[0][1]["file_type"] == "json"


def test_falls_back_to_csv_without_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    calls = _patch_get(monkeypatch, lambda url, p: _Resp(text="observation_date,DGS10\n2026-09-28,4.12\n2026-09-29,\n"))
    assert fx.fetch_fred_series_range("DGS10", "2026-09-01", "2026-09-29") == {"2026-09-28": 4.12}
    assert calls[0][0] == "https://fred.stlouisfed.org/graph/fredgraph.csv"


def test_api_failure_falls_back_to_csv(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", _SECRET)

    def handler(url, p):
        if "api.stlouisfed.org" in url:
            return _Resp(status=500)
        return _Resp(text="observation_date,T10Y2Y\n2026-09-28,0.55\n")

    _patch_get(monkeypatch, handler)
    assert fx.fetch_fred_series_range("T10Y2Y", "2026-09-01", "2026-09-29") == {"2026-09-28": 0.55}


def test_error_message_never_contains_api_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", _SECRET)

    def handler(url, p):
        raise requests.ConnectionError(f"failed {url}?api_key={p.get('api_key')}")

    _patch_get(monkeypatch, handler)
    with pytest.raises(fx.FredFetchError) as info:
        fx.fetch_fred_series_range("DGS10", "2026-09-01", "2026-09-29")
    assert _SECRET not in str(info.value)
    assert _SECRET not in repr(info.value)


def test_fred_missing_codes_flags_failed_or_absent_indicators():
    rows = [
        {"code": "DGS10"}, {"code": "T10Y2Y"}, {"code": "BAMLH0A0HYM2"},
        {"code": "DEXKOUS · KRW=X"},  # DEXKOUS는 KRW=X로 보완돼 칸은 있지만 FRED 조회는 실패
    ]
    warnings = ["DEXKOUS 조회 실패(캐시 값 사용): FRED API HTTP 403"]
    assert _fred_missing_codes(rows, warnings) == ["DFEDTARU", "DEXKOUS"]
    all_rows = rows + [{"code": "DFEDTARU"}]
    assert _fred_missing_codes(all_rows, []) == []


def test_briefing_shows_fred_and_sheet_alert_lines():
    summary = {"as_of": None, "fred_missing": ["DGS10", "T10Y2Y"], "input_errors": ["체결 기록 오류: 헤더 오류 - x"]}
    text = briefing.build_briefing_text(summary, {})
    assert "⚠️ FRED 지표 수집 실패: DGS10, T10Y2Y" in text
    assert "⚠️ 시트 오류: 체결 기록 오류: 헤더 오류 - x" in text
    clean = briefing.build_briefing_text({"as_of": None}, {})
    assert "⚠️" not in clean
