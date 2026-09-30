"""engine.daily._build_macro_rows의 DEXKOUS 보완 로직 테스트 (P3.8 4번 판단 필요 후속).

DEXKOUS(FRED)는 보통 1주일 정도 늦게 갱신되므로, 그 뒤 며칠은 기존 KRW=X로
채우고 실제 최신 날짜를 그대로 보여야 한다. 네트워크 없이 fetch 함수를 모두
모킹해서 돈다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from data import fx
from engine import daily

_ROOT = Path(__file__).resolve().parents[1]
with open(_ROOT / "config.yaml", encoding="utf-8") as _f:
    _CFG = {"macro": {**yaml.safe_load(_f)["macro"], "enabled": True}}  # 기준값은 실제 config.yaml 그대로


def test_dexkous_is_filled_with_recent_krwx_and_shows_actual_latest_date(monkeypatch):
    """DEXKOUS 마지막 날짜가 일주일 전이어도, 그 뒤는 KRW=X로 채워 실제 최신 날짜를 보여준다."""

    def fake_fred(series_id, as_of, lookback_days=400, fetch_provider=None):
        if series_id != "DEXKOUS":
            return {"series": {}, "warning": None}
        # DEXKOUS는 일주일 전(09-18)까지만 있다고 가정
        return {"series": {"2026-09-15": 1385.0, "2026-09-16": 1386.0, "2026-09-17": 1387.0, "2026-09-18": 1388.0}, "warning": None}

    def fake_krwx_range(start, end):
        # KRW=X는 그 뒤로도(09-19~09-25) 최신 값이 있다고 가정
        return {"2026-09-18": 1388.5, "2026-09-19": 1389.0, "2026-09-22": 1390.0, "2026-09-23": 1391.0, "2026-09-25": 1392.0}

    monkeypatch.setattr(daily.macrodata, "fetch_fred_indicator", fake_fred)
    monkeypatch.setattr(daily.fx, "fetch_usd_krw_range", fake_krwx_range)
    monkeypatch.setattr(daily.macrodata, "get_fear_greed", lambda *a, **k: {"value": 50, "as_of": "2026-09-25", "series_1y": [50], "is_fallback": False, "use_vix_fallback": False, "warning": None})

    rows, warnings = daily._build_macro_rows(_CFG, date(2026, 9, 25))
    fx_row = next(r for r in rows if r["name"] == "원/달러 환율")

    assert fx_row["as_of"] == "2026-09-25"  # DEXKOUS만 썼으면 09-18에 머물렀을 것
    assert fx_row["value"] == 1392.0
    assert fx_row["code"] == "DEXKOUS · KRW=X"
    # 겹치는 09-18은 DEXKOUS(1388.0)가 우선하고 KRW=X(1388.5)로 덮어쓰지 않는다
    assert 1388.0 in fx_row["series"]
    assert not any(w for w in warnings if "DEXKOUS 최근" in w)


def test_dexkous_krwx_fetch_failure_falls_back_to_dexkous_only_with_warning(monkeypatch):
    def fake_fred(series_id, as_of, lookback_days=400, fetch_provider=None):
        if series_id != "DEXKOUS":
            return {"series": {}, "warning": None}
        return {"series": {"2026-09-18": 1388.0}, "warning": None}

    def failing_krwx_range(start, end):
        raise RuntimeError("network down")

    monkeypatch.setattr(daily.macrodata, "fetch_fred_indicator", fake_fred)
    monkeypatch.setattr(daily.fx, "fetch_usd_krw_range", failing_krwx_range)
    monkeypatch.setattr(daily.macrodata, "get_fear_greed", lambda *a, **k: {"value": 50, "as_of": "2026-09-25", "series_1y": [50], "is_fallback": False, "use_vix_fallback": False, "warning": None})

    rows, warnings = daily._build_macro_rows(_CFG, date(2026, 9, 25))
    fx_row = next(r for r in rows if r["name"] == "원/달러 환율")

    assert fx_row["as_of"] == "2026-09-18"  # 보완 실패 시 DEXKOUS 값 그대로
    assert any("DEXKOUS 최근" in w for w in warnings)
