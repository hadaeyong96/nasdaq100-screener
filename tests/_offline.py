"""engine.daily.run()을 부르는 테스트용: 시장 온도·환율 네트워크 호출을 막는다.

run()은 시세·종목 목록 말고도 시장 온도(CNN 공포·탐욕, FRED, yfinance KRW=X)와
원/달러 환율(yfinance KRW=X)을 받는다. 테스트는 네트워크 없이 돌아야 하므로
(CLAUDE.md 테스트 원칙) 이 함수로 한 번에 가짜로 바꾼다. 시장 온도 자체는
test_macro_daily_build.py가, 환율은 test_fx.py가 따로 확인한다.
"""

from __future__ import annotations

from data.fx import FxRateResult

FAKE_USD_KRW = 1400.0


def block_engine_network(monkeypatch) -> None:
    """engine.daily의 시장 온도·환율 조회를 네트워크 없는 가짜로 바꾼다.

    입력: monkeypatch. 출력: 없음 (시장 온도는 빈 칸, 환율은 FAKE_USD_KRW 고정).
    """
    from engine import daily as engine_daily

    monkeypatch.setattr(engine_daily, "_build_macro_rows", lambda cfg, as_of_date: ([], []))
    monkeypatch.setattr(engine_daily.fx, "get_usd_krw_rate", lambda d: FxRateResult(FAKE_USD_KRW, str(d), False))
    monkeypatch.setattr(engine_daily.fx, "get_usd_krw_rate_map", lambda dates: {})
