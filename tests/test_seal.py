"""core/seal.py 및 로더의 봉인 구간 차단 테스트 (AI 펀드 F2, docs/design/fund_sim.md 2.2).

네트워크 없음 — 봉인 검사는 실제 요청 전에 일어나므로, 봉인 구간을 요청하는
테스트는 티커·인증 없이도 SealedDataError만 확인하면 된다.
"""

from __future__ import annotations

from datetime import date

import pytest

from core.seal import SealedDataError, enforce_not_sealed
from data import backtest_prices as bp
from data import fx


# ── core.seal.enforce_not_sealed (순수 함수) ────────────────────────────────


def test_enforce_not_sealed_passes_when_end_before_seal_date():
    enforce_not_sealed(date(2021, 12, 31), seal_date=date(2022, 1, 1))  # 예외 없음


def test_enforce_not_sealed_passes_when_end_equals_seal_date():
    enforce_not_sealed(date(2022, 1, 1), seal_date=date(2022, 1, 1))  # 경계값 포함(이 날짜까지는 허용)


def test_enforce_not_sealed_raises_when_end_after_seal_date():
    with pytest.raises(SealedDataError, match="봉인"):
        enforce_not_sealed(date(2022, 1, 2), seal_date=date(2022, 1, 1))


def test_enforce_not_sealed_unseal_flag_bypasses_block():
    enforce_not_sealed(date(2026, 9, 30), seal_date=date(2022, 1, 1), unseal=True)  # 예외 없음


def test_enforce_not_sealed_no_seal_date_never_blocks():
    """seal_date를 안 넘기면(기존 P5/P6 호출부) 이 함수는 아무것도 확인하지 않는다."""
    enforce_not_sealed(date(2099, 1, 1), seal_date=None)  # 예외 없음


# ── data/backtest_prices.py 로더 ────────────────────────────────────────────


def test_fetch_history_raises_before_any_network_call_when_sealed():
    """봉인 검사가 캐시·네트워크 접근보다 먼저 일어나 — 실존하지 않는 티커로도 검증 가능."""
    with pytest.raises(SealedDataError):
        bp.fetch_history("__NO_SUCH_TICKER__", date(2021, 1, 1), date(2022, 6, 1), seal_date=date(2022, 1, 1))


def test_fetch_history_no_seal_date_param_is_unaffected():
    """seal_date를 안 넘기면 기존 동작 그대로 — 봉인 구간 날짜를 요청해도 이 검사 때문에
    막히지 않는다(실제로는 캐시·네트워크가 없어 ValueError로 실패하지만 SealedDataError는 아니다)."""
    with pytest.raises(Exception) as exc_info:
        bp.fetch_history("__NO_SUCH_TICKER__", date(2021, 1, 1), date(2022, 6, 1))
    assert not isinstance(exc_info.value, SealedDataError)


def test_fetch_universe_history_raises_before_iterating_tickers_when_sealed():
    with pytest.raises(SealedDataError):
        bp.fetch_universe_history(["AAA", "BBB"], date(2021, 1, 1), date(2022, 6, 1), seal_date=date(2022, 1, 1))


def test_fetch_dividends_raises_when_sealed():
    with pytest.raises(SealedDataError):
        bp.fetch_dividends("__NO_SUCH_TICKER__", date(2021, 1, 1), date(2022, 6, 1), seal_date=date(2022, 1, 1))


def test_fetch_history_unseal_true_does_not_raise_sealed_error():
    """unseal=True면 SealedDataError는 안 난다(그 뒤 네트워크·캐시 단계에서 다른 예외는 날 수 있다)."""
    with pytest.raises(Exception) as exc_info:
        bp.fetch_history("__NO_SUCH_TICKER__", date(2021, 1, 1), date(2022, 6, 1), seal_date=date(2022, 1, 1), unseal=True)
    assert not isinstance(exc_info.value, SealedDataError)


# ── data/fx.py 로더 ─────────────────────────────────────────────────────────


def test_fetch_usd_krw_range_raises_before_network_when_sealed():
    with pytest.raises(SealedDataError):
        fx.fetch_usd_krw_range(date(2021, 1, 1), date(2022, 6, 1), seal_date=date(2022, 1, 1))


def test_fetch_usd_krw_range_no_seal_date_is_unaffected(monkeypatch):
    """seal_date를 안 넘기면 기존 서명 그대로 동작한다 — yfinance 호출까지 도달하는지만 확인(네트워크는 막는다)."""
    called = []

    class _FakeTicker:
        def history(self, **kwargs):
            called.append(kwargs)
            import pandas as pd

            return pd.DataFrame()

    monkeypatch.setattr("yfinance.Ticker", lambda symbol: _FakeTicker())
    result = fx.fetch_usd_krw_range(date(2021, 1, 1), date(2022, 6, 1))
    assert called  # SealedDataError 없이 실제 fetch 단계까지 도달했다
    assert result == {}
