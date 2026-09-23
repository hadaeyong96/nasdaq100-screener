"""P3.2 2·3번: 데이터 지연 모드, last_processed_date 가드·건너뛴 거래일 catch-up 테스트.

engine/daily.py의 run()을 실제로 부르되 네트워크·실제 종목 목록/시세는 모두
가짜로 바꿔치기한다 (universe·가격·실적·체결·DB 경로·현재 시각).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data.fills import FillsResult
from data.prices import US_EASTERN, PriceFetchResult
from engine import daily as engine_daily
from store import db


def _make_price_df(end_date: str, periods: int = 60, seed: int = 0) -> pd.DataFrame:
    """네트워크 없는 합성 OHLCV (compute_indicators 입력용). 마지막 날짜를 고정한다."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.05, scale=1.2, size=periods)
    close = 100 + np.cumsum(steps)
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0.1, 1.5, size=periods)
    low = close - rng.uniform(0.1, 1.5, size=periods)
    low = np.minimum(low, close - 0.01)
    open_ = low + rng.uniform(0, 1, size=periods) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, size=periods)
    index = pd.bdate_range(end=end_date, periods=periods, name="date")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )
    df["close_source"] = "yahoo"
    df.attrs["warnings"] = []
    df.attrs["close_meta_time"] = None
    return df


def _patch_io(monkeypatch, tmp_path, last_date: str, fixed_now_et: datetime):
    """run()의 네트워크·DB·현재 시각 접근을 모두 가짜로 바꾼다."""
    monkeypatch.setattr(engine_daily, "get_universe", lambda: pd.DataFrame({"ticker": ["TEST"], "name_kr": ["테스트"]}))
    monkeypatch.setattr(
        engine_daily,
        "fetch_universe_prices",
        lambda tickers, cfg: PriceFetchResult(prices={"TEST": _make_price_df(last_date)}),
    )
    monkeypatch.setattr(engine_daily, "get_earnings_dates", lambda tickers: {t: None for t in tickers})
    monkeypatch.setattr(engine_daily, "load_fills", lambda: FillsResult())
    monkeypatch.setattr(engine_daily, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "state.db")

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now_et if tz is not None else fixed_now_et.replace(tzinfo=None)

    monkeypatch.setattr(engine_daily, "datetime", _FrozenDateTime)


def test_stale_data_skips_state_changes_and_marks_summary(monkeypatch, tmp_path, cfg):
    """이번 기준일이 기대 기준일보다 오래됐으면 상태 변화 없이 지연 요약만 낸다."""
    # 실행 시각: 목요일(9/24) 오전(장중) -> 기대 기준일은 그 전 마감 거래일인 9/23.
    now_et = datetime(2026, 9, 24, 7, 0, tzinfo=US_EASTERN)
    # 그런데 데이터는 9/22(화)까지만 있다 -> 지연.
    _patch_io(monkeypatch, tmp_path, "2026-09-22", now_et)

    summary = engine_daily.run(cfg, "live", do_replay=False, dry_run=False)

    assert summary["stale"] is True
    assert summary["expected_date"] == "2026-09-23"
    assert summary["actual_date"] == "2026-09-22"
    assert summary["buy_count"] == 0
    assert summary["sell_rows"] == []
    assert summary["all_events"] == []
    assert not (tmp_path / "state.db").exists()  # DB에 아예 손대지 않았다 (상태 전이 0)
    assert summary["report_path"].exists()
    assert "데이터 지연" in summary["report_path"].read_text(encoding="utf-8")


def test_stale_summary_sends_single_delay_notice_no_report_attached(monkeypatch, tmp_path, cfg):
    from notify import telegram

    now_et = datetime(2026, 9, 24, 7, 0, tzinfo=US_EASTERN)
    _patch_io(monkeypatch, tmp_path, "2026-09-22", now_et)
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy")

    summary = engine_daily.run(cfg, "live", do_replay=False, dry_run=False)

    calls = []
    monkeypatch.setattr(telegram, "_send_text", lambda *a, **k: calls.append("text") or True)
    monkeypatch.setattr(
        telegram, "_send_document", lambda *a, **k: calls.append("doc") or pytest.fail("보고서를 첨부하면 안 된다")
    )
    path = telegram.send_delay_notice(summary, cfg, force_no_send=False)

    assert calls == ["text"]  # 문서 첨부 없이 글 한 통만
    assert "데이터 지연" in path.read_text(encoding="utf-8")


def test_already_processed_date_is_a_no_op(monkeypatch, tmp_path, cfg):
    """이번 기준일이 last_processed_date 이하이면 아무것도 하지 않는다 (같은 날 재실행)."""
    now_et = datetime(2026, 9, 22, 20, 0, tzinfo=US_EASTERN)  # 화요일 밤(마감 후) -> 기대 기준일 9/22
    _patch_io(monkeypatch, tmp_path, "2026-09-22", now_et)

    conn = db.connect(db.db_path_for_mode("live"))
    db.set_meta(conn, "last_processed_date", "2026-09-22")
    conn.close()

    summary = engine_daily.run(cfg, "live", do_replay=False, dry_run=False)

    assert summary["skipped"] is True
    assert "이미 처리됨" in summary["warnings"][0]
    assert summary["all_events"] == []

    conn = db.connect(db.db_path_for_mode("live"))
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0
    conn.close()


def test_first_run_bootstraps_last_processed_date_to_today_only(monkeypatch, tmp_path, cfg):
    """last_processed_date가 없으면(첫 실행) 과거로 replay하지 않고 오늘 하루만 처리한다."""
    now_et = datetime(2026, 9, 22, 20, 0, tzinfo=US_EASTERN)
    captured = {}
    real_simulate_since = engine_daily.simulate_since

    def _spy(indicator_map, per_ticker_dates, *a, **k):
        captured["per_ticker_dates"] = {t: list(d) for t, d in per_ticker_dates.items()}
        return real_simulate_since(indicator_map, per_ticker_dates, *a, **k)

    _patch_io(monkeypatch, tmp_path, "2026-09-22", now_et)
    monkeypatch.setattr(engine_daily, "simulate_since", _spy)

    summary = engine_daily.run(cfg, "live", do_replay=False, dry_run=False)

    assert not summary.get("stale")
    assert not summary.get("skipped")
    assert captured["per_ticker_dates"]["TEST"] == [pd.Timestamp("2026-09-22")]  # 오늘 하루만

    conn = db.connect(db.db_path_for_mode("live"))
    assert db.get_meta(conn, "last_processed_date") == "2026-09-22"
    conn.close()


def test_catchup_run_processes_all_skipped_trading_days_in_order(monkeypatch, tmp_path, cfg):
    """며칠 걸러 실행해도(last_processed_date가 며칠 전) 건너뛴 거래일을 모두 순서대로 처리한다."""
    now_et = datetime(2026, 9, 24, 20, 0, tzinfo=US_EASTERN)  # 목요일 밤(마감 후) -> 기대 기준일 9/24
    _patch_io(monkeypatch, tmp_path, "2026-09-24", now_et)

    conn = db.connect(db.db_path_for_mode("live"))
    db.set_meta(conn, "last_processed_date", "2026-09-21")  # 월요일까지만 처리한 상태
    conn.close()

    captured = {}
    real_simulate_since = engine_daily.simulate_since

    def _spy(indicator_map, per_ticker_dates, *a, **k):
        captured["per_ticker_dates"] = {t: list(d) for t, d in per_ticker_dates.items()}
        return real_simulate_since(indicator_map, per_ticker_dates, *a, **k)

    monkeypatch.setattr(engine_daily, "simulate_since", _spy)

    summary = engine_daily.run(cfg, "live", do_replay=False, dry_run=False)

    # 화(9/22)·수(9/23)·목(9/24) 사흘이 순서대로 모두 포함돼야 한다.
    assert captured["per_ticker_dates"]["TEST"] == [
        pd.Timestamp("2026-09-22"),
        pd.Timestamp("2026-09-23"),
        pd.Timestamp("2026-09-24"),
    ]
    assert summary["as_of"] == pd.Timestamp("2026-09-24")

    conn = db.connect(db.db_path_for_mode("live"))
    assert db.get_meta(conn, "last_processed_date") == "2026-09-24"
    conn.close()
