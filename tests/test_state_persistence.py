"""엔진 재실행 시 이미 저장된 보유 상태가 사라지지 않는지 확인한다 (회귀 방지).

발견된 버그: engine.daily.run()의 기본(P3) 경로가 last_processed_date까지 저장된
실제 포지션을 db.load_all_positions로 이어받지 않고 매번 core.state.init_state
(대기, 무포지션)로 states를 새로 만든 뒤, 그 다음 거래일~오늘까지의 증분 구간만
simulate_since로 계산했다. 그 증분 구간에 해당 종목의 새 이벤트가 없으면
states[ticker]는 계속 초기값(대기)에 머물렀고, 그 값이 그대로 db.save_position으로
전 종목에 대해 매번 덮어써져 — 이미 보유 중이던 포지션(단계·수량·평단가·손절가)이
새 이벤트가 없다는 이유만으로 조용히 지워졌다. --replay(레거시) 경로는 원래부터
db.load_all_positions로 이어받고 있어 이 문제가 없었다.

수정: 기본 경로도 db.load_all_positions로 시작 상태를 이어받고, DB에 없는(신규)
종목만 init_state로 시작한다.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from data.fills import FillsResult
from data.prices import US_EASTERN, PriceFetchResult
from engine import daily as engine_daily
from store import db


def _flat_price_df(end_date: str, periods: int = 260, seed: int = 0) -> pd.DataFrame:
    """RSI/MACD가 교차를 만들지 않도록 완만하게 우상향하는 합성 시세(E1/E2/E3 미발생용)."""
    rng = np.random.default_rng(seed)
    close = 100 + np.linspace(0, 5, periods) + rng.normal(0, 0.05, periods)
    high = close + 0.2
    low = close - 0.2
    open_ = close - 0.05
    volume = np.full(periods, 2_000_000)
    index = pd.bdate_range(end=end_date, periods=periods, name="date")
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)
    df["close_source"] = "yahoo"
    df.attrs["warnings"] = []
    df.attrs["close_meta_time"] = None
    return df


def _patch_io(monkeypatch, tmp_path, last_date: str, fixed_now_et: datetime, mode: str = "live"):
    monkeypatch.setattr(engine_daily, "get_universe", lambda: pd.DataFrame({"ticker": ["TEST"], "name_kr": ["테스트"]}))
    monkeypatch.setattr(
        engine_daily,
        "fetch_universe_prices",
        lambda tickers, cfg: PriceFetchResult(prices={"TEST": _flat_price_df(last_date)}),
    )
    monkeypatch.setattr(engine_daily, "get_earnings_dates", lambda tickers: {t: None for t in tickers})
    monkeypatch.setattr(engine_daily, "load_fills", lambda: FillsResult())
    monkeypatch.setattr(engine_daily, "load_plan", lambda: (pd.DataFrame(columns=["ticker", "budget_krw", "memo"]), []))
    monkeypatch.setattr(engine_daily, "OUTPUT_DIR", tmp_path)
    # live 모드는 구글 시트를 먼저 시도한다 — 개발자의 실제 .env에 시트 인증 정보가
    # 있어도 이 테스트가 실제 네트워크를 타지 않도록 강제로 없앤다
    # (CLAUDE.md 네트워크 없는 테스트 원칙).
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.delenv("GOOGLE_SHEETS_ID", raising=False)
    db_attr = "DB_PATH" if mode == "live" else "PAPER_DB_PATH"
    monkeypatch.setattr(db, db_attr, tmp_path / f"{mode}_state.db")

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now_et if tz is not None else fixed_now_et.replace(tzinfo=None)

    monkeypatch.setattr(engine_daily, "datetime", _FrozenDateTime)


def _seeded_holding() -> dict:
    """"추세보유"(B, 시간 만료 규칙 없음 — E1/E2/E3로만 벗어난다)로 이미 보유 중인 상태.

    "정찰"(A1)은 a1_to_a2_expiry_days가 지나면 스스로 만료되는 정상 규칙이 있어
    (core/state.py 2번), 그 자연 만료와 이 버그를 혼동하지 않도록 추세보유를 쓴다.
    """
    return {
        "ticker": "TEST", "name_kr": "테스트", "state": "추세보유",
        "units": {"B": 30}, "entries": {"B": 100.0}, "stop": 90.0,
        "a1_date": None, "cooldown_until": None, "sent_alerts": [],
        "updated_at": "2026-09-10T09:00:00", "b_entry_date": "2026-09-10", "b_total_qty": 30,
        "grade": None, "pending": None,
    }


def _run_and_reload(monkeypatch, tmp_path, cfg, last_processed_date: str, as_of_date: str, mode: str = "live"):
    now_et = datetime.fromisoformat(f"{as_of_date}T20:00:00").replace(tzinfo=US_EASTERN)
    _patch_io(monkeypatch, tmp_path, as_of_date, now_et, mode=mode)

    conn = db.connect(db.db_path_for_mode(mode))
    db.set_meta(conn, "last_processed_date", last_processed_date)
    seeded = _seeded_holding()
    db.save_position(conn, seeded)
    conn.close()

    engine_daily.run(cfg, mode, do_replay=False, dry_run=False)

    conn = db.connect(db.db_path_for_mode(mode))
    after = db.load_all_positions(conn)["TEST"]
    conn.close()
    return seeded, after


def test_held_position_survives_multi_day_catchup_with_no_new_events(monkeypatch, tmp_path, cfg):
    """며칠 건너뛴 뒤 한 번에 캐치업 실행해도(9/21 처리 -> 9/25 실행, 9/22~9/25 재계산)
    이미 보유 중이던 포지션이 그대로 남아야 한다."""
    seeded, after = _run_and_reload(monkeypatch, tmp_path, cfg, "2026-09-21", "2026-09-25")

    assert after["state"] == seeded["state"]
    assert after["units"] == seeded["units"]
    assert after["entries"] == seeded["entries"]
    assert after["stop"] == seeded["stop"]


def test_held_position_survives_single_day_increment_with_no_new_events(monkeypatch, tmp_path, cfg):
    """정상적인 하루 단위 증분 실행(9/24 처리 -> 9/25 실행)에서도 같은 결과여야 한다
    — 이게 실제 매일 운영과 가장 가까운 경우."""
    seeded, after = _run_and_reload(monkeypatch, tmp_path, cfg, "2026-09-24", "2026-09-25")

    assert after["state"] == seeded["state"]
    assert after["units"] == seeded["units"]
    assert after["entries"] == seeded["entries"]
    assert after["stop"] == seeded["stop"]


def test_held_position_survives_in_paper_mode_too(monkeypatch, tmp_path, cfg):
    """paper 모드도 같은 run() 경로를 쓰므로 같은 수정이 적용돼야 한다."""
    seeded, after = _run_and_reload(monkeypatch, tmp_path, cfg, "2026-09-24", "2026-09-25", mode="paper")

    assert after["state"] == seeded["state"]
    assert after["units"] == seeded["units"]
    assert after["entries"] == seeded["entries"]
    assert after["stop"] == seeded["stop"]
