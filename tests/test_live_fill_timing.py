"""라이브 체결 날짜 처리 (2026-09-30 ODFL 사례 재현). 네트워크 없이 engine.daily.run()을
날짜를 바꿔 가며 여러 번 돌려, 매일 실행되는 실제 흐름(DB·fill_ledger 포함)을 그대로 본다.

ODFL: 9/28 A1 신호(주문대기) -> 9/29 실제 1차 매수 4주 @ $178.41 (체결 날짜 9/29).
기존 코드는 체결을 process_day 뒤에 반영해 9/29 처리에서 _resolve_pending이 먼저
UNFILLED로 되돌렸다. 여기서는 합성 데이터의 날짜로 같은 상황을 만든다:
  9/22 rsi 20 -> 9/23 rsi 32 (A1 신호) -> 9/24 (신호 다음 날, 실제 매수일) -> 9/25 -> 9/28.
"""

from __future__ import annotations

import copy
from datetime import datetime

import pandas as pd
import pytest

from data import fx as fx_module
from data import sheets as sheets_module
from data.fills import FillsResult
from store import db
from tests.test_state import make_df

_SIGNAL_DAY = "2026-09-23"
_NEXT_DAY = "2026-09-24"


def _indicator_df() -> pd.DataFrame:
    """2026-09-28(월)까지 60거래일. 9/23에 A1(RSI 30 상향 돌파)이 난다."""
    index = pd.bdate_range(end="2026-09-28", periods=60, name="date")
    rows = []
    for d in index:
        row = {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "rsi": 40.0}
        if d == pd.Timestamp("2026-09-22"):
            row["rsi"] = 20.0
        if d == pd.Timestamp(_SIGNAL_DAY):
            row.update({"rsi": 32.0, "swing_low": 94.0})
        rows.append(row)
    df = make_df(rows, start=str(index[0].date()))
    df["close_source"] = "yahoo"
    return df


def _fill_row(date: str, side: str, qty: int, price: float) -> dict:
    """실제 구글 시트 "체결" 탭 형식 (날짜, 종목, 매수매도, 수량, 체결가, 차수, 메모)."""
    return {"날짜": date, "종목": "TEST", "매수매도": side, "수량": str(qty), "체결가": str(price), "차수": "1차", "메모": ""}


class _Client:
    def __init__(self, fills_records):
        self.fills_records = fills_records

    def open_by_key(self, sheet_id):
        client = self

        class _Sheet:
            def worksheet(self, name):
                class _Ws:
                    def get_all_records(self_inner):
                        if name == "계획":
                            return [{
                                "티커": "TEST", "계획금액": "3,000,000", "등록일": "", "기준가($)": "", "환율": "",
                                "1차(주)": "", "2차(주)": "", "3차(주)": "", "합계(주)": "", "보유(주)": "",
                                "남은(주)": "", "메모": "",
                            }]
                        return client.fills_records

                return _Ws()

        return _Sheet()


@pytest.fixture
def live(monkeypatch, tmp_path, cfg):
    """run(today, fills)로 그날 KST 아침 실행을 흉내 낸다 (그날 ET 20:00 기준, 그날까지 확정 봉)."""
    from data.prices import US_EASTERN, PriceFetchResult
    from engine import daily as engine_daily

    full = _indicator_df()
    local_cfg = copy.deepcopy(cfg)
    local_cfg["macro"]["enabled"] = False
    state = {"today": None}

    def _prices(tickers, cfg_):
        df = full.loc[: state["today"]].copy()
        df.attrs["warnings"] = []
        df.attrs["close_meta_time"] = None
        return PriceFetchResult(prices={"TEST": df})

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            now_et = datetime.fromisoformat(f"{state['today']}T20:00:00").replace(tzinfo=US_EASTERN)
            return now_et if tz is not None else now_et.replace(tzinfo=None)

    monkeypatch.setattr(engine_daily, "get_universe", lambda: pd.DataFrame({"ticker": ["TEST"], "name_kr": ["테스트"]}))
    monkeypatch.setattr(engine_daily, "fetch_universe_prices", _prices)
    monkeypatch.setattr(engine_daily, "compute_indicators", lambda df, cfg_: df)
    monkeypatch.setattr(engine_daily, "get_earnings_dates", lambda tickers: {t: None for t in tickers})
    monkeypatch.setattr(engine_daily, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(engine_daily, "datetime", _Frozen)
    monkeypatch.setattr(engine_daily.fx, "get_usd_krw_rate", lambda d: fx_module.FxRateResult(1400.0, str(d), False))
    monkeypatch.setattr(engine_daily.fx, "get_usd_krw_rate_map", lambda dates: {})
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "state.db")
    monkeypatch.setattr(fx_module, "_load_cache", lambda: {})
    monkeypatch.setattr(sheets_module, "_default_fx_provider", lambda start, end: {"2026-09-01": 1400.0})
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "sheet-123")

    def run(today: str, fills_records: list[dict]) -> dict:
        state["today"] = today
        return engine_daily.run(local_cfg, "live", do_replay=False, dry_run=False, sheets_client=_Client(fills_records))

    return run


def _position():
    conn = db.connect(db.db_path_for_mode("live"))
    try:
        return db.load_all_positions(conn)["TEST"], db.get_fill_ledger(conn, "live")
    finally:
        conn.close()


def test_fill_dated_day_after_signal_confirms_scouting(live):
    """신호일(9/23) 다음 날 날짜(9/24)로 입력한 체결 -> 9/24 처리에서 UNFILLED가 아니라 정찰."""
    first = live(_SIGNAL_DAY, [])
    assert any(e["kind"] == "A1" for e in first["all_events"])
    assert _position()[0]["state"] == "주문대기"

    fills = [_fill_row(_NEXT_DAY, "매수", 4, 178.41)]
    second = live(_NEXT_DAY, fills)

    pos, ledger = _position()
    assert second["input_errors"] == []
    assert not any(e["kind"] == "UNFILLED" for e in second["all_events"])
    assert pos["state"] == "정찰"
    assert pos["units"]["1"] == 4 and pos["entries"]["1"] == pytest.approx(178.41)
    assert pd.Timestamp(pos["a1_date"]) == pd.Timestamp(_SIGNAL_DAY)  # 진입일은 신호일
    assert second["rebuilt_from"] is None  # 제때 입력한 체결은 재계산 없이 반영된다
    assert [(r["date"], r["qty"]) for r in ledger] == [(_NEXT_DAY, 4)]
    assert any(r["티커"] == "TEST" for r in second["hold_rows"])


def test_late_entered_fill_is_backfilled_after_unfilled(live):
    """9/24 실행 때는 체결을 안 적어 UNFILLED(대기 복귀) -> 9/25에 9/24 날짜로 늦게 입력 ->
    소급 반영돼 정찰(1차 보유)로 복구된다."""
    live(_SIGNAL_DAY, [])
    second = live(_NEXT_DAY, [])
    assert any(e["kind"] == "UNFILLED" for e in second["all_events"])
    assert _position()[0]["state"] == "대기"

    third = live("2026-09-25", [_fill_row(_NEXT_DAY, "매수", 4, 178.41)])

    pos, ledger = _position()
    assert third["rebuilt_from"] == _NEXT_DAY
    assert pos["state"] == "정찰"
    assert pos["units"]["1"] == 4
    assert pd.Timestamp(pos["a1_date"]) == pd.Timestamp(_SIGNAL_DAY)
    assert [(r["date"], r["qty"]) for r in ledger] == [(_NEXT_DAY, 4)]


def test_same_fill_is_not_applied_twice(live):
    """이미 반영한 체결(fill_ledger에 있음)은 다음 실행들에서 다시 반영하지 않는다 —
    매도가 두 번 반영되면 보유가 0이 되므로 매도로 확인한다."""
    buy = _fill_row(_NEXT_DAY, "매수", 4, 178.41)
    sell = _fill_row("2026-09-25", "매도", 1, 180.0)
    live(_SIGNAL_DAY, [])
    live(_NEXT_DAY, [buy])
    live("2026-09-25", [buy, sell])
    assert _position()[0]["units"]["1"] == 3

    fourth = live("2026-09-28", [buy, sell])

    pos, ledger = _position()
    assert fourth["rebuilt_from"] is None
    assert not any(e["kind"] == "FILL" for e in fourth["all_events"])
    assert pos["units"]["1"] == 3
    assert len(ledger) == 2

    # 같은 기준일 재실행도 체결이 그대로면 건너뛴다 (반영 0건)
    assert live("2026-09-28", [buy, sell]).get("skipped")
    assert len(_position()[1]) == 2


def test_sheet_header_error_does_not_wipe_holdings(live):
    """체결 탭 헤더가 깨져 체결을 못 읽은 날은 '체결이 지워졌다'로 보고 보유를 지우지 않는다."""
    buy = _fill_row(_NEXT_DAY, "매수", 4, 178.41)
    live(_SIGNAL_DAY, [])
    live(_NEXT_DAY, [buy])

    broken = live("2026-09-25", [{"Date": _NEXT_DAY, "Symbol": "TEST"}])

    assert any("헤더 오류" in e for e in broken["input_errors"])
    assert broken["rebuilt_from"] is None
    assert _position()[0]["units"]["1"] == 4
