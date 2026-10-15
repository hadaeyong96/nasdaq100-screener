"""가상 장부 5개 테스트 — 네트워크 없이 합성 데이터만 쓴다.

중복 실행, 전환 체결 시점(t 신호 → t+1 종가), 미래 데이터 방지, 장부 오류 격리, 세금·분배금,
시트 탭 보호, 브리핑 국면 한 줄.
"""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core import virtual_books as vbc
from data import virtual_books_sheet as vbsheet
from engine import virtual_books as vbe
from notify import briefing, telegram
from notify import virtual_books as vbtext
from store import virtual_books as vbstore

TICKERS = ["QQQ", "QLD", "TQQQ", "SGOV"]
ET = ZoneInfo("America/New_York")
NOW_ET = datetime(2030, 1, 10, 18, 0, tzinfo=ET)


def _index(n: int, start: str = "2023-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _qqq_path(n: int, flips: list[int]) -> np.ndarray:
    """250일 완만한 상승 뒤 하루 ±3%로, flips 위치마다 방향을 바꿔 200일선을 오가는 QQQ 종가.

    기본 flips [250, 262, 280] → 200일선 교차(신호일) 253(아래로)·271(위로)·288(아래로).
    """
    px = np.empty(n)
    px[0] = 100.0
    direction = 1
    for i in range(1, n):
        if i in flips:
            direction *= -1
        step = 0.001 if i < 250 else 0.03
        px[i] = px[i - 1] * (1 + direction * step)
    return px


def _frames(n: int = 300, flips: list[int] | None = None, start: str = "2023-01-02"):
    idx = _index(n, start)
    q = _qqq_path(n, flips if flips is not None else [250, 262, 280])
    r = np.r_[0.0, q[1:] / q[:-1] - 1]
    closes = pd.DataFrame({
        "QQQ": q,
        "QLD": 50 * np.cumprod(1 + 2 * r),
        "TQQQ": 30 * np.cumprod(1 + 3 * r),
        "SGOV": 100.0 + 0.0 * q,
    }, index=idx)
    divs = pd.DataFrame(0.0, index=idx, columns=TICKERS)
    return closes, divs


def _books():
    return [
        vbc.BookSpec("QQQ 그냥 보유", vbc.HOLD, "QQQ"),
        vbc.BookSpec("QLD 그냥 보유", vbc.HOLD, "QLD"),
        vbc.BookSpec("TQQQ 그냥 보유", vbc.HOLD, "TQQQ"),
        vbc.BookSpec("QLD 200일선 전환", vbc.SMA_SWITCH, "QLD"),
        vbc.BookSpec("TQQQ 200일선 전환", vbc.SMA_SWITCH, "TQQQ"),
    ]


def _params(**kw):
    return vbc.LedgerParams(initial_usd=70_000.0, **kw)


def _year_ends(idx: pd.DatetimeIndex) -> set:
    s = pd.Series(idx, index=idx)
    return set(s.groupby(idx.year).max())


# ── 코어: 체결 규칙 ─────────────────────────────────────────────────────────


def test_hold_book_buys_integer_shares_at_start_close_with_commission():
    closes, divs = _frames()
    start = closes.index[230]
    led = vbc.run_ledger(_books(), closes, divs, start, _params(), set())
    t = led.books["QQQ 그냥 보유"].trades
    assert len(t) == 1 and t[0]["side"] == vbc.SIDE_BUY and t[0]["fill_date"] == start
    px = closes.at[start, "QQQ"]
    n = math.floor(70_000 / (px * 1.0007))
    assert t[0]["qty"] == n
    assert t[0]["commission"] == pytest.approx(n * px * 0.0007)
    d0 = led.books["QQQ 그냥 보유"].daily.iloc[0]
    assert d0["value"] == pytest.approx(70_000 - n * px * 0.0007)  # 첫날 손실 = 수수료만


def test_switch_fills_at_next_day_close_after_signal():
    closes, divs = _frames()
    start = closes.index[230]
    led = vbc.run_ledger(_books(), closes, divs, start, _params(), set())
    assert led.switches, "합성 데이터에 전환 신호가 있어야 한다"
    sig = led.switches[0]
    idx = closes.index
    assert sig["fill_date"] == idx[idx.get_loc(sig["signal_date"]) + 1]
    trades = [t for t in led.books["TQQQ 200일선 전환"].trades if t["fill_date"] == sig["fill_date"]]
    sell = next(t for t in trades if t["side"] == vbc.SIDE_SELL)
    buy = next(t for t in trades if t["side"] == vbc.SIDE_BUY)
    assert sell["signal_date"] == sig["signal_date"] and buy["signal_date"] == sig["signal_date"]
    assert sell["ticker"] == "TQQQ" and buy["ticker"] == "SGOV"  # 위 → 아래 전환
    assert sell["price"] == pytest.approx(closes.at[sig["fill_date"], "TQQQ"])
    assert buy["price"] == pytest.approx(closes.at[sig["fill_date"], "SGOV"])
    # 신호일 당일 종가에는 아직 이전 종목을 들고 있다
    assert led.books["TQQQ 200일선 전환"].daily.at[sig["signal_date"], "holding"] == "TQQQ"
    assert led.books["TQQQ 200일선 전환"].daily.at[sig["fill_date"], "holding"] == "SGOV"


def test_switch_initial_buy_uses_previous_day_signal():
    closes, divs = _frames()
    idx = closes.index
    sig = vbc.run_ledger(_books(), closes, divs, idx[230], _params(), set()).switches[0]
    start = sig["fill_date"]  # 시작일 전날이 전환 신호일 → 첫 매수는 새 국면 종목
    led = vbc.run_ledger(_books(), closes, divs, start, _params(), set())
    first = led.books["QLD 200일선 전환"].trades[0]
    assert first["signal_date"] == sig["signal_date"] and first["fill_date"] == start
    assert first["ticker"] == ("QLD" if sig["above"] else "SGOV")
    assert first["reason"].startswith("시작 매수")


def test_switch_book_needs_sma_before_start():
    closes, divs = _frames()
    with pytest.raises(ValueError, match="200일선 신호가 없습니다"):
        vbc.run_ledger(_books(), closes, divs, closes.index[100], _params(), set())


def test_equal_to_sma_counts_as_below():
    s = pd.Series([1.0, 2.0, 3.0, 2.0])
    reg = vbc.regime_frame(s, 3)
    assert reg["above"].iloc[2] == 1.0 and reg["above"].iloc[3] == 0.0  # 3 > 2 위, 2 == 7/3? 아래
    s2 = pd.Series([2.0, 2.0, 2.0])
    assert vbc.regime_frame(s2, 3)["above"].iloc[2] == 0.0  # 같으면 아래


def test_missing_close_after_start_raises():
    closes, divs = _frames()
    closes.iloc[260, closes.columns.get_loc("SGOV")] = np.nan
    with pytest.raises(ValueError, match="빈 날"):
        vbc.run_ledger(_books(), closes, divs, closes.index[230], _params(), set())


# ── 코어: 분배금·세금 ───────────────────────────────────────────────────────


def test_dividend_credited_after_withholding_to_shares_held_before_ex_date():
    closes, divs = _frames()
    start = closes.index[230]
    ex = closes.index[235]
    divs.loc[ex, "QQQ"] = 1.0
    led = vbc.run_ledger(_books(), closes, divs, start, _params(), set())
    d = led.books["QQQ 그냥 보유"].daily
    qty = d["qty"].iat[0]
    assert d.at[ex, "dividends_cum"] == pytest.approx(qty * 1.0 * 0.85)
    assert d.at[ex, "cash"] - d.at[closes.index[234], "cash"] == pytest.approx(qty * 0.85)


def test_year_end_tax_formula_on_realized_gain():
    closes, divs = _frames(n=300, start="2022-01-03")  # 첫 전환 체결(254)과 2022 연말(259)이 겹치게
    start = closes.index[230]
    led = vbc.run_ledger(_books(), closes, divs, start, _params(), _year_ends(closes.index))
    book = led.books["TQQQ 200일선 전환"]
    sells = [t for t in book.trades if t["side"] == vbc.SIDE_SELL and t["kind"] == vbc.KIND_SWITCH]
    year = sells[0]["fill_date"].year
    realized = sum(t["realized_pnl"] for t in sells if t["fill_date"].year == year)
    assert book.taxes[year]["tax"] == pytest.approx(max(realized, 0) * 0.22)
    hold = led.books["QQQ 그냥 보유"]
    assert all(v["tax"] == 0 for v in hold.taxes.values())  # 그냥 보유는 실현 없음


def _tax_case():
    """QLD로 이익을 낸 뒤 12/20 SGOV로 전환 → 12/30 연말 세금이 남은 현금보다 커서 SGOV 일부 매도."""
    idx = pd.bdate_range("2022-12-01", "2023-01-13")
    k = np.arange(len(idx), dtype=float)
    drop = idx >= pd.Timestamp("2022-12-19")
    qqq = np.where(drop, 50.0, 100.0 + k)
    qld = np.where(idx >= pd.Timestamp("2022-12-21"), 80.0, 50.0 + 2 * k)
    closes = pd.DataFrame({"QQQ": qqq, "QLD": qld, "TQQQ": qld, "SGOV": 100.0}, index=idx)
    divs = pd.DataFrame(0.0, index=idx, columns=TICKERS)
    books = [vbc.BookSpec("QLD 200일선 전환", vbc.SMA_SWITCH, "QLD")]
    params = vbc.LedgerParams(initial_usd=70_000.0, sma_days=3)
    led = vbc.run_ledger(books, closes, divs, pd.Timestamp("2022-12-07"), params, {pd.Timestamp("2022-12-30")})
    return led.books["QLD 200일선 전환"], led


def test_year_end_tax_deducted_from_cash_selling_shares_when_short():
    book, led = _tax_case()
    d = book.daily
    ye, before = pd.Timestamp("2022-12-30"), pd.Timestamp("2022-12-29")
    switch_sell = next(t for t in book.trades if t["kind"] == vbc.KIND_SWITCH and t["side"] == vbc.SIDE_SELL)
    assert switch_sell["fill_date"] == pd.Timestamp("2022-12-20")  # 12/19 신호 → 12/20 종가
    tax = book.taxes[2022]["tax"]
    assert tax == pytest.approx(0.22 * switch_sell["realized_pnl"]) and tax > d.at[before, "cash"]
    tax_sale = next(t for t in book.trades if t["kind"] == vbc.KIND_TAX)
    assert tax_sale["fill_date"] == ye and tax_sale["ticker"] == "SGOV" and tax_sale["reason"].startswith("2022년 세금 납부 매도")
    short = tax - d.at[before, "cash"]
    assert tax_sale["qty"] == math.ceil(short / (100.0 * (1 - 0.0007)))
    assert d.at[ye, "qty"] == d.at[before, "qty"] - tax_sale["qty"]
    assert 0 <= d.at[ye, "cash"] < 100.0
    assert d.at[ye, "cash"] == pytest.approx(d.at[before, "cash"] + tax_sale["amount"] - tax)
    assert d.at[ye, "value"] == pytest.approx(d.at[before, "value"] - tax - tax_sale["commission"])  # SGOV 가격 그대로
    assert d.at[ye, "tax_cum"] == pytest.approx(tax)
    # 납부용 매도 손익(매수 수수료 몫만큼 손실)은 다음 해 장부로
    assert d.at[ye, "realized_ytd"] == pytest.approx(tax_sale["realized_pnl"]) and tax_sale["realized_pnl"] < 0
    assert d["realized_ytd"].iat[-1] == pytest.approx(tax_sale["realized_pnl"])
    assert led.summaries[0]["switches"] == 1  # 세금 납부 매도는 전환 횟수에서 뺀다
    assert led.summaries[0]["tax_cum"] == pytest.approx(tax)


def test_liquidation_after_tax_estimate_for_hold_book():
    closes, divs = _frames()
    led = vbc.run_ledger(_books(), closes, divs, closes.index[230], _params(), set())
    d = led.books["QQQ 그냥 보유"].daily.iloc[-1]
    basis = led.books["QQQ 그냥 보유"].trades[0]["amount"]
    liq = d["qty"] * d["close"] * (1 - 0.0007)
    expect = d["cash"] + liq - max(liq - basis, 0) * 0.22
    assert d["value_after_tax_liq"] == pytest.approx(expect)


def test_summary_counts_switches_and_drawdown_breach():
    closes, divs = _frames()
    led = vbc.run_ledger(_books(), closes, divs, closes.index[230], _params(), set(), min_days_for_annualized=60)
    by = {s["book"]: s for s in led.summaries}
    assert by["QQQ 그냥 보유"]["switches"] == 0
    assert by["TQQQ 200일선 전환"]["switches"] == len([s for s in led.switches if s["fill_date"] is not None])
    d = led.books["TQQQ 그냥 보유"].daily
    breach = vbc.first_drawdown_breach(d, -50)
    assert breach is not None and d.at[breach, "drawdown_pct"] <= -50
    assert (d.loc[: breach]["drawdown_pct"].iloc[:-1] > -50).all()


# ── 미래 데이터 방지 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("cut", [231, 250, 262, 263, 270, 281, 299])
def test_no_lookahead_ledger_cut_equals_full(cut):
    closes, divs = _frames()
    divs.loc[closes.index[240], "QQQ"] = 0.5
    start = closes.index[230]
    full = vbc.run_ledger(_books(), closes, divs, start, _params(), set())
    part = vbc.run_ledger(_books(), closes.iloc[:cut], divs.iloc[:cut], start, _params(), set())
    t = closes.index[cut - 1]
    for name in full.books:
        pd.testing.assert_frame_equal(part.books[name].daily, full.books[name].daily.loc[:t])
        assert [x for x in full.books[name].trades if x["fill_date"] <= t] == part.books[name].trades
    pd.testing.assert_frame_equal(part.regime, full.regime.loc[:t])


# ── 문구 ─────────────────────────────────────────────────────────────────


def test_regime_line_format():
    closes, divs = _frames()
    led = vbc.run_ledger(_books(), closes, divs, closes.index[230], _params(), set())
    reg = vbc.regime_today(vbc.run_ledger(_books(), closes.iloc[:250], divs.iloc[:250], closes.index[230], _params(), set()), _books())
    line = vbtext.regime_line(reg, "QQQ", "SGOV", ["QLD", "TQQQ"])
    assert line.startswith("오늘 국면: QQQ 200일선 위(+") and line.endswith("· 전환 장부 QLD·TQQQ 보유 중")
    sig = led.switches[0]
    i = closes.index.get_loc(sig["signal_date"]) + 1
    reg2 = vbc.regime_today(vbc.run_ledger(_books(), closes.iloc[:i], divs.iloc[:i], closes.index[230], _params(), set()), _books())
    line2 = vbtext.regime_line(reg2, "QQQ", "SGOV", ["QLD", "TQQQ"])
    assert "200일선 아래(-" in line2 and "QLD·TQQQ 보유 중 → 오늘 전환 신호, 다음 거래일 종가에 SGOV로" in line2


def test_briefing_puts_regime_line_under_title_and_caption_keeps_it():
    summary = {"as_of": pd.Timestamp("2026-10-13"), "mode_label": "실전", "regime_line": "오늘 국면: QQQ 200일선 위(+3.2%) · 전환 장부 QLD·TQQQ 보유 중"}
    text = briefing.build_briefing_text(summary, {})
    lines = text.split("\n")
    assert lines[1] == summary["regime_line"]
    cap = telegram.report_caption(text)
    assert cap.split("\n") == [lines[0], summary["regime_line"]]
    assert "오늘 국면" not in briefing.build_briefing_text({**summary, "regime_line": None}, {})


# ── 시트 탭 보호 ────────────────────────────────────────────────────────────


class _FakeWs:
    def __init__(self, title):
        self.title, self.values, self.row_count, self.col_count = title, None, 1000, 26

    def clear(self):
        self.values = None

    def resize(self, rows, cols):
        self.row_count, self.col_count = rows, cols

    def update(self, values, range_name, value_input_option):
        self.values = values


class _FakeSpreadsheet:
    def __init__(self):
        self.tabs = {}

    def worksheet(self, name):
        import gspread

        if name not in self.tabs:
            raise gspread.WorksheetNotFound(name)
        return self.tabs[name]

    def add_worksheet(self, title, rows, cols):
        self.tabs[title] = _FakeWs(title)
        return self.tabs[title]


class _FakeClient:
    def __init__(self):
        self.ss = _FakeSpreadsheet()
        self.opened = []

    def open_by_key(self, key):
        self.opened.append(key)
        return self.ss


def test_sheet_writer_rejects_other_tabs_before_writing():
    client = _FakeClient()
    with pytest.raises(vbsheet.VirtualBooksSheetForbiddenError):
        vbsheet.write_tables(client, "sid", {"전략요약": [["a"]], "계획": [["x"]]})
    assert client.opened == [] and client.ss.tabs == {}


# ── 엔진: 중복 실행·알림 한 번·오류 격리 ──────────────────────────────────


def _yf_like(closes, divs):
    def history_fn(ticker, start):
        idx = closes.index.tz_localize("America/New_York")
        return pd.DataFrame({"Close": closes[ticker].values, "Dividends": divs[ticker].values}, index=idx)
    return history_fn


def _cfg(start_date=None):
    return {"virtual_books": {
        "enabled": True, "start_date": start_date, "start_krw": 100_000_000, "start_fx": None,
        "signal_ticker": "QQQ", "sma_days": 200, "defensive_ticker": "SGOV", "commission_pct": 0.07,
        "slippage_pct": 0.0, "capital_gains_rate_pct": 22, "dividend_withholding_pct": 15,
        "min_days_for_annualized": 60, "benchmark_book": "QQQ 그냥 보유",
        "drawdown_alert_book": "TQQQ 그냥 보유", "drawdown_alert_pct": -50,
        "sheet_id_env": "VB_TEST_SHEET_ID",
        "books": [{"name": b.name, "kind": b.kind, "asset": b.asset} for b in _books()],
    }}


def _run_kwargs(closes, divs, tmp_path, client):
    return dict(db_path=tmp_path / "vb.db", sheets_client=client, now_et=NOW_ET,
                history_fn=_yf_like(closes, divs), fx_fn=lambda d: (1400.0, d.isoformat(), None),
                year_end_fn=lambda a, b: _year_ends(closes.index), next_day_fn=lambda d: None)


def test_same_day_double_run_no_duplicates_and_alerts_once(tmp_path, monkeypatch):
    monkeypatch.setenv("VB_TEST_SHEET_ID", "copy-sheet")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    closes, divs = _frames(n=283)
    start = closes.index[230].date().isoformat()
    cfg = _cfg(start)
    client = _FakeClient()
    sent_texts = []

    def sender(text, force_no_send=False):
        sent_texts.append(text)
        return True

    r1 = vbe.run(cfg, **_run_kwargs(closes, divs, tmp_path, client))
    s1 = vbe.send_alerts(r1, sender=sender)
    conn = vbstore.connect(tmp_path / "vb.db")
    n_daily, n_trades = vbstore.count_rows(conn, "daily"), vbstore.count_rows(conn, "trades")
    conn.close()
    sheet1 = {k: v.values for k, v in client.ss.tabs.items()}

    r2 = vbe.run(cfg, **_run_kwargs(closes, divs, tmp_path, client))
    s2 = vbe.send_alerts(r2, sender=sender)
    conn = vbstore.connect(tmp_path / "vb.db")
    assert vbstore.count_rows(conn, "daily") == n_daily == 5 * (283 - 230)
    assert vbstore.count_rows(conn, "trades") == n_trades
    runs = vbstore.runs_for_base_date(conn, closes.index[-1].date().isoformat())
    conn.close()
    assert len(runs) == 2 and len({r["run_id"] for r in runs}) == 2 and all(r["status"] == "ok" for r in runs)
    assert r2["same_day_runs"] == 1
    def _no_stamp(tabs):  # 전략요약의 "갱신" 시각 줄만 실행마다 다르다
        return {k: [r for r in v if not (r and r[0] == "갱신")] for k, v in tabs.items()}

    assert _no_stamp({k: v.values for k, v in client.ss.tabs.items()}) == _no_stamp(sheet1)
    assert set(client.ss.tabs) == set(vbsheet.ALLOWED_TABS)
    assert len(s1) >= 1 and s2 == []  # 낙폭·전환 알림은 한 번만
    assert len(sent_texts) == len(s1)


def test_unsent_alert_is_retried_next_run(tmp_path, monkeypatch):
    monkeypatch.setenv("VB_TEST_SHEET_ID", "copy-sheet")
    closes, divs = _frames(n=283)
    cfg = _cfg(closes.index[230].date().isoformat())
    r1 = vbe.run(cfg, **_run_kwargs(closes, divs, tmp_path, _FakeClient()))
    assert vbe.send_alerts(r1, force_no_send=True, sender=lambda t, force_no_send=False: not force_no_send) == []
    r2 = vbe.run(cfg, **_run_kwargs(closes, divs, tmp_path, _FakeClient()))
    assert len(vbe.send_alerts(r2, sender=lambda t, force_no_send=False: True)) == len(r1["alerts"]) >= 1


def test_first_run_without_start_date_starts_on_base_date_and_remembers(tmp_path, monkeypatch):
    monkeypatch.delenv("VB_TEST_SHEET_ID", raising=False)
    closes, divs = _frames(n=260)
    r1 = vbe.run(_cfg(None), **_run_kwargs(closes, divs, tmp_path, None))
    assert r1["start"] == closes.index[-1] and r1["alerts"] == []
    assert any("시트 기록을 건너뜁니다" in w for w in r1["warnings"])
    closes2, divs2 = _frames(n=265)
    r2 = vbe.run(_cfg(None), **_run_kwargs(closes2, divs2, tmp_path, None))
    assert r2["start"] == closes.index[-1]  # DB에 기억한 시작일 그대로
    tr = [t for t in r2["ledger"].books["QQQ 그냥 보유"].trades]
    assert len(tr) == 1 and tr[0]["fill_date"] == closes.index[-1]
    assert r2["ledger"].summaries[0]["initial_usd"] == pytest.approx(100_000_000 / 1400.0)


def test_run_safely_isolates_errors_and_alerts_separately(tmp_path):
    def broken_history(ticker, start):
        raise ConnectionError("network down")

    ops = []
    out = vbe.run_safely(_cfg("2023-11-01"), db_path=tmp_path / "vb.db", history_fn=broken_history,
                         now_et=NOW_ET, ops_sender=ops.append)
    assert out is None
    assert len(ops) == 1 and "가상 장부 오류 (기존 브리핑은 그대로 나감)" in ops[0] and "TQQQ" in ops[0]
    conn = vbstore.connect(tmp_path / "vb.db")
    assert vbstore.count_rows(conn, "runs") == 1 and vbstore.count_rows(conn, "daily") == 0
    conn.close()


def test_sheet_failure_is_partial_error_ledger_still_saved(tmp_path, monkeypatch):
    monkeypatch.setenv("VB_TEST_SHEET_ID", "copy-sheet")
    closes, divs = _frames(n=260)

    class Boom:
        def open_by_key(self, key):
            raise PermissionError("403")

    ops = []
    kw = _run_kwargs(closes, divs, tmp_path, Boom())
    out = vbe.run_safely(_cfg(closes.index[230].date().isoformat()), ops_sender=ops.append, **kw)
    assert out is not None and out["regime_line"].startswith("오늘 국면:")
    assert out["errors"] and "시트 기록 실패" in ops[0]
    conn = vbstore.connect(tmp_path / "vb.db")
    assert vbstore.count_rows(conn, "daily") == 5 * 30
    conn.close()


def test_send_alerts_safely_never_raises(tmp_path):
    def bad_sender(text, force_no_send=False):
        raise RuntimeError("telegram down")

    ops = []
    res = {"alerts": [{"kind": "switch", "key": "x", "text": "t"}], "db_path": tmp_path / "vb.db", "dry_run": False}
    assert vbe.send_alerts_safely(res, sender=bad_sender, ops_sender=ops.append) == []
    assert ops and "알림 발송 오류" in ops[0]


def test_disabled_or_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("VB_TEST_SHEET_ID", "copy-sheet")
    cfg = _cfg(None)
    cfg["virtual_books"]["enabled"] = False
    assert vbe.run_safely(cfg, db_path=tmp_path / "vb.db") is None
    closes, divs = _frames(n=260)
    client = _FakeClient()
    r = vbe.run(_cfg("2023-11-01"), dry_run=True, **_run_kwargs(closes, divs, tmp_path, client))
    assert r["regime_line"] and client.opened == []
    conn = vbstore.connect(tmp_path / "vb.db")
    assert vbstore.count_rows(conn, "daily") == 0 and vbstore.count_rows(conn, "runs") == 0
    conn.close()


def test_ci_without_drive_file_and_start_date_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS", raising=False)
    with pytest.raises(RuntimeError, match="매 실행이 새 장부로"):
        vbe.run(_cfg(None), db_path=tmp_path / "vb.db")


def test_future_start_date_is_not_started(tmp_path):
    closes, divs = _frames(n=260)
    r = vbe.run(_cfg("2031-01-02"), **_run_kwargs(closes, divs, tmp_path, None))
    assert r["status"] == "not_started" and r["regime_line"] is None


def test_unconfirmed_bar_after_last_closed_day_is_dropped():
    from data import virtual_books_data as vbdata

    closes, divs = _frames(n=260, start="2026-01-02")
    last_closed = datetime(2026, 12, 1, 18, 0, tzinfo=ET)  # 이후 행은 미확정으로 버린다
    c, _ = vbdata.fetch_closes_and_dividends(TICKERS, closes.index[0].date(), now_et=last_closed,
                                             history_fn=_yf_like(closes, divs))
    assert c.index[-1] <= pd.Timestamp("2026-12-01")


def test_fetch_collects_all_failures():
    from data import virtual_books_data as vbdata

    def h(t, start):
        if t in ("QLD", "SGOV"):
            raise ConnectionError("x")
        closes, divs = _frames(n=10)
        return _yf_like(closes, divs)(t, start)

    with pytest.raises(vbdata.VirtualBooksDataError) as e:
        vbdata.fetch_closes_and_dividends(TICKERS, datetime(2023, 1, 1).date(), now_et=NOW_ET, history_fn=h)
    assert "QLD" in str(e.value) and "SGOV" in str(e.value)


def test_trailing_empty_close_row_is_trimmed_with_warning():
    """마감 직후 yfinance가 마지막 날 행을 종가 없이 줄 때(2026-10-15 실제 사례) — 잘라내고 경고."""
    from data import virtual_books_data as vbdata

    closes, divs = _frames(n=260)
    holed = closes.copy()
    holed.iloc[-1] = np.nan  # 네 종목 모두 마지막 날 종가 없음
    c, d = vbdata.fetch_closes_and_dividends(TICKERS, closes.index[0].date(), now_et=NOW_ET,
                                             history_fn=_yf_like(holed, divs))
    assert c.index[-1] == closes.index[-2] and d.index[-1] == closes.index[-2]
    assert c.notna().all().all()
    pd.testing.assert_frame_equal(c, closes.iloc[:-1], check_freq=False)
    (w,) = c.attrs["warnings"]
    assert str(closes.index[-1].date()) in w and "잘라" in w


def test_trailing_empty_close_in_one_ticker_trims_whole_day():
    from data import virtual_books_data as vbdata

    closes, divs = _frames(n=260)
    holed = closes.copy()
    holed.iloc[-2:, holed.columns.get_loc("SGOV")] = np.nan  # 맨 끝 이틀 SGOV만 비어도 그 이틀을 자른다
    c, _ = vbdata.fetch_closes_and_dividends(TICKERS, closes.index[0].date(), now_et=NOW_ET,
                                             history_fn=_yf_like(holed, divs))
    assert c.index[-1] == closes.index[-3]
    assert "SGOV" in c.attrs["warnings"][0] and "2일" in c.attrs["warnings"][0]


def test_no_trailing_hole_means_no_warning():
    from data import virtual_books_data as vbdata

    closes, divs = _frames(n=260)
    c, _ = vbdata.fetch_closes_and_dividends(TICKERS, closes.index[0].date(), now_et=NOW_ET,
                                             history_fn=_yf_like(closes, divs))
    assert len(c) == len(closes) and c.attrs["warnings"] == []


def test_middle_empty_close_after_start_still_raises(tmp_path):
    """중간 날짜가 비면 채우지 않고 지금처럼 멈춘다 (run_safely가 잡아 오류 알림)."""
    closes, divs = _frames(n=300)
    holed = closes.copy()
    holed.iloc[-5, holed.columns.get_loc("QLD")] = np.nan
    with pytest.raises(ValueError, match="종가가 빈 날"):
        vbe.run(_cfg(str(closes.index[250].date())), **_run_kwargs(holed, divs, tmp_path, None))


def test_run_with_trailing_hole_succeeds_and_catches_up_next_run(tmp_path, monkeypatch):
    """끝 빈 행이 있는 날은 전날 기준으로 계산·경고, 다음 실행에서 그날까지 따라잡는다."""
    monkeypatch.delenv("VB_TEST_SHEET_ID", raising=False)
    closes, divs = _frames(n=300)
    holed = closes.copy()
    holed.iloc[-1] = np.nan
    cfg = _cfg(str(closes.index[250].date()))
    r1 = vbe.run(cfg, **_run_kwargs(holed, divs, tmp_path, None))
    assert r1["status"] == "ok" and r1["base_date"] == closes.index[-2]
    assert any("잘라" in w for w in r1["warnings"])
    r2 = vbe.run(cfg, **_run_kwargs(closes, divs, tmp_path, None))
    assert r2["base_date"] == closes.index[-1] and not any("잘라" in w for w in r2["warnings"])
    full = vbe.run(cfg, **_run_kwargs(closes, divs, tmp_path / "fresh", None))
    for name in r2["ledger"].books:
        pd.testing.assert_frame_equal(r2["ledger"].books[name].daily, full["ledger"].books[name].daily)


def test_start_day_itself_empty_is_not_started(tmp_path):
    """시작일 당일 종가가 아직 비어 있으면 시작 전으로 처리(장부·시작값 저장 안 함)."""
    closes, divs = _frames(n=300)
    holed = closes.copy()
    holed.iloc[-1] = np.nan
    r = vbe.run(_cfg(str(closes.index[-1].date())), **_run_kwargs(holed, divs, tmp_path, None))
    assert r["status"] == "not_started"


# ── daily 연결: 국면 줄 붙이기 실패해도 브리핑 보호 ──────────────────────────


def test_attach_regime_line_guards(monkeypatch, tmp_path):
    from engine import daily

    summary = {"as_of": pd.Timestamp("2026-10-13"), "report_path": "orig.html"}
    daily._attach_regime_line(summary, None, {})
    assert "regime_line" not in summary and summary["report_path"] == "orig.html"

    other_day = {"regime_line": "오늘 국면: x", "base_date": pd.Timestamp("2026-10-12")}
    daily._attach_regime_line(summary, other_day, {})
    assert "regime_line" not in summary

    def boom(*a, **k):
        raise RuntimeError("template")

    monkeypatch.setattr(daily.report_html, "render_report", boom)
    same_day = {"regime_line": "오늘 국면: x", "base_date": pd.Timestamp("2026-10-13")}
    daily._attach_regime_line(summary, same_day, {})
    assert "regime_line" not in summary and summary["report_path"] == "orig.html"

    monkeypatch.setattr(daily.report_html, "render_report", lambda s, c, o: tmp_path / "new.html")
    daily._attach_regime_line(summary, same_day, {})
    assert summary["regime_line"] == "오늘 국면: x" and summary["report_path"] == tmp_path / "new.html"
