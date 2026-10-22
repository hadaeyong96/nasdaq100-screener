"""8번 장부 「가치평가 따라하기」 (core/value_book.py, engine/value_book.py, data/value_live.py) — 합성 데이터만, 네트워크 없음.

로컬 테스트는 dry-run·주입 함수만 쓴다(텔레그램 실제 전송 없음).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from core import value_book as vbk
from core import value_score as vs
from data import value_live as vl
from data.market_calendar import trading_days_between
from engine import value_book as ev
from notify import letter
from store import virtual_books as vbstore
from tests.legacy import legacy_cfg

P = vbk.ValueParams(name="가치평가 따라하기")
FX = 1400.0


def _board(scores: dict[str, float], sic: dict[str, str] | None = None, universe=None) -> dict:
    df = pd.DataFrame({"total": scores, "peg_value": {t: 1.0 for t in scores},
                       "universe": {t: (universe is None or t in universe) for t in scores},
                       "sic2": {t: (sic or {}).get(t, f"{i:02d}") for i, t in enumerate(scores)}})
    return vbk.board(df)


def test_display_scale_and_thresholds():
    assert vbk.display_score(49.0) == pytest.approx(70.0)
    assert vbk.display_score(35.0) == pytest.approx(50.0)
    assert vbk.display_score(56.0) == pytest.approx(80.0)
    assert vbk.threshold_text(P) == "매수 70 · 유지 50 · 갈아타기 80점 이상·약 14.3점 차"


def test_start_picks_threshold_sector_cap_and_lump_sum():
    b = vbk.new_book(P.name, "2026-10-19")
    bd = _board({"A": 60, "B": 59, "C": 58, "D": 57, "E": 48}, {"A": "28", "B": "28", "C": "28", "D": "28", "E": "35"})
    closes = {t: 100.0 for t in "ABCDE"}
    b2, row, tr = vbk.process_day(b, "2026-10-16", closes=closes, fx=FX, dividends={}, splits={}, p=P, start_scores=bd)
    assert row is None and b2["status"] == vbk.WAITING  # 시작일 전
    b3, row, tr = vbk.process_day(b, "2026-10-19", closes=closes, fx=FX, dividends={}, splits={}, p=P, start_scores=bd)
    assert b3["status"] == vbk.RUNNING and b3["start_fx"] == FX
    assert b3["initial_usd"] == pytest.approx(1e8 / FX * 0.999)  # 1억 원 일시금, 환전 0.1%
    assert sorted(t["ticker"] for t in tr) == ["A", "B", "C"]  # 업종 28 최대 3, E는 49점 미만
    each = b3["initial_usd"] / 10
    assert all(t["qty"] == int(each // (100 * 1.001 * 1.0007)) for t in tr)


def _running(positions: dict, cash=50_000.0, start="2025-01-02") -> dict:
    b = vbk.new_book(P.name, start)
    b.update(status=vbk.RUNNING, start_date=pd.Timestamp(start), start_fx=FX, initial_usd=70_000.0, cash=cash, peak=70_000.0)
    for t, (q, cost_usd, first) in positions.items():
        b["positions"][t] = {"qty": q, "cost_krw": cost_usd * FX, "first_buy": pd.Timestamp(first), "last_close": 1.0,
                             "flag": None, "missing": 0}
    return b


def test_quarter_rule_needs_two_quarters_and_one_year():
    b = _running({"A": (100, 1000.0, "2026-02-02")})
    low = _board({"A": 30.0})
    b, _, _ = vbk.process_day(b, "2026-09-30", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P, quarter_scores=low)
    assert b["sell_next"] == []  # 첫 번째 미만
    b, _, _ = vbk.process_day(b, "2026-12-31", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P, quarter_scores=low,
                              is_year_end=True)
    assert b["sell_next"] == ["A"]
    b2, _, tr = vbk.process_day(b, "2027-01-04", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P)
    assert tr == [] and "A" in b2["positions"]  # 첫 매수 2026-02-02 → 365일 안 됨
    b, _, _ = vbk.process_day(b2, "2027-03-31", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P, quarter_scores=low)
    b3, _, tr = vbk.process_day(b, "2027-04-01", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P)
    assert [t["kind"] for t in tr] == [vbk.K_QTR] and "A" not in b3["positions"]


def test_unknown_quarter_breaks_streak():
    b = _running({"A": (100, 1000.0, "2025-01-02")})
    b, _, _ = vbk.process_day(b, "2026-06-30", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P, quarter_scores=_board({"A": 30.0}))
    b, _, _ = vbk.process_day(b, "2026-09-30", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P, quarter_scores=_board({"B": 60.0}))
    b, _, _ = vbk.process_day(b, "2026-12-31", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P, quarter_scores=_board({"A": 30.0}))
    assert b["sell_next"] == []


def test_december_swap_qty_limited_by_deduction_then_harvest():
    """2026-12-29: 보유 A(40점, 주당 이익 큼) → 후보 C(60점). 수량 = 내림(남은 공제 ÷ 1주당 이익) < 절반, 그 뒤 남은 공제로 공제 소진."""
    b = _running({"A": (1000, 10_000.0, "2025-03-03"), "B": (500, 5_000.0, "2025-03-03")}, cash=100.0)
    nov = _board({"A": 40.0, "B": 50.0, "C": 60.0}, {"A": "28", "B": "35", "C": "28"})
    closes = {"A": 30.0, "B": 12.0, "C": 50.0}
    b2, _, tr = vbk.process_day(b, "2026-12-29", closes=closes, fx=FX, dividends={}, splits={}, p=P, nov_scores=nov,
                                is_dec_exec=True)
    g = 30.0 * 0.999 * 0.9993 * FX - 10.0 * FX  # A 1주당 원화 이익
    q = int(2_500_000 // g)
    swap_out = [t for t in tr if t["kind"] == vbk.K_SWAP_OUT]
    assert len(swap_out) == 1 and swap_out[0]["ticker"] == "A" and swap_out[0]["qty"] == q and q < 500
    assert any(t["kind"] == vbk.K_SWAP_IN and t["ticker"] == "C" for t in tr)
    assert b2["swaps"][0]["sell"] == "A" and b2["swaps"][0]["buy"] == "C"
    assert b2["realized_krw"]["2026"] <= 2_500_000 + 1e-6  # 공제 한도 안
    assert b2["positions"]["A"]["first_buy"] == pd.Timestamp("2025-03-03")


def test_december_no_swap_when_deduction_used_up():
    b = _running({"A": (1000, 10_000.0, "2025-03-03")})
    b["realized_krw"]["2026"] = 3_000_000.0
    nov = _board({"A": 40.0, "C": 60.0}, {"A": "28", "C": "28"})
    _, _, tr = vbk.process_day(b, "2026-12-29", closes={"A": 30.0, "C": 50.0}, fx=FX, dividends={}, splits={}, p=P,
                               nov_scores=nov, is_dec_exec=True)
    assert not any(t["kind"].startswith("VAL_SWAP") for t in tr)  # 남은 공제 ≤ 0 → 건너뜀


def test_tax_common_rule_paid_first_may_day():
    b = _running({"A": (100, 1000.0, "2025-01-02")}, cash=10_000.0)
    b["realized_krw"]["2026"] = 12_500_000.0
    b, _, _ = vbk.process_day(b, "2026-12-31", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P, is_year_end=True)
    assert b["pending_tax"][1] == pytest.approx(2_200_000.0)  # (1,250만 − 250만) × 22%
    b, _, _ = vbk.process_day(b, "2027-05-03", closes={"A": 10.0}, fx=FX, dividends={}, splits={}, p=P, is_may_first=True)
    assert b["pending_tax"] is None and b["tax_cum_krw"] == pytest.approx(2_200_000.0)
    assert b["cash"] == pytest.approx(10_000.0 - 2_200_000.0 / (FX * 0.999))


def test_calendar_dec_exec_2026_is_dec_29():
    cal = ev.calendar(2026, 2027)
    assert cal["dec_exec"][2026] == pd.Timestamp("2026-12-29")
    assert cal["nov_end"][2026] == pd.Timestamp("2026-11-30")
    assert pd.Timestamp("2027-05-03") in cal["may_first"]
    assert ev.last_month_end_before(pd.Timestamp("2026-10-19")) == pd.Timestamp("2026-09-30")


# ── engine: 시작일 고정·DB 보호·점수 저장 ───────────────────────────────────


def _cfg():
    with open("config.yaml", encoding="utf-8") as f:
        return legacy_cfg(yaml.safe_load(f))


DAYS = trading_days_between(date(2026, 10, 14), date(2026, 10, 23))


def _ctx(base):
    return {"base_date": pd.Timestamp(base), "closes": pd.DataFrame({"QQQ": 1.0}, index=DAYS)}


def _hist(t, start):
    return pd.DataFrame({"Close": 100.0, "Dividends": 0.0, "Stock Splits": 0.0}, index=DAYS)


def _scores(calls):
    def fn(me, held):
        calls.append(me)
        df = pd.DataFrame({"total": {"AAA": 60.0, "BBB": 55.0, "CCC": 50.0}, "peg_value": {"AAA": 1.0, "BBB": 1.0, "CCC": 1.0},
                           "universe": {"AAA": True, "BBB": True, "CCC": True}, "sic2": {"AAA": "10", "BBB": "20", "CCC": "30"}})
        return df, []
    return fn


def test_config_pins_start_date_and_threshold_values():
    c = _cfg()["virtual_books"]
    assert c["value_score"]["start_date"] == "2026-10-19" and c["value_score"]["enabled"] is True
    assert c["momentum"]["enabled"] is False  # 6개월 강세 꺼 둠(코드·DB는 그대로)
    p = ev.value_params(_cfg())
    assert (p.min_pick, p.min_keep, p.min_swap, p.swap_gap, p.sector_cap) == (49.0, 35.0, 56.0, 10.0, 3)
    assert (p.commission_pct, p.slippage_pct, p.fx_spread_pct, p.deduction_krw) == (0.07, 0.1, 0.1, 2_500_000.0)


def test_engine_waits_then_starts_on_pinned_date_with_prev_month_end_scores(tmp_path):
    cfg = _cfg()
    conn = vbstore.connect(tmp_path / "v.db")
    calls = []
    kw = dict(history_fn=_hist, fx_range_fn=lambda a, b: {d.date().isoformat(): FX for d in DAYS}, score_fn=_scores(calls))
    r = ev.run_value_score(cfg, conn, _ctx("2026-10-16"), dry_run=False, **kw)
    assert r["extra"] == [] and any("대기" in w for w in r["warnings"]) and calls == []
    r = ev.run_value_score(cfg, conn, _ctx("2026-10-20"), dry_run=False, **kw)
    assert not r["errors"], r["errors"]
    assert calls == [pd.Timestamp("2026-09-30")]  # 시작일 전 마지막 월말 점수
    x = r["extra"][0]
    assert x["start"] == pd.Timestamp("2026-10-19") and sorted(x["positions"]) == ["AAA", "BBB", "CCC"]
    assert x["meta"]["start_fx"] == FX
    assert "AAA(86)" in x["summary"]["holdings_list"]  # 60점 × 100/70 표시
    assert "문턱 매수 70" in x["summary"]["rules"]
    first = [t for t in vbstore.picks_rows(conn, "picks_trades", "가치평가 따라하기")]
    assert {t["fill_date"] for t in first} == {"2026-10-19"}
    # 같은 날 다시 실행해도 다시 사지 않는다, 점수표는 DB에서 다시 읽는다
    r2 = ev.run_value_score(cfg, conn, _ctx("2026-10-20"), dry_run=False, **kw)
    assert len(vbstore.picks_rows(conn, "picks_trades", "가치평가 따라하기")) == len(first) and calls == [pd.Timestamp("2026-09-30")]
    assert vbstore.load_value_scores(conn, "가치평가 따라하기", "2026-09-30") is not None


def test_engine_refuses_restart_when_start_fx_pinned_but_db_missing(tmp_path):
    cfg = _cfg()
    cfg["virtual_books"]["value_score"]["start_fx"] = 1400.0
    conn = vbstore.connect(tmp_path / "v.db")
    r = ev.run_value_score(cfg, conn, _ctx("2026-10-20"), dry_run=False, history_fn=_hist,
                           fx_range_fn=lambda a, b: {}, score_fn=_scores([]))
    assert r["errors"] and "새로 시작하지 않습니다" in r["errors"][0]


def test_engine_dry_run_writes_nothing(tmp_path):
    conn = vbstore.connect(tmp_path / "v.db")
    r = ev.run_value_score(_cfg(), conn, _ctx("2026-10-20"), dry_run=True, history_fn=_hist,
                           fx_range_fn=lambda a, b: {d.date().isoformat(): FX for d in DAYS}, score_fn=_scores([]))
    assert r["extra"] and vbstore.load_picks_book(conn, "가치평가 따라하기") is None
    assert vbstore.load_value_scores(conn, "가치평가 따라하기", "2026-09-30") is None


# ── 단체방판: 종목명 노출 금지 ───────────────────────────────────────────────


def test_group_edition_hides_value_book_tickers(tmp_path):
    cfg = _cfg()
    conn = vbstore.connect(tmp_path / "v.db")
    r = ev.run_value_score(cfg, conn, _ctx("2026-10-19"), dry_run=True, history_fn=_hist,
                           fx_range_fn=lambda a, b: {d.date().isoformat(): FX for d in DAYS}, score_fn=_scores([]))
    books = letter.letter_books(cfg, pd.Timestamp("2026-10-19"), [], {}, r["extra"], None)
    vb_row = next(b for b in books if b["name"] == "가치평가 따라하기")
    assert vb_row["family"] == "rebalance" and vb_row["trades_today"]
    group_text = "\n".join(letter.strategy_lines(books, cfg, letter.GROUP))
    _, rows = letter.table_rows(books, cfg, letter.GROUP)
    all_group = group_text + "\n" + "\n".join(" ".join(r) for r in rows)
    for t in ("AAA", "BBB", "CCC"):
        assert t not in all_group
    assert "3종목" in all_group and "오늘 가상 거래" in all_group
    personal = "\n".join(letter.strategy_lines(books, cfg, letter.PERSONAL))
    assert "AAA" in personal  # 개인판에만


# ── 점수: 미래 정보 금지 (라이브 계산) ────────────────────────────────────────


def _facts(years, filed_lag=60):
    rows = []
    for i, y in enumerate(years):
        end = f"{y}-12-31"
        start = f"{y}-01-01"
        filed = (pd.Timestamp(end) + pd.Timedelta(days=filed_lag)).date().isoformat()
        for key, v in {"rev": 100 * 1.15 ** i, "ni": 10 * 1.1 ** i, "eps": 1.0 * 1.1 ** i, "opinc": 15.0, "ocf": 14.0,
                       "capex": 3.0, "gp": 50.0, "da": 5.0, "shares": 10.0, "equity": 60.0, "cash": 10.0, "assets": 150.0}.items():
            flow = key in vs.FLOW_KEYS
            rows.append({"tag": vs.TAGS[key][0], "start": start if flow else None, "end": end, "val": v, "filed": filed,
                         "form": "10-K", "unit": vs.UNITS.get(key, "USD")})
    return vs.prepare_facts(pd.DataFrame(rows))


def test_live_scores_ignore_filings_after_asof():
    full = _facts([2019, 2020, 2021, 2022, 2023, 2024, 2025])
    asof = pd.Timestamp("2025-01-31")  # 2024 10-K 접수(2025-03-01) 전
    kw = dict(universe_fn=lambda d: {"X", "Y"}, price_fn=lambda t, d: (20.0, pd.Series(dtype=float)), sic_fn=lambda t: "2800")
    a, _ = vl.compute_scores(asof, facts_fn=lambda t: full, **kw)
    b, _ = vl.compute_scores(asof, facts_fn=lambda t: full[full["filed"] < "2025-01-31"], **kw)
    cols = vs.ITEMS + ["total", "n_na"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    assert bool(a.loc["X", "universe"])


def test_live_scores_failures_reported_not_silent():
    kw = dict(universe_fn=lambda d: {"X", "BAD"}, price_fn=lambda t, d: (20.0, pd.Series(dtype=float)), sic_fn=lambda t: "2800")

    def facts(t):
        if t == "BAD":
            raise ConnectionError("EDGAR 실패")
        return _facts([2019, 2020, 2021, 2022, 2023])

    df, warns = vl.compute_scores(pd.Timestamp("2025-01-31"), facts_fn=facts, **kw)
    assert "BAD" not in df.index and any(w.startswith("BAD:") for w in warns)


def test_financials_and_cheap_stocks_excluded_from_universe():
    kw = dict(universe_fn=lambda d: {"BANK", "PENNY", "OK"}, facts_fn=lambda t: _facts([2019, 2020, 2021, 2022, 2023]),
              price_fn=lambda t, d: (3.0 if t == "PENNY" else 20.0, pd.Series(dtype=float)),
              sic_fn=lambda t: "6022" if t == "BANK" else "3571")
    df, _ = vl.compute_scores(pd.Timestamp("2025-01-31"), **kw)
    assert df["universe"].to_dict() == {"BANK": False, "OK": True, "PENNY": False}


def test_sp500_rollback_with_changes_table():
    cur = pd.DataFrame({"Symbol": ["AAA", "BRK.B", "NEW"]})
    ch = pd.DataFrame([["October 6, 2026", "NEW", "OLD"]],
                      columns=pd.MultiIndex.from_tuples([("Effective Date", "Effective Date"), ("Added", "Ticker"), ("Removed", "Ticker")]))
    assert vl.sp500_on(date(2026, 9, 30), [cur], [ch]) == {"AAA", "BRK-B", "OLD"}
    assert vl.sp500_on(date(2026, 10, 7), [cur], [ch]) == {"AAA", "BRK-B", "NEW"}


def test_missing_sec_user_agent_skips_book8_with_one_warning(tmp_path, monkeypatch):
    """Actions에 SEC_USER_AGENT Secret이 아직 없을 때: 8번만 건너뛰고 경고 한 줄, 오류(운영 알림) 없음, DB에 아무것도 안 씀."""
    from data import edgar, value_live

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.setattr(edgar, "get_user_agent", lambda: (_ for _ in ()).throw(edgar.EdgarConfigError("없음")))
    monkeypatch.setattr(value_live, "universe_on", lambda d: {"AAA"})
    conn = vbstore.connect(tmp_path / "v.db")
    r = ev.run_value_score(_cfg(), conn, _ctx("2026-10-19"), dry_run=False, history_fn=_hist,
                           fx_range_fn=lambda a, b: {d.date().isoformat(): FX for d in DAYS})
    assert r["errors"] == [] and r["extra"] == []
    assert len([w for w in r["warnings"] if "SEC_USER_AGENT" in w]) == 1
    assert vbstore.load_picks_book(conn, "가치평가 따라하기") is None  # 다음 실행이 시작일부터 따라잡는다


def test_virtual_books_run_keeps_other_books_when_book8_skipped(monkeypatch):
    """8번이 건너뛰어도 run 결과의 오류가 없어 운영 오류 알림이 나가지 않는다(다른 장부·편지는 그대로)."""
    from engine import virtual_books as evb

    sent = []
    monkeypatch.setattr(evb, "run", lambda cfg, **kw: {"errors": [], "warnings": ["가치평가 따라하기: SEC_USER_AGENT가 없어 오늘은 건너뜁니다"],
                                                         "alerts": [], "picks": {"extra": []}})
    out = evb.run_safely(_cfg(), ops_sender=sent.append)
    assert out is not None and sent == []


def test_universe_is_sp500_union_nasdaq100():
    """[사용자 결정 2026-10-18 23:12] 앞으로 기록 대상 = 그날 S&P 500 ∪ 나스닥 100 (S&P 100도 빼지 않음). 금융 제외는 compute_scores가 그대로 한다."""
    cur = pd.DataFrame({"Symbol": [f"S{i}" for i in range(460)] + ["AAPL", "JPM"]})
    ch = pd.DataFrame([["October 6, 2026", "S0", "OLD"]],
                      columns=pd.MultiIndex.from_tuples([("Effective Date", "Effective Date"), ("Added", "Ticker"), ("Removed", "Ticker")]))
    ndx = {f"N{i}" for i in range(99)} | {"AAPL"}
    u = vl.universe_on(date(2026, 9, 30), [cur], ndx_fn=lambda d: ndx, change_tables=[ch])
    assert {"AAPL", "JPM", "OLD", "N0"} <= u and "S0" not in u
    assert len(u) == 460 + 2 + len(ndx) - 1  # S0 되돌림(−1), OLD(+1), AAPL 겹침(−1)
    with pytest.raises(vl.ValueLiveError):
        vl.universe_on(date(2026, 9, 30), [cur], ndx_fn=lambda d: {"AAPL"}, change_tables=[ch])  # 나스닥 100이 비면 멈춤
