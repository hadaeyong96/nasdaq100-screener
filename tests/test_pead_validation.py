"""실적급등 장부 기록 꼬리표·참고 기록 (core/pead_tags, core/pead_validation, engine 연결) — 네트워크 없음."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core import pead_tags as ptag
from core import pead_timing as pt
from core import pead_validation as pv
from data import pead_tags_data as tg
from store import pead_watch as st
from store import virtual_books as vbstore

DAYS = pd.bdate_range("2024-01-01", "2026-12-31")


def _ohlc(seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(DAYS)))
    return pd.DataFrame({"open": c * 0.999, "high": c * 1.01, "low": c * 0.98, "close": c,
                         "volume": rng.uniform(1e6, 2e6, len(DAYS))}, index=DAYS)


# ── 꼬리표 ──────────────────────────────────────────────────────────────────


def test_sma200_and_high52_values():
    px = _ohlc()
    d0 = DAYS[400]
    above, dist = ptag.sma200(px["close"], d0)
    m = px["close"].iloc[201:401].mean()
    assert dist == pytest.approx((px["close"].iloc[400] / m - 1) * 100) and above == int(px["close"].iloc[400] > m)
    assert ptag.high52(px["close"], d0) == pytest.approx((px["close"].iloc[400] / px["close"].iloc[149:401].max() - 1) * 100)
    assert ptag.sma200(px["close"], DAYS[100]) == (None, None) and ptag.high52(px["close"], DAYS[200]) is None


def test_price_tags_no_future_data():
    """d0까지 자른 자료와 전체 자료의 꼬리표가 같다."""
    px = _ohlc(1)
    for i in (260, 400, 600):
        d0 = DAYS[i]
        full, _ = ptag.price_tags(px, d0)
        cut, _ = ptag.price_tags(px.iloc[:i + 1], d0)
        assert full == cut


def test_close_position_and_weekday():
    assert ptag.close_position(110, 100, 107.5) == pytest.approx(0.75)
    assert ptag.close_position(100, 100, 100) is None and ptag.close_position(110, 100, 111) is None
    assert ptag.weekday_kr("2026-11-12") == "목"


def test_count_8k_202_distinct_cik_excludes_amendments():
    hits = [{"form": "8-K", "items": ["2.02", "9.01"], "ciks": ["0000000001"]},
            {"form": "8-K", "items": ["2.02", "9.01"], "ciks": ["0000000001"]},  # 같은 접수의 첨부
            {"form": "8-K", "items": ["7.01"], "ciks": ["0000000002"]},
            {"form": "8-K/A", "items": ["2.02"], "ciks": ["0000000003"]},
            {"form": "8-K", "items": ["2.02"], "ciks": ["0000000004"]}]
    assert ptag.count_8k_202(hits) == 2


def test_q_before_close_rules():
    fl = [{"form": "10-Q", "filing_date": "2026-10-20", "accn": "q-old"},  # 8-K 전 → 직전 분기
          {"form": "10-Q/A", "filing_date": "2026-10-22", "accn": "qa"},
          {"form": "10-Q", "filing_date": "2026-10-22", "accn": "q1"}]
    c = ptag.q_report_candidates(fl, "2026-10-21", "2026-10-22")
    assert [f["accn"] for f in c] == ["q1"]
    assert ptag.q_before_close(c, "2026-10-22", {"q1": pd.Timestamp("2026-10-22 15:59")}) == 1
    assert ptag.q_before_close(c, "2026-10-22", {"q1": pd.Timestamp("2026-10-22 16:05")}) == 0
    assert ptag.q_before_close(c, "2026-10-22", {"q1": None}) is None
    assert ptag.q_before_close([], "2026-10-22", {}) == 0
    c2 = ptag.q_report_candidates([{"form": "10-K", "filing_date": "2026-10-21", "accn": "k"}], "2026-10-21", "2026-10-22")
    assert ptag.q_before_close(c2, "2026-10-22", {}) == 1
    # 반응일 뒤 보고서는 후보가 아니다(미래 정보 안 씀)
    assert ptag.q_report_candidates([{"form": "10-Q", "filing_date": "2026-10-23", "accn": "z"}], "2026-10-21", "2026-10-22") == []


def test_lm_tone_and_html_words():
    w = ptag.html_words("<html><style>p{x:1}</style><p>Strong GROWTH &amp; record loss; may be uncertain 2026</p></html>")
    assert w == ["STRONG", "GROWTH", "RECORD", "LOSS", "MAY", "BE", "UNCERTAIN"]
    lists = {"positive": {"STRONG"}, "negative": {"LOSS"}, "uncertainty": {"MAY", "UNCERTAIN"}}
    t = ptag.lm_tone(w, lists)
    assert t["words"] == 7 and t["pos"] == pytest.approx(100 / 7) and t["unc"] == pytest.approx(200 / 7)
    assert ptag.lm_tone([], lists) is None


def test_lm_word_list_file_matches_source_counts():
    L = tg.load_lm_lists()
    assert (len(L["positive"]), len(L["negative"]), len(L["uncertainty"])) == (347, 2345, 297)
    assert "STRONG" in L["positive"] and "LOSS" in L["negative"] and "UNCERTAIN" in L["uncertainty"]


def test_parse_index_docs_finds_ex991():
    html = """<table class="tableFile"><tr><th>Seq</th></tr>
      <tr><td>1</td><td>8-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/x/a.htm">a.htm</a></td><td>8-K</td><td>1</td></tr>
      <tr><td>2</td><td>EX-99.1</td><td><a href="/Archives/edgar/data/1/x/ex991.htm">ex991.htm</a></td><td>EX-99.1</td><td>2</td></tr>
    </table>"""
    d = tg.parse_index_docs(html)
    assert [x["type"] for x in d] == ["8-K", "EX-99.1"]
    assert d[0]["url"] == "https://www.sec.gov/Archives/edgar/data/1/x/a.htm"
    assert d[1]["url"].endswith("/ex991.htm")


def test_store_tag_columns_match_core_and_migrate(tmp_path):
    assert st.TAG_COLS == ptag.TAG_COLS
    import sqlite3

    p = tmp_path / "old.db"
    c = sqlite3.connect(p)
    c.execute("""CREATE TABLE pead_watch_signals (ticker TEXT, d0 TEXT, accn TEXT, label TEXT, accepted_et TEXT,
                 ret REAL, excess REAL, vol_ratio REAL, signal INTEGER, status TEXT, judged_as_of TEXT, PRIMARY KEY (ticker, d0))""")
    c.execute("INSERT INTO pead_watch_signals VALUES ('X','2026-10-01','a','둘 다',NULL,0.1,0.1,3,1,'신호','2026-10-01')")
    c.commit()
    c.close()
    conn = vbstore.connect(p)
    st.ensure_schema(conn)
    st.put_signals(conn, [{"ticker": "Y", "d0": "2026-10-02", "signal": 1, "tag_weekday": "금", "tag_hi52": -3.0}])
    rows = {r["ticker"]: r for r in st.load_signals(conn)}
    assert rows["X"]["tag_weekday"] is None and rows["Y"]["tag_weekday"] == "금" and rows["Y"]["tag_hi52"] == -3.0
    conn.close()


# ── 포트폴리오 ──────────────────────────────────────────────────────────────


def _tr(t, buy, plan, bpx, spx, q_buy=100.0, status="closed", d0=None):
    return {"ticker": t, "d0": d0 or buy, "status": status, "buy_day": buy, "plan_day": plan, "buy_px": bpx,
            "q_buy": q_buy, "sell_px": spx}


def test_pf_step_buy_tenth_of_cash_then_sell_into_qqq():
    p = pv.PfParams()
    s = pv.pf_init(1000.0, 100.0, p)  # 1억 원 / 1000 × 0.999 = 99,900달러, QQQ 999단위
    assert s["start_usd"] == pytest.approx(99_900) and s["min_usd"] == pytest.approx(1000)
    trades = [_tr("B", "2026-01-05", "2026-02-05", 50.0, 60.0), _tr("A", "2026-01-05", "2026-03-05", 10.0, 9.0)]
    s1, ev, row = pv.pf_step(s, "2026-01-05", trades, 100.0, 100.0, {"A": 10.0, "B": 50.0}, p)
    a, b = ev[0], ev[1]
    assert (a["ticker"], b["ticker"]) == ("A", "B")  # 종목 기호 순
    sh_a = math.floor(99_900 * 0.1 / (10 * 1.0007))
    assert a["shares"] == sh_a
    cash_after_a = 99_900 - sh_a * 10 * 1.0007
    assert b["shares"] == math.floor(cash_after_a * 0.1 / (50 * 1.0007))  # 그때 남은 현금의 1/10
    assert row["value_usd"] == pytest.approx(99_900 - (sh_a * 10 + b["shares"] * 50) * 0.0007)
    s2, ev2, row2 = pv.pf_step(s1, "2026-02-05", trades, 100.0, 125.0, {"A": 10.0, "B": 60.0}, p)
    sell = [e for e in ev2 if e["action"] == "sell"]
    assert [e["ticker"] for e in sell] == ["B"] and sell[0]["amount_usd"] == pytest.approx(b["shares"] * 60 * 0.9993)
    assert "B|2026-01-05" not in s2["pos"] and row2["qqq100_usd"] == pytest.approx(999 * 125)
    # 같은 매수를 두 번 하지 않는다
    s3, ev3, _ = pv.pf_step(s2, "2026-02-06", trades, 125.0, 125.0, {"A": 10.0}, p)
    assert ev3 == []


def test_pf_step_skip_below_minimum_and_records():
    p = pv.PfParams(start_krw=5_000_000, min_buy_krw=1_000_000)  # 예산 50만 원 < 100만 원
    s = pv.pf_init(1000.0, 100.0, p)
    _, ev, row = pv.pf_step(s, "2026-01-05", [_tr("A", "2026-01-05", "2026-03-05", 10.0, 9.0)], 100.0, 100.0, {}, p)
    assert ev[0]["action"] == "skip" and ev[0]["note"] == "최소 금액 미만" and row["n_pos"] == 0
    p2 = pv.PfParams(min_buy_krw=0)
    _, ev2, _ = pv.pf_step(pv.pf_init(1000.0, 100.0, p2), "2026-01-05",
                           [_tr("A", "2026-01-05", "2026-03-05", 50_000.0, 1.0)], 100.0, 100.0, {}, p2)
    assert ev2[0]["note"] == "1주 값보다 예산이 작음"


def test_pf_step_ignores_pending_and_future_buys():
    p = pv.PfParams()
    s = pv.pf_init(1000.0, 100.0, p)
    trades = [_tr("A", "2026-01-06", "2026-03-05", 10.0, 9.0), {**_tr("B", "2026-01-05", "2026-03-05", 10.0, 9.0), "status": "pending", "buy_px": None}]
    _, ev, _ = pv.pf_step(s, "2026-01-05", trades, 100.0, 100.0, {}, p)
    assert ev == []


def test_simulate_matches_step_by_step_and_no_trades_equals_qqq():
    days = DAYS[:300]
    q = _ohlc(3).reindex(days)
    bench = pt.align(q, days)
    rows, ev = pv.simulate([], {}, bench, days, 0, 299, 1300.0)
    assert np.allclose(rows["vs_qqq_pct"], 0.0) and ev.empty
    a = _ohlc(4).reindex(days)
    al = {"A": pt.align(a, days)}
    tr = [{"ticker": "A", "d0": days[9].date().isoformat(), "status": "closed", "buy_day": days[10].date().isoformat(),
           "plan_day": days[100].date().isoformat(), "buy_px": float(a["open"].iloc[10]), "q_buy": float(q["open"].iloc[10]),
           "sell_px": float(a["close"].iloc[100])}]
    rows, ev = pv.simulate(tr, al, bench, days, 0, 299, 1300.0)
    assert list(ev["action"]) == ["buy", "sell"]
    assert rows["n_pos"].iloc[50] == 1 and rows["n_pos"].iloc[150] == 0
    s = pv.pf_summary(rows, 1300.0)
    assert s["vs_qqq_pct"] == pytest.approx(rows["vs_qqq_pct"].iloc[-1])


# ── 다음 발표 · 무작위 보고 ─────────────────────────────────────────────────


def test_next_announcement_and_split():
    assert pv.next_announcement(["2026-01-10", "2026-04-20", "2026-07-20"], "2026-01-12", "2026-05-30") == pd.Timestamp("2026-04-20")
    assert pv.next_announcement(["2026-01-10"], "2026-01-12", "2026-05-30") is None
    days = DAYS[:200]
    a, q = _ohlc(5).reindex(days), _ohlc(6).reindex(days)
    A, Q = pt.align(a, days), pt.align(q, days)
    ib, ie, k = 10, 100, 70
    r = pv.split_at_next(A, Q, ib, ie, float(a["open"].iloc[ib]), float(a["close"].iloc[ie]), k)
    assert r["ex_a"] == pytest.approx(a["close"].iloc[k - 1] / a["open"].iloc[ib] - q["close"].iloc[k - 1] / q["open"].iloc[ib])
    assert r["ex_b"] == pytest.approx(a["close"].iloc[ie] / a["close"].iloc[k - 1] - q["close"].iloc[ie] / q["close"].iloc[k - 1])
    assert r["len_a"] + r["len_b"] == ie - ib + 1
    assert pv.excess_to(A, Q, ib, k, float(a["open"].iloc[ib]), float(q["open"].iloc[ib])) == pytest.approx(
        a["close"].iloc[k] / a["open"].iloc[ib] - q["close"].iloc[k] / q["open"].iloc[ib])


def test_random_summary_and_bootstrap():
    d = np.arange(10_000) / 10_000
    s = pv.random_summary(0.5, d)
    assert s["pct"] == pytest.approx(50.005) and s["lo"] == pytest.approx(0.025, abs=1e-3) and s["hi"] == pytest.approx(0.975, abs=1e-3)
    lo, hi = pv.bootstrap_ci(np.r_[np.zeros(50), np.ones(50)], 2000, 1)
    assert lo < 0.5 < hi and hi - lo < 0.3
    assert pv.bootstrap_ci(np.ones(5), 100, 1) == (1.0, 1.0)


# ── 엔진 연결 ───────────────────────────────────────────────────────────────


def test_engine_writes_tags_portfolio_and_next_announcement(tmp_path, monkeypatch):
    from tests import test_pead_watch_engine as te

    from engine import pead_watch as ew

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    db = tmp_path / "vb.db"
    src = te._src()
    hits = {"2026-10-21": [{"form": "8-K", "items": ["2.02"], "ciks": ["1"]}, {"form": "8-K", "items": ["2.02"], "ciks": ["9"]}],
            "2026-10-20": [{"form": "8-K", "items": ["2.02"], "ciks": ["2"]}]}
    idx = ('<table><tr><td>1</td><td>8-K</td><td><a href="/a.htm">a</a></td><td>8-K</td><td>1</td></tr>'
           '<tr><td>2</td><td>EX-99.1</td><td><a href="/ex.htm">e</a></td><td>EX-99.1</td><td>1</td></tr></table>'
           '<div class="infoHead">Accepted</div>\n<div class="info">2026-10-21 15:00:00</div>')
    src.efts_day = lambda day: hits.get(day, [])
    src.recent_q = lambda cik: [{"form": "10-Q", "filing_date": "2026-10-21", "accn": "q-a"}]
    src.index_html = lambda cik, accn: idx
    src.fetch_text = lambda url: "<p>Record strong growth despite loss</p>"
    src.lm_lists = lambda: {"positive": {"STRONG"}, "negative": {"LOSS"}, "uncertainty": set()}
    r1 = ew.run(te.CFG, te.D0, db_path=db, src=src, today_kst="2026-10-22")
    a = next(j for j in r1["judged"] if j["ticker"] == "A")
    assert a["tag_n202_d0"] == 2 and a["tag_n202_prev"] == 1 and a["tag_weekday"] == "수"
    assert a["tag_q_before_close"] == 1 and a["tag_lm_pos"] == pytest.approx(20.0) and a["tag_lm_words"] == 5
    assert a["tag_close_pos"] is None and "종가 위치" in a["tag_errors"]  # 가짜 일봉에 고가·저가 없음
    conn = vbstore.connect(db)
    sig = {s["ticker"]: s for s in st.load_signals(conn)}
    assert sig["A"]["tag_n202_d0"] == 2 and sig["B"]["tag_weekday"] is None  # 조건 미달은 빈칸
    assert [r["date"] for r in st.load_portfolio(conn)] == ["2026-10-21"]
    conn.close()
    # 매수일 체결 → 포트폴리오 매수, 다음 실적 발표(가짜: 2027-01-20 장전 접수) 반응일 지나면 기록
    filings = {1: [{"form": "8-K", "items": "2.02", "filing_date": "2026-07-22", "accn": "a-1"},
                   {"form": "8-K", "items": "2.02", "filing_date": "2026-10-21", "accn": "a-2"},
                   {"form": "8-K", "items": "2.02", "filing_date": "2027-01-20", "accn": "a-3"}],
               2: [{"form": "8-K", "items": "2.02", "filing_date": "2026-10-20", "accn": "b-1"}]}
    acc = {"a-1": pd.Timestamp("2026-07-22 16:05"), "a-2": pd.Timestamp("2026-10-21 07:30"),
           "a-3": pd.Timestamp("2027-01-20 07:00"), "b-1": pd.Timestamp("2026-10-20 16:10")}
    src.recent_8k = lambda cik: filings[cik]
    src.accepted = lambda cik, accn: acc[accn]
    r2 = ew.run(te.CFG, pd.Timestamp("2027-01-22"), db_path=db, src=src, today_kst="2027-01-23")
    conn = vbstore.connect(db)
    ev = st.load_pf_events(conn)
    assert [(e["action"], e["ticker"], e["date"]) for e in ev] == [("buy", "A", "2026-10-22")]
    na = st.load_next_ann(conn)
    assert [(n["ticker"], n["next_d0"]) for n in na] == [("A", "2027-01-20")]
    px, q = te._px(1, te.D0), te._px(3)
    ib, dn = pd.Timestamp("2026-10-22"), pd.Timestamp("2027-01-20")
    assert na[0]["ex_to_next"] == pytest.approx(px.at[dn, "close"] / px.at[ib, "open"] - q.at[dn, "close"] / q.at[ib, "open"])
    rows = st.load_portfolio(conn)
    assert rows[-1]["date"] == "2027-01-22" and len(rows) == len(te.DAYS[(te.DAYS >= "2026-10-21") & (te.DAYS <= "2027-01-22")])
    conn.close()
    assert r2["status"] == "ok"


def test_is_ex991_number_forms_only():
    assert ptag.is_ex991("EX-99.1") and ptag.is_ex991("ex-99.01")
    assert not ptag.is_ex991("EX-99") and not ptag.is_ex991("EX-99.2") and not ptag.is_ex991("EX-99.10")
