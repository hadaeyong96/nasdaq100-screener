"""실적급등 매도 조건 비교(pead_exits) — 규칙(A 예정일 고정·최소 5일·최대 90일, B 종가 판정·다음 날 시가·90일),
15칸 포트폴리오, 가이던스 꼬리표, 비교 통계, 엔진·시트 연결. 네트워크 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import pead_exits as px
from core import pead_timing as pt
from core import pead_validation as pv
from engine import pead_exits as ex
from engine import pead_watch as ew
from store import pead_exits as xst
from store import pead_watch as st
from store import virtual_books as vbstore

DAYS = pd.bdate_range("2026-01-02", "2027-12-31")
P = px.Params()


def _al(close, open_=None):
    c = np.asarray(close, dtype=float)
    o = c if open_ is None else np.asarray(open_, dtype=float)
    n = len(c)
    return pt.align(pd.DataFrame({"open": o, "close": c}, index=DAYS[:n]), DAYS)


def _flat(n=200, v=100.0):
    return np.full(n, v)


# ── 사전 등록 ───────────────────────────────────────────────────────────────


def test_prereg_hash_and_params():
    pre = ex.verify_prereg()
    p = px.params_from(pre)
    assert (p.min_hold, p.max_hold, p.default_hold, p.a_offset, p.target_mult, p.hit_last, p.slots) == (5, 90, 60, 2, 1.2, 89, 15)
    assert (p.seed, p.iterations, p.fdr, p.alpha) == (20261113, 10000, 0.10, 0.05)


def test_prereg_hash_mismatch_raises(tmp_path):
    import shutil

    for rel in ex.PREREG_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ex.ROOT / rel, tmp_path / rel)
    (tmp_path / ex.PREREG_YAML).write_text("바뀜", encoding="utf-8")
    with pytest.raises(RuntimeError):
        ex.verify_prereg(tmp_path)


# ── 조건 A ───────────────────────────────────────────────────────────────────


def test_pick_next_date_skips_current_report_within_30_days():
    ds = ["2026-07-20", "2026-10-21", "2026-10-22", "2027-01-26", "2027-04-27"]
    assert px.pick_next_date(ds, "2026-10-21") == pd.Timestamp("2027-01-26")
    assert px.pick_next_date(["2026-10-22"], "2026-10-21") is None


def test_estimate_from_gaps_mean_of_last_four():
    rd = ["2025-07-24", "2025-10-23", "2026-01-29", "2026-04-23", "2026-07-23", "2025-04-24"]
    e, n = px.estimate_from_gaps(rd, 4)
    gaps = [91, 98, 84, 91]  # 최근 4개 간격 (2025-07-24 ~ 2026-07-23)
    assert n == 4 and e == pd.Timestamp("2026-07-23") + pd.Timedelta(days=round(np.mean(gaps)))
    assert px.estimate_from_gaps(["2026-07-23"]) == (None, 0)


def test_a_plan_two_days_before_and_min5_max90_default60():
    ib = 100
    e = DAYS[ib + 30]
    assert px.a_plan_index(DAYS, ib, e) == ib + 28
    sat = DAYS[ib + 30] + pd.Timedelta(days=(5 - DAYS[ib + 30].dayofweek) % 7 or 7)  # 다음 토요일 — 휴장일도 같은 식
    assert px.a_plan_index(DAYS, ib, sat) == int(DAYS.searchsorted(sat)) - 2
    assert px.a_plan_index(DAYS, ib, DAYS[ib + 3]) == ib + 5  # 최소 5거래일
    assert px.a_plan_index(DAYS, ib, DAYS[ib - 10]) == ib + 5
    assert px.a_plan_index(DAYS, ib, DAYS[ib + 200]) == ib + 90  # 최대 90거래일
    assert px.a_plan_index(DAYS, ib, None) == ib + 60  # 기본 60일


def _sig(t="X", d0=None):
    return {"ticker": t, "d0": (d0 or DAYS[20]).date().isoformat(), "label": "나스닥100"}


def test_a_sell_day_fixed_even_if_next_date_changes_later():
    a, q = _al(_flat(300)), _al(_flat(300))
    sig = _sig()
    plans = {("X", sig["d0"]): {"next_date": DAYS[51].date().isoformat(), "source": px.SRC_YF}}
    tr, _ = px.step_exit_day([], DAYS[20], [sig], {"X": a}, q, DAYS, "exit_a", plans)
    assert tr[0]["plan_day"] == DAYS[49].date().isoformat() and tr[0]["next_source"] == px.SRC_YF
    later = {("X", sig["d0"]): {"next_date": DAYS[80].date().isoformat(), "source": px.SRC_YF}}  # 매수 뒤 예정일이 바뀜
    for i in range(21, 60):
        tr, _ = px.step_exit_day(tr, DAYS[i], [], {"X": a}, q, DAYS, "exit_a", later)
    assert tr[0]["status"] == "closed" and tr[0]["sell_day"] == DAYS[49].date().isoformat()


def test_engine_keeps_first_next_date_in_db(tmp_path, monkeypatch):
    from tests.test_pead_watch_engine import CFG, D0, _src

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    db = tmp_path / "vb.db"
    s = _src()
    s.earnings_dates = lambda t: ["2026-10-21", "2027-01-20"]
    r1 = ew.run(CFG, D0, db_path=db, src=s, today_kst="2026-10-22")
    a1 = r1["exits"]["trades"]["exit_a"][0]
    assert r1["exits"]["start"] == "2026-10-21" and a1["next_source"] == px.SRC_YF and a1["next_date"] == "2027-01-20"
    s.earnings_dates = lambda t: ["2026-12-01"]  # 다음 실행에서 예정일이 바뀌어도
    r2 = ew.run(CFG, pd.Timestamp("2026-10-28"), db_path=db, src=s, today_kst="2026-10-29")
    a2 = r2["exits"]["trades"]["exit_a"][0]
    assert a2["plan_day"] == a1["plan_day"] and a2["next_date"] == "2027-01-20"
    conn = vbstore.connect(db)
    pl = xst.load_plans(conn)[("A", "2026-10-21")]
    assert pl["next_date"] == "2027-01-20" and pl["source"] == px.SRC_YF and pl["fetched_as_of"] == "2026-10-21"
    assert st.get_meta(conn, xst.META_START) == "2026-10-21"
    conn.close()


def test_engine_fallback_gap_then_default(tmp_path):
    from tests.test_pead_watch_engine import CFG, D0, _src

    s = _src()
    r = ew.run(CFG, D0, db_path=tmp_path / "a.db", src=s, dry_run=True, today_kst="2026-10-22")
    pl = r["exits"]["plans"][("A", "2026-10-21")]
    assert pl["source"].startswith(px.SRC_GAP) and pl["n_gaps"] == 1  # earnings_dates 없음 → 8-K 간격(07-23 → 10-21)
    assert pl["next_date"] == (pd.Timestamp("2026-10-21") + pd.Timedelta(days=90)).date().isoformat()
    s.recent_8k = lambda cik: [{"form": "8-K", "items": "2.02", "filing_date": "2026-10-21", "accn": "a-2"}]
    r = ew.run(CFG, D0, db_path=tmp_path / "b.db", src=s, dry_run=True, today_kst="2026-10-22")
    pl = r["exits"]["plans"][("A", "2026-10-21")]
    t = r["exits"]["trades"]["exit_a"][0]
    assert pl["source"] == px.SRC_DEFAULT and pl["next_date"] is None
    from tests.test_pead_watch_engine import DAYS as EDAYS

    ib = EDAYS.get_loc(pd.Timestamp(t["buy_day"]))
    assert t["plan_day"] == EDAYS[ib + 60].date().isoformat()


# ── 조건 B ───────────────────────────────────────────────────────────────────


def test_b_uses_close_only_and_sells_next_open():
    c = _flat(300)
    o = _flat(300)
    hi_only = c.copy()
    a = pt.align(pd.DataFrame({"open": o, "close": c, "high": hi_only * 1.5}, index=DAYS[:300]), DAYS)  # 고가 칸은 아예 안 읽는다
    q = _al(_flat(300, 50.0), np.full(300, 49.0))
    ib = 21
    assert px.b_exit(a, q, ib, ib + 30, 100.0) is None
    c2 = c.copy()
    c2[ib + 10] = 119.99  # 목표 바로 밑
    c2[ib + 12] = 120.0  # 종가 = 목표 → 다음 날 시가
    o2 = o.copy()
    o2[ib + 13] = 118.0
    a2 = _al(c2, o2)
    assert px.b_exit(a2, q, ib, ib + 12, 100.0) is None  # 다음 날 시가가 아직 없음
    x = px.b_exit(a2, q, ib, ib + 13, 100.0)
    assert x["sell_idx"] == ib + 13 and x["sell_px"] == 118.0 and x["sell_at"] == "open" and x["kind"] == "target"
    assert x["hit_idx"] == ib + 12 and x["q_sell"] == 49.0  # QQQ도 그날 시가


def test_b_sells_at_day90_close_when_not_reached_and_counts_day89_hit():
    q = _al(_flat(300, 50.0))
    ib = 21
    c = _flat(300)
    c[ib + 90] = 105.0
    a = _al(c)
    assert px.b_exit(a, q, ib, ib + 89, 100.0) is None
    x = px.b_exit(a, q, ib, ib + 90, 100.0)
    assert x["sell_idx"] == ib + 90 and x["sell_px"] == 105.0 and x["kind"] == "max" and x["sell_at"] == "close"
    c2 = _flat(300)
    c2[ib + 89] = 125.0  # 89거래일째 종가 도달 → 90거래일째 시가
    x2 = px.b_exit(_al(c2), q, ib, ib + 90, 100.0)
    assert x2["kind"] == "target" and x2["sell_idx"] == ib + 90 and x2["sell_at"] == "open"
    c3 = _flat(300)
    c3[ib + 90] = 130.0  # 90거래일째 종가는 목표 판정에 안 씀 → 그날 종가 매도
    x3 = px.b_exit(_al(c3), q, ib, ib + 90, 100.0)
    assert x3["kind"] == "max" and x3["sell_px"] == 130.0 and x3["hit_idx"] is None


def test_b_ledger_step_records_target_hit_and_frees_ticker():
    c = _flat(300)
    c[30] = 125.0
    a, q = _al(c), _al(_flat(300, 50.0))
    sig = _sig(d0=DAYS[20])
    tr, _ = px.step_exit_day([], DAYS[20], [sig], {"X": a}, q, DAYS, "exit_b")
    assert tr[0]["status"] == "pending" and tr[0]["plan_day"] == DAYS[111].date().isoformat()
    for i in range(21, 32):
        tr, _ = px.step_exit_day(tr, DAYS[i], [], {"X": a}, q, DAYS, "exit_b")
    t = tr[0]
    assert t["target_px"] == pytest.approx(120.0) and t["hit_day"] == DAYS[30].date().isoformat() and t["hit_days"] == 9
    assert t["status"] == "closed" and t["sell_day"] == DAYS[31].date().isoformat() and t["exit_kind"] == "target"
    tr, _ = px.step_exit_day(tr, DAYS[40], [_sig(d0=DAYS[40])], {"X": a}, q, DAYS, "exit_b")
    assert tr[-1]["status"] == "pending"  # 일찍 팔았으니 같은 종목 새 신호를 받는다


# ── 15칸 포트폴리오 ─────────────────────────────────────────────────────────


def _closed(t, buy, sell, bpx=10.0, spx=12.0, at="close"):
    return {"ticker": t, "d0": buy, "status": "closed", "buy_day": buy, "plan_day": sell, "sell_day": sell, "buy_px": bpx,
            "q_buy": 100.0, "sell_px": spx, "sell_at": at}


def test_slots_equal_split_full_skip_and_open_sell_first():
    p = pv.PfParams(start_krw=1_500_000_000, min_buy_krw=0)
    s0 = pv.pf_init(1000.0, 100.0, p)
    trades = [_closed(f"T{i:02d}", "2026-11-16", "2026-12-31") for i in range(16)]
    s1, ev, row = px.pf_step_slots(s0, "2026-11-16", trades, 100.0, 100.0, {}, p, slots=15)
    buys = [e for e in ev if e["action"] == "buy"]
    assert len(buys) == 15 and row["n_pos"] == 15
    assert ev[-1]["action"] == "skip" and ev[-1]["note"] == "15칸 다 참" and ev[-1]["ticker"] == "T15"
    first = buys[0]["amount_usd"]
    assert first == pytest.approx(s0["start_usd"] / 15, rel=1e-3)  # 처음 칸 = 1/15
    # 시가 매도(B)가 먼저 칸을 비우면 같은 날 시가 매수가 들어간다
    t_open = _closed("T00", "2026-11-16", "2026-11-17", at="open")
    new = _closed("NEW", "2026-11-17", "2026-12-31")
    s2, ev2, _ = px.pf_step_slots(s1, "2026-11-17", [t_open] + trades[1:15] + [new], 100.0, 101.0, {}, p, slots=15)
    assert [e["action"] for e in ev2][:2] == ["sell", "buy"] and ev2[1]["ticker"] == "NEW"


def test_v1_slot_book_excludes_pre_start_signal(tmp_path):
    from tests.test_pead_watch_engine import CFG, D0, _src

    r = ew.run(CFG, pd.Timestamp("2026-10-28"), db_path=tmp_path / "a.db", src=_src(), dry_run=True,
               start="2026-10-01", exits_start="2026-10-22", today_kst="2026-10-29")
    assert r["exits"]["start"] == "2026-10-22"
    assert r["exits"]["trades"]["exit_a"] == [] and r["exits"]["trades"]["exit_b"] == []  # 10/21 신호는 시작 전
    assert any(j["ticker"] == "A" and j["signal"] for j in r["judged"])  # V1에는 신호·매매가 있다
    assert all(e["ticker"] != "A" for e in r["exits"]["pf"]["main"]["events"])  # 15칸 V1에도 안 들어감(같은 신호 범위)
    assert r["exits"]["pf"]["main"]["rows"][0]["date"] == "2026-10-22"


# ── 꼬리표 ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text,want", [
    ("<p>The Company raises full-year 2026 revenue guidance to $5.0 billion.</p>", 1),
    ("We are raising our fiscal 2027 outlook for adjusted EPS.", 1),
    ("Increased full year revenue expectations to $10B.", 1),
    ("Revenue increased 12% year over year. Full-year guidance reaffirmed.", 0),
    ("The company did not raise its guidance.", 0),
    ("Full-year revenue increased 12% to $3.1 billion.", 0),
    ("Net income rose 5%.", 0),
])
def test_guidance_raise_rules(text, want):
    g, s = px.guidance_raise(text)
    assert g == want and ((s is not None) == (want == 1))


def test_guidance_raise_empty_is_blank():
    assert px.guidance_raise("<html><script>x</script></html>") == (None, None)


def test_prev_below_sma200_uses_previous_day_only():
    idx = pd.bdate_range("2025-01-01", periods=260)
    c = pd.Series(np.linspace(100, 50, 260), index=idx)  # 내리막 → 전날 종가 < 200일 평균
    d0 = idx[-1]
    assert px.prev_below_sma200(c, d0) == 1
    c2 = c.copy()
    c2.iloc[-1] = 1e6  # 반응일 종가가 튀어도 결과 같음
    assert px.prev_below_sma200(c2, d0) == 1
    assert px.prev_below_sma200(c.iloc[:150], idx[149]) is None


# ── 비교 통계 ───────────────────────────────────────────────────────────────


def test_clusters_link_same_ticker_or_day():
    g = px.clusters(["A", "B", "A", "C", "D"], ["d1", "d2", "d3", "d3", "d4"])
    assert g[0] == g[2] == g[3] and g[1] != g[0] and g[4] not in (g[0], g[1])


def test_holm_and_bh():
    assert px.holm([0.01, 0.04]) == pytest.approx([0.02, 0.04])
    assert px.holm([0.03, 0.02]) == pytest.approx([0.04, 0.04])
    assert px.bh_reject([0.001, 0.02, 0.03, 0.5], 0.10) == [True, True, True, False]
    assert px.bh_reject([0.2, 0.3], 0.10) == [False, False]


def test_sign_flip_and_perm_deterministic():
    rng = np.random.default_rng(0)
    d = rng.normal(0.02, 0.01, 40)
    g = np.arange(40)
    p1 = px.sign_flip_p(d, g, 2000, 20261113)
    assert p1 == px.sign_flip_p(d, g, 2000, 20261113) and p1 < 0.01
    assert px.sign_flip_p(d - d.mean(), g, 2000, 1) > 0.5
    x = np.r_[np.full(10, 0.05), np.zeros(30)]
    mask = np.r_[np.ones(10, bool), np.zeros(30, bool)]
    assert px.perm_p(x, mask, 2000, 1) < 0.01


def test_metrics_and_bins():
    x = np.array([0.1, -0.05, 0.02, 0.5, -0.2, 0.0, 0.03, 0.01, -0.01, 0.04])
    m = px.metrics(x, iters=500)
    assert m["n"] == 10 and m["max_loss"] == pytest.approx(-20.0) and m["mean_ex_top10"] == pytest.approx(np.sort(x)[:9].mean() * 100)
    assert m["win"] == pytest.approx(60.0) and m["ci_lo"] <= m["mean"] <= m["ci_hi"]
    assert px.in_bin(-10, "x<=-10") and not px.in_bin(-10, "-10<x<=0") and px.in_bin(0, "-10<x<=0")
    assert px.in_bin(3, "2<=x<=3") and px.in_bin(4, "x>=4") and px.in_bin("월", "월") and px.in_bin(1, "1")
    assert px.monotone([1, 2, 2, 5]) and px.monotone([3, 1, 0]) and not px.monotone([1, 3, 2])


# ── 엔진·시트 ───────────────────────────────────────────────────────────────


def test_exits_failure_keeps_v1_ledger(tmp_path, monkeypatch):
    from tests.test_pead_watch_engine import CFG, D0, _src

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(ex, "advance", boom)
    db = tmp_path / "vb.db"
    r = ew.run(CFG, D0, db_path=db, src=_src(), today_kst="2026-10-22")
    assert r["status"] == "ok" and r["exits"] is None
    assert any("매도 조건 A·B 기록 실패" in w for w in r["warnings"])
    conn = vbstore.connect(db)
    assert st.load_trades(conn) and st.get_meta(conn, xst.META_START) is None
    conn.close()


def test_exits_save_failure_rolls_back_only_exits(tmp_path, monkeypatch):
    from tests.test_pead_watch_engine import CFG, D0, _src

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    real = xst.put_plans

    def bad(conn, rows):
        real(conn, rows)
        raise RuntimeError("저장 실패")

    monkeypatch.setattr(xst, "put_plans", bad)
    db = tmp_path / "vb.db"
    r = ew.run(CFG, D0, db_path=db, src=_src(), today_kst="2026-10-22")
    assert any("매도 조건 A·B 저장 실패" in w for w in r["warnings"])
    conn = vbstore.connect(db)
    assert st.load_trades(conn) and st.get_meta(conn, st.META_LAST) == "2026-10-21"
    assert xst.load_trades(conn, "exit_a") == [] and xst.load_plans(conn) == {} and st.get_meta(conn, xst.META_START) is None
    conn.close()


def test_letter_unchanged_by_exits(tmp_path):
    from tests.test_pead_watch_engine import CFG, D0, _src

    r = ew.run(CFG, D0, db_path=tmp_path / "a.db", src=_src(), dry_run=True, today_kst="2026-10-22")
    assert "exits" not in r["letter"] and r["exits"]["trades"]["exit_a"]
    assert set(r["letter"]["scorecards"]) == {"main", "compare"}


def test_sheet_rows_have_exit_and_slot_cells(tmp_path, monkeypatch):
    from core import pead_sheet as ps
    from tests.test_pead_watch_engine import CFG, D0, _src

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    db = tmp_path / "vb.db"
    s = _src()
    s.earnings_dates = lambda t: ["2027-01-20"]
    ew.run(CFG, D0, db_path=db, src=s, today_kst="2026-10-22")
    r = ew.run(CFG, pd.Timestamp("2026-10-28"), db_path=db, src=s, today_kst="2026-10-29")
    conn = vbstore.connect(db)
    from tests.test_pead_watch_engine import DAYS as EDAYS

    raw = {t: s.prices(t).loc[:"2026-10-28"] for t in ("A", "QQQ")}
    al = {t: pt.align(d, EDAYS) for t, d in raw.items()}
    daily, trades = ew.sheet_rows(conn, al, al["QQQ"], EDAYS, pd.Timestamp("2026-10-28"), {})
    conn.close()
    assert len(trades[0]) == len(ps.TRADE_HEADER) and len(daily[0]) == len(ps.DAILY_HEADER)
    row = trades[0]
    h = ps.TRADE_HEADER
    assert row[h.index("A 매도 예정일")].endswith("(yfinance)") and row[h.index("A 결과")].startswith("보유 중")
    assert row[h.index("B 목표가($)")] == round(r["exits"]["trades"]["exit_b"][0]["buy_px"] * 1.2, 2)
    assert row[h.index("B 목표 도달")] in ("아직", "아니오") or row[h.index("B 목표 도달")].startswith("예")
    dh = ps.DAILY_HEADER
    assert daily[-1][0] == "2026-10-28" and all(daily[-1][dh.index(c)] != "" for c in dh[9:])


def test_report_pairs_holm_and_tags():
    rng = np.random.default_rng(3)

    def tr(i, ret, hit=None):
        d0 = (DAYS[10] + pd.Timedelta(days=i)).date().isoformat()
        return {"ticker": f"T{i}", "d0": d0, "status": "closed", "buy_px": 100.0, "sell_px": 100 * (1 + ret), "q_buy": 100.0,
                "q_sell": 100.0, "fx_buy": 1400.0, "fx_sell": 1400.0, "sell_day": "2027-03-01", "hit_days": hit}

    n = 60
    rets = rng.normal(0.0, 0.05, n)
    v1 = [tr(i, r) for i, r in enumerate(rets)]
    a = [tr(i, r + 0.03) for i, r in enumerate(rets)]  # A가 매번 3% 낫다
    b = [tr(i, r, hit=5 if i % 2 else None) for i, r in enumerate(rets)]  # B는 같음
    sigs = [{"ticker": t["ticker"], "d0": t["d0"], "tag_weekday": "월", "tag_sma200_dist": float(i % 20 - 10),
             "tag_lm_pos": 1.0, "tag_lm_neg": float(i % 3)} for i, t in enumerate(v1)]
    rep = px.report({"main": v1, "exit_a": a, "exit_b": b}, sigs, {}, "2000-01-01", px.Params(iterations=500))
    assert rep["pairs"]["exit_a"]["verdict"] == "낫다" and rep["pairs"]["exit_b"]["verdict"] == "차이 없음"
    assert rep["pairs"]["exit_a"]["p_holm"] >= rep["pairs"]["exit_a"]["p"]
    assert rep["books"]["exit_b"]["hit_rate"] == pytest.approx(50.0) and rep["milestone"] == "첫 성적"
    tags = {r["tag"]: r for r in rep["analysis"]["tags"]}
    assert tags["tag_weekday"]["n"][0] == n and tags["tag_weekday"]["tests"][1]["note"] == "표본 부족"
    assert tags["tag_sma200_dist"]["monotone"] in (True, False) and rep["books"]["main"]["pre_tax"]["n"] == n
    assert "after_tax" in rep["books"]["main"] and rep["analysis"]["sma200_prev"]["n_below"] == 0


def test_milestone_report_once(tmp_path, monkeypatch):
    conn = vbstore.connect(tmp_path / "vb.db")
    st.ensure_schema(conn)
    xst.ensure_schema(conn)
    pre = ex.verify_prereg()

    def closed(i):
        return {"ticker": f"T{i}", "d0": f"2027-01-{(i % 28) + 1:02d}", "status": "closed", "buy_px": 10.0, "sell_px": 11.0,
                "q_buy": 10.0, "q_sell": 10.0, "fx_buy": 1400.0, "fx_sell": 1400.0, "sell_day": "2027-06-01"}

    tr = {r: [closed(i) for i in range(49)] for r in ex.RULES}
    p = px.Params(iterations=200)
    assert ex.milestone_reports(conn, pre, p, "2027-01-01", tr["exit_a"], tr, [], {}, "2027-06-01") == []
    tr = {r: [closed(i) for i in range(50)] for r in ex.RULES}
    reps = ex.milestone_reports(conn, pre, p, "2027-01-01", tr["exit_a"], tr, [], {}, "2027-06-01")
    assert [r["milestone"] for r in reps] == ["first"]
    xst.put_report(conn, "first", "2027-06-01", reps[0]["counts"], reps[0]["report"])
    assert ex.milestone_reports(conn, pre, p, "2027-01-01", tr["exit_a"], tr, [], {}, "2027-06-02") == []  # 한 번만
    assert xst.load_reports(conn)["first"]["counts"]["exit_a"] == 50
    conn.close()


def test_sheet_tab_widens_columns_for_new_header():
    from core import pead_sheet as ps
    from data import pead_sheet
    from tests.test_pead_sheet import FakeSpreadsheet

    sp = FakeSpreadsheet(["매매이력"])
    ws = sp.tabs["매매이력"]
    ws.col_count = 19  # 예전 머리글 칸 수
    ws.resize = lambda rows=None, cols=None: setattr(ws, "col_count", cols or ws.col_count) or setattr(ws, "row_count", rows or ws.row_count)
    pead_sheet.write_tab(sp, "매매이력", [])
    assert ws.col_count == len(ps.TRADE_HEADER) == 25 and ws.grid[0] == ps.TRADE_HEADER
