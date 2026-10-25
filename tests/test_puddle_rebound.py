"""웅덩이 반등 확인 가상 장부 (core/puddle_rebound.py, engine/puddle_rebound.py) — 합성 데이터, 네트워크 막음."""

from __future__ import annotations

import copy
import json
import socket

import numpy as np
import pandas as pd
import pytest

from core import puddle_rebound as pr
from core import puddle_v31 as pv
from engine import puddle_rebound as eng
from engine import puddle_v31 as e31


@pytest.fixture(autouse=True)
def _block_socket(monkeypatch):
    def deny(*a, **k):
        raise RuntimeError("테스트 중 네트워크 금지")

    monkeypatch.setattr(socket.socket, "connect", deny)


P = pr.Params()
D0 = pd.Timestamp("2026-11-02")


def _ind(sma60=100.0, sma120=95.0, sma200=90.0, prev=88.0, hi=120.0, above=1.0, rsi=50.0, rsi_prev=50.0, gc3=0.0):
    return {"sma60": sma60, "sma120": sma120, "sma200": sma200, "sma200_prev20": prev, "hi_prev": hi, "above_prev": above,
            "rsi": rsi, "rsi_prev": rsi_prev, "gc3": gc3}


DIP = dict(above=0.0, rsi=35.0, rsi_prev=38.0)  # 대기 중 RSI 40 이하
CONFIRM = dict(above=0.0, rsi=42.0, rsi_prev=38.0, gc3=1.0)  # RSI 40 재돌파 + 최근 3일 골든크로스


def _day(idx=0, closes=None, ind=None, top=None, qqq=400.0, qqq_sma=380.0, **kw):
    closes = closes if closes is not None else {"AAA": 99.0}
    return pv.Day(date=D0 + pd.tseries.offsets.BDay(idx), idx=idx, fx=1400.0, qqq=qqq, qqq_sma200=qqq_sma, qqq_ret20=0.01,
                  closes=closes, ind=ind if ind is not None else {t: _ind() for t in closes},
                  top10=top if top is not None else list(closes), **kw)


def _ledger(**kw):
    return pr.new_ledger("rebound", _day(), P, **kw)


def _no(tk):
    return False


def _buys(L):
    return [(o["tk"], o["tranche"]) for o in L["pending"] if o["type"] == "buy"]


def _open(L):
    """0일째: 웅덩이 열림(60일선 아래 처음, RSI 50) → 대기 시작."""
    ev = pr.step(L, _day(0), P, filing=_no)
    assert ev["opened"] == ["AAA"] and not _buys(L)
    return ev


# ── 지표 ──────────────────────────────────────────────────────────────────────


def test_rebound_indicators_no_future_data_and_gc_window():
    rng = np.random.default_rng(5)
    cal = pd.bdate_range("2025-01-01", periods=300)
    c = pd.DataFrame({"A": 100 + rng.normal(0, 1, 300).cumsum(), "B": 80 + rng.normal(0, 1, 300).cumsum()}, index=cal)
    full = pr.rebound_indicators(c)
    for t in (60, 150, 299):
        part = pr.rebound_indicators(c.iloc[:t + 1])
        for k in full:
            assert np.allclose(part[k].iloc[-1], full[k].iloc[t], equal_nan=True), k
    from core.indicators import macd_lines

    m, s = macd_lines(c["A"])
    cross = (m.shift(1) <= s.shift(1)) & (m > s)
    days = list(np.flatnonzero(cross.values))
    d = [x for x in days if x > 40][0]
    g = full["gc3"]["A"]
    assert g.iloc[d] == g.iloc[d + 1] == g.iloc[d + 2] == 1.0  # 그날 포함 3거래일
    if d + 3 not in days and d + 2 not in days and d + 1 not in days and d + 3 < len(g):
        assert g.iloc[d + 3] == 0.0


def test_rsi_cross_up_rule():
    assert pr.rsi_cross_up(39.9, 40.0) and pr.rsi_cross_up(30, 45)
    assert not pr.rsi_cross_up(40.0, 45) and not pr.rsi_cross_up(35, 39.9) and not pr.rsi_cross_up(None, 45)


# ── 1차: 열림·대기·확인·닫힘 ─────────────────────────────────────────────────


def test_open_wait_then_confirm_buys_first_tranche():
    L = _ledger()
    _open(L)
    pr.step(L, _day(3, ind={"AAA": _ind(**DIP)}), P, filing=_no)
    assert not _buys(L) and L["waits"]["AAA"]["dipped"]
    ev = pr.step(L, _day(5, ind={"AAA": _ind(**CONFIRM)}), P, filing=_no)
    assert _buys(L) == [("AAA", 1)] and ev["buys"][0]["rsi"] == 42.0 and "AAA" not in L["waits"]
    assert L["pending"][0]["usd"] == pytest.approx(0.05 * 1e8 / 1400)


def test_confirmation_needs_dip_cross_and_recent_golden_cross():
    for ind in (dict(above=0.0, rsi=42.0, rsi_prev=41.0, gc3=1.0),   # 재돌파 아님(전날 ≥ 40) — 하지만 그날 dip도 없음
                dict(above=0.0, rsi=42.0, rsi_prev=38.0, gc3=0.0),   # 골든크로스 없음
                dict(above=0.0, rsi=39.0, rsi_prev=38.0, gc3=1.0)):  # 40 위로 못 올라옴
        L = _ledger()
        _open(L)
        if ind["rsi_prev"] < 40 and ind["rsi"] >= 40:
            pr.step(L, _day(2, ind={"AAA": _ind(**DIP)}), P, filing=_no)
        pr.step(L, _day(4, ind={"AAA": _ind(**ind)}), P, filing=_no)
        assert not _buys(L), ind
    L = _ledger()  # 대기 중 40 이하가 한 번도 없으면(전날만 38이고 대기 밖) 안 산다
    pr.step(L, _day(0, ind={"AAA": _ind(rsi=42.0, rsi_prev=38.0, gc3=1.0)}), P, filing=_no)
    assert L["waits"]["AAA"]["dipped"] is False and not _buys(L)


def test_wait_closes_after_20_trading_days():
    L = _ledger()
    _open(L)
    pr.step(L, _day(3, ind={"AAA": _ind(**DIP)}), P, filing=_no)
    pr.step(L, _day(19, ind={"AAA": _ind(above=0.0)}), P, filing=_no)  # 20일째 — 아직 열림
    assert "AAA" in L["waits"]
    pr.step(L, _day(20, ind={"AAA": _ind(**CONFIRM)}), P, filing=_no)  # 21일째 — 닫힘, 매수 없음
    assert "AAA" not in L["waits"] and not _buys(L)


def test_no_open_without_20_days_above_or_unqualified():
    for ind in (_ind(above=0.0), _ind(sma200=100.0), _ind(prev=91.0)):
        L = _ledger()
        ev = pr.step(L, _day(0, ind={"AAA": ind}), P, filing=_no)
        assert not ev["opened"] and not L["waits"]
    ev = pr.step(_ledger(), _day(0, ind={"AAA": _ind(sma200=100.0)}), P, filing=_no)
    assert ev["unqualified"] == ["AAA"]


def test_index_condition_blocks_buy_but_wait_stays():
    L = _ledger()
    _open(L)
    pr.step(L, _day(3, ind={"AAA": _ind(**DIP)}), P, filing=_no)
    pr.step(L, _day(4, ind={"AAA": _ind(**CONFIRM)}, qqq=370.0), P, filing=_no)
    assert not _buys(L) and "AAA" in L["waits"]


# ── 실적 필터 ────────────────────────────────────────────────────────────────


def test_earnings_filter_skip_and_lookup_failure():
    L = _ledger()
    _open(L)
    pr.step(L, _day(3, ind={"AAA": _ind(**DIP)}), P, filing=_no)
    ev = pr.step(L, _day(4, ind={"AAA": _ind(**CONFIRM)}), P, filing=lambda tk: True)
    assert ev["skipped"] == [{"tk": "AAA", "tranche": 1}] and not _buys(L) and "AAA" in L["waits"]
    ev = pr.step(L, _day(5, ind={"AAA": _ind(**CONFIRM)}), P, filing=lambda tk: None)
    assert ev["failed"] == ["AAA"] and not _buys(L)
    pr.step(L, _day(6, ind={"AAA": _ind(**CONFIRM)}), P, filing=_no)  # 대기 기간 안에 다시 확인되면 산다
    assert _buys(L) == [("AAA", 1)]


# ── 2차·한도 ─────────────────────────────────────────────────────────────────


def _held(entry_idx=0):
    L = _ledger()
    _open(L)
    pr.step(L, _day(2, ind={"AAA": _ind(**DIP)}), P, filing=_no)
    pr.step(L, _day(3, ind={"AAA": _ind(**CONFIRM)}), P, filing=_no)
    pr.step(L, _day(4, closes={"AAA": 98.0}, ind={"AAA": _ind(above=0.0)}), P, filing=_no)  # 체결
    assert L["pos"]["AAA"]["tranches"] == 1
    L["pending"] = []
    L["pos"]["AAA"]["entry_idx"] = entry_idx
    return L


def test_second_tranche_opens_below_sma120_and_needs_confirmation():
    L = _held()
    below120 = dict(sma120=99.0, sma200=90.0, above=0.0)
    ev = pr.step(L, _day(10, closes={"AAA": 97.0}, ind={"AAA": _ind(**below120)}), P, filing=_no)
    assert isinstance(L["pos"]["AAA"]["w2"], dict) and ev["waiting"][0]["tranche"] == 2 and not _buys(L)
    pr.step(L, _day(12, closes={"AAA": 96.0}, ind={"AAA": _ind(**below120, rsi=35.0, rsi_prev=37.0)}), P, filing=_no)
    pr.step(L, _day(13, closes={"AAA": 96.5}, ind={"AAA": _ind(**below120, rsi=41.0, rsi_prev=37.0, gc3=1.0)}), P, filing=_no)
    assert _buys(L) == [("AAA", 2)] and L["pos"]["AAA"]["w2"] == "done"


def test_second_tranche_window_expires_once():
    L = _held()
    below120 = dict(sma120=99.0, sma200=90.0, above=0.0)
    pr.step(L, _day(10, closes={"AAA": 97.0}, ind={"AAA": _ind(**below120)}), P, filing=_no)
    pr.step(L, _day(30, closes={"AAA": 97.0}, ind={"AAA": _ind(**below120, rsi=41.0, rsi_prev=35.0, gc3=1.0)}), P, filing=_no)
    assert not _buys(L) and L["pos"]["AAA"]["w2"] == "done"  # 21일째 — 닫힘, 다시 열리지 않음
    pr.step(L, _day(31, closes={"AAA": 97.0}, ind={"AAA": _ind(**below120)}), P, filing=_no)
    assert L["pos"]["AAA"]["w2"] == "done"


def test_cap_10pct_and_max_two_names():
    L = _held()
    L["pos"]["AAA"]["units"] *= 2.2  # 이미 10% 넘음
    below120 = dict(sma120=99.0, sma200=90.0, above=0.0)
    pr.step(L, _day(10, closes={"AAA": 97.0}, ind={"AAA": _ind(**below120, rsi=35.0)}), P, filing=_no)
    pr.step(L, _day(11, closes={"AAA": 97.0}, ind={"AAA": _ind(**below120, rsi=41.0, rsi_prev=35.0, gc3=1.0)}), P, filing=_no)
    assert not _buys(L)
    L = _ledger()
    names = {t: 99.0 for t in ("A", "B", "C")}
    pr.step(L, _day(0, closes=names), P, filing=_no)
    pr.step(L, _day(2, closes=names, ind={t: _ind(**DIP) for t in names}), P, filing=_no)
    pr.step(L, _day(3, closes=names, ind={t: _ind(**CONFIRM) for t in names}), P, filing=_no)
    assert _buys(L) == [("A", 1), ("B", 1)]  # 상위 10 순서로 빈자리 2개


# ── 매도·재매수 금지 ─────────────────────────────────────────────────────────


def test_min_hold_and_three_sell_reasons():
    hi = {"AAA": _ind(hi=100.0, above=0.0)}
    L = _held()
    pr.step(L, _day(59, closes={"AAA": 130.0}, ind=hi), P, filing=_no)
    assert not [o for o in L["pending"] if o["type"] == "sell"]
    for idx, closes, ind, top, reason in ((60, {"AAA": 130.0}, hi, None, "high"),
                                          (60, {"AAA": 99.0}, {"AAA": _ind(above=0.0)}, ["ZZZ"], "rank"),
                                          (250, {"AAA": 99.0}, {"AAA": _ind(above=0.0)}, None, "max_hold")):
        L = _held()
        pr.step(L, _day(idx, closes=closes, ind=ind, top=top), P, filing=_no)
        assert [o["reason"] for o in L["pending"] if o["type"] == "sell"] == [reason]


def test_sell_then_cooldown_20_days():
    L = _held()
    pr.step(L, _day(60, closes={"AAA": 130.0}, ind={"AAA": _ind(hi=100.0, above=0.0)}), P, filing=_no)
    ev = pr.step(L, _day(61, closes={"AAA": 131.0}, ind={"AAA": _ind(hi=131.0, above=0.0)}), P, filing=_no)
    assert "AAA" not in L["pos"] and ev["sells"][0]["reason"] == pv.REASONS["high"] and L["last_sell"]["AAA"] == 61
    for start, allowed in ((63, False), (80, True)):
        L2 = copy.deepcopy(L)
        pr.step(L2, _day(start), P, filing=_no)  # 열림
        pr.step(L2, _day(start + 1, ind={"AAA": _ind(**DIP)}), P, filing=_no)
        pr.step(L2, _day(start + 2, ind={"AAA": _ind(**CONFIRM)}), P, filing=_no)  # start=63 → 65(금지), 80 → 82(허용)
        assert bool(_buys(L2)) is allowed


def test_random_path_buys_only_on_given_days():
    R = pr.new_ledger("r0", _day(), P, random=True)
    pr.step(R, _day(), P, filing=_no, random_n=0)
    assert not R["pending"]
    rng = np.random.default_rng(P.seed)
    pr.step(R, _day(closes={"A": 120.0, "B": 120.0}), P, filing=_no, random_n=1, rng=rng)
    assert len(R["pending"]) == 1 and R["pending"][0]["tk"] in ("A", "B")


def test_judge_needs_v31():
    assert pr.judge(1.1e8, 1.05e8, 1.08e8, 1.09e8, -0.1, -0.1)["pass"]
    j = pr.judge(1.1e8, 1.05e8, None, 1.09e8, -0.1, -0.1)
    assert not j["pass"] and j["v31_missing"]


# ── 엔진 ─────────────────────────────────────────────────────────────────────


def _history_world(n=330, last="2026-11-13"):
    cal = pd.bdate_range(end=last, periods=n)
    t = np.arange(n)
    q = pd.DataFrame({"close": 300 * 1.001 ** t, "volume": 1e7}, index=cal)
    a = 100 * 1.002 ** t
    k = n - 16  # 열엿새 전부터 웅덩이: 이틀 크게 빠지고(60일선 아래 처음·RSI 40 아래) 엿새 옆걸음(MACD 돌아섬) 뒤 하루 1%씩 반등
    a[k:] = a[k - 1] * np.cumprod([0.92, 0.92] + [1.0] * 6 + [1.01] * 8)
    data = {"QQQ": q, "AAA": pd.DataFrame({"close": a, "volume": 5e6}, index=cal),
            "BBB": pd.DataFrame({"close": 50 * 1.001 ** t, "volume": 1e6}, index=cal)}
    return cal, data


def _run(tmp_path, as_of, monkeypatch, actions=True, filings=None, **kw):
    cal, data = _history_world()
    if actions:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS", "x")
    cfg = {"puddle_rebound": {"enabled": True, "start_date": "", "start_fx": ""}}
    return eng.run(cfg, as_of, db_path=tmp_path / "vb.db", history_fn=lambda t, c: data[t], universe_fn=lambda: ["AAA", "BBB"],
                   fx_fn=lambda ds: {str(pd.Timestamp(d).date()): 1400.0 for d in ds},
                   year_days_fn=lambda y: pd.bdate_range(f"{y}-01-01", f"{y}-12-31"),
                   filings_fn=filings or (lambda tk: []), sheet_write=False, p=pr.Params(n_random=5), **kw)


def test_engine_catchup_buys_after_rebound_and_is_idempotent(tmp_path, monkeypatch):
    cal, data = _history_world()
    r1 = _run(tmp_path, str(cal[-17].date()), monkeypatch)
    assert r1["status"] == "ok" and len(r1["rows"]) == 1
    r2 = _run(tmp_path, str(cal[-1].date()), monkeypatch)
    rows = r2["rows"]
    assert len(rows) == 16 and rows[0]["opened"] == ["AAA"]
    sig = [r["date"] for r in rows if r["buys"]]
    fills = [(r["date"], f["tk"], f["tranche"]) for r in rows for f in r["fills"]]
    assert sig and fills and fills[0][1:] == ("AAA", 1)
    assert pd.Timestamp(fills[0][0]) == cal[cal.get_loc(pd.Timestamp(sig[0])) + 1]  # 신호 다음 거래일 종가
    sig_row = next(r for r in rows if r["buys"])
    i = data["AAA"].index.get_loc(pd.Timestamp(sig_row["date"]))
    ind = pr.rebound_indicators(data["AAA"][["close"]].rename(columns={"close": "AAA"}))
    assert ind["rsi_prev"]["AAA"].iloc[i] < 40 <= ind["rsi"]["AAA"].iloc[i] and ind["gc3"]["AAA"].iloc[i] == 1.0
    assert _run(tmp_path, str(cal[-1].date()), monkeypatch)["status"] == "already"


def test_engine_does_not_touch_v31_records(tmp_path, monkeypatch):
    """v3.1 장부를 먼저 돌려 저장한 뒤 반등 확인 장부를 돌려도 v3.1 상태·일별 기록은 그대로(읽기만).
    시작일이 같으면 B는 같은 기준일의 v3.1 평가금액 그대로(2026-10-25 수정: 전에는 1억으로 다시 맞춰 100,000,000이 찍혔다)."""
    cal, data = _history_world()
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS", "x")
    kw = dict(db_path=tmp_path / "vb.db", history_fn=lambda t, c: data[t], universe_fn=lambda: ["AAA", "BBB"],
              fx_fn=lambda ds: {str(pd.Timestamp(d).date()): 1400.0 for d in ds},
              year_days_fn=lambda y: pd.bdate_range(f"{y}-01-01", f"{y}-12-31"), filings_fn=lambda tk: [], sheet_write=False)
    start, end = str(cal[-17].date()), str(cal[-1].date())
    e31.run({"puddle_v31": {"enabled": True}}, start, p=pv.Params(n_random=5), **kw)
    e31.run({"puddle_v31": {"enabled": True}}, end, p=pv.Params(n_random=5), **kw)
    from store import virtual_books as vbstore

    def snapshot():
        conn = vbstore.connect(tmp_path / "vb.db")
        s = (vbstore.get_meta(conn, e31.META_STATE), conn.execute("SELECT * FROM puddle_v31_daily ORDER BY date").fetchall())
        conn.close()
        return s

    before = snapshot()
    eng.run({"puddle_rebound": {"enabled": True}}, start, p=pr.Params(n_random=5), **kw)
    r = eng.run({"puddle_rebound": {"enabled": True}}, end, p=pr.Params(n_random=5), **kw)
    assert snapshot() == before
    v31 = {d: json.loads(j)["v31"] for d, j in before[1]}
    last = r["rows"][-1]
    assert last["v31"] == pytest.approx(v31[end])
    assert r["rows"][0]["v31"] == pytest.approx(v31[str(cal[-16].date())])


def test_engine_filing_failure_blocks_buy_and_marks_check(tmp_path, monkeypatch):
    def boom(tk):
        raise ConnectionError("x")

    cal, _ = _history_world()
    _run(tmp_path, str(cal[-17].date()), monkeypatch, filings=boom)
    r = _run(tmp_path, str(cal[-1].date()), monkeypatch, filings=boom)
    assert not any(row["fills"] for row in r["rows"]) and {t for row in r["rows"] for t in row["failed"]} == {"AAA"}
    assert r["letter"]["check_personal"] == "실적 확인 실패(AAA)" and r["letter"]["check_group"] == "실적 확인 실패(1종목)"


def test_engine_local_run_is_preview_only(tmp_path, monkeypatch):
    cal, _ = _history_world()
    assert _run(tmp_path, str(cal[-1].date()), monkeypatch, actions=False)["status"] == "preview"
    from store import virtual_books as vbstore

    conn = vbstore.connect(tmp_path / "vb.db")
    assert eng.load_state(conn) is None
    conn.close()


# ── 표시: 후보 칸·편지·시트 ──────────────────────────────────────────────────


def _row(**kw):
    r = {"date": "2026-11-04", "value": 1.004e8, "qqq": 1.0e8, "v31": 1.001e8, "random": {"50": 1.0e8, "80": 1.01e8},
         "weights": {"AMD": 0.05}, "qqq_weight": 0.95, "fills": [{"tk": "AMD", "tranche": 1, "usd": 1.0, "date": "2026-11-04"}],
         "sells": [], "skipped": [{"tk": "META", "tranche": 1}], "failed": [],
         "buys": [{"tk": "NVDA", "tranche": 1, "rsi": 41.23}], "waiting": [{"tk": "AAPL", "tranche": 1, "day": 7, "rsi": 36.04}],
         "opened": ["MSFT"], "unqualified": ["TSLA", "INTC"], "realized_year": 0.0}
    r.update(kw)
    return r


def test_candidate_lines_stages_and_group_without_tickers():
    p = eng.candidate_lines(_row(), True)
    assert p == ["반등 확인 — 오늘 가상 매수: NVDA (RSI 41.2, MACD 골든크로스)",
                 "반등 확인 대기: AAPL (웅덩이 7일째, RSI 36.0)",
                 "웅덩이 진입: MSFT (60일선 아래 처음)",
                 "실적 발표 직후라 제외: META",
                 "자격 미달(200일선 아래 등): TSLA, INTC"]
    g = eng.candidate_lines(_row(), False)
    assert g == ["반등 확인 1종목 · 대기 1종목 · 진입 1종목"]
    assert eng.candidate_lines(_row(buys=[], waiting=[], opened=[], skipped=[], unqualified=[]), True) == ["오늘은 웅덩이 후보가 없습니다."]
    out = eng.letter_lines([_row()])
    assert out["personal"] == ["웅덩이 반등 확인: QQQ보다 0.4%p 앞섭니다. AMD 5% 담고 있습니다. AMD를 처음 샀습니다. "
                               "META는 실적 발표 직후라 사지 않았습니다."]
    for t in ("AMD", "NVDA", "AAPL", "MSFT", "META", "TSLA", "INTC"):
        assert all(t not in x for x in out["group"] + out["cand_group"]), t
    assert out["vs_v31_pp"] == pytest.approx(0.3)


def test_letter_section_order_caution_and_monday_line():
    from notify import letter_plain as lp

    rb = eng.letter_lines([_row()])
    base = {"as_of": pd.Timestamp("2026-11-06"), "letter_today_kst": pd.Timestamp("2026-11-09"), "mode": "live",
            "letter_check": {"lag": 0, "notes": []}, "puddle_rebound": rb,
            "puddle_v31": {"personal": ["웅덩이 v3.1: QQQ와 같습니다."], "group": ["웅덩이 v3.1: QQQ와 같습니다."]}}
    d = lp.build(base, {"research_news": {}}, lp.PERSONAL)
    heads = [s["header"] for s in d["sections"]]
    assert heads[-2:] == ["■ 웅덩이 후보 (가상 실험 신호, 실제 매수 권유 아님)", "■ 가상 실험 (실제 돈 아님)"]
    lab = d["sections"][-1]["lines"]
    assert lab[0] == "웅덩이 v3.1: QQQ와 같습니다." and lab[1].startswith("웅덩이 반등 확인: ")
    assert "웅덩이 v3.1보다 0.3%p 앞섭니다." in lab  # 월요일
    t = lp.to_text(d)
    assert t.endswith("※ 웅덩이 후보는 12개월 시험 중인 가상 전략의 신호입니다. 검증 전이라 실제 매수 권유가 아닙니다.")
    tue = lp.build({**base, "letter_today_kst": pd.Timestamp("2026-11-10")}, {"research_news": {}}, lp.PERSONAL)
    assert not any("v3.1보다" in x for x in tue["sections"][-1]["lines"])
    g = lp.to_text(lp.build(base, {"research_news": {}}, lp.GROUP))
    assert "▶ 반등 확인 1종목 · 대기 1종목 · 진입 1종목" in g
    for tk in ("AMD", "NVDA", "AAPL", "MSFT", "META", "TSLA"):
        assert tk not in g, tk


class _WS:
    def __init__(self, title, values=None):
        self.title, self.values, self.row_count, self.calls = title, values or [], 400, []

    def get_all_values(self):
        return self.values

    def batch_update(self, updates, value_input_option=None):
        self.calls.append(updates)

    def resize(self, rows=None, cols=None):
        self.row_count = rows


class _Book:
    def __init__(self):
        self.sheets = {"QQQM 모으기": _WS("QQQM 모으기", [["x"]]), "웅덩이 v3.1 (가상)": _WS("웅덩이 v3.1 (가상)", [["y"]])}

    def worksheet(self, name):
        if name not in self.sheets:
            raise KeyError(name)
        return self.sheets[name]

    def add_worksheet(self, title, rows, cols):
        self.sheets[title] = _WS(title)
        return self.sheets[title]


def test_sheet_writes_only_its_tab_and_upserts_by_date():
    from data import puddle_rebound_sheet as ps

    book = _Book()
    client = type("C", (), {"open_by_key": lambda self, k: book})()
    ps.write_rows(client, "sid", [eng.sheet_row(_row())])
    ws = book.sheets["웅덩이 반등 확인 (가상)"]
    ups = ws.calls[0]
    assert ups[0] == {"range": "A1", "values": [["웅덩이 반등 확인 — 가상 1억, 실제 돈 아님"]]}
    assert ups[1]["values"] == [["날짜", "평가금액(원)", "QQQ 보유(원)", "v3.1(원)", "QQQ 대비(%p)", "v3.1 대비(%p)", "무작위 중앙값 대비(%p)",
                                 "QQQ 비중", "보유 종목(비중)", "오늘 매매", "반등 확인 대기 종목", "올해 실현 이익(원)"]]
    row = ups[2]["values"][0]
    assert ups[2]["range"] == "A3" and row[0] == "2026-11-04" and row[10] == "AAPL 7일째" and len(row) == 12
    assert book.sheets["QQQM 모으기"].calls == [] and book.sheets["웅덩이 v3.1 (가상)"].calls == []
    ws.values = [["t"], ["h"], ["2026-11-04"]]
    ps.write_rows(client, "sid", [eng.sheet_row(_row())])
    assert ws.calls[1][2]["range"] == "A3"
    for other in ("QQQM 모으기", "웅덩이 v3.1 (가상)"):
        with pytest.raises(ps.WrongTabError):
            ps.upsert_rows(book.sheets[other], [["x"]])
