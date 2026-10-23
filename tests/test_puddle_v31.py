"""상승장 대형주 웅덩이 v3.1 가상 장부 (core/puddle_v31.py, engine/puddle_v31.py) — 합성 데이터, 네트워크 막음."""

from __future__ import annotations

import copy
import socket

import numpy as np
import pandas as pd
import pytest

from core import puddle_v31 as pv
from engine import puddle_v31 as eng


@pytest.fixture(autouse=True)
def _block_socket(monkeypatch):
    def deny(*a, **k):
        raise RuntimeError("테스트 중 네트워크 금지")

    monkeypatch.setattr(socket.socket, "connect", deny)


P = pv.Params()
D0 = pd.Timestamp("2026-11-02")


def _ind(sma60=100.0, sma120=95.0, sma200=90.0, prev=88.0, hi=120.0, above=1.0):
    return {"sma60": sma60, "sma120": sma120, "sma200": sma200, "sma200_prev20": prev, "hi_prev": hi, "above_prev": above}


def _day(idx=0, closes=None, ind=None, top=None, qqq=400.0, qqq_sma=380.0, date=None, **kw):
    closes = closes if closes is not None else {"AAA": 99.0}
    return pv.Day(date=date or D0 + pd.tseries.offsets.BDay(idx), idx=idx, fx=1400.0, qqq=qqq, qqq_sma200=qqq_sma,
                  qqq_ret20=0.01, closes=closes, ind=ind if ind is not None else {t: _ind() for t in closes},
                  top10=top if top is not None else list(closes), **kw)


def _ledger(filter_=True, **kw):
    return pv.new_ledger("v31", _day(), P, earnings_filter=filter_, **kw)


def _no_filing(tk):
    return False


# ── 지표·필터 판정 ────────────────────────────────────────────────────────────


def test_indicators_no_future_data():
    rng = np.random.default_rng(3)
    cal = pd.bdate_range("2025-01-01", periods=320)
    c = pd.DataFrame({"A": 100 + rng.normal(0, 1, 320).cumsum(), "B": 80 + rng.normal(0, 1, 320).cumsum()}, index=cal)
    v = c * 0 + 1e6
    full = pv.indicators(c, v)
    for t in (230, 280, 319):
        part = pv.indicators(c.iloc[:t + 1], v.iloc[:t + 1])
        for k in full:
            assert np.allclose(part[k].iloc[-1], full[k].iloc[t], equal_nan=True), k


def test_above_prev_means_all_prior_20_days_above_sma60():
    cal = pd.bdate_range("2025-01-01", periods=120)
    x = pd.Series(np.linspace(100, 160, 120), index=cal)
    x.iloc[-5] = 50  # 5일 전 한 번 60일선 아래
    ind = pv.indicators(x.to_frame("A"), x.to_frame("A") * 0 + 1)
    assert ind["above_prev"]["A"].iloc[-6] == 1.0 and ind["above_prev"]["A"].iloc[-1] == 0.0


def test_has_earnings_filing_forms_and_window():
    f = [{"form": "8-K", "filingDate": "2026-10-28", "items": "5.02,9.01"},
         {"form": "10-Q/A", "filingDate": "2026-10-29", "items": ""}]
    assert not pv.has_earnings_filing(f, "2026-10-20", "2026-11-02")
    for form, items in (("8-K", "2.02,9.01"), ("10-Q", ""), ("10-K", "")):
        g = f + [{"form": form, "filingDate": "2026-10-20", "items": items}]
        assert pv.has_earnings_filing(g, "2026-10-20", "2026-11-02")  # 창 첫날 포함
        assert not pv.has_earnings_filing(g, "2026-10-21", "2026-11-02")
    assert not pv.has_earnings_filing([{"form": "10-Q", "filingDate": "2026-11-03"}], "2026-10-20", "2026-11-02")  # 신호 뒤 접수는 안 봄


def test_top10_merges_share_classes():
    dv = {"GOOG": 9.0, "GOOGL": 8.0, "AAPL": 10.0, "X": float("nan")}
    assert pv.top_by_dollar_volume(dv, list(dv), 10) == ["AAPL", "GOOG"]


# ── 매수 ─────────────────────────────────────────────────────────────────────


def test_first_tranche_on_first_cross_below_sma60():
    L = _ledger()
    pv.step(L, _day(), P, filing=_no_filing)
    o = L["pending"][0]
    assert (o["tk"], o["tranche"]) == ("AAA", 1) and o["usd"] == pytest.approx(0.05 * 1e8 / 1400)
    L2 = _ledger()
    pv.step(L2, _day(ind={"AAA": _ind(above=0.0)}), P, filing=_no_filing)  # 직전 20일 중 아래였던 날 있음
    assert not L2["pending"]
    L3 = _ledger()
    pv.step(L3, _day(closes={"AAA": 101.0}), P, filing=_no_filing)  # 60일선 위
    assert not L3["pending"]


def test_qualification_and_index_condition():
    for ind in (_ind(sma200=100.0), _ind(prev=91.0)):  # 종가 < 200일선 / 200일선이 20일 전보다 낮음
        L = _ledger()
        pv.step(L, _day(ind={"AAA": ind}), P, filing=_no_filing)
        assert not L["pending"]
    L = _ledger()
    pv.step(L, _day(qqq=370.0), P, filing=_no_filing)  # QQQ 200일선 아래
    assert not L["pending"]


def test_execution_next_day_close_with_costs_and_second_tranche_cap():
    L = _ledger()
    pv.step(L, _day(0), P, filing=_no_filing)
    pv.step(L, _day(1, closes={"AAA": 98.0}, ind={"AAA": _ind(above=0.0)}), P, filing=_no_filing)
    pos = L["pos"]["AAA"]
    usd = 0.05 * 1e8 / 1400
    assert pos["units"] == pytest.approx(usd * 0.999 / 1.001 / 98.0)  # QQQ 팔 때·종목 살 때 각각 0.1%
    assert pos["entry_idx"] == 1 and pos["tranches"] == 1
    # 2차: 120일선 아래·200일선 위
    pv.step(L, _day(2, closes={"AAA": 94.0}, ind={"AAA": _ind(above=0.0)}), P, filing=_no_filing)
    o = [o for o in L["pending"] if o["type"] == "buy"]
    assert o and o[0]["tranche"] == 2
    total = pv.nav_usd(L, _day(2, closes={"AAA": 94.0}))
    assert o[0]["usd"] == pytest.approx(min(0.05 * total, 0.10 * total - pos["units"] * 94.0))
    # 종목이 이미 10% 넘으면 2차 없음
    L2 = copy.deepcopy(L)
    L2["pending"] = []
    L2["pos"]["AAA"]["units"] *= 3
    pv.step(L2, _day(2, closes={"AAA": 94.0}, ind={"AAA": _ind(above=0.0)}), P, filing=_no_filing)
    assert not [o for o in L2["pending"] if o["type"] == "buy"]


def test_max_two_names():
    closes = {"A": 99.0, "B": 99.0, "C": 99.0}
    L = _ledger()
    pv.step(L, _day(closes=closes), P, filing=_no_filing)
    assert [o["tk"] for o in L["pending"]] == ["A", "B"]


def test_earnings_filter_skip_and_failure():
    L = _ledger()
    ev = pv.step(L, _day(), P, filing=lambda tk: True)
    assert not L["pending"] and ev["skipped"] == [{"tk": "AAA", "tranche": 1}]
    L = _ledger()
    ev = pv.step(L, _day(), P, filing=lambda tk: None)
    assert not L["pending"] and ev["failed"] == ["AAA"]
    L3 = _ledger(filter_=False)  # v3는 필터 없음 — filing을 부르지도 않는다
    pv.step(L3, _day(), P, filing=lambda tk: 1 / 0)
    assert L3["pending"]


# ── 매도·재매수 금지 ─────────────────────────────────────────────────────────


def _held(entry_idx=0):
    L = _ledger()
    pv.step(L, _day(0), P, filing=_no_filing)
    pv.step(L, _day(1, closes={"AAA": 98.0}, ind={"AAA": _ind(above=0.0)}), P, filing=_no_filing)
    L["pending"] = []
    L["pos"]["AAA"]["entry_idx"] = entry_idx
    return L


def test_min_hold_60_then_high_or_rank_and_max_hold_250():
    hi_ind = {"AAA": _ind(hi=100.0, above=0.0)}
    L = _held()
    pv.step(L, _day(59, closes={"AAA": 130.0}, ind=hi_ind), P, filing=_no_filing)
    assert not [o for o in L["pending"] if o["type"] == "sell"]  # 60일 전엔 최고가 돌파도 안 팜
    L = _held()
    pv.step(L, _day(60, closes={"AAA": 130.0}, ind=hi_ind), P, filing=_no_filing)
    assert [o["reason"] for o in L["pending"] if o["type"] == "sell"] == ["high"]
    L = _held()
    pv.step(L, _day(60, closes={"AAA": 99.0}, ind={"AAA": _ind(above=0.0)}, top=["ZZZ"]), P, filing=_no_filing)
    assert [o["reason"] for o in L["pending"] if o["type"] == "sell"] == ["rank"]
    L = _held()
    pv.step(L, _day(250, closes={"AAA": 99.0}, ind={"AAA": _ind(above=0.0)}), P, filing=_no_filing)
    assert [o["reason"] for o in L["pending"] if o["type"] == "sell"] == ["max_hold"]
    # 가격 손절 없음: 60일 전 크게 떨어져도 안 팜
    L = _held()
    pv.step(L, _day(30, closes={"AAA": 50.0}, ind={"AAA": _ind(above=0.0)}, top=["ZZZ"]), P, filing=_no_filing)
    assert not [o for o in L["pending"] if o["type"] == "sell"]


def test_sell_executes_to_qqq_and_cooldown_20_days():
    L = _held()
    pv.step(L, _day(60, closes={"AAA": 130.0}, ind={"AAA": _ind(hi=100.0, above=0.0)}), P, filing=_no_filing)
    ev = pv.step(L, _day(61, closes={"AAA": 131.0}, ind={"AAA": _ind(hi=131.0, above=0.0)}), P, filing=_no_filing)
    assert "AAA" not in L["pos"] and ev["sells"][0]["reason"] == pv.REASONS["high"]
    assert L["last_sell"]["AAA"] == 61
    for idx, allowed in ((81, False), (82, True)):  # 체결일 + 20거래일까지 금지
        L2 = copy.deepcopy(L)
        pv.step(L2, _day(idx), P, filing=_no_filing)
        assert bool([o for o in L2["pending"] if o["type"] == "buy"]) is allowed


def test_random_path_only_buys_on_given_days():
    R = pv.new_ledger("r0", _day(), P, earnings_filter=True, random=True)
    pv.step(R, _day(), P, filing=_no_filing, random_n=0)
    assert not R["pending"]
    rng = np.random.default_rng(P.seed)
    pv.step(R, _day(closes={"A": 120.0, "B": 120.0}), P, filing=_no_filing, random_n=1, rng=rng)  # 60일선 위여도 자격이면 후보
    assert len(R["pending"]) == 1 and R["pending"][0]["tk"] in ("A", "B")


def test_judge():
    ok = pv.judge(1.10e8, 1.05e8, 1.08e8, 1.09e8, -0.12, -0.11)
    assert ok["pass"]
    bad = pv.judge(1.10e8, 1.05e8, 1.08e8, 1.09e8, -0.14, -0.11)
    assert not bad["pass"] and not bad["3_mdd_within_2pp"]


# ── 엔진: 시작·따라잡기·저장 ──────────────────────────────────────────────────


def _history_world(n=330, last="2026-11-04"):
    cal = pd.bdate_range(end=last, periods=n)
    t = np.arange(n)
    q = pd.DataFrame({"close": 300 * 1.001 ** t, "volume": 1e7}, index=cal)
    a = 100 * 1.002 ** t
    a[-3:] = a[-4] * np.array([0.9, 0.89, 0.88])  # 마지막 사흘 60일선 아래로
    data = {"QQQ": q, "AAA": pd.DataFrame({"close": a, "volume": 5e6}, index=cal),
            "BBB": pd.DataFrame({"close": 50 * 1.001 ** t, "volume": 1e6}, index=cal)}
    return cal, data


def _run(tmp_path, as_of, monkeypatch, actions=True, filings=None, **kw):
    cal, data = _history_world()
    if actions:
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS", "x")
    cfg = {"puddle_v31": {"enabled": True, "start_date": "", "start_fx": ""}}
    return eng.run(cfg, as_of, db_path=tmp_path / "vb.db", history_fn=lambda t, c: data[t], universe_fn=lambda: ["AAA", "BBB"],
                   fx_fn=lambda ds: {str(pd.Timestamp(d).date()): 1400.0 for d in ds},
                   year_days_fn=lambda y: pd.bdate_range(f"{y}-01-01", f"{y}-12-31"),
                   filings_fn=filings or (lambda tk: []), sheet_write=False, p=pv.Params(n_random=5), **kw)


def test_engine_start_catchup_and_idempotent(tmp_path, monkeypatch):
    r1 = _run(tmp_path, "2026-10-28", monkeypatch)
    assert r1["status"] == "ok" and r1["start_date"] == "2026-10-28" and len(r1["rows"]) == 1
    assert r1["rows"][0]["v31"] == pytest.approx(1e8 * (1 - 0.001), rel=1e-3)  # 시작일 세후 = 1억 − 청산 비용
    r2 = _run(tmp_path, "2026-11-04", monkeypatch)
    assert [r["date"] for r in r2["rows"]] == ["2026-10-29", "2026-10-30", "2026-11-02", "2026-11-03", "2026-11-04"]
    fills = [(r["date"], f["tk"], f["tranche"]) for r in r2["rows"] for f in r["fills"]]
    assert fills[0] == ("2026-11-03", "AAA", 1)  # 11-02 처음 60일선 아래 → 다음 거래일 종가
    assert fills[1] == ("2026-11-04", "AAA", 2)  # 11-03 120일선 아래(200일선 위) → 2차
    assert r2["rows"][-1]["weights"]["AAA"] == pytest.approx(0.10, abs=0.005)
    r3 = _run(tmp_path, "2026-11-04", monkeypatch)
    assert r3["status"] == "already" and r3["letter"]["personal"]


def test_engine_local_run_is_preview_only(tmp_path, monkeypatch):
    r = _run(tmp_path, "2026-10-28", monkeypatch, actions=False)
    assert r["status"] == "preview"
    from store import virtual_books as vbstore

    conn = vbstore.connect(tmp_path / "vb.db")
    assert eng.load_state(conn) is None  # 로컬은 저장하지 않는다
    conn.close()


def test_engine_filing_failure_blocks_buy_and_marks_check(tmp_path, monkeypatch):
    def boom(tk):
        raise ConnectionError("x")

    _run(tmp_path, "2026-10-28", monkeypatch, filings=boom)
    r = _run(tmp_path, "2026-11-04", monkeypatch, filings=boom)
    fails = {t for row in r["rows"] for t in row["failed"]}
    assert fails == {"AAA"} and not any(row["fills"] for row in r["rows"])
    assert r["letter"]["check_personal"] == "실적 확인 실패(AAA)" and r["letter"]["check_group"] == "실적 확인 실패(1종목)"


# ── 표시: 편지·시트 ──────────────────────────────────────────────────────────


def _row(**kw):
    r = {"date": "2026-11-04", "v31": 1.02e8, "qqq": 1.01e8, "v3": 1.025e8, "random": {"50": 1.015e8},
         "weights": {"AAPL": 0.051, "MSFT": 0.10}, "qqq_weight": 0.849,
         "fills": [{"tk": "AAPL", "tranche": 1, "usd": 1.0, "date": "2026-11-04"}],
         "sells": [{"tk": "NVDA", "reason": "월초 상위 10 밖"}], "skipped": [{"tk": "AMZN", "tranche": 1}], "failed": [],
         "realized_year": 123456.0}
    r.update(kw)
    return r


def test_letter_lines_personal_and_group_without_tickers():
    out = eng.letter_lines([_row()])
    assert out["personal"] == ["웅덩이 v3.1: 가상 1억이 QQQ보다 1.0%p 앞서요.", "지금 MSFT 10%, AAPL 5%를 담고 있어요.",
                               "오늘 AAPL을 처음 샀어요(1차).", "오늘 NVDA를 모두 팔았어요(월초 상위 10 밖).",
                               "오늘은 AMZN이 내려왔지만 실적 발표 직후라 사지 않았어요."]
    g = out["group"]
    assert g == ["웅덩이 v3.1: 가상 1억이 QQQ보다 1.0%p 앞서요.", "지금 2종목을 합계 15% 담고 있어요.", "오늘 가상으로 1종목을 샀어요.",
                 "오늘 가상으로 1종목을 팔았어요.", "오늘 1종목이 내려왔지만 실적 발표 직후라 사지 않았어요."]
    for t in ("AAPL", "MSFT", "NVDA", "AMZN"):
        assert all(t not in line for line in g), t
    quiet = eng.letter_lines([_row(v31=0.99e8, weights={}, fills=[], sells=[], skipped=[])])
    assert quiet["personal"] == ["웅덩이 v3.1: 가상 1억이 QQQ보다 2.0%p 뒤처져요.", "지금은 종목 없이 QQQ만 들고 있어요."]
    one = eng.letter_lines([_row(weights={"AMD": 0.05}, fills=[], sells=[], skipped=[])])
    assert one["personal"][1] == "지금 AMD를 5% 담고 있어요." and one["group"][1] == "지금 1종목을 5% 담고 있어요."


def test_josa_for_tickers():
    from notify import letter_plain as lp

    assert [lp.josa(t, ("을", "를")) for t in ("AAPL", "AMD", "AMZN", "META", "QQQM")] == ["AAPL을", "AMD를", "AMZN을", "META를", "QQQM을"]
    assert lp.josa("ISA", ("은", "는")) == "ISA는" and lp.josa("웅덩이", ("이", "가")) == "웅덩이가"


def test_report_group_edition_has_no_tickers(cfg, tmp_path):
    from notify import report_html
    from tests.test_fresh_start import _summary

    s = _summary(copy.deepcopy(cfg))
    s["puddle_v31"] = eng.letter_lines([_row()])
    personal = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    assert "[가상 실험 — 실제 돈 아님]" in personal and "웅덩이 v3.1: 가상 1억이 QQQ보다 1.0%p 앞서요." in personal
    assert "오늘 AAPL을 처음 샀어요(1차)." in personal
    group = report_html.render_report(s, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    body = group[group.index("<header>"):group.index("<script>")]
    assert "웅덩이 v3.1: 가상 1억이" in body
    for t in ("AAPL", "MSFT", "NVDA", "AMZN"):
        assert t not in body, t


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
        self.sheets = {"QQQM 모으기": _WS("QQQM 모으기", [["x"]])}

    def worksheet(self, name):
        if name not in self.sheets:
            raise KeyError(name)
        return self.sheets[name]

    def add_worksheet(self, title, rows, cols):
        self.sheets[title] = _WS(title)
        return self.sheets[title]


def test_sheet_writes_only_its_tab_and_upserts_by_date():
    from data import puddle_v31_sheet as ps

    book = _Book()
    client = type("C", (), {"open_by_key": lambda self, k: book})()
    ps.write_rows(client, "sid", [eng.sheet_row(_row())])
    ws = book.sheets[ps.TAB]
    ups = ws.calls[0]
    assert ups[0] == {"range": "A1", "values": [[ps.TITLE]]} and ups[1]["values"] == [ps.HEADER]
    assert ups[2]["range"] == "A3" and ups[2]["values"][0][0] == "2026-11-04"
    assert book.sheets["QQQM 모으기"].calls == []  # 다른 탭은 건드리지 않음
    ws.values = [[ps.TITLE], ps.HEADER, ["2026-11-04"]]
    ps.write_rows(client, "sid", [eng.sheet_row(_row())])
    assert ws.calls[1][2]["range"] == "A3"  # 같은 날짜는 덮어씀
    with pytest.raises(ps.WrongTabError):
        ps.upsert_rows(book.sheets["QQQM 모으기"], [["x"]])
    assert ps.HEADER == ["날짜", "v3.1 평가금액(원)", "QQQ 보유(원)", "v3(원)", "QQQ 대비(%p)", "무작위 중앙값 대비(%p)", "QQQ 비중",
                         "보유 종목(비중)", "오늘 매매", "올해 실현 이익(원)"]


def test_research_news_in_progress_on_monday(cfg):
    from notify import letter

    lines = letter.research_lines(cfg, pd.Timestamp("2026-10-26"))
    assert lines[-1] == "연구 중: 상승장 대형주 웅덩이 v3.1(가상 기록, 12개월)"
    assert letter.research_lines(cfg, pd.Timestamp("2026-10-27")) == []
