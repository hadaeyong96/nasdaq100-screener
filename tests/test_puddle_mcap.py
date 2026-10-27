"""웅덩이 v3.2·반등 확인 v2 (시가총액 상위 10, 2026-10-27) — 순위·주식 수·옛 장부 꺼짐·새 장부 기록·시트 탭 교체. 합성 데이터, 네트워크 막음."""

from __future__ import annotations

import json
import socket

import numpy as np
import pandas as pd
import pytest

from core import puddle_rebound as pr
from core import puddle_v31 as pv
from data import shares as dshares
from engine import puddle_rebound as erb
from engine import puddle_rebound_v2 as erb2
from engine import puddle_v31 as e31
from engine import puddle_v32 as e32
from store import virtual_books as vbstore


@pytest.fixture(autouse=True)
def _block_socket(monkeypatch):
    def deny(*a, **k):
        raise RuntimeError("테스트 중 네트워크 금지")

    monkeypatch.setattr(socket.socket, "connect", deny)


# ── 시가총액 순위 (순수 함수) ─────────────────────────────────────────────────


def test_market_cap_rank_orders_by_close_times_shares():
    closes = {"AAA": 100.0, "BBB": 10.0, "CCC": 50.0}
    shares = {"AAA": 1e6, "BBB": 1e8, "CCC": 1e7}
    top, caps, missing = pv.top_by_market_cap(closes, shares, {}, ["AAA", "BBB", "CCC"], n=2)
    assert top == ["BBB", "CCC"] and caps == {"BBB": 1e9, "CCC": 5e8} and missing == []


def test_same_company_caps_are_summed_and_trade_ticker_has_bigger_dollar_volume():
    closes = {"GOOG": 100.0, "GOOGL": 101.0, "MSFT": 400.0}
    shares = {"GOOG": 5.5e9, "GOOGL": 5.8e9, "MSFT": 1.4e9}  # 알파벳 한 종류만이면 MSFT(5.6e11)보다 작지만 합치면 1.1e12
    top, caps, _ = pv.top_by_market_cap(closes, shares, {"GOOG": 3e9, "GOOGL": 2e9, "MSFT": 1e9}, ["GOOGL", "GOOG", "MSFT"], n=2)
    assert top == ["GOOG", "MSFT"]  # 거래대금 큰 GOOG로 거래
    assert caps["GOOG"] == pytest.approx(100 * 5.5e9 + 101 * 5.8e9)
    top, _, _ = pv.top_by_market_cap(closes, shares, {"GOOG": 1e9, "GOOGL": 2e9}, ["GOOGL", "GOOG", "MSFT"], n=1)
    assert top == ["GOOGL"]
    top, _, _ = pv.top_by_market_cap(closes, shares, {}, ["GOOGL", "GOOG", "MSFT"], n=1)
    assert top == ["GOOG"]  # 거래대금 모르면 알파벳 앞


def test_missing_shares_excluded_and_whole_company_out_if_one_class_missing():
    closes = {"AAA": 10.0, "GOOG": 100.0, "GOOGL": 100.0, "ZZZ": 1.0, "NOPX": float("nan")}
    shares = {"AAA": 1e6, "GOOG": None, "GOOGL": 5e9, "ZZZ": 0.0, "NOPX": 1e12}
    top, caps, missing = pv.top_by_market_cap(closes, shares, {}, list(closes), n=10)
    assert top == ["AAA"] and missing == ["GOOG", "ZZZ"]  # 알파벳은 GOOG 주식 수가 없어 회사 전체(GOOGL 포함) 제외, 점검에는 못 받은 티커만
    assert "NOPX" not in missing


def test_edgar_whole_company_shares_not_double_counted():
    closes = {"GOOG": 100.0, "GOOGL": 110.0}
    shares = {"GOOG": 12e9, "GOOGL": 5.8e9}  # GOOG 값은 EDGAR 회사 전체
    _, caps, _ = pv.top_by_market_cap(closes, shares, {"GOOG": 1.0, "GOOGL": 2.0}, ["GOOG", "GOOGL"], whole={"GOOG"})
    assert caps == {"GOOGL": pytest.approx(12e9 * 110.0)}  # 회사 전체 주식 수 × 거래 티커 종가 하나


def test_ties_break_by_ticker():
    top, _, _ = pv.top_by_market_cap({"BBB": 1.0, "AAA": 1.0}, {"BBB": 5.0, "AAA": 5.0}, {}, ["BBB", "AAA"], n=2)
    assert top == ["AAA", "BBB"]


# ── 주식 수 받기 ──────────────────────────────────────────────────────────────


def test_edgar_total_shares_latest_filing_sum():
    facts = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2026-04-20", "val": 100, "filed": "2026-04-25", "accn": "a1"},
        {"end": "2026-07-20", "val": 5_800, "filed": "2026-07-25", "accn": "a2"},  # Class A
        {"end": "2026-07-20", "val": 800, "filed": "2026-07-25", "accn": "a2"},  # Class B
        {"end": "2026-07-20", "val": 5_400, "filed": "2026-07-25", "accn": "a2"},  # Class C
        {"end": "2026-07-20", "val": 999, "filed": "2026-07-01", "accn": "a0"},  # 같은 날 값의 이른 접수 — 안 씀
    ]}}}}}
    assert dshares.edgar_total_shares(facts) == 12_000
    assert dshares.edgar_total_shares({"facts": {}}) is None


def test_fetch_shares_falls_back_to_edgar_and_reports_failures():
    calls = []

    def yf(t):
        calls.append(t)
        if t == "BAD":
            raise ValueError("x")
        return {"AAA": 10.0, "BBB": None, "BAD": None, "NONE": 0.0}[t]

    def ed(t):
        if t == "NONE":
            raise RuntimeError("edgar down")
        return {"BBB": 20.0, "BAD": 30.0}[t]

    out, whole, failed = dshares.fetch_shares(["AAA", "BBB", "BAD", "NONE", "AAA"], yf_fn=yf, edgar_fn=ed, memo=False)
    assert out == {"AAA": 10.0, "BBB": 20.0, "BAD": 30.0, "NONE": None}
    assert whole == {"BBB", "BAD"} and failed == ["NONE"] and calls.count("AAA") == 1


# ── 옛 장부 꺼짐 ─────────────────────────────────────────────────────────────


def test_old_ledgers_are_off_and_new_ones_on_in_config(cfg):
    assert cfg["puddle_v31"]["enabled"] is False and cfg["puddle_rebound"]["enabled"] is False
    assert cfg["puddle_v31"]["start_date"] == "2026-10-23"  # 보관본 시작값은 그대로
    assert cfg["puddle_v32"]["enabled"] is True and cfg["puddle_rebound_v2"]["enabled"] is True
    assert not cfg["puddle_v32"].get("start_date") and not cfg["puddle_rebound_v2"].get("start_date")  # 첫 정규 실행 뒤 고정
    assert e31.run_safely(cfg, "2026-10-27") is None and erb.run_safely(cfg, "2026-10-27") is None


def test_daily_attaches_new_ledgers_not_old(monkeypatch):
    from engine import daily
    from engine import puddle_rebound_v2, puddle_v32

    seen = []
    monkeypatch.setattr(puddle_v32, "run_safely", lambda cfg, as_of: seen.append("v32") or {"letter": {"personal": ["a"]}})
    monkeypatch.setattr(puddle_rebound_v2, "run_safely", lambda cfg, as_of: seen.append("rb2") or {"letter": {"personal": ["b"]}})
    monkeypatch.setattr(e31, "run_safely", lambda *a, **k: pytest.fail("옛 v3.1 장부가 돌았습니다"))
    monkeypatch.setattr(erb, "run_safely", lambda *a, **k: pytest.fail("옛 반등 확인 장부가 돌았습니다"))
    s = {"as_of": pd.Timestamp("2026-10-27"), "letter_check": {"lag": 0, "notes": []}}
    daily._attach_puddle_v31(s, {})
    daily._attach_puddle_rebound(s, {})
    assert seen == ["v32", "rb2"] and s["puddle_v31"]["personal"] == ["a"] and s["puddle_rebound"]["personal"] == ["b"]


def test_daily_error_note_names_new_ledgers():
    from engine import daily
    from notify import letter_plain as lp

    s = {"as_of": pd.Timestamp("2026-10-27"), "letter_check": {"lag": 0, "notes": []}}
    err = {"status": "error", "letter": {}, "rows": []}
    daily._attach_puddle_v31(s, {}, run_fn=lambda cfg, a: err)
    daily._attach_puddle_rebound(s, {}, run_fn=lambda cfg, a: err)
    assert s["letter_check"]["notes"] == ["웅덩이 v3.2 장부 오류", "웅덩이 반등 확인 v2 장부 오류"]
    probs = lp.problems(s, lp.load_phrases(), lp.PERSONAL)
    assert probs == ["웅덩이 v3.2 장부를 계산하지 못했습니다.", "웅덩이 반등 확인 v2 장부를 계산하지 못했습니다."]


# ── 새 장부 기록 (엔진) ──────────────────────────────────────────────────────


def _world(n=330, last="2026-11-04"):
    """test_puddle_v31과 같은 세계 + CCC(주식 수 없음). AAA는 거래대금 1위·마지막 사흘 웅덩이, BBB는 시가총액 1위."""
    cal = pd.bdate_range(end=last, periods=n)
    t = np.arange(n)
    q = pd.DataFrame({"close": 300 * 1.001 ** t, "volume": 1e7}, index=cal)
    a = 100 * 1.002 ** t
    a[-3:] = a[-4] * np.array([0.9, 0.89, 0.88])
    data = {"QQQ": q, "AAA": pd.DataFrame({"close": a, "volume": 5e6}, index=cal),
            "BBB": pd.DataFrame({"close": 50 * 1.001 ** t, "volume": 1e6}, index=cal),
            "CCC": pd.DataFrame({"close": 20 * 1.001 ** t, "volume": 1e6}, index=cal)}
    return cal, data


SHARES = ({"AAA": 1e6, "BBB": 1e9, "CCC": None}, set(), ["CCC"])


def _kw(tmp_path, data, calls=None):
    def shares_fn(tickers):
        if calls is not None:
            calls.append(list(tickers))
        return SHARES

    return dict(db_path=tmp_path / "vb.db", history_fn=lambda t, c: data[t], universe_fn=lambda: ["AAA", "BBB", "CCC"],
                fx_fn=lambda ds: {str(pd.Timestamp(d).date()): 1400.0 for d in ds},
                year_days_fn=lambda y: pd.bdate_range(f"{y}-01-01", f"{y}-12-31"), filings_fn=lambda tk: [], sheet_write=False,
                shares_fn=shares_fn)


def _actions(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS", "x")


def test_v32_ranks_by_market_cap_records_to_new_tables_and_archives_old(tmp_path, monkeypatch):
    _actions(monkeypatch)
    cal, data = _world()
    kw = _kw(tmp_path, data)
    # 옛 v3.1 장부 기록(보관본)을 먼저 만든다
    e31.run({"puddle_v31": {"enabled": True}}, "2026-10-28", p=pv.Params(n_random=3), **{k: v for k, v in kw.items() if k != "shares_fn"})
    conn = vbstore.connect(tmp_path / "vb.db")
    old = (vbstore.get_meta(conn, e31.META_STATE), conn.execute("SELECT * FROM puddle_v31_daily").fetchall())
    conn.close()
    calls = []
    kw = _kw(tmp_path, data, calls)
    p1 = pv.Params(n_random=5, top_n=1)
    r1 = e32.run({"puddle_v32": {"enabled": True}}, "2026-10-28", p=p1, **kw)
    assert r1["status"] == "ok" and r1["start_date"] == "2026-10-28"
    row = r1["rows"][0]
    assert row["top10"] == ["BBB"]  # 거래대금 1위 AAA가 아니라 시가총액 1위 BBB
    assert row["caps"] == {"BBB": pytest.approx(float(data["BBB"]["close"].loc["2026-10-28"]) * 1e9)}
    assert row["shares_missing"] == ["CCC"] and len(calls) == 1
    assert r1["letter"]["personal"][0].startswith("웅덩이 v3.2: ")
    assert r1["letter"]["shares_personal"] == "(CCC)" and r1["letter"]["shares_group"] == "(1종목)"
    assert "CCC" not in " ".join(r1["letter"]["group"])
    r2 = e32.run({"puddle_v32": {"enabled": True}}, "2026-11-04", p=p1, **kw)
    rows = {r["date"]: r for r in r2["rows"]}
    assert "caps" not in rows["2026-10-29"] and "caps" in rows["2026-11-02"]  # 순위 날 = 시작일·매달 첫 거래일만
    assert not [f for r in r2["rows"] for f in r["fills"]]  # AAA 웅덩이는 대상(시총 상위 1) 밖이라 안 산다
    conn = vbstore.connect(tmp_path / "vb.db")
    try:
        assert (vbstore.get_meta(conn, e31.META_STATE), conn.execute("SELECT * FROM puddle_v31_daily").fetchall()) == old  # 옛 기록 그대로
        assert [d for (d,) in conn.execute("SELECT date FROM puddle_v32_daily ORDER BY date")][0] == "2026-10-28"
        assert json.loads(vbstore.get_meta(conn, "puddle_v32_state"))["start_date"] == "2026-10-28"
        arch = json.loads(vbstore.get_meta(conn, e31.META_ARCHIVED))
        assert set(arch) == {"puddle_v31", "puddle_rebound"} and arch["puddle_v31"]["tag"] == "archive-puddle-v31-20261027"
    finally:
        conn.close()
    assert e32.run({"puddle_v32": {"enabled": True}}, "2026-11-04", p=p1, **kw)["status"] == "already"


def test_v32_same_rules_as_v31_when_ranks_agree(tmp_path, monkeypatch):
    """대상만 다르다: 시가총액 순위도 AAA가 1위면 v3.1과 매매가 같다(11-03 1차, 11-04 2차)."""
    _actions(monkeypatch)
    cal, data = _world()
    kw = _kw(tmp_path, data)
    kw["shares_fn"] = lambda tks: ({"AAA": 1e12, "BBB": 1e6, "CCC": 1e6}, set(), [])
    e32.run({"puddle_v32": {"enabled": True}}, "2026-10-28", p=pv.Params(n_random=5), **kw)
    r = e32.run({"puddle_v32": {"enabled": True}}, "2026-11-04", p=pv.Params(n_random=5), **kw)
    fills = [(x["date"], f["tk"], f["tranche"]) for x in r["rows"] for f in x["fills"]]
    assert fills[:2] == [("2026-11-03", "AAA", 1), ("2026-11-04", "AAA", 2)]


def test_v32_local_run_is_preview_only(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    cal, data = _world()
    r = e32.run({"puddle_v32": {"enabled": True}}, "2026-10-28", p=pv.Params(n_random=3), **_kw(tmp_path, data))
    assert r["status"] == "preview"
    conn = vbstore.connect(tmp_path / "vb.db")
    try:
        assert e31.load_state(conn, e31.BOOK_V32) is None and vbstore.get_meta(conn, e31.META_ARCHIVED) is None
    finally:
        conn.close()


def test_rebound_v2_reads_v32_values_as_comparison_b(tmp_path, monkeypatch):
    _actions(monkeypatch)
    cal, data = _world()
    kw = _kw(tmp_path, data)
    for d in ("2026-10-28", "2026-11-04"):
        e32.run({"puddle_v32": {"enabled": True}}, d, p=pv.Params(n_random=3), **kw)
    conn = vbstore.connect(tmp_path / "vb.db")
    v32 = {d: json.loads(j)["v31"] for d, j in conn.execute("SELECT date, row_json FROM puddle_v32_daily")}
    conn.close()
    erb2.run({"puddle_rebound_v2": {"enabled": True}}, "2026-10-28", p=pr.Params(n_random=3), **kw)
    r = erb2.run({"puddle_rebound_v2": {"enabled": True}}, "2026-11-04", p=pr.Params(n_random=3), **kw)
    assert r["rows"][-1]["v31"] == pytest.approx(v32["2026-11-04"]) and r["rows"][-1]["top10"][0] == "BBB"
    assert r["letter"]["personal"][0].startswith("웅덩이 반등 확인 v2: ") and r["letter"]["b_label"] == "v3.2"
    conn = vbstore.connect(tmp_path / "vb.db")
    try:
        assert [d for (d,) in conn.execute("SELECT date FROM puddle_rebound_v2_daily ORDER BY date")][0] == "2026-10-28"
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='puddle_rebound_daily'").fetchall() == []  # 옛 표는 만들지도 않음
    finally:
        conn.close()


# ── 편지 ─────────────────────────────────────────────────────────────────────


def _rb_row(**kw):
    r = {"date": "2026-11-06", "value": 100_300_000.0, "qqq": 100_000_000.0, "v31": 100_000_000.0, "random": {"50": 1e8},
         "weights": {}, "qqq_weight": 1.0, "fills": [], "sells": [], "skipped": [], "failed": [], "buys": [], "waiting": [],
         "opened": [], "unqualified": [], "realized_year": 0.0, "shares_missing": ["CCC"]}
    r.update(kw)
    return r


def test_letter_monday_line_names_v32_and_shares_problem_line():
    from notify import letter_plain as lp

    rb = erb2.letter_lines([_rb_row()])
    v = e32.letter_lines([{"date": "2026-11-06", "v31": 1e8, "qqq": 1e8, "v3": 1e8, "random": {"50": 1e8}, "weights": {}, "qqq_weight": 1.0,
                           "fills": [], "sells": [], "skipped": [], "failed": [], "shares_missing": ["CCC"]}])
    base = {"as_of": pd.Timestamp("2026-11-06"), "letter_today_kst": pd.Timestamp("2026-11-09"), "mode": "live",
            "letter_check": {"lag": 0, "notes": []}, "puddle_rebound": rb, "puddle_v31": v}
    d = lp.build(base, {"research_news": {}}, lp.PERSONAL)
    lab = d["sections"][-1]["lines"]
    assert lab[0].startswith("웅덩이 v3.2: ") and any(x.startswith("웅덩이 반등 확인 v2: ") for x in lab)
    assert "웅덩이 v3.2보다 0.3%p 앞섭니다." in lab
    assert d["problem"].endswith("주식 수를 못 받아 시가총액 순위에서 뺀 종목이 있습니다(CCC).")
    g = lp.build(base, {"research_news": {}}, lp.GROUP)
    assert g["problem"].endswith("주식 수를 못 받아 시가총액 순위에서 뺀 종목이 있습니다(1종목).") and "CCC" not in lp.to_text(g)


def test_research_news_in_progress_lists_new_ledgers(cfg):
    from notify import letter

    lines = letter.research_lines(cfg, pd.Timestamp("2026-11-02"))
    assert "연구 중: 웅덩이 v3.2(시가총액 상위 10, 가상 기록, 12개월)" in " ".join(lines) or \
        any("웅덩이 v3.2(시가총액 상위 10" in x for x in lines)
    assert not any("v3.1" in x for x in lines)


# ── 시트 탭 교체 ─────────────────────────────────────────────────────────────


class _WS:
    def __init__(self, title, values=None):
        self.title, self.values, self.row_count, self.calls, self.cleared = title, values or [], 400, [], []

    def get_all_values(self):
        return self.values

    def batch_update(self, updates, value_input_option=None):
        self.calls.append(updates)

    def batch_clear(self, ranges):
        self.cleared.append(ranges)
        self.values = []

    def update_title(self, title):
        self.title = title

    def resize(self, rows=None, cols=None):
        self.row_count = rows


class _Spreadsheet:
    def __init__(self, tabs):
        self.ws = {t.title: t for t in tabs}

    def worksheet(self, name):
        for w in self.ws.values():
            if w.title == name:
                return w
        raise KeyError(name)

    def add_worksheet(self, title, rows, cols):
        w = _WS(title)
        self.ws[title] = w
        return w


def _client(book):
    return type("C", (), {"open_by_key": lambda self, k: book})()


@pytest.mark.parametrize("mod_name", ["data.puddle_v32_sheet", "data.puddle_rebound_v2_sheet"])
def test_sheet_renames_old_tab_clears_and_writes_new_rows(mod_name):
    import importlib

    ps = importlib.import_module(mod_name)
    old = _WS(ps.OLD_TAB, [["옛 제목"], ["옛 머리글"], ["2026-10-23", 1], ["2026-10-26", 2]])
    other = _WS("QQQM 모으기", [["x"]])
    book = _Spreadsheet([old, other])
    ps.write_rows(_client(book), "sid", [["2026-10-28", 100_000_000]])
    assert old.title == ps.TAB and old.cleared  # 같은 탭을 새 이름으로 바꾸고 비움
    ups = old.calls[-1]
    assert ups[0] == {"range": "A1", "values": [[ps.TITLE]]} and ups[1] == {"range": "A2", "values": [ps.HEADER]}
    assert ups[2] == {"range": "A3", "values": [["2026-10-28", 100_000_000]]} and len(ups) == 3
    assert other.calls == [] and other.title == "QQQM 모으기"  # 다른 탭은 건드리지 않음
    assert len(book.ws) == 2  # 새 탭을 따로 만들지 않음
    # 두 번째 실행: 새 탭이 이미 있으면 바꾸거나 비우지 않고 날짜 줄만 덮어쓰기/붙이기
    old.values = [[ps.TITLE], ps.HEADER, ["2026-10-28", 1]]
    n_clear = len(old.cleared)
    ps.write_rows(_client(book), "sid", [["2026-10-28", 5], ["2026-10-29", 6]])
    assert len(old.cleared) == n_clear
    assert [u["range"] for u in old.calls[-1][2:]] == ["A3", "A4"]


def test_sheet_creates_new_tab_when_no_old_tab_and_refuses_other_tabs():
    from data import puddle_v32_sheet as ps

    book = _Spreadsheet([_WS("QQQM 모으기")])
    ps.write_rows(_client(book), "sid", [["2026-10-28", 1]])
    assert ps.TAB in book.ws and book.ws[ps.TAB].calls
    with pytest.raises(ps.WrongTabError):
        ps.upsert(book.ws["QQQM 모으기"], [["x"]], ps.TAB, ps.TITLE, ps.HEADER)
    assert ps.TAB == "웅덩이 v3.2 (가상·시총)"
    from data import puddle_rebound_v2_sheet as ps2

    assert ps2.TAB == "웅덩이 반등 확인 v2 (가상·시총)" and ps2.OLD_TAB == "웅덩이 반등 확인 (가상)" and ps.OLD_TAB == "웅덩이 v3.1 (가상)"


# ── 미래 데이터 방지(순위) ────────────────────────────────────────────────────


def test_rank_uses_only_that_day_close(tmp_path):
    cal, data = _world()
    w = e31.World({}, ["AAA", "BBB", "CCC"], set(), lambda t, c: data[t], pv.Params(top_n=3), ranking="mcap",
                  shares_fn=lambda tks: SHARES)
    d = cal[-10]
    full = w.rank(d)
    cut = {k: v[v.index <= d] for k, v in data.items()}
    w2 = e31.World({}, ["AAA", "BBB", "CCC"], set(), lambda t, c: cut[t], pv.Params(top_n=3), ranking="mcap",
                   shares_fn=lambda tks: SHARES)
    assert w2.rank(d) == full and full["list"] == ["BBB", "AAA"] and full["missing"] == ["CCC"]
