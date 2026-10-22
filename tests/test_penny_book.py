"""10번 「동전주 거래량 수렴 전략」·그림자 「무작위 동전주 20」 (docs/prereg/penny_volume_squeeze.md) — 네트워크 없이 합성 데이터만."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core import penny_book as pc
from data import penny_universe as pu
from engine import penny_book as pe
from engine.daily import load_config
from notify import letter
from notify import virtual_books as vbtext
from store import virtual_books as vbstore
from tests.legacy import legacy_cfg

P = pc.PennyParams(name="동전주 거래량 수렴 따라하기")
REAL_FETCH_LISTS = pu.fetch_lists  # conftest가 테스트마다 네트워크 함수를 막기 전(모듈 읽을 때) 진짜 함수
DAYS = pd.bdate_range("2026-03-02", "2026-10-30")


@pytest.fixture
def cfg():
    return legacy_cfg(load_config())  # 10번은 새 출발(2026-10-22)로 꺼 둠 — 코드 동작은 그대로 시험


def _frame(close, vol, days=DAYS):
    close = np.broadcast_to(np.asarray(close, dtype=float), (len(days),)).copy()
    vol = np.broadcast_to(np.asarray(vol, dtype=float), (len(days),)).copy()
    return pd.DataFrame({"close": close, "volume": vol}, index=days)


def _squeezer(level=3.0, early_vol=1_000_000, late_vol=400_000, n_late=20, days=DAYS, p_date="2026-10-19"):
    """p_date까지 최근 20행(과 그 뒤) 거래량이 작은(수렴) 종목."""
    vol = np.full(len(days), float(early_vol))
    vol[days >= days[days <= pd.Timestamp(p_date)][-n_late]] = late_vol
    return _frame(level, vol, days)


# ── 대상·고르기 ──────────────────────────────────────────────────────────────


def test_stats_filters_price_dollar_volume_history():
    d = DAYS[-1]
    assert pc.stats(_frame(3.0, 1_000_000), d, P)["ratio"] == pytest.approx(1.0)
    assert pc.stats(_frame(0.9, 5_000_000), d, P) is None  # 1달러 미만
    assert pc.stats(_frame(5.5, 5_000_000), d, P) is None  # 5달러 초과
    assert pc.stats(_frame(3.0, 300_000), d, P) is None  # 거래대금 90만 달러
    assert pc.stats(_frame(3.0, 1_000_000, DAYS[-119:]), d, P) is None  # 기록 119행
    assert pc.stats(_frame(3.0, 1_000_000, DAYS[-120:]), d, P) is not None  # 120행
    assert pc.stats(_frame(3.0, 1_000_000, DAYS[:-1]), d, P) is None  # 전날(P) 행 없음


def test_stats_no_future_data():
    f = _squeezer()
    f.iloc[-1, 0] = 50.0  # 마지막 날 값이 바뀌어도 그 전날 판정은 같다
    d = DAYS[-2]
    assert pc.stats(f, d, P) == pc.stats(f.iloc[:-1], d, P)


def test_squeeze_pick_ratio_width_order_and_cap():
    d = DAYS[-1]
    frames = {f"S{i:02d}": _squeezer(late_vol=400_000 + i * 1000) for i in range(25)}  # 비율 0.4…
    frames["WIDE"] = _squeezer()
    frames["WIDE"].iloc[-10:, 0] = 4.0  # 20일 가격 폭 > 25%
    frames["LOOSE"] = _frame(3.0, 1_000_000)  # 비율 1.0
    elig = pc.eligible(frames, d, P)
    picks = pc.squeeze_pick(elig, P)
    assert len(picks) == 20 and picks[0] == "S00" and picks[-1] == "S19"
    assert "WIDE" not in picks and "LOOSE" not in picks and "WIDE" in elig


def test_random_pick_seeded_and_from_same_pool():
    elig = {f"T{i:03d}": {} for i in range(100)}
    a = pc.random_pick(elig, "2026-10-21", P)
    assert a == pc.random_pick(dict(reversed(list(elig.items()))), "2026-10-21", P)  # 정렬 뒤 뽑기 — 같은 결과
    assert len(set(a)) == 20 and set(a) <= set(elig)
    assert a != pc.random_pick(elig, "2027-01-04", P)  # 선정일마다 다른 뽑기
    assert pc.random_pick({"A": {}, "B": {}}, "2026-10-21", P) in (["A", "B"], ["B", "A"])


def test_select_days_quarter_first_trading_day():
    assert pc.is_select_day("2027-01-04", "2026-12-31") and pc.is_select_day("2026-10-01", "2026-09-30")
    assert not pc.is_select_day("2026-10-02", "2026-10-01") and not pc.is_select_day("2026-11-02", "2026-10-30")


# ── 하루 처리 ────────────────────────────────────────────────────────────────


def _start(targets, closes, fx=1400.0, d="2026-10-20"):
    b = pc.new_book(P.name)
    return pc.process_day(b, d, closes=closes, fx=fx, dividends={}, splits={}, p=P, targets=targets)


def test_start_equal_amount_and_bil_for_empty_slots():
    closes = {"AAA": 2.0, "BBB": 4.0, "BIL": 91.5}
    b, row, trades = _start(["AAA", "BBB"], closes)
    init = 100_000_000 / 1400 * (1 - 0.001)
    assert b["initial_usd"] == pytest.approx(init)
    per = init / 20
    assert b["positions"]["AAA"]["qty"] == int(per / (2.0 * 1.005 * 1.0007))
    assert b["positions"]["BBB"]["qty"] == int(per / (4.0 * 1.005 * 1.0007))
    assert b["positions"]["BIL"]["qty"] > 0 and b["cash"] < 91.5 * 1.0007  # 18칸 몫은 BIL, 잔돈만 현금
    assert [t["ticker"] for t in trades] == ["AAA", "BBB"]  # BIL 매매는 기록하지 않는다
    assert row["holding"] == "2종목" and row["value"] < init  # 비용만큼 빠짐
    assert b["positions"]["AAA"]["entry_px"] == pytest.approx(2.0 * 1.005)


def test_half_sell_next_day_once_per_holding():
    b, _, _ = _start(["AAA"], {"AAA": 2.0, "BIL": 91.5})
    q0 = b["positions"]["AAA"]["qty"]
    b, _, tr = pc.process_day(b, "2026-10-21", closes={"AAA": 4.05, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P)
    assert tr == [] and b["pending_half"] == [["AAA", pd.Timestamp("2026-10-21")]]  # 판정일엔 팔지 않는다
    b, _, tr = pc.process_day(b, "2026-10-22", closes={"AAA": 3.0, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P)
    assert [(t["side"], t["qty"], t["kind"]) for t in tr] == [("매도", q0 // 2, pc.K_HALF)]
    assert tr[0]["price"] == pytest.approx(3.0 * 0.995)  # 다음 거래일 종가(슬리피지)
    b, _, tr = pc.process_day(b, "2026-10-23", closes={"AAA": 9.0, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P)
    b, _, tr = pc.process_day(b, "2026-10-26", closes={"AAA": 9.0, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P)
    assert tr == []  # 1회만


def test_gone_after_five_missing_days_zero_value():
    b, _, _ = _start(["AAA", "BBB"], {"AAA": 2.0, "BBB": 2.0, "BIL": 91.5})
    for i, d in enumerate(pd.bdate_range("2026-10-21", periods=5)):
        b, row, tr = pc.process_day(b, d, closes={"BBB": 2.0, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P)
        if i < 4:
            assert "AAA" in b["positions"] and tr == []
    assert "AAA" not in b["positions"] and tr[0]["kind"] == pc.K_GONE and tr[0]["price"] == 0.0
    assert b["realized_krw"]["2026"] < 0  # 원화 취득가 전액 손실


def test_reselect_keeps_overlap_sells_dropped():
    b, _, _ = _start(["AAA", "BBB"], {"AAA": 2.0, "BBB": 2.0, "BIL": 91.5})
    qa = b["positions"]["AAA"]["qty"]
    b, _, tr = pc.process_day(b, "2027-01-04", closes={"AAA": 2.2, "BBB": 1.8, "CCC": 3.0, "BIL": 92.0}, fx=1400, dividends={},
                              splits={}, p=P, targets=["CCC", "AAA"])
    assert b["positions"]["AAA"]["qty"] == qa  # 겹치는 종목은 그대로(비중 맞추기 없음)
    assert "BBB" not in b["positions"] and "CCC" in b["positions"]
    assert [(t["ticker"], t["kind"]) for t in tr] == [("BBB", pc.K_OUT), ("CCC", pc.K_IN)]


def test_split_reverse_and_dividend():
    b, _, _ = _start(["AAA"], {"AAA": 2.0, "BIL": 91.5})
    q, e = b["positions"]["AAA"]["qty"], b["positions"]["AAA"]["entry_px"]
    b, _, _ = pc.process_day(b, "2026-10-21", closes={"AAA": 20.0, "BIL": 91.5}, fx=1400, dividends={"BIL": 0.3}, splits={"AAA": 0.1},
                             p=P)
    assert b["positions"]["AAA"]["qty"] == q // 10 and b["positions"]["AAA"]["entry_px"] == pytest.approx(e * 10)
    assert b["div_cum"] == pytest.approx(b["positions"]["BIL"]["qty"] * 0.3 * 0.85)
    assert b["pending_half"] == []  # 병합 뒤 같은 가격은 +100%가 아니다


def test_tax_year_end_and_may_payment_and_december_harvest():
    b, _, _ = _start(["AAA"], {"AAA": 2.0, "BIL": 91.5})
    # 12월 실행일: 이익 난 종목을 공제 한도만큼 팔고 같은 날 다시 산다
    b, _, tr = pc.process_day(b, "2026-12-29", closes={"AAA": 3.9, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P,
                              is_dec_exec=True)
    kinds = [t["kind"] for t in tr]
    assert kinds == [pc.K_HARV_OUT, pc.K_HARV_IN]
    assert b["realized_krw"]["2026"] == pytest.approx(2_500_000, rel=0.01) and b["realized_krw"]["2026"] <= 2_500_000
    assert b["positions"]["AAA"]["entry_px"] == pytest.approx(2.0 * 1.005)  # 매수가 유지
    b, _, _ = pc.process_day(b, "2026-12-31", closes={"AAA": 3.9, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P,
                             is_year_end=True)
    assert b["pending_tax"] is None  # 공제 안이라 세금 없음
    b["realized_krw"]["2027"] = 12_500_000.0
    b, _, _ = pc.process_day(b, "2027-12-31", closes={"AAA": 3.9, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P,
                             is_year_end=True)
    assert b["pending_tax"][1] == pytest.approx(10_000_000 * 0.22)
    cash_before = b["cash"] + b["positions"]["BIL"]["qty"] * 91.5
    b, _, tr = pc.process_day(b, "2028-05-01", closes={"AAA": 3.9, "BIL": 91.5}, fx=1400, dividends={}, splits={}, p=P,
                              is_may_first=True)
    assert b["tax_cum_krw"] == pytest.approx(2_200_000) and b["pending_tax"] is None
    assert all(t["ticker"] != "AAA" for t in tr) or cash_before < 2_200_000 / 1400  # BIL부터 판다


def test_alert_hits_jump_or_volume():
    days = DAYS[-25:]
    f = _frame(2.0, 100_000, days)
    g = _frame(2.0, 100_000, days)
    f.iloc[-1, 0] = 2.7  # +35%
    g.iloc[-1, 1] = 600_000  # 6배
    h = _frame(2.0, 100_000, days)
    hits = pc.alert_hits({"F": f, "G": g, "H": h}, ["F", "G", "H"], days[-1])
    assert [x["ticker"] for x in hits] == ["F", "G"] and hits[0]["ret"] == pytest.approx(0.35) and hits[1]["vol_x"] == pytest.approx(6)


def test_judge_rules_and_date():
    assert pc.judge(10.0, 5.0, 8.0, -30.0)["pass"]
    assert not pc.judge(10.0, 12.0, 8.0, -30.0)["pass"]  # 무작위보다 낮음
    assert not pc.judge(10.0, 5.0, 11.0, -30.0)["pass"]  # QQQ보다 낮음
    assert not pc.judge(10.0, 5.0, 8.0, -50.0)["pass"]  # 낙폭 −50% 이하
    days = pd.bdate_range("2027-10-01", "2027-10-29")
    assert pc.judge_date("2026-10-20", days) == pd.Timestamp("2027-10-20")
    assert pc.judge_date("2026-10-23", days) == pd.Timestamp("2027-10-25")  # 토요일 → 월요일
    q = pc.qqq_compare_params(P)
    assert (q.slots, q.slippage_pct, q.half_rule, q.cash_ticker) == (1, 0.0, False, None)


# ── 목록 해석 ────────────────────────────────────────────────────────────────

NASDAQ_TXT = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust, Series 1|G|N|N|100|Y|N
ZZZW|Zeta Corp - Warrant|G|N|N|100|N|N
ZZZU|Zeta Corp - Units|G|N|N|100|N|N
TSTX|Test Co - Common Stock|G|Y|N|100|N|N
ADRX|Foo Ltd - American Depositary Shares|G|N|N|100|N|N
NOTE|Bar Inc - 8.50% Senior Notes due 2029|G|N|N|100|N|N
File Creation Time: 1021202608:46|||||||
"""
OTHER_TXT = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK=B
BAC$K|Bank of America Depositary Shares Preferred|N|BAC$K|N|100|N|BACpK
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
ARCA|Arca Only Co Common Stock|P|ARCA|N|100|N|ARCA
AMX|Small Co Common Stock|A|AMX|N|100|N|AMX
XYZ.U|XYZ Acquisition Units|N|XYZ.U|N|100|N|XYZ=U
CEF|Some Income Fund|N|CEF|N|100|N|CEF
File Creation Time: 1021202608:46|||||||
"""


def test_parse_lists_common_only():
    assert pu.parse_lists(NASDAQ_TXT, OTHER_TXT) == ["AAPL", "ADRX", "AMX", "BRK-B"]
    with pytest.raises(pu.PennyListError):
        pu.parse_lists("Symbol|x\n", OTHER_TXT)


def test_fetch_lists_failure_raises_list_error():
    def boom(url, timeout):
        raise ConnectionError("x")

    with pytest.raises(pu.PennyListError, match="nasdaqlisted.txt 받기 실패: ConnectionError"):
        REAL_FETCH_LISTS(get=boom)

    class R:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    assert REAL_FETCH_LISTS(get=lambda url, timeout: R(NASDAQ_TXT if "nasdaqlisted" in url else OTHER_TXT)) == [
        "AAPL", "ADRX", "AMX", "BRK-B"]


# ── 엔진 (주입, 네트워크 없음) ─────────────────────────────────────────────────


def _universe():
    fr = {f"S{i:02d}": _squeezer(late_vol=400_000 + i * 1000) for i in range(25)}
    fr.update({f"R{i:02d}": _frame(3.0, 1_000_000) for i in range(30)})
    return fr


def _hist(fr):
    def history(t, start):
        if t == "BIL":
            f = _frame(91.5, 1_000_000)
        else:
            f = fr[t].copy()
        f["dividend"], f["split"] = 0.0, 0.0
        return f[f.index >= pd.Timestamp(start)]
    return history


def _ctx(base):
    idx = DAYS[DAYS <= pd.Timestamp(base)]
    closes = pd.DataFrame({"QQQ": np.linspace(500, 520, len(idx))}, index=idx)
    return {"base_date": pd.Timestamp(base), "closes": closes, "divs": pd.DataFrame({"QQQ": 0.0}, index=idx)}


def _cal(first, last):
    return {"dec_exec": {}, "may_first": set(), "year_ends": set()}


def _fns(fr, calls, list_ok=True):
    def list_fn():
        calls.append("list")
        if not list_ok:
            raise pu.PennyListError("nasdaqlisted.txt 받기 실패: ConnectionError")
        return sorted(fr)

    def batch_fn(tickers, start, end):
        calls.append("batch")
        return {t: fr[t] for t in tickers}, {}, 3.0

    return {"list_fn": list_fn, "batch_fn": batch_fn, "history_fn": _hist(fr),
            "fx_range_fn": lambda s, e: {d.date().isoformat(): 1400.0 for d in pd.bdate_range(s, e)}, "calendar_fn": _cal}


def test_engine_start_then_weekday_only_holdings(cfg, tmp_path):
    fr = _universe()
    conn = vbstore.connect(tmp_path / "vb.db")
    calls: list = []
    out = pe.run_penny(cfg, conn, _ctx("2026-10-20"), dry_run=False, **_fns(fr, calls))
    assert out["errors"] == [] and calls == ["list", "batch"]
    names = [x["name"] for x in out["extra"]]
    assert names == ["동전주 거래량 수렴 따라하기", "무작위 동전주 20"]
    main, shadow = out["extra"]
    assert len(main["positions"]) == 20 and set(main["positions"]) == {f"S{i:02d}" for i in range(20)}
    assert len(shadow["positions"]) == 20 and main["start"] == pd.Timestamp("2026-10-20")
    assert out["selection"]["2026-10-20"]["main"] == 20 and out["selection"]["2026-10-20"]["eligible"] == 55
    assert len(vbstore.penny_picks(conn, "동전주 거래량 수렴 따라하기")) == 20
    calls.clear()
    out2 = pe.run_penny(cfg, conn, _ctx("2026-10-22"), dry_run=False, **_fns(fr, calls))
    assert calls == [] and out2["errors"] == []  # 평일: 목록·전체 가격 안 받음
    assert list(out2["extra"][0]["daily"].index) == list(pd.to_datetime(["2026-10-20", "2026-10-21", "2026-10-22"]))
    out3 = pe.run_penny(cfg, conn, _ctx("2026-10-22"), dry_run=False, **_fns(fr, calls))
    assert len(out3["extra"][0]["daily"]) == 3  # 같은 날 다시 실행해도 그대로


def test_engine_list_failure_defers_and_notes(cfg, tmp_path):
    fr = _universe()
    conn = vbstore.connect(tmp_path / "vb.db")
    out = pe.run_penny(cfg, conn, _ctx("2026-10-20"), dry_run=False, **_fns(fr, [], list_ok=False))
    assert out["extra"] == [] and out["notes"] == ["동전주 선정 못함(목록 못 받음)"] and out["errors"] == []
    assert vbstore.load_picks_book(conn, "동전주 거래량 수렴 따라하기")["status"] == pc.WAITING
    out = pe.run_penny(cfg, conn, _ctx("2026-10-21"), dry_run=False, **_fns(fr, []))
    assert out["extra"][0]["start"] == pd.Timestamp("2026-10-21")  # 다음 실행일에 시작


def test_engine_quarter_reselect_and_deferred_reselect(cfg, tmp_path):
    days = pd.bdate_range("2026-03-02", "2027-01-08")
    fr = {t: _frame(f["close"].iat[0], 1_000_000, days) for t, f in _universe().items()}
    for i in range(25):  # 10월엔 S00~S19, 1월엔 S05~S24가 수렴 상위
        v = np.full(len(days), 1_000_000.0)
        v[(days >= "2026-09-21") & (days <= "2026-10-19")] = 400_000 + i * 1000
        v[days >= "2026-12-01"] = 350_000 + (24 - i) * 1000 if i >= 5 else 1_000_000  # 거래대금 100만 달러는 넘게
        fr[f"S{i:02d}"]["volume"] = v
    conn = vbstore.connect(tmp_path / "vb.db")

    def ctx(base):
        idx = days[days <= pd.Timestamp(base)]
        return {"base_date": pd.Timestamp(base), "closes": pd.DataFrame({"QQQ": 500.0}, index=idx),
                "divs": pd.DataFrame({"QQQ": 0.0}, index=idx)}

    fns = _fns(fr, [])
    fns["history_fn"] = lambda t, s: (lambda f: f.assign(dividend=0.0, split=0.0))(
        _frame(91.5, 1, days) if t == "BIL" else fr[t].copy()).loc[pd.Timestamp(s):]
    pe.run_penny(cfg, conn, ctx("2026-10-20"), dry_run=False, **fns)
    before = set(vbstore.load_picks_book(conn, "동전주 거래량 수렴 따라하기")["positions"]) - {"BIL"}
    bad = _fns(fr, [], list_ok=False)
    bad["history_fn"] = fns["history_fn"]
    out = pe.run_penny(cfg, conn, ctx("2027-01-04"), dry_run=False, **bad)  # 1월 첫 거래일 — 목록 실패
    assert out["notes"] and vbstore.load_picks_book(conn, "동전주 거래량 수렴 따라하기")["select_due"] is True
    out = pe.run_penny(cfg, conn, ctx("2027-01-05"), dry_run=False, **fns)  # 다음 실행일이 새 선정일
    b = vbstore.load_picks_book(conn, "동전주 거래량 수렴 따라하기")
    after = set(b["positions"]) - {"BIL"}
    assert b["select_due"] is False and b["selections"] == 2 and before != after and before & after
    trades = out["extra"][0]["trades"]
    assert {t["kind"] for t in trades if pd.Timestamp(t["fill_date"]) == pd.Timestamp("2027-01-05")} == {pc.K_OUT, pc.K_IN}


def test_engine_alerts_only_main_holdings_to_factory(cfg, tmp_path):
    fr = _universe()
    conn = vbstore.connect(tmp_path / "vb.db")
    pe.run_penny(cfg, conn, _ctx("2026-10-20"), dry_run=False, **_fns(fr, []))
    fr["S03"].loc["2026-10-21", "close"] = 4.2  # +40%
    out = pe.run_penny(cfg, conn, _ctx("2026-10-21"), dry_run=False, **_fns(fr, []))
    assert [a["key"] for a in out["alerts"]] == ["2026-10-21:S03"] and out["alerts"][0]["kind"] == pe.ALERT_KIND
    assert "S03" in out["alerts"][0]["text"] and "실제 주문 아님" in out["alerts"][0]["text"]

    from engine import virtual_books as ev

    got = {"personal": [], "factory": []}
    result = {"alerts": out["alerts"], "db_path": tmp_path / "vb.db", "dry_run": False}
    ev.send_alerts(result, sender=lambda t, force_no_send=False: got["personal"].append(t) or True,
                   penny_sender=lambda t, force_no_send=False: got["factory"].append(t) or True)
    assert got["personal"] == [] and len(got["factory"]) == 1  # 전략분석기에만
    ev.send_alerts(result, sender=lambda t, force_no_send=False: True,
                   penny_sender=lambda t, force_no_send=False: got["factory"].append(t) or True)
    assert len(got["factory"]) == 1  # 같은 종목·같은 날 한 번만


def test_factory_send_text_never_real_locally(monkeypatch, capsys):
    from factory import notify as fn

    monkeypatch.setenv("TELEGRAM_FACTORY_CHAT_ID", "1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("SCREENER_REAL_SEND", raising=False)
    monkeypatch.setattr(fn, "_post", lambda *a: (_ for _ in ()).throw(AssertionError("실제 발송")))
    assert fn.send_text_safely("글") is False and "dry-run" in capsys.readouterr().out
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "1")
    assert fn.send_text_safely("글", force=True) is False  # 단체방과 같은 chat id면 보내지 않는다


# ── 편지·시트 ────────────────────────────────────────────────────────────────


def _penny_extra(cfg, name, cum, n=20, trades=()):
    daily = pd.DataFrame({"value": [70000.0, 70000 * (1 + cum / 100)], "cum_return_pct": [0.0, cum], "drawdown_pct": [0.0, 0.0]},
                         index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
    return {"name": name, "kind": "penny" if "수렴" in name else "penny_random", "start": pd.Timestamp("2026-10-20"),
            "daily": daily, "trades": list(trades), "positions": {f"T{i}": {} for i in range(n)},
            "summary": {"cum_return_pct": cum, "value": float(daily["value"].iat[-1]), "mdd_pct": 0.0}}


def test_letter_lines_six_to_ten_and_shadow_pp_no_names(cfg):
    from tests.test_letter import _books

    tr = [{"side": "매수", "ticker": "PNYX", "fill_date": pd.Timestamp("2026-10-16")}]
    off = copy.deepcopy(cfg)
    off["virtual_books"]["penny"]["enabled"] = False
    base_books = _books(off)
    bench = pd.DataFrame({"value": [100.0, 101.0]}, index=pd.to_datetime(["2026-10-20", "2026-10-21"]))
    extra = [_penny_extra(cfg, "동전주 거래량 수렴 따라하기", 3.0, trades=tr), _penny_extra(cfg, "무작위 동전주 20", 0.5, trades=tr)]
    pen = letter.letter_books(cfg, "2026-10-16", [], {}, extra, bench)
    books = base_books + [b for b in pen if b["family"] in ("penny", "shadow")]
    for ed in (letter.PERSONAL, letter.GROUP):
        lines = letter.book_lines(books, cfg, ed)
        picks = next(x for x in lines if x.startswith("종목 고르기(6~10)"))
        assert picks.endswith("동전주 거래량 수렴 전략 +2.0%p")  # QQQ(시작일부터 +1%) 대비
        assert "비교: 무작위 동전주 20(비교용) −0.5%p (10번 판정용, QQQ 대비)" in lines
        assert "PNYX" not in " ".join(lines)
    _, rows = letter.table_rows(books, cfg, letter.PERSONAL)
    assert [r[-1] for r in rows if r[0].startswith(("동전주", "무작위"))] == ["20종목", "20종목"]  # 개인판도 종목 수만
    assert "PNYX" in letter.trade_detail_line(books) and letter.trade_detail_line(books).count("PNYX") == 1  # 그림자 거래는 빼고
    info = letter.check_info(books, "2026-10-21", "2026-10-21", 0, {}, ["동전주 선정 못함(목록 못 받음)"])
    assert "무작위 동전주 20(비교용)" not in info["recorded"]
    assert letter.check_line(info).endswith(" · 데이터 지연 없음 · 동전주 선정 못함(목록 못 받음)")


def test_waiting_penny_shows_start_wait(cfg):
    from engine import daily

    w = daily._waiting_books(cfg, pd.Timestamp("2026-10-21"), set())
    assert w["동전주 거래량 수렴 따라하기"] == pd.Timestamp("2026-10-21") and "무작위 동전주 20" in w


def test_sheet_summary_has_penny_rows(cfg):
    from core.virtual_books import BookSpec, LedgerResult

    idx = pd.to_datetime(["2026-10-20", "2026-10-21"])
    q = pd.DataFrame({"value": [100.0, 101], "cum_return_pct": [0.0, 1.0], "drawdown_pct": [0.0, 0.0], "holding": ["QQQ"] * 2},
                     index=idx)
    led = LedgerResult(start=idx[0], base_date=idx[-1], books={"QQQ 그냥 보유": type("B", (), {"daily": q, "trades": []})()},
                       summaries=[], regime=pd.DataFrame({"close": [1.0, 1], "sma": [1.0, 1], "above": [1.0, 1]}, index=idx),
                       switches=[])
    ex = []
    for name, kind in (("동전주 거래량 수렴 따라하기", "penny"), ("무작위 동전주 20", "penny_random")):
        x = _penny_extra(cfg, name, 1.0)
        x["kind"] = kind
        x["summary"] = {"book": name, "start": idx[0], "end": idx[-1], "initial_usd": 70000.0, "value": 70700.0, "cum_return_pct": 1.0,
                        "cagr_pct": None, "mdd_pct": 0.0, "mdd_date": idx[0], "switches": "-", "holding": "20종목", "tax_cum": 0.0,
                        "value_after_tax_liq": 70600.0, "rebalances": 1, "holdings_list": "A, B", "rules": "규칙"}
        x["meta"] = {"start_krw": 1e8, "start_fx": 1400.0, "fx_date": "2026-10-20", "initial_usd": 70000.0}
        ex.append(x)
    meta = {"start_krw": 1e8, "start_fx": 1340.0, "fx_date": "2026-10-14", "initial_usd": 74000.0, "updated_at": "x",
            "min_days_for_annualized": 60, "notes": []}
    t = vbtext.rename_tables({"s": vbtext.summary_table(led, meta, ex),
                              "d": vbtext.daily_table(led, [BookSpec("QQQ 그냥 보유", "hold", "QQQ")], "QQQ 그냥 보유", ex)},
                             vbtext.display_map(cfg))
    firsts = [r[0] for r in t["s"] if r]
    assert "동전주 거래량 수렴 전략" in firsts and "무작위 동전주 20(비교용)" in firsts
    assert "config 고정값(penny)" in firsts and "규칙(동전주 거래량 수렴)" in firsts
    assert "동전주 거래량 수렴 전략 QQQ 대비(%p)" in t["d"][0] and "무작위 동전주 20(비교용) 누적(%)" in t["d"][0]


def test_params_match_prereg(cfg):
    p, s = pe.penny_params(cfg)
    assert (p.slots, p.min_price, p.max_price, p.min_dollar_volume, p.min_history, p.max_ratio, p.max_width) == (
        20, 1.0, 5.0, 1_000_000.0, 120, 0.5, 0.25)
    assert (p.half_gain, p.gone_days, p.commission_pct, p.slippage_pct, p.fx_spread_pct, p.seed, p.cash_ticker) == (
        1.0, 5, 0.07, 0.5, 0.1, 20261021, "BIL")
    assert s.name == "무작위 동전주 20" and replace(s, name=p.name) == p


def test_engine_judgement_after_twelve_months_once(cfg, tmp_path):
    days = pd.bdate_range("2026-03-02", "2027-10-29")
    fr = {f"S{i:02d}": _squeezer(late_vol=400_000 + i * 1000, days=days) for i in range(25)}
    fr.update({f"R{i:02d}": _frame(3.0, 1_000_000, days) for i in range(30)})
    for t in list(fr)[:5]:  # 조금 오르는 종목
        fr[t]["close"] = np.linspace(3.0, 3.6, len(days))
    conn = vbstore.connect(tmp_path / "vb.db")

    def ctx(base):
        idx = days[days <= pd.Timestamp(base)]
        return {"base_date": pd.Timestamp(base), "closes": pd.DataFrame({"QQQ": np.linspace(500, 600, len(idx))}, index=idx),
                "divs": pd.DataFrame({"QQQ": 0.0}, index=idx)}

    fns = _fns(fr, [])
    fns["history_fn"] = lambda t, s: (lambda f: f.assign(dividend=0.0, split=0.0))(
        _frame(91.5, 1, days) if t == "BIL" else fr[t].copy()).loc[pd.Timestamp(s):]
    out = pe.run_penny(cfg, conn, ctx("2026-10-20"), dry_run=False, **fns)
    assert "판정 예정 2027-10-20 이후 첫 거래일" in out["extra"][0]["summary"]["rules"]
    out = pe.run_penny(cfg, conn, ctx("2027-10-21"), dry_run=False, **fns)
    assert out["errors"] == []
    r = __import__("json").loads(vbstore.get_meta(conn, pe.JUDGE_KEY))
    assert r["date"] == "2027-10-20" and 8 < r["qqq_cum"] < 11.4 and r["pass"] is False  # QQQ 약 +11% 세전 → 세후 약 +9%
    assert any("12개월 판정 2027-10-20" in w for w in out["warnings"])
    out = pe.run_penny(cfg, conn, ctx("2027-10-22"), dry_run=False, **fns)
    assert not any("12개월 판정" in w for w in out["warnings"]) and "판정 2027-10-20: 불통과" in out["extra"][0]["summary"]["rules"]
