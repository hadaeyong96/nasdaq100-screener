"""내부자 집단 매수 — 차단·누수·장부·판정·전체 실행 테스트 (docs/prereg/insider_cluster.md). 네트워크 없이 합성 데이터만."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from core.tax import capital_gains_tax
from scripts import smallmid_prices_check as spc
from scripts.insider_cluster import book as bk
from scripts.insider_cluster import data as dm
from scripts.insider_cluster import run as rn
from scripts.insider_cluster import signals as sg
from scripts.insider_cluster.gate import CUTOFF, CutoffError, assert_cutoff, assert_dates, gate, gate_filings

DAYS = sg.nyse_days("2015-01-01", "2021-12-31")


def sec_zip(rows_sub, rows_own, rows_tr) -> bytes:
    def tsv(cols, rows):
        return "\t".join(cols) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", tsv(["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"], rows_sub))
        z.writestr("REPORTINGOWNER.tsv", tsv(["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"], rows_own))
        z.writestr("NONDERIV_TRANS.tsv", tsv(["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_ACQUIRED_DISP_CD", "TRANS_SHARES", "TRANS_PRICEPERSHARE"], rows_tr))
    return buf.getvalue()


def tables(items):
    """items: (accession, 접수일, 문서, 회사, 소유자 [(cik, 관계)], 거래 [(날짜, 코드, A/D)], period)."""
    f, o, t = [], [], []
    for acc, filed, doc, cik, owners, trs, *per in items:
        f.append({"accession": acc, "filing_date": pd.Timestamp(filed), "doc_type": doc, "issuer_cik": cik, "period": per[0] if per else filed})
        o += [{"accession": acc, "owner_cik": c, "relationship": rel} for c, rel in owners]
        t += [{"accession": acc, "trans_date": pd.Timestamp(d), "code": code, "ad": ad, "shares": 100.0, "price": 10.0} for d, code, ad in trs]
    return pd.DataFrame(f), pd.DataFrame(o), pd.DataFrame(t, columns=["accession", "trans_date", "code", "ad", "shares", "price"])


def sigs(items, min_people=3):
    pur, stats = sg.eligible_purchases(*tables(items))
    return sg.cluster_signals(pur, DAYS, min_people), stats


D = "Director"
O = "Officer"


def buy(acc, filed, cik, owner, rel=D, date=None, code="P", doc="4", ad="A", per=None):
    return (acc, filed, doc, cik, [(owner, rel)], [(date or filed, code, ad)], per or (date or filed))


THREE = [buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12, O), buy("a3", "2018-03-08", 1, 13)]


# ── 1. 2022 행 자르기 ─────────────────────────────────────────────────────────


def test_gate_drops_2022_keeps_2021_12_31():
    s = pd.Series(1.0, index=pd.to_datetime(["2021-12-30", "2021-12-31", "2022-01-03"]))
    assert gate(s).index.max() == CUTOFF
    assert pd.Timestamp("2022-01-03") not in gate(s).index
    f = pd.DataFrame({"filing_date": pd.to_datetime(["2021-12-31", "2022-01-03"]), "a": [1, 2]})
    assert list(gate_filings(f)["a"]) == [1]


def test_parse_sec_zip_cuts_2022_filings_before_saving():
    raw = sec_zip([["x1", "31-DEC-2021", "30-DEC-2021", "4", "5", "ABC"], ["x2", "03-JAN-2022", "30-DEC-2021", "4", "5", "ABC"], ["x3", "31-DEC-2021", "30-DEC-2021", "3", "5", "ABC"]],
                  [["x1", "9", D], ["x2", "9", D], ["x3", "9", D]],
                  [["x1", "30-DEC-2021", "P", "A", "10", "5"], ["x2", "30-DEC-2021", "P", "A", "10", "5"], ["x1", "30-DEC-2021", "S", "D", "10", "5"]])
    f, o, t = dm.parse_sec_zip(raw)
    assert list(f["accession"]) == ["x1"] and list(o["accession"]) == ["x1"] and list(t["accession"]) == ["x1"]
    assert list(t["code"]) == ["P"] and list(f["symbol"]) == ["ABC"]


def test_sec_fetch_refuses_2022_quarter(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "get_user_agent", lambda: "테스트")
    with pytest.raises(ValueError):
        dm.fetch_sec_all(tmp_path, ["2022q1"], get=lambda *a, **k: None)
    assert all(q <= "2021q4" for q in dm.QUARTERS) and dm.QUARTERS[0] == "2015q4" and dm.QUARTERS[-1] == "2021q4"


# ── 2. 계산 함수 2022 예외 ────────────────────────────────────────────────────


def test_compute_raises_on_2022():
    with pytest.raises(CutoffError):
        assert_cutoff(pd.Series(1.0, index=pd.to_datetime(["2022-01-03"])))
    with pytest.raises(CutoffError):
        assert_dates(pd.Series(pd.to_datetime(["2022-01-03"])))
    late = [buy("z", "2022-01-03", 1, 11, date="2021-12-30")]
    with pytest.raises(CutoffError):
        sg.eligible_purchases(*tables(late))
    d = pd.bdate_range("2021-12-01", "2022-01-05")
    px = np.ones((len(d), 1))
    with pytest.raises(CutoffError):
        bk.Market(d, px[:, 0], px[:, 0], px, np.array([len(d) - 1]), [1], np.zeros_like(px, bool), np.zeros(len(d)))
    with pytest.raises(CutoffError):
        rn.build_market({}, pd.Series(1.0, index=d), pd.Series(1.0, index=d), dm.Zone({}, {}))


# ── 3. 시점 ───────────────────────────────────────────────────────────────────


def test_use_and_exec_positions():
    t = pd.Timestamp("2018-03-08")  # 목요일
    use, ex = sg.use_and_exec(t, DAYS)
    assert DAYS[use] == pd.Timestamp("2018-03-09") and DAYS[ex] == pd.Timestamp("2018-03-12")
    # 금요일 휴장(Good Friday 2018-03-30) 접수 → 다음 거래일이 t+1
    use, ex = sg.use_and_exec(pd.Timestamp("2018-03-30"), DAYS)
    assert DAYS[use] == pd.Timestamp("2018-04-02") and DAYS[ex] == pd.Timestamp("2018-04-03")
    assert sg.use_and_exec(pd.Timestamp("2021-12-31"), DAYS) == (None, None)
    use, ex = sg.use_and_exec(pd.Timestamp("2021-12-30"), DAYS)
    assert DAYS[use] == CUTOFF and ex is None


def test_signal_uses_third_filing_date():
    s, _ = sigs(THREE)
    assert list(s["confirm"]) == [pd.Timestamp("2018-03-08")] and s["people"].iloc[0] == 3


def test_third_filing_late_confirm_date():
    items = [buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", "2018-04-20", 1, 13, date="2018-03-07")]
    s, _ = sigs(items)
    assert list(s["confirm"]) == [pd.Timestamp("2018-04-20")]


def test_nyse_calendar_counts_and_holidays():
    d = sg.nyse_days("2017-01-01", "2021-12-31")
    assert {y: int((d.year == y).sum()) for y in range(2017, 2022)} == {2017: 251, 2018: 251, 2019: 252, 2020: 253, 2021: 252}
    for closed in ("2017-01-02", "2018-12-05", "2020-07-03", "2021-04-02", "2021-12-24", "2021-01-18"):
        assert pd.Timestamp(closed) not in d
    assert pd.Timestamp("2021-12-31") in d  # 2022-01-01 토요일이지만 대체 휴장 없음


# ── 4. 사람·거래 필터 ─────────────────────────────────────────────────────────


def test_same_person_counts_once():
    s, _ = sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 11), buy("a3", "2018-03-07", 1, 11), buy("a4", "2018-03-08", 1, 12)])
    assert s.empty


def test_ten_percent_only_excluded():
    assert sg.is_insider("Director,TenPercentOwner") and sg.is_insider("Officer")
    assert not sg.is_insider("TenPercentOwner") and not sg.is_insider("Other") and not sg.is_insider("TenPercentOwnerOther")
    s, _ = sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", "2018-03-07", 1, 13, "TenPercentOwner")])
    assert s.empty
    s, _ = sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", "2018-03-07", 1, 13, "Director,TenPercentOwner")])
    assert len(s) == 1


def test_only_code_p_acquired():
    s, _ = sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", "2018-03-07", 1, 13, code="S", ad="D"), buy("a4", "2018-03-08", 1, 14, code="A"), buy("a5", "2018-03-08", 1, 15, code="M")])
    assert s.empty
    s, _ = sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", "2018-03-07", 1, 13, ad="D")])
    assert s.empty


def test_form5_not_used_and_bad_date_dropped():
    items = [buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", "2018-03-07", 1, 13, doc="5")]
    assert sigs(items)[0].empty
    s, stats = sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", "2018-03-07", 1, 13, date="2028-03-07")])
    assert s.empty and stats["bad_trans_date"] == 1


def test_joint_filing_counts_one_person():
    joint = ("a3", "2018-03-07", "4", 1, [(13, D), (14, D)], [("2018-03-07", "P", "A")], "2018-03-07")
    assert sigs([buy("a1", "2018-03-05", 1, 11), joint])[0].empty
    assert sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), joint])[0]["people"].iloc[0] == 3


def test_window_30_trading_days():
    first = pd.Timestamp("2018-03-05")
    i = int(np.searchsorted(DAYS.values, np.datetime64(first)))
    inside, outside = DAYS[i + 29].strftime("%Y-%m-%d"), DAYS[i + 30].strftime("%Y-%m-%d")
    assert len(sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", inside, 1, 13)])[0]) == 1
    assert sigs([buy("a1", "2018-03-05", 1, 11), buy("a2", "2018-03-06", 1, 12), buy("a3", outside, 1, 13)])[0].empty


def test_amendment_replaces_original_from_its_date():
    # 원 신고 P → 4/A가 코드 정정(P 사라짐): 정정 뒤에는 그 매수가 세어지지 않는다
    orig = buy("a1", "2018-03-05", 1, 11)
    fix = ("a1f", "2018-03-06", "4/A", 1, [(11, D)], [("2018-03-05", "S", "D")], "2018-03-05")
    assert sigs([orig, fix, buy("a2", "2018-03-07", 1, 12), buy("a3", "2018-03-08", 1, 13)])[0].empty
    # 정정이 신호 뒤에 와도 이미 난 신호는 바뀌지 않는다(미래 정보 금지)
    late = ("a1g", "2018-03-20", "4/A", 1, [(11, D)], [("2018-03-05", "S", "D")], "2018-03-05")
    s, _ = sigs([orig, buy("a2", "2018-03-07", 1, 12), buy("a3", "2018-03-08", 1, 13), late])
    assert list(s["confirm"]) == [pd.Timestamp("2018-03-08")]
    # 같은 내용 정정은 1건으로 남아 사람 수를 늘리지 않는다
    same = ("a1h", "2018-03-06", "4/A", 1, [(11, D)], [("2018-03-05", "P", "A")], "2018-03-05")
    assert sigs([orig, same, buy("a2", "2018-03-07", 1, 12)])[0].empty


def test_new_cluster_needs_new_purchases():
    second = [buy("b1", "2018-09-05", 1, 11), buy("b2", "2018-09-06", 1, 12)]
    assert len(sigs(THREE + second)[0]) == 1  # 소진 뒤 2명뿐
    assert len(sigs(THREE + second + [buy("b3", "2018-09-07", 1, 13)])[0]) == 2
    # 소진된 거래를 같은 내용으로 정정해도 되살아나지 않는다
    again = ("a1r", "2018-03-20", "4/A", 1, [(11, D)], [("2018-03-05", "P", "A")], "2018-03-05")
    assert len(sigs(THREE + [again, buy("c1", "2018-03-21", 1, 12), buy("c2", "2018-03-22", 1, 13)])[0]) == 1


def test_preheat_consumes_but_no_signal_before_2017():
    pre = [buy("a1", "2016-12-05", 1, 11), buy("a2", "2016-12-06", 1, 12), buy("a3", "2016-12-07", 1, 13)]
    assert sigs(pre)[0].empty
    # 2016-12 신호로 소진 → 2017 초 2명 추가만으로는 신호 없음
    assert sigs(pre + [buy("b1", "2017-01-04", 1, 14), buy("b2", "2017-01-05", 1, 15)])[0].empty
    # 2016-12 매수 2명 + 2017 접수 1명 → 2017 신호
    s = sigs([buy("a1", "2016-12-28", 1, 11), buy("a2", "2016-12-29", 1, 12), buy("a3", "2017-01-04", 1, 13)])[0]
    assert list(s["confirm"]) == [pd.Timestamp("2017-01-04")]


# ── 5. 명단 ───────────────────────────────────────────────────────────────────


def zone_for(ciks, first="2015-01-01", last="2021-12-31", big=None, n=5):
    return dm.Zone({c: [(pd.Timestamp(first), pd.Timestamp(last), "sp600", n)] for c in ciks}, big or {})


def test_zone_filter_rejects_non_members():
    s = pd.DataFrame({"cik": [1, 2, 3, 4], "confirm": pd.to_datetime(["2018-03-08"] * 4), "people": 3, "amount_usd": 1.0, "first_trade": pd.NaT, "last_trade": pd.NaT})
    zone = dm.Zone({1: [(pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31"), "sp600", 5)],
                    2: [(pd.Timestamp("2015-01-01"), pd.Timestamp("2018-01-01"), "sp600", 5)],
                    3: [(pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31"), "sp400", 1)],
                    4: [(pd.Timestamp("2015-01-01"), pd.Timestamp("2021-12-31"), "sp400", 5)]},
                   {4: (pd.Timestamp("2018-05-01"), pd.Timestamp("2019-01-01"))})
    assert list(rn.zone_signals(s, zone, DAYS)["cik"]) == [1]  # 2 편출, 3 보고 1번, 4 앞뒤 92일 대형


def test_member_flags_same_as_contains():
    zone = dm.Zone({1: [(pd.Timestamp("2017-02-01"), pd.Timestamp("2019-01-01"), "sp600", 5), (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-01"), "sp400", 1)]},
                   {1: (pd.Timestamp("2018-06-01"), pd.Timestamp("2018-07-01"))})
    d = sg.nyse_days("2017-01-03", "2021-12-31")
    v = dm.member_flags(1, zone, d)
    assert list(v) == [zone.contains(1, x) for x in d]


def test_liquid_flags_same_as_liquid_before():
    idx = sg.nyse_days("2019-01-02", "2019-06-28")
    rng = np.random.default_rng(1)
    px = pd.DataFrame({"close": 6.0 + rng.normal(0, 1.5, len(idx)), "volume": rng.integers(100_000, 2_000_000, len(idx)).astype(float)}, index=idx).drop(idx[40])
    v = dm.liquid_flags(px, idx)
    for k, d in enumerate(idx):
        expect = spc.liquid_before(px, d)[0] if d in px.index else False
        assert bool(v[k]) == bool(expect), d


def test_entry_requires_zone_and_liquidity():
    m = market(n_stocks=3, eligible_cols=[0])
    z = pd.DataFrame({"cik": [100, 101, 999], "confirm": [m.days[10]] * 3, "people": 3, "amount_usd": 1.0, "first_trade": pd.NaT, "last_trade": pd.NaT, "exec_day": [m.days[12]] * 3})
    t, entries, und = rn.classify(z, m)
    assert list(t["status"]) == ["후보", "유동성 미달", "판정 불가"]
    assert [e.col for e in entries] == [0] and und[0]["cik"] == 999


# ── 6. 누수 ───────────────────────────────────────────────────────────────────


def test_no_lookahead_in_signals():
    rng = np.random.default_rng(3)
    items = []
    k = 0
    for f in sg.nyse_days("2017-01-03", "2021-12-30")[::3]:
        k += 1
        items.append(buy(f"r{k}", f.strftime("%Y-%m-%d"), int(rng.integers(1, 6)), int(rng.integers(10, 40))))
    full = sigs(items)[0]
    t = pd.Timestamp("2019-06-28")
    assert len(full[full["confirm"] <= t]) > 0
    early = [x for x in items if pd.Timestamp(x[1]) <= t]
    cut = sigs(early)[0]
    pd.testing.assert_frame_equal(full[full["confirm"] <= t].reset_index(drop=True), cut.reset_index(drop=True))
    changed = early + [buy("late1", "2020-01-06", 1, 91), buy("late2", "2020-01-07", 1, 92), buy("late3", "2020-01-08", 1, 93)]
    pd.testing.assert_frame_equal(sigs(changed)[0].query("confirm <= @t").reset_index(drop=True), cut.reset_index(drop=True))


def test_liquidity_and_basket_no_lookahead():
    m1 = market(n_stocks=4, eligible_cols=[0, 1, 2, 3])
    px2 = m1.px.copy()
    px2[100:, :] *= 3.0
    a, b = bk.basket_returns(m1.px, m1.eligible), bk.basket_returns(px2, m1.eligible)
    assert (a[:100] == b[:100]).all()
    idx = sg.nyse_days("2019-01-02", "2019-12-31")
    p = pd.DataFrame({"close": 10.0, "volume": 1e6}, index=idx)
    q = p.copy()
    q.loc[idx[60]:, "close"] = 1.0
    va, vb = dm.liquid_flags(p, idx), dm.liquid_flags(q, idx)
    assert (va[:61] == vb[:61]).all()


# ── 장부용 합성 시장 ──────────────────────────────────────────────────────────


def market(n_stocks=3, start="2017-01-03", end="2021-12-31", eligible_cols=None, drift=0.0, fx=1000.0, seed=None, last=None):
    days = sg.nyse_days(start, end)
    T = len(days)
    if seed is None:
        px = np.tile(100.0 * (1.0 + drift) ** np.arange(T)[:, None], (1, n_stocks))
    else:
        rng = np.random.default_rng(seed)
        px = 50.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.02, (T, n_stocks)), axis=0)
    last_idx = np.full(n_stocks, T - 1)
    if last:
        for c, i in last.items():
            px[i + 1:, c] = np.nan
            last_idx[c] = i
    el = np.zeros((T, n_stocks), bool)
    el[:, eligible_cols if eligible_cols is not None else list(range(n_stocks))] = True
    return bk.Market(days, np.full(T, 100.0), np.full(T, fx), px, last_idx, [100 + i for i in range(n_stocks)], el, bk.basket_returns(px, el))


# ── 7. 슬롯·청산·끝날·판정 불가 ───────────────────────────────────────────────


def test_ten_position_cap_and_skip_count():
    m = market(n_stocks=12)
    es = [bk.Entry(10, 12, c) for c in range(12)]
    r = bk.run_strategy(es, m, want_path=True)
    assert r["counts"]["skipped_full"] == 2
    assert len(r["trades"]) == 10
    assert r["sat_weight"][12] == pytest.approx(0.2, abs=0.002)


def test_hold_252_days_then_back_to_qqq():
    m = market(n_stocks=1)
    r = bk.run_strategy([bk.Entry(10, 12, 0)], m, want_path=True)
    t = r["trades"][0]
    assert t["exec_idx"] == 12 and t["exit_idx"] == 12 + 252 and t["reason"] == "252거래일"
    assert r["sat_weight"][12 + 251] > 0.0 and r["sat_weight"][12 + 252] == 0.0


def test_delisted_sells_at_last_close():
    m = market(n_stocks=1, last={0: 50})
    r = bk.run_strategy([bk.Entry(10, 12, 0)], m, want_path=True)
    t = r["trades"][0]
    assert t["exit_idx"] == 50 and t["reason"] == "가격 끊김" and r["counts"]["delisted"] == 1
    assert r["sat_weight"][50] == 0.0


def test_exec_on_last_price_day_not_bought():
    m = market(n_stocks=1, last={0: 12})
    r = bk.run_strategy([bk.Entry(10, 12, 0)], m, want_path=True)
    assert r["counts"]["no_price_at_exec"] == 1 and r["trades"] == [] and np.isfinite(r["end_krw"])
    m2 = market(n_stocks=1, last={0: 11})
    assert bk.run_strategy([bk.Entry(10, 12, 0)], m2)["counts"]["no_price_at_exec"] == 1


def test_forced_close_on_last_day():
    m = market(n_stocks=1)
    T = len(m.days)
    r = bk.run_strategy([bk.Entry(T - 12, T - 10, 0)], m)
    assert r["counts"]["forced_end"] == 1 and r["trades"][0]["reason"] == "끝날 강제 청산" and r["trades"][0]["exit_idx"] == T - 1
    # 252거래일이 끝날에 차도 강제 청산
    r2 = bk.run_strategy([bk.Entry(T - 255, T - 253, 0)], m)
    assert r2["trades"][0]["exit_idx"] == T - 1 and r2["counts"]["forced_end"] == 1


def test_holding_signal_ignored_and_reentry_after_exit():
    m = market(n_stocks=1)
    es = [bk.Entry(10, 12, 0), bk.Entry(20, 22, 0), bk.Entry(10 + 2 + 251, 10 + 2 + 253, 0), bk.Entry(12 + 252, 12 + 254, 0)]
    r = bk.run_strategy(es, m)
    assert r["counts"]["ignored_holding"] == 2  # 보유 중 + 확인일에 아직 보유(체결 전 청산)
    assert len(r["trades"]) == 2


def test_undecided_main_stays_qqq_conservative_buys():
    m = market(n_stocks=1, eligible_cols=[0])
    z = pd.DataFrame({"cik": [100, 999], "confirm": [m.days[10], m.days[20]], "people": 3, "amount_usd": 1.0, "first_trade": pd.NaT, "last_trade": pd.NaT, "exec_day": [m.days[12], m.days[22]]})
    t, entries, und = rn.classify(z, m)
    assert [u["cik"] for u in und] == [999]
    main = bk.run_strategy(entries, m)
    assert len(main["trades"]) == 1
    m_c, syn = rn.with_synthetic(m, und, -0.30)
    cons = bk.run_strategy(rn.order(entries + syn), m_c)
    assert len(cons["trades"]) == 2
    tr = [x for x in cons["trades"] if x["key"] == "판정불가0"][0]
    assert tr["ret"] == pytest.approx(-0.30) and tr["exit_idx"] == und[0]["exec_idx"] + 252
    assert cons["end_krw"] < main["end_krw"]


def test_observed_p10_excludes_forced():
    tr = [{"ret": float(x), "reason": "252거래일"} for x in range(11)] + [{"ret": -99.0, "reason": "끝날 강제 청산"}]
    assert rn.observed_p10(tr) == pytest.approx(1.0)
    assert rn.observed_p10([]) == 0.0


def test_judge_rules():
    a = {"cagr": 0.2, "mdd": -0.3}
    lo = {"cagr": 0.1, "mdd": -0.28}
    v = bk.judge(a, lo, lo, 0.01, a, lo, a, lo, a, lo, 30)
    assert v["result"] == "PASS"
    assert bk.judge(a, lo, lo, 0.011, a, lo, a, lo, a, lo, 30)["c3"] is False
    assert bk.judge(a, lo, lo, 0.01, a, lo, a, lo, a, lo, 29)["result"] == "FAIL"
    assert bk.judge(a, lo, lo, 0.01, lo, a, a, lo, a, lo, 30)["c5"] is False
    assert bk.judge(a, lo, lo, 0.01, a, lo, a, lo, lo, a, 30)["c6"] is False
    assert bk.judge(a, {"cagr": 0.1, "mdd": -0.27}, lo, 0.01, a, lo, a, lo, a, lo, 30)["c4"] is False  # 정확히 3%p 깊음
    assert bk.mdd_ok(-0.30, -0.28) and not bk.mdd_ok(-0.31, -0.28)


# ── 8. seed·세금 ──────────────────────────────────────────────────────────────


def test_same_seed_same_random():
    m = market(n_stocks=30, seed=5)
    es = [bk.Entry(10 + 40 * i, 12 + 40 * i, i) for i in range(8)]
    a, b = bk.random_control(es, {}, m, 15), bk.random_control(es, {}, m, 15)
    assert (a == b).all() and bk.SEED == 20261107
    assert not (bk.random_control(es, {}, m, 15, seed=1) == a).all()
    p, r = bk.p_value(float(a.max()) + 1.0, a)
    assert r == 0 and p == pytest.approx(1 / 16)
    assert bk.p_value(float(a.min()) - 1.0, a)[0] == pytest.approx(1.0)


def test_random_excludes_actual_signal_stock():
    m = market(n_stocks=3, seed=5)
    es = [bk.Entry(10, 12, 0)]
    seen = set()
    orig = bk.run_strategy

    def spy(entries, mm, want_path=False):
        seen.update(e.col for e in entries)
        return orig(entries, mm, want_path)

    bk.run_strategy = spy
    try:
        bk.random_control(es, {10: {0}}, m, 30)
    finally:
        bk.run_strategy = orig
    assert seen == {1, 2}


def test_tax_qqq_only_flat_and_gain():
    m = market(n_stocks=1, end="2017-12-29")
    r = bk.run_strategy([], m)
    assert r["end_krw"] == pytest.approx(1e8 * 0.999)  # 비용만, 세금 없음
    T = len(m.days)
    m2 = bk.Market(m.days, 100.0 * np.linspace(1.0, 1.1, T), m.fx, m.px, m.last_idx, m.ciks, m.eligible, m.basket)
    r2 = bk.run_strategy([], m2)
    krw = 1.1e8
    cost = krw * 0.001
    assert r2["end_krw"] == pytest.approx(krw - cost - capital_gains_tax(1e7 - cost))
    assert capital_gains_tax(1e7 - cost) == pytest.approx((1e7 - cost - 2_500_000) * 0.22)


def test_tax_realized_paid_first_may_next_year():
    days = sg.nyse_days("2017-01-03", "2018-06-29")
    T = len(days)
    ex = 12
    exit_i = ex + 252
    stock = np.full(T, 100.0)
    stock[ex + 1:] = 200.0  # 체결 다음 날 2배
    m = bk.Market(days, np.full(T, 100.0), np.full(T, 1000.0), stock[:, None], np.array([T - 1]), [100], np.ones((T, 1), bool), np.zeros(T))
    assert days[exit_i].year == 2018  # 252거래일 뒤는 다음 해 초
    r = bk.run_strategy([bk.Entry(10, ex, 0)], m)
    c = 0.001
    x = 0.02 * 1e5  # 달러
    sell = c * x
    buy = (x - sell) / (1 + c)
    cost_in = sell + c * buy
    gain_2017 = -cost_in * 1000.0  # 2017 실현 = 진입 비용뿐
    assert capital_gains_tax(gain_2017) == 0.0
    sale = buy * 2.0
    sc = c * sale
    rebuy = (sale - sc) / (1 + c)
    gain_2018 = (sale - sc) * 1000.0 - buy * 1000.0 - c * rebuy * 1000.0
    # 2018 실현은 2019년 납부라 장부 끝(2018-06)에는 그해 전량 매도 세금으로 함께 계산
    nav = (1e5 - x) + rebuy
    krw = nav * 1000.0
    liq = krw * c
    assert r["end_krw"] == pytest.approx(krw - liq - capital_gains_tax(gain_2018 - liq))
    assert r["counts"]["taxes_paid_krw"] == 0.0


def test_tax_paid_in_may_reduces_nav():
    days = sg.nyse_days("2017-01-03", "2018-06-29")
    T = len(days)
    ex = 12
    exit_i = 100
    stock = np.full(T, 100.0)
    stock[ex + 1:] = 400.0  # 이익이 250만 원 공제보다 크도록 4배
    m = bk.Market(days, np.full(T, 100.0), np.full(T, 1000.0), stock[:, None], np.array([exit_i]), [100], np.ones((T, 1), bool), np.zeros(T))
    r = bk.run_strategy([bk.Entry(10, ex, 0)], m)
    assert r["trades"][0]["reason"] == "가격 끊김"
    c = 0.001
    x = 0.02 * 1e5
    buy = (x - c * x) / (1 + c)
    sale = buy * 4.0
    rebuy = (sale - c * sale) / (1 + c)
    gain = (sale - c * sale) * 1000.0 - buy * 1000.0 - (c * x + c * buy) * 1000.0 - c * rebuy * 1000.0
    tax = capital_gains_tax(gain)
    assert tax > 0.0 and r["counts"]["taxes_paid_krw"] == pytest.approx(tax)
    nav = (1e5 - x) + rebuy - tax / 1000.0 / (1 - c)
    krw = nav * 1000.0
    liq = krw * c
    # 납부 매도 손익은 0(원가=시가), 비용만 그해 실현
    tax_cost = c * tax / 1000.0 / (1 - c) * 1000.0
    assert r["end_krw"] == pytest.approx(krw - liq - capital_gains_tax(-tax_cost - liq))


def test_krw_gain_uses_fx():
    days = sg.nyse_days("2017-01-03", "2017-03-31")
    T = len(days)
    fx = np.full(T, 1000.0)
    fx[-1] = 1200.0
    m = bk.Market(days, np.full(T, 100.0), fx, np.full((T, 1), 100.0), np.array([T - 1]), [1], np.ones((T, 1), bool), np.zeros(T))
    krw = 1.2e8
    assert bk.run_strategy([], m)["end_krw"] == pytest.approx(krw - krw * 0.001 - capital_gains_tax(2e7 - krw * 0.001))


def test_basket_zero_weight_equals_qqq_and_exposure_matters():
    m = market(n_stocks=3, seed=9)
    q = bk.run_strategy([], m)
    T = len(m.days)
    assert bk.run_basket(np.zeros(T), np.zeros(T), m)["end_krw"] == pytest.approx(q["end_krw"])
    up = bk.Market(m.days, m.qqq, m.fx, m.px, m.last_idx, m.ciks, m.eligible, np.full(T, 0.001))
    assert bk.run_basket(np.full(T, 0.2), np.zeros(T), up)["end_krw"] > q["end_krw"]


def test_cost_charged_in_basket_like_strategy():
    m = market(n_stocks=1)
    T = len(m.days)
    frac = np.zeros(T)
    frac[10] = 0.001
    r = bk.run_basket(np.zeros(T), frac, m)
    assert r["end_krw"] == pytest.approx(1e8 * 0.999 * 0.999)


# ── 메타데이터·가격 파일 ──────────────────────────────────────────────────────


def test_meta_covers_and_name_match():
    c, e = pd.Timestamp("2018-03-08"), pd.Timestamp("2018-03-12")
    good = {"exists": True, "start": "2005-01-01", "end": "2019-01-01", "name": "ACME WIDGETS INC"}
    assert dm.meta_covers(good, "Acme Widgets Corp.;Acme", c, e)
    assert not dm.meta_covers({**good, "name": "ZZZ TOTALLY OTHER"}, "Acme Widgets Corp.", c, e)
    assert not dm.meta_covers({**good, "end": "2018-03-09"}, "Acme Widgets Corp.", c, e)
    assert not dm.meta_covers({**good, "start": "2018-03-09"}, "Acme Widgets Corp.", c, e)
    assert not dm.meta_covers({"exists": False}, "Acme Widgets Corp.", c, e)


def test_covered_by_cache():
    c, e = pd.Timestamp("2018-03-08"), pd.Timestamp("2018-03-12")
    row = {"same_company": True, "price_first": "2010-01-01", "price_last": "2021-12-31"}
    assert dm.covered_by_cache(row, c, e)
    assert not dm.covered_by_cache({**row, "same_company": False}, c, e)
    assert not dm.covered_by_cache({**row, "price_last": "2018-03-09"}, c, e)
    assert not dm.covered_by_cache({**row, "price_first": "2018-03-09"}, c, e)


class FakeResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text

    def json(self):
        return self._p


def test_tiingo_meta_no_body_and_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "tiingo_key", lambda: "k")
    urls = []

    def get(url, **k):
        urls.append(url)
        return FakeResp(200, {"startDate": "2000-01-01", "endDate": "2020-01-01", "name": "A"}, "{}")

    meta = dm.fetch_tiingo_meta(["AAA"], tmp_path / "m.json", get=get, sleep=lambda s: None)
    assert meta["AAA"]["exists"] and urls == ["https://api.tiingo.com/tiingo/daily/AAA"]  # /prices 아님
    with pytest.raises(dm.TiingoLimit):
        dm.fetch_tiingo_meta(["BBB"], tmp_path / "m.json", get=lambda url, **k: FakeResp(200, None, "You have run over your 500 symbol look up for this month."), sleep=lambda s: None)
    with pytest.raises(dm.TiingoLimit):
        dm.fetch_tiingo_meta(["CCC"], tmp_path / "m.json", get=lambda url, **k: FakeResp(403, None, "no"), sleep=lambda s: None)


def test_tiingo_body_saves_same_company_only(tmp_path, monkeypatch):
    monkeypatch.setattr(dm, "tiingo_key", lambda: "k")
    d = sg.nyse_days("2016-06-01", "2019-12-31")
    recs = [{"date": x.strftime("%Y-%m-%dT00:00:00.000Z"), "open": 10.0, "close": 10.0, "volume": 1000, "splitFactor": 1.0} for x in d]
    row = {"first": "2017-01-31", "last": "2019-12-31"}
    got = dm.fetch_tiingo_body(7, row, ["AAA"], "2010-01-01", tmp_path, get=lambda url, **k: FakeResp(200, recs, "[]"), sleep=lambda s: None)
    assert got == "AAA" and (tmp_path / "7.csv").exists()
    px = dm.read_cached_price(7, tmp_path, tmp_path / "없음")
    assert px.index.max() == pd.Timestamp("2019-12-31") and list(px.columns) == ["close", "volume"]
    # 가격이 편입 기간보다 한참 늦게 시작 = 다른 회사
    late = [r for r in recs if r["date"] >= "2019-06"]
    assert dm.fetch_tiingo_body(8, row, ["BBB"], "2010-01-01", tmp_path, get=lambda url, **k: FakeResp(200, late, "[]"), sleep=lambda s: None) is None


def test_read_cached_price_trims_stale_and_2022(tmp_path):
    idx = pd.to_datetime(["2021-12-29", "2021-12-30", "2021-12-31", "2022-01-03"])
    pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0], "volume": [5.0, 5.0, 5.0, 5.0]}, index=idx).to_csv(tmp_path / "1.csv")
    assert dm.read_cached_price(1, tmp_path, tmp_path).index.max() == CUTOFF
    pd.DataFrame({"close": [1.0, 2.0, 3.0, 3.0], "volume": [5.0, 5.0, 0.0, 0.0]}, index=idx).to_csv(tmp_path / "2.csv")
    assert dm.read_cached_price(2, tmp_path, tmp_path).index.max() == pd.Timestamp("2021-12-30")


def test_normalize_symbols():
    assert dm.normalize_symbols(" snv ") == ["SNV"]
    assert dm.normalize_symbols("BRK.B") == ["BRK-B"]
    assert dm.normalize_symbols("ABC, DEF") == ["ABC", "DEF"]
    for bad in ("NONE", "N/A", "", "nan", "12345", "ABC-WT"):
        assert dm.normalize_symbols(bad) == [], bad


def test_issuer_symbols_and_candidates():
    f = pd.DataFrame({"issuer_cik": [1, 1, 1, 2], "symbol": ["SNV", "SNV", "SYN", "NONE"]})
    sym = dm.issuer_symbols(f)
    assert sym == {1: ["SNV", "SYN"]}
    tk = pd.DataFrame({"ticker_hist": ["SYN"], "ticker_now": [""]}, index=[1])
    assert dm.candidate_tickers_for(1, tk, {}, sym) == ["SYN", "SNV"]  # 기존 후보 다음, 중복 없음
    assert dm.candidate_tickers_for(1, tk, {"SYN": "빈 응답"}, sym) == ["SNV"]  # 이미 받은 티커는 결과를 그대로 쓴다
    assert dm.candidate_tickers_for(1, tk, {}) == ["SYN"]
    assert dm.candidate_tickers_for(9, tk, {}, {9: ["XYZ"]}) == ["XYZ"]  # tickers.csv에 없어도 신고 티커는 후보


def test_gap_status_uses_symbol_with_same_name_rule():
    c, e = pd.Timestamp("2018-03-08"), pd.Timestamp("2018-03-12")
    z = pd.DataFrame({"cik": [2, 3], "confirm": [c, c], "exec_day": [e, e]})
    check = pd.DataFrame({"same_company": [None, None], "price": ["없음", "없음"], "price_first": [None, None], "price_last": [None, None], "names": ["Beta Corp", "Gamma Corp"]}, index=[2, 3])
    tickers = pd.DataFrame({"ticker_hist": ["OLD", "OLG"], "ticker_now": ["", ""]}, index=[2, 3])
    meta = {"NEW": {"exists": True, "start": "2000-01-01", "end": "2020-01-01", "name": "BETA CORP"},
            "OTH": {"exists": True, "start": "2000-01-01", "end": "2020-01-01", "name": "ZZZ UNRELATED"}}
    st = rn.gap_status(z, check, tickers, {"OLD": "빈 응답", "OLG": "빈 응답"}, meta, {2: ["NEW"], 3: ["OTH"]})
    assert list(st["source"]) == ["Tiingo 메타", "없음"] and st["tiingo"].iloc[0] == "NEW"  # 이름 유사도 조건은 그대로


def test_gap_status_labels():
    c, e = pd.Timestamp("2018-03-08"), pd.Timestamp("2018-03-12")
    z = pd.DataFrame({"cik": [1, 2, 3, 4], "confirm": [c] * 4, "exec_day": [e] * 4})
    check = pd.DataFrame({"same_company": [True, None, None, False], "price": ["yfinance", "없음", "없음", "tiingo"], "price_first": ["2010-01-01", None, None, "2010-01-01"],
                          "price_last": ["2021-12-31", None, None, "2021-12-31"], "names": ["A", "Beta Corp", "Gamma Corp", "D"]}, index=[1, 2, 3, 4])
    tickers = pd.DataFrame({"ticker_hist": ["A", "BET", "GAM", "D"], "ticker_now": ["", "", "", ""]}, index=[1, 2, 3, 4])
    meta = {"BET": {"exists": True, "start": "2000-01-01", "end": "2020-01-01", "name": "BETA CORP"}, "GAM": {"exists": False}}
    st = rn.gap_status(z, check, tickers, {}, meta)
    assert list(st["source"]) == ["캐시", "Tiingo 메타", "없음", "없음"] and st["tiingo"].iloc[1] == "BET"


# ── 합성 전체 실행 (태그 뒤 코드 오류 방지) ───────────────────────────────────


def synth_inputs(n=40, gaps=2):
    days = sg.nyse_days("2016-06-01", "2021-12-31")
    rng = np.random.default_rng(11)
    prices = {}
    for c in range(1000, 1000 + n):
        px = 30.0 * np.cumprod(1.0 + rng.normal(0.0006, 0.02, len(days)))
        df = pd.DataFrame({"close": px, "volume": 1e6}, index=days)
        if c % 7 == 0:
            df = df.iloc[: 800 + (c % 300)]  # 중간 상장폐지
        prices[c] = df
    qqq = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0.0007, 0.012, len(days))), index=days)
    fx = pd.Series(1100.0 + rng.normal(0, 20, len(days)), index=days)
    zone = zone_for(list(prices) + [9000 + i for i in range(gaps)], first="2015-01-01", last="2021-12-31")
    d17 = sg.nyse_days("2017-01-03", "2021-12-30")

    def sig(step, offset):
        rows = []
        cs = list(prices)
        for k, i in enumerate(range(offset, len(d17) - 5, step)):
            rows.append({"cik": cs[k % n], "confirm": d17[i], "people": 3, "amount_usd": float(1000 + k), "first_trade": d17[max(i - 3, 0)], "last_trade": d17[i]})
        rows += [{"cik": 9000 + g, "confirm": d17[100 + 50 * g], "people": 3, "amount_usd": 5.0, "first_trade": d17[90], "last_trade": d17[100]} for g in range(gaps)]
        rows.append({"cik": cs[0], "confirm": d17[-1], "people": 3, "amount_usd": 5.0, "first_trade": d17[-3], "last_trade": d17[-1]})  # 체결 불가
        s = pd.DataFrame(rows).sort_values(["confirm", "cik"]).reset_index(drop=True)
        return rn.zone_signals(s, zone, sg.nyse_days())

    return sig(25, 5), sig(12, 3), sig(60, 7), prices, qqq, fx, zone


def test_end_to_end_synthetic(tmp_path):
    z3, z2, z4, prices, qqq, fx, zone = synth_inputs()
    res = rn.compute(z3, z2, z4, prices, qqq, fx, zone, n_random=10)
    # 가격 없는 CIK 2개 + 이미 상장폐지된 종목의 신호도 판정 불가(해석 18)
    assert res["undecided"] >= 2 and res["signals"]["status"].isin(["진입", "후보", "판정 불가", "유동성 미달", "기간 끝 체결 불가"]).all()
    assert (res["signals"]["status"] == "진입").sum() == len(res["excess"]) and len(res["excess"]) > 0
    assert (res["signals"]["status"] == "기간 끝 체결 불가").sum() == 1
    assert len(res["random"]) == 10 and set(res["verdict"]) == {"c1", "c2", "c3", "c4", "c5", "c6", "enough", "result"}
    out = tmp_path / "v1"
    gap = {"signals": len(z3), "by_year": {2017: 1}, "gap_rate": 0.05, "from_cache": 1, "from_tiingo_meta": 0, "no_source": 0, "tiingo_meta_tickers": 0}
    info = {"tag_commit": "abcdef0123", "commit_time": "t", "pushed_at": "t", "first_price": "t"}
    rn.write_outputs(out, res, gap, info, [{"name": "x", "sha256": "y"}], {"QQQ": qqq.index.max()}, {"ciks": 40, "new": 0, "old": 40})
    for name in ("report.md", "signals.csv", "trades.csv", "equity_curves.csv", "data_manifest.csv"):
        assert (out / name).stat().st_size > 0
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "최종 판정" in report and "합격 기준" in report and "2017" in report
    eq = pd.read_csv(out / "equity_curves.csv")
    assert eq["month"].iloc[-1] == "2021-12" and len(eq) == 61
    tr = pd.read_csv(out / "trades.csv")
    assert tr["sell"].max() <= "2021-12-31" and tr["buy"].min() >= "2017-01-03"
    with pytest.raises(FileExistsError):
        rn.write_outputs(out, res, gap, info, [], {}, {})


def test_end_to_end_no_entries_writes(tmp_path):
    _, _, _, prices, qqq, fx, zone = synth_inputs()
    empty = pd.DataFrame(columns=["cik", "confirm", "people", "amount_usd", "first_trade", "last_trade", "exec_day"]).astype({"confirm": "datetime64[ns]", "exec_day": "datetime64[ns]", "first_trade": "datetime64[ns]", "last_trade": "datetime64[ns]"})
    res = rn.compute(empty, empty, empty, prices, qqq, fx, zone, n_random=3)
    assert res["verdict"]["result"] == "FAIL" and res["verdict"]["enough"] is False
    out = tmp_path / "v1"
    rn.write_outputs(out, res, {}, {"tag_commit": "abcdef0", "commit_time": "t", "pushed_at": "t", "first_price": "t"}, [{"name": "x"}], {"QQQ": qqq.index.max()}, {"ciks": 1, "new": 0, "old": 1})
    assert (out / "report.md").exists()
