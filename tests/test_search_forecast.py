"""검색량 실적 예고 테스트 (합성 데이터, 네트워크 없음) — docs/prereg/search_forecast.md 8장."""

import re

import numpy as np
import pandas as pd
import pytest
import yaml

from core import search_forecast as sf
from engine import search_forecast as eng
from store import search_forecast as st

WEEKS = pd.date_range("2019-01-06", "2027-12-26", freq="W-SUN")


def _weekly(seed: int, scale: float = 1.0, growth: float = 0.0) -> pd.Series:
    r = np.random.default_rng(seed)
    t = np.arange(len(WEEKS)) / 52
    return pd.Series(scale * (50 + 5 * np.sin(2 * np.pi * t)) * (1 + growth) ** t + r.normal(0, 0.5, len(WEEKS)), index=WEEKS)


# ── YoY는 같은 검색어(같은 요청) 안에서만 ──


def test_company_yoy_is_median_of_each_terms_own_yoy():
    a, b, c = _weekly(1, 1.0, 0.10), _weekly(2, 10.0, -0.05), _weekly(4, 3.0, 0.60)  # 크기가 다른 세 검색어
    s, e = pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")
    ya, yb, yc = sf.term_yoy(a, s, e), sf.term_yoy(b, s, e), sf.term_yoy(c, s, e)
    three = {"a": a, "b": b, "c": c}
    assert np.isclose(sf.company_yoy(three, s, e), np.median([ya, yb, yc]))  # v1.1 기본 = 중앙값
    assert np.isclose(sf.company_yoy(three, s, e, agg="mean"), (ya + yb + yc) / 3)  # v1 참고
    assert not np.isclose(np.median([ya, yb, yc]), (ya + yb + yc) / 3)
    assert np.isclose(sf.company_yoy({"a": a}, s, e), ya)  # 검색어 1개면 그대로
    mixed = a + b  # 값을 섞어 YoY를 내면 큰 검색어가 지배 → 다른 값
    assert not np.isclose(sf.term_yoy(mixed, s, e), (ya + yb) / 2, atol=1e-3)


def test_latest_pull_never_mixes_requests(tmp_path):
    conn = st.connect(tmp_path / "t.db")
    old = _weekly(1).loc["2024-01-01":"2025-12-31"] * 3  # 다른 요청 = 다른 0~100 기준
    new = _weekly(1).loc["2024-01-01":"2025-12-31"]
    st.add_trends(conn, "AAA", "x", old, "2026-11-01T01:30:00Z")
    st.add_trends(conn, "AAA", "x", new.iloc[:-10], "2026-11-08T01:30:00Z")
    got = st.latest_pull(conn, "x")
    pd.testing.assert_series_equal(got, new.iloc[:-10], check_names=False, check_freq=False)


def test_score_is_yoy_minus_median_of_previous_8():
    w = _weekly(3, 1.0, 0.10)
    ends = list(pd.date_range("2022-03-31", "2026-09-30", freq="QE"))
    q = sf.quarter_windows(ends)
    r = sf.company_score({"t": w}, q[-9:])
    base = [sf.term_yoy(w, *x) for x in q[-9:-1]]
    assert r["base_n"] == 8 and np.isclose(r["score"], sf.term_yoy(w, *q[-1]) - np.median(base))


def test_score_v1_mean_reference_alongside_median():
    ws = {"a": _weekly(3, 1.0, 0.10), "b": _weekly(5, 2.0, -0.08), "c": _weekly(6, 1.0, 0.40)}
    q = sf.quarter_windows(list(pd.date_range("2022-03-31", "2026-09-30", freq="QE")))[-9:]
    r = sf.company_score(ws, q)
    y = lambda agg, w: sf.company_yoy(ws, *w, agg=agg)
    assert np.isclose(r["score"], y("median", q[-1]) - np.median([y("median", w) for w in q[:-1]]))
    assert np.isclose(r["score_v1"], y("mean", q[-1]) - np.median([y("mean", w) for w in q[:-1]]))
    assert not np.isclose(r["score"], r["score_v1"])


def test_connect_adds_v1_columns_to_old_table(tmp_path):
    import sqlite3

    db = tmp_path / "old.db"
    old = sqlite3.connect(db)  # v1.1 전 모양의 표
    old.execute("CREATE TABLE search_predictions (ticker TEXT, fiscal_end TEXT, season TEXT, quarter_start TEXT, yoy REAL, "
                "baseline REAL, score REAL, base_n INTEGER, computed_at TEXT, score_lock_hash TEXT, prediction TEXT, "
                "pred_locked_at TEXT, pred_lock_hash TEXT, PRIMARY KEY (ticker, fiscal_end))")
    old.close()
    conn = st.connect(db)
    assert list(st.predictions(conn).columns) == st.PRED_COLS
    conn.close()


# ── 분기·시즌·3등분 ──


def test_project_quarter_ends_month_end_and_52_week():
    cal = list(pd.date_range("2024-03-31", "2026-09-30", freq="QE"))
    assert sf.project_quarter_ends(cal, pd.Timestamp("2026-11-01"))[-1] == pd.Timestamp("2026-12-31")
    wk = [pd.Timestamp("2025-02-01") + pd.Timedelta(days=91 * k) for k in range(8)]
    out = sf.project_quarter_ends(wk, wk[-1] + pd.Timedelta(days=1))
    assert out[-1] == wk[-4] + pd.Timedelta(days=364)


def test_terciles():
    lab = sf.terciles({"A": 3.0, "B": 2.0, "C": 1.0, "D": 0.0, "E": -1.0, "F": -2.0, "G": np.nan})
    assert [lab[k] for k in "ABCDEFG"] == [sf.ACCEL, sf.ACCEL, sf.NONE, sf.NONE, sf.DECEL, sf.DECEL, sf.NONE]
    assert sf.season_of(pd.Timestamp("2026-11-01")) == "2026-Q4"


def test_quarterly_revenue_derives_q4_from_annual_minus_ytd():
    facts = [{"start": "2025-01-01", "end": "2025-03-31", "val": 10, "filed": "2025-05-01"},
             {"start": "2025-04-01", "end": "2025-06-30", "val": 11, "filed": "2025-08-01"},
             {"start": "2025-01-01", "end": "2025-09-30", "val": 33, "filed": "2025-11-01"},
             {"start": "2025-07-01", "end": "2025-09-30", "val": 12, "filed": "2025-11-01"},
             {"start": "2025-01-01", "end": "2025-12-31", "val": 46, "filed": "2026-02-01"}]
    q = sf.quarterly_revenue(facts)
    assert list(q["revenue"]) == [10, 11, 12, 13]


def test_announcement_date_prefers_first_8k_202():
    f = [{"form": "8-K", "filingDate": "2026-01-10", "items": "5.02"}, {"form": "8-K", "filingDate": "2026-01-28", "items": "2.02,9.01"},
         {"form": "10-K", "filingDate": "2026-02-05", "items": ""}]
    assert sf.announcement_date(f, pd.Timestamp("2025-12-31")) == pd.Timestamp("2026-01-28")


# ── 잠금 시각 이후 발표만 유효 ──


def test_validity_lock_before_announcement_only():
    lock = pd.Timestamp("2027-01-10T01:30:00Z")  # 미 동부 2027-01-09 저녁
    assert sf.is_valid(lock, pd.Timestamp("2027-01-28")) is True
    assert sf.is_valid(lock, pd.Timestamp("2027-01-09")) is False  # 같은 날 = 무효
    assert sf.is_valid(lock, pd.Timestamp("2027-01-05")) is False
    assert sf.is_valid(lock, None) is None


# ── 날짜 시뮬레이션: 첫 시즌 2026-Q4부터 잠금이 자동으로 걸리는지 ──


def _fake_cfg() -> dict:
    cfg = eng.load_config()
    cfg["companies"] = {"AAA": {"terms": ["a1", "a2"], "wiki": ["A_page"]}, "BBB": {"terms": ["b1"], "wiki": []},
                        "WKS": {"terms": ["w1"], "wiki": []}}
    return cfg


ROST_LIKE = [pd.Timestamp("2019-02-02") + pd.Timedelta(days=91 * k) for k in range(35)]  # 52주 회계(분기 끝 토요일)
CAL = list(pd.date_range("2019-03-31", "2027-03-31", freq="QE"))


def _facts(ends, now):
    """now에 EDGAR가 알고 있을 분기 매출(분기 끝 + 30일에 공시)."""
    out, rev = [], 100.0
    for i in range(1, len(ends)):
        rev *= 1.02 + 0.01 * (i % 3)
        filed = ends[i] + pd.Timedelta(days=30)
        if filed <= now:
            out.append({"start": (ends[i - 1] + pd.Timedelta(days=1)).date().isoformat(), "end": ends[i].date().isoformat(),
                        "val": rev, "filed": filed.date().isoformat()})
    return out


def _sources(clock: dict) -> eng.Sources:
    series = {t: _weekly(i, 1 + i, 0.03 * (i - 2)) for i, t in enumerate(["a1", "a2", "b1", "w1"])}
    fil = {"AAA": [{"form": "8-K", "filingDate": "2027-01-28", "items": "2.02"}],
           "BBB": [{"form": "10-K", "filingDate": "2027-02-20", "items": ""}],
           "WKS": [{"form": "8-K", "filingDate": "2026-11-19", "items": "2.02"}]}

    def edgar(t):
        now = clock["now"].tz_localize(None)
        ends = ROST_LIKE if t == "WKS" else CAL
        return {"revenue_facts": _facts(ends, now), "filings": [f for f in fil[t] if pd.Timestamp(f["filingDate"]) <= now]}

    days = pd.bdate_range("2026-01-01", "2027-06-30")
    px = pd.Series(np.linspace(100, 120, len(days)), index=days)
    return eng.Sources(trends=lambda term: series[term][series[term].index <= clock["now"].tz_localize(None)],
                       wiki=lambda a: pd.Series({pd.Timestamp("2026-09-01"): 10}), edgar=edgar,
                       prices=lambda t: px[px.index <= clock["now"].tz_localize(None)])


def test_first_season_locks_automatically(tmp_path):
    cfg, clock = _fake_cfg(), {}
    src, db, locks = _sources(clock), tmp_path / "vb.db", tmp_path / "locks"
    for sunday in pd.date_range("2026-11-01", "2027-03-28", freq="W-SUN"):
        clock["now"] = pd.Timestamp(sunday.date().isoformat() + "T01:30:00", tz="UTC")
        eng.run(cfg, db, clock["now"], src, locks)
        if sunday == pd.Timestamp("2026-11-01"):
            conn = st.connect(db)
            assert st.predictions(conn).empty  # 2026-Q3 분기(9-30 끝)는 시작 시즌 전이라 잠그지 않음
            conn.close()
    conn = st.connect(db)
    p = st.predictions(conn).set_index(["ticker", "season"])
    s = st.scores(conn).set_index("ticker")
    conn.close()
    assert not (p.index.get_level_values("season") < "2026-Q4").any()
    wk = p.loc[("WKS", "2026-Q4")]
    assert wk["fiscal_end"] == "2026-10-24" and wk["computed_at"] == "2026-11-08T01:30:00Z"  # 끝 + 10일 = 11-03 → 다음 일요일
    for t in ("AAA", "BBB"):
        assert p.loc[(t, "2026-Q4"), "computed_at"] == "2027-01-10T01:30:00Z"  # 12-31 + 10일
    q4 = p.xs("2026-Q4", level="season")
    assert (q4["pred_locked_at"] == "2027-01-10T01:30:00Z").all() and q4["prediction"].notna().all()
    assert sorted(q4["prediction"]) == sorted([sf.ACCEL, sf.NONE, sf.DECEL])  # 3곳 → 1·1·1
    assert ("WKS", "2027-Q1") in p.index  # 다음 분기(2027-01-23 끝)도 자동으로
    # 채점: 잠금 뒤 발표만 유효
    assert s.loc["AAA", "valid"] == 1 and s.loc["AAA", "announced"] == "2027-01-28"
    assert s.loc["WKS", "valid"] == 1 and s.loc["WKS", "valid_pred_lock"] == 0  # 점수 잠금(11-08) < 발표(11-19) < 시즌 잠금(01-10)
    assert not pd.isna(s.loc["AAA", "ex_1d"]) and pd.isna(s.loc["AAA", "ex_6m"])  # 6개월은 아직 안 참
    # 잠금 파일: 시각·해시만
    txt = (locks / "2026-Q4.txt").read_text(encoding="utf-8")
    lines = txt.strip().splitlines()
    assert len(lines) == 3  # WKS 점수 · AAA·BBB 점수 · 시즌 예측
    assert all(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ sha256=[0-9a-f]{64}", ln) for ln in lines)
    for bad in ("AAA", "BBB", "WKS", "가속", "감속", "예측", "score"):
        assert bad not in txt


def test_lock_hash_matches_db_rows(tmp_path):
    cfg, clock = _fake_cfg(), {}
    src, db, locks = _sources(clock), tmp_path / "vb.db", tmp_path / "locks"
    clock["now"] = pd.Timestamp("2026-11-08T01:30:00", tz="UTC")
    eng.run(cfg, db, clock["now"], src, locks)
    conn = st.connect(db)
    p = st.predictions(conn)
    conn.close()
    row = p.iloc[0].to_dict()
    keys = ["ticker", "fiscal_end", "season", "quarter_start", "yoy", "baseline", "score", "base_n", "computed_at",
            "yoy_v1", "baseline_v1", "score_v1"]
    digest = sf.lock_hash([{k: row[k] for k in keys}])
    assert row["score_lock_hash"] == digest and digest in (locks / "2026-Q4.txt").read_text(encoding="utf-8")


def test_collection_failures_are_logged_not_raised(tmp_path):
    cfg, clock = _fake_cfg(), {"now": pd.Timestamp("2026-11-08T01:30:00", tz="UTC")}
    src = _sources(clock)

    def boom(term):
        raise RuntimeError("429")

    src.trends = boom
    rep = eng.run(cfg, tmp_path / "vb.db", clock["now"], src, tmp_path / "locks")
    assert len(rep["terms_fail"]) == 4 and rep["terms_ok"] == 0
    conn = st.connect(tmp_path / "vb.db")
    n = conn.execute("SELECT COUNT(*) FROM search_fetch_log WHERE ok = 0").fetchone()[0]
    conn.close()
    assert n == 4


# ── 편지: 수집 실패가 아침 편지를 막지 않음 · 월요일 개인판만 ──


def test_daily_letter_unaffected_by_search_forecast_failure():
    from engine import daily

    summary = {}

    def broken():
        raise RuntimeError("DB 깨짐")

    daily._attach_search_forecast(summary, counts_fn=broken)
    assert summary["search_forecast"] is None and "letter_check" not in summary  # 「자료 문제」 줄도 없음


def test_letter_counts_none_without_db(tmp_path):
    assert st.letter_counts(tmp_path / "없음.db") is None
    st.connect(tmp_path / "vb.db").close()
    assert st.letter_counts(tmp_path / "vb.db") == {"locked": 0, "scored": 0, "hit": None}


def test_monday_line_personal_only_without_ticker():
    from notify import letter_plain as lp

    ph = lp.load_phrases()
    s = {"letter_today_kst": "2026-11-02", "search_forecast": {"locked": 5, "scored": 2, "hit": 0.5}}  # 월요일
    per = lp.extra_lines(s, {}, ph, lp.PERSONAL)
    grp = lp.extra_lines(s, {}, ph, lp.GROUP)
    assert "검색량 예고: 잠금 5건 · 채점 2건 · 가속 적중 50%" in per
    assert not any("검색량" in x for x in grp)
    tue = lp.extra_lines({**s, "letter_today_kst": "2026-11-03"}, {}, ph, lp.PERSONAL)
    assert not any("검색량" in x for x in tue)
    assert not any("검색량" in x for x in lp.extra_lines({**s, "search_forecast": None}, {}, ph, lp.PERSONAL))


def test_workflow_shares_db_group_and_schedule():
    load = lambda n: yaml.safe_load(open(eng.ROOT / ".github" / "workflows" / n, encoding="utf-8"))
    wf, daily = load("search_forecast.yml"), load("daily.yml")
    # v1.1: 같은 가상 장부 DB를 쓰므로 편지와 같은 그룹, 대기만(취소 없음)
    assert wf["concurrency"] == daily["concurrency"] == {"group": "daily-screener", "cancel-in-progress": False}
    assert wf[True]["schedule"][0]["cron"] == "30 1 * * 0"  # 일요일 KST 10:30 (yaml의 on: → True)


def test_config_targets_and_user_agent():
    cfg = eng.load_config()
    assert "CHTR" not in cfg["companies"] and "CHTR" in cfg["excluded"]
    assert len(cfg["companies"]) == 17
    assert cfg["term_agg"] == "median" and cfg["collect"]["request_delay_sec"] == 40  # v1.1
    assert "@" not in cfg["wiki"]["user_agent"]
    assert cfg["companies"]["PEP"]["terms"] == ["Doritos", "Cheetos", "Lay's", "Gatorade", "Pepsi"]
