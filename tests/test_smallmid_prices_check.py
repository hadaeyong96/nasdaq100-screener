"""중소형 가격·발표 시각·신호 건수 조사 스크립트의 순수 함수 (네트워크 없음)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import smallmid_prices_check as pc  # noqa: E402


def test_xbrl_prefix():
    assert pc.xbrl_prefix("ads-20121231.xml") == "ADS"
    assert pc.xbrl_prefix("brkr-20191231.htm") == "BRKR"
    assert pc.xbrl_prefix("bf.b-20150430.xml") == "BF.B"
    assert pc.xbrl_prefix("d123456d10k.htm") is None
    assert pc.xbrl_prefix("Financial_Report.xlsx") is None


def test_candidate_tickers_order_and_dots():
    assert pc.candidate_tickers({"ticker_hist": "BF.B", "ticker_now": "BF-B;BF-A"}) == ["BF-B", "BF-A"]
    assert pc.candidate_tickers({"ticker_hist": None, "ticker_now": "ABC"}) == ["ABC"]
    assert pc.candidate_tickers({"ticker_hist": float("nan"), "ticker_now": float("nan")}) == []
    assert pc.candidate_tickers({"ticker_hist": "AHL", "ticker_now": "AHL-PD;AHL-PE"}) == ["AHL"]
    assert pc.candidate_tickers({"ticker_hist": None, "ticker_now": "CDR-PB;VAL-WT"}) == []


def test_check_seal_stops_on_rows_after_2021():
    ok = pd.DataFrame({"close": [1.0]}, index=pd.to_datetime(["2021-12-31"]))
    pc.check_seal(ok, "x")
    bad = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.to_datetime(["2021-12-31", "2022-01-03"]))
    with pytest.raises(pc.SealError):
        pc.check_seal(bad, "x")


def test_tiingo_adjust_and_raw_close_roundtrip():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    raw = pd.DataFrame({"raw_open": [100.0, 102.0, 51.0], "raw_close": [101.0, 104.0, 52.0],
                        "volume": [10.0, 10.0, 20.0], "split_factor": [1.0, 1.0, 2.0]}, index=idx)
    adj = pc.tiingo_adjust(raw)
    assert list(adj["close"]) == [50.5, 52.0, 52.0]  # 2:1 분할 전날까지 반으로
    assert list(adj["volume"]) == [20.0, 20.0, 20.0]
    assert list(adj["open"]) == [50.0, 51.0, 51.0]
    assert list(pc.raw_close(adj)) == [101.0, 104.0, 52.0]


def test_overlap_and_gap():
    px = pd.DataFrame({"close": 1.0}, index=pd.to_datetime(["2015-01-02", "2015-01-05", "2015-02-02", "2016-01-04"]))
    assert pc.overlap_days(px, "2015-01-01", "2015-12-31", pad_days=0) == 3
    assert pc.overlap_days(px, "2015-01-01", "2015-12-31") == 4  # 한 분기 넓히면 2016-01-04 포함
    assert pc.overlap_days(px, "2015-01-05", "2015-01-05") == 3  # 보고 한 번만 보인 종목
    assert pc.max_gap_days(px, "2015-01-01", "2015-12-31") == 28


def test_next_hour_wait_seconds():
    assert pc.next_hour_wait_seconds(3600 * 10 + 1800, margin_sec=120) == 1920


def test_exit_reason():
    f = pd.DataFrame({"form": ["10-K", "8-K", "25-NSE"], "filing_date": ["2015-03-01", "2015-09-01", "2015-09-20"],
                      "items": ["", "2.01,9.01", ""]})
    assert pc.exit_reason(f, "2015-08-31", False) == "인수·합병(상장폐지)"
    assert pc.exit_reason(f, "2015-08-31", True) == "끝까지 편입(2021-12)"
    bk = pd.DataFrame({"form": ["10-Q", "8-K"], "filing_date": ["2016-05-01", "2016-06-01"], "items": ["", "1.03"]})
    assert pc.exit_reason(bk, "2016-05-31", False) == "파산(8-K 1.03)"
    alive = pd.DataFrame({"form": ["10-K"], "filing_date": ["2018-03-01"], "items": [""]})
    assert pc.exit_reason(alive, "2016-05-31", False) == "지수 편출만(회사는 계속 공시)"
    assert pc.exit_reason(pd.DataFrame(columns=["form", "filing_date", "items"]), "2016-05-31", False) == "불명"


def test_liquid_before_uses_days_before_d0_only():
    idx = pd.bdate_range("2020-01-01", periods=25)
    px = pd.DataFrame({"close": [10.0] * 24 + [1.0], "volume": [600_000.0] * 24 + [0.0]}, index=idx)
    ok, price, dv = pc.liquid_before(px, idx[-1])
    assert ok and price == 10.0 and dv == 6_000_000.0  # 반응일(종가 1달러) 자체는 안 봄
    px.loc[idx[-2], "close"] = 4.0
    assert not pc.liquid_before(px, idx[-1])[0]


def test_member_on_excludes_single_report_intervals():
    ivs = [(pd.Timestamp("2015-03-31"), pd.Timestamp("2016-06-30"), "sp600", 6),
           (pd.Timestamp("2018-03-31"), pd.Timestamp("2018-03-31"), "sp400", 1)]
    assert pc.member_on(ivs, pd.Timestamp("2015-06-01")) == (True, "sp600")
    assert pc.member_on(ivs, pd.Timestamp("2018-03-31")) == (False, None)
    assert pc.member_on(ivs, pd.Timestamp("2017-01-03")) == (False, None)


# ── Tiingo 받기 순서·멈춤 ─────────────────────────────────────────────────────

def _ivs():
    full = (pd.Timestamp("2012-01-01"), pd.Timestamp("2021-12-31"), "sp600", 30)
    return {1: [full], 2: [full], 3: [(pd.Timestamp("2012-01-01"), pd.Timestamp("2016-12-31"), "sp400", 20)],
            4: [(pd.Timestamp("2018-01-01"), pd.Timestamp("2018-01-01"), "sp400", 1)], 5: [full]}


def _events():
    rows = []
    for cik, days in {1: ["2018-02-01", "2019-02-01", "2013-02-01"],
                      2: ["2017-05-01", "2018-05-01", "2019-05-01", "2020-05-01"],
                      3: ["2013-03-01", "2014-03-01"],
                      4: ["2018-01-01"],  # 보고 1번만 보인 구간 → 안 셈
                      5: ["2019-06-01", "2020-06-01"]}.items():  # 대형 겹침 → 안 셈
        rows += [{"cik": cik, "filing_date": d, "accepted_et": f"{d} 16:05:00"} for d in days]
    return pd.DataFrame(rows)


def test_tiingo_priority_validation_first_then_counts():
    big = {5: (pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-31"))}
    pr = pc.tiingo_priority(_events(), _ivs(), big)
    assert pr == {1: (0, -2), 2: (0, -4), 3: (1, -2)}
    assert pc.big_overlap(big[5], pd.Timestamp("2021-03-15"))  # 끝 + 92일 안
    assert not pc.big_overlap(None, pd.Timestamp("2021-03-15"))


def test_event_day_falls_back_to_filing_date():
    assert pc.event_day({"accepted_et": "2018-02-01 17:00:00", "filing_date": "2018-02-02"}) == pd.Timestamp("2018-02-01")
    assert pc.event_day({"accepted_et": float("nan"), "filing_date": "2018-02-02"}) == pd.Timestamp("2018-02-02")


def test_order_todo_zone_order_and_skips_progress():
    need = pd.DataFrame([{"cik": 9, "ticker_hist": "REST", "ticker_now": None},
                         {"cik": 3, "ticker_hist": "EXP", "ticker_now": None},
                         {"cik": 1, "ticker_hist": "VA", "ticker_now": "VA2"},
                         {"cik": 2, "ticker_hist": "VB", "ticker_now": "VA"}])
    prio = {1: (0, -2), 2: (0, -4), 3: (1, -2)}
    assert pc.order_todo(need, {}, prio) == ["VB", "VA", "VA2", "EXP", "REST"]
    assert pc.order_todo(need, {"VB": "있음 10", "EXP": "빈 응답"}, prio) == ["VA", "VA2", "REST"]


def test_is_hard_stop():
    assert pc.is_hard_stop(403, "Forbidden")
    assert pc.is_hard_stop(429, "You have exceeded your monthly unique symbol limit")
    assert not pc.is_hard_stop(429, "hourly request allocation exceeded")
    assert not pc.is_hard_stop(404, "")


class _Resp:
    def __init__(self, status, text="", data=None):
        self.status_code, self.text, self._data = status, text, data

    def json(self):
        return self._data


def _tiingo_env(tmp_path, monkeypatch):
    """임시 캐시: CIK 1(이미 받았지만 안 붙인 OLD), CIK 2(받을 NEW1), CIK 3(받을 NEW2)."""
    cache = tmp_path / "smallmid"
    (cache / "tiingo").mkdir(parents=True)
    (cache / "prices").mkdir()
    pd.DataFrame([{"cik": 1, "ticker_hist": "OLD", "ticker_now": None, "first": "2015-01-01", "last": "2015-12-31"},
                  {"cik": 2, "ticker_hist": "NEW1", "ticker_now": None, "first": "2018-01-01", "last": "2018-12-31"},
                  {"cik": 3, "ticker_hist": "NEW2", "ticker_now": None, "first": "2013-01-01", "last": "2013-12-31"}]
                 ).to_csv(cache / "tickers.csv", index=False)
    pd.DataFrame({"cik": [1, 2, 3], "source": ["없음"] * 3}).to_csv(cache / "price_status.csv", index=False)
    days = pd.bdate_range("2015-01-02", "2015-12-31")
    pd.DataFrame({"raw_open": 10.0, "raw_close": 10.0, "volume": 1000, "split_factor": 1.0},
                 index=pd.Index(days, name="date")).to_csv(cache / "tiingo" / "OLD.csv")
    (cache / "tiingo_progress.json").write_text('{"OLD": "있음 250"}', encoding="utf-8")
    for name, val in {"TI": cache / "tiingo", "PX": cache / "prices", "TICKERS_CSV": cache / "tickers.csv",
                      "PRICE_STATUS_CSV": cache / "price_status.csv", "TI_PROGRESS": cache / "tiingo_progress.json",
                      "TI_STOP": cache / "tiingo_stop.json"}.items():
        monkeypatch.setattr(pc, name, val)
    monkeypatch.setattr(pc, "load_tiingo_priority", lambda: {3: (1, -4), 2: (0, -4)})
    monkeypatch.setenv("TIINGO_API_KEY", "testkey")
    return cache


def test_step_tiingo_attaches_old_at_start_and_stops_on_403(tmp_path, monkeypatch):
    cache = _tiingo_env(tmp_path, monkeypatch)
    calls = []

    def get(url, params, timeout):
        calls.append(url.split("/")[-2])
        assert (cache / "prices" / "1.csv").exists()  # 첫 요청 전에 이미 붙어 있어야 함
        if len(calls) == 1:
            return _Resp(200, "", [{"date": "2018-03-01T00:00:00Z", "open": 5, "close": 5, "volume": 1}] * 1)
        return _Resp(403, "Forbidden testkey")

    pc.step_tiingo(False, get=get, sleep=lambda s: None)
    assert calls == ["NEW1", "NEW2"]  # 검증 ① 먼저, 받은 OLD는 건너뜀
    prog = json.loads((cache / "tiingo_progress.json").read_text(encoding="utf-8"))
    assert set(prog) == {"OLD", "NEW1"}
    stop = json.loads((cache / "tiingo_stop.json").read_text(encoding="utf-8"))
    assert stop["status"] == 403 and stop["last_success"] == "NEW1" and stop["next_ticker"] == "NEW2"
    assert stop["remaining"] == 1 and "testkey" not in stop["message"]


def test_step_tiingo_stops_on_monthly_429_without_waiting(tmp_path, monkeypatch):
    cache = _tiingo_env(tmp_path, monkeypatch)
    slept = []
    pc.step_tiingo(True, get=lambda url, params, timeout: _Resp(429, "monthly unique symbol limit"),
                   sleep=slept.append)
    assert slept == []
    stop = json.loads((cache / "tiingo_stop.json").read_text(encoding="utf-8"))
    assert stop["last_success"] is None and stop["next_ticker"] == "NEW1" and stop["remaining"] == 2


def test_judge_period_smallmid_new_listing_month():
    # 분사·신규 상장: 첫 보고 달 안(+10일)에 가격 시작, 첫 10-K·10-Q가 1년 안이거나 뒤 → 통과
    assert pc.judge_period_smallmid("2012-06-30", "2021-12-31", "2012-06-14", "2021-12-31", "2012-04-02") == "맞음(상장 첫 달)"
    assert pc.judge_period_smallmid("2015-06-30", "2021-12-31", "2015-07-01", "2021-12-31", "2015-08-11") == "맞음(상장 첫 달)"
    assert pc.judge_period_smallmid("2014-11-30", "2021-12-31", "2014-11-13", "2021-12-31", "2014-11-14") == "맞음(상장 첫 달)"
    # 오래된 회사의 Tiingo 앞 기록 잘림 → 그대로 앞이 빔
    assert pc.judge_period_smallmid("2011-05-31", "2016-06-30", "2016-01-04", "2016-08-15", "2009-06-01") == "앞이 빔"
    # 첫 달 안에 시작해도 오래된 CIK(티커 재사용 가능) → 앞이 빔
    assert pc.judge_period_smallmid("2014-09-30", "2021-12-31", "2014-09-24", "2021-12-31", "2001-01-01") == "앞이 빔"
    # 첫 보고 달 + 10일 넘게 늦으면 앞이 빔
    assert pc.judge_period_smallmid("2015-06-30", "2021-12-31", "2015-07-20", "2021-12-31", "2015-03-04") == "앞이 빔"
    # 원래 맞음은 그대로, 제출일 모름도 원래 판정
    assert pc.judge_period_smallmid("2012-06-30", "2021-12-31", "2011-06-01", "2021-12-31", None) == "맞음"
    assert pc.judge_period_smallmid("2012-06-30", "2021-12-31", "2012-06-14", "2021-12-31", None) == "앞이 빔"
