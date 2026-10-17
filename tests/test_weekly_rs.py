"""이번 주 QQQ보다 강한 종목·업종 (작업 5-1) — 네트워크 없이 합성 데이터만 쓴다.

5거래일 초과 계산, 상위 10, 업종 묶음(2종목 미만 제외·상위 3·하위 3), 날짜 없는 종목 제외,
미래 데이터 방지, 보고서 접기(기본 닫힘·추천 아래), 텔레그램 첨부 설명·공개 보고서에 안 들어감.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from core import weekly_rs as wrs
from engine import daily
from notify import briefing, report_html, telegram

IDX = pd.bdate_range("2026-09-01", periods=30)
AS_OF = IDX[-1]


def _cfg():
    return copy.deepcopy(daily.load_config())


def _close(rets_last5: float, n: int = 30, base: float = 100.0) -> pd.DataFrame:
    """마지막 5거래일 동안 rets_last5만큼 오르는 종가 (그 전은 평평)."""
    px = np.full(n, base)
    px[-5:] = base * (1 + rets_last5 * np.arange(1, 6) / 5)
    return pd.DataFrame({"close": px}, index=IDX[:n])


def _qqq(ret: float = 0.02) -> pd.Series:
    return _close(ret)["close"]


def test_excess_is_stock_5d_return_minus_qqq_5d_return():
    prices = {"NVDA": _close(0.10), "PEP": _close(-0.01)}
    v = wrs.compute_weekly_rs(prices, _qqq(0.02), AS_OF, _cfg())
    assert v["status"] == "ok" and v["window"] == (IDX[-6], AS_OF)
    assert v["qqq_return"] == pytest.approx(0.02)
    nv = next(r for r in v["rows"] if r["ticker"] == "NVDA")
    assert nv["ret"] == pytest.approx(0.10) and nv["excess_pp"] == pytest.approx(8.0)
    assert nv["group"] == "반도체"
    assert next(r for r in v["rows"] if r["ticker"] == "PEP")["excess_pp"] == pytest.approx(-3.0)


def test_top_10_sorted_by_excess():
    prices = {f"T{i:02d}": _close(i / 100) for i in range(15)}
    v = wrs.compute_weekly_rs(prices, _qqq(0.0), AS_OF, _cfg())
    assert [r["ticker"] for r in v["top"]] == [f"T{i:02d}" for i in range(14, 4, -1)]
    assert v["top"][0]["group"] == wrs.NO_GROUP


def test_groups_need_two_members_and_split_top3_bottom3():
    rets = {"MU": 0.10, "WDC": 0.08,  # 메모리 평균 +9-2 = +7
            "AMAT": 0.06, "LRCX": 0.04,  # 반도체장비 +3
            "NVDA": 0.03, "AMD": 0.03,  # 반도체 +1
            "PEP": 0.0, "KDP": 0.0,  # 식음료 -2
            "AEP": -0.01, "EXC": -0.01,  # 유틸리티 -3
            "MSFT": -0.04, "ADBE": -0.02,  # 소프트웨어 -5
            "GOOGL": -0.06, "META": -0.06,  # 인터넷 -8
            "FANG": 0.50}  # 에너지 1종목뿐 → 표시 안 함
    v = wrs.compute_weekly_rs({t: _close(r) for t, r in rets.items()}, _qqq(0.02), AS_OF, _cfg())
    assert [x["group"] for x in v["groups_top"]] == ["메모리", "반도체장비", "반도체"]
    assert [x["group"] for x in v["groups_bottom"]] == ["인터넷", "소프트웨어", "유틸리티"]  # 약한 순
    assert v["groups_top"][0]["avg_excess_pp"] == pytest.approx(7.0) and v["groups_top"][0]["count"] == 2
    assert "에너지" not in {x["group"] for x in v["groups_top"] + v["groups_bottom"]}


def test_few_groups_do_not_repeat_in_bottom():
    prices = {t: _close(r) for t, r in {"MU": 0.1, "WDC": 0.1, "PEP": 0.0, "KDP": 0.0}.items()}
    v = wrs.compute_weekly_rs(prices, _qqq(0.0), AS_OF, _cfg())
    assert [x["group"] for x in v["groups_top"]] == ["메모리", "식음료"] and v["groups_bottom"] == []


def test_ticker_missing_window_dates_is_excluded_and_not_in_group_count():
    short = _close(0.2).iloc[-3:]  # 5거래일 전 종가 없음
    stale = _close(0.2).iloc[:-1]  # 기준일 종가 없음
    holed = _close(0.2)
    holed.iloc[-1, 0] = np.nan
    prices = {"MU": _close(0.1), "WDC": short, "STX": stale, "SNDK": holed, "PEP": _close(0.0)}
    v = wrs.compute_weekly_rs(prices, _qqq(0.0), AS_OF, _cfg())
    assert sorted(v["excluded"]) == ["SNDK", "STX", "WDC"]
    assert "메모리" not in {x["group"] for x in v["groups_top"] + v["groups_bottom"]}  # 계산된 메모리는 MU 하나


def test_qqq_without_as_of_is_no_qqq():
    v = wrs.compute_weekly_rs({"MU": _close(0.1)}, _qqq().iloc[:-1], AS_OF, _cfg())
    assert v["status"] == "no_qqq" and v["top"] == []


@pytest.mark.parametrize("cut", [-8, -3, -1])
def test_no_lookahead_cut_equals_full(cut):
    rng = np.random.default_rng(0)
    full = {t: pd.DataFrame({"close": 100 * np.cumprod(1 + rng.normal(0, 0.02, 30))}, index=IDX)
            for t in ["MU", "WDC", "PEP", "KDP", "NVDA", "AMD"]}
    q = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 30)), index=IDX)
    d = IDX[cut]
    a = wrs.compute_weekly_rs(full, q, d, _cfg())
    b = wrs.compute_weekly_rs({t: df.loc[:d] for t, df in full.items()}, q.loc[:d], d, _cfg())
    assert a["rows"] == b["rows"] and a["groups_top"] == b["groups_top"] and a["window"] == b["window"]


# ── engine view·보고서 ───────────────────────────────────────────────────


def _summary(view, mode="live"):
    s = daily._empty_run_summary(mode, AS_OF.date(), [])
    s["as_of"] = AS_OF
    s["weekly_rs"] = view
    return s


def _view(qqq_fetch=None):
    prices = {"MU": _close(0.10), "WDC": _close(0.08), "PEP": _close(0.0), "KDP": _close(-0.01)}
    return daily.build_weekly_rs_view(prices, AS_OF, _cfg(), {"MU": "마이크론"}, qqq_fetch=qqq_fetch or (lambda: _qqq(0.02)))


def test_view_adds_names_and_reports_qqq_failure():
    v = _view()
    assert v["names"]["MU"] == "마이크론" and v["names"]["PEP"] == "PEP" and v["error"] is None

    def broken():
        raise ConnectionError("down")

    e = _view(qqq_fetch=broken)
    assert e["status"] == "error" and "QQQ 가격을 받지 못해" in e["error"] and "down" in e["error"]


def test_report_fold_closed_below_recommendations_with_caption(tmp_path):
    html = report_html.render_report(_summary(_view()), _cfg(), tmp_path).read_text(encoding="utf-8")
    assert '<details class="fold" id="fold-weekly-rs">' in html  # open 속성 없음 = 기본 닫힘
    assert "▶ 이번 주 강한 종목·업종 (관찰용)" in html
    # 짧은 판(2026-10-17): 맨 위 세 칸 다음 첫 접기, 스크리닝(기본 숨김)은 없음
    assert html.index('aria-label="오늘 할 일"') < html.index('id="fold-weekly-rs"') < html.index('id="fold-fred"')
    assert 'id="fold-candidates"' not in html
    assert "관찰용 · 매수 신호 아님 (1주 강세는 과거 시험에서 다음 상승을 예측하지 못함)" in html
    assert "이번 주 QQQ보다 강한 종목" in html and "이번 주 강한 업종" in html
    assert "마이크론" in html and "+6.0%p" in html and "메모리" in html


def test_report_shows_failure_text_inside_fold(tmp_path):
    def broken():
        raise ConnectionError("down")

    html = report_html.render_report(_summary(_view(qqq_fetch=broken)), _cfg(), tmp_path).read_text(encoding="utf-8")
    assert "QQQ 가격을 받지 못해" in html


def test_not_in_telegram_caption_or_public_report(tmp_path):
    s = _summary(_view())
    cfg = _cfg()
    text = briefing.build_briefing_text(s, cfg)
    assert "QQQ보다 강한" not in text and "QQQ보다 강한" not in telegram.report_caption(text)
