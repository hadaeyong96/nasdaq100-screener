"""「QQQ 이기는 힘이 세지는 종목」 점수 (작업 6) — 네트워크 없이 합성 데이터만.

점수 계산(RS선·최근 힘·직전 힘), 60거래일 미만·5달러 미만 제외, 상위 10, 업종 묶음 상위 3·하위 3,
미래 데이터 방지, 브리핑 접기 안 표 2개(관찰용), 텔레그램 첨부 설명·공개 보고서에 안 들어감.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from core import accel
from engine import daily
from notify import briefing, report_html, telegram

IDX = pd.bdate_range("2026-06-01", periods=90)
D = IDX[-1]


def _s(values, idx=IDX) -> pd.Series:
    return pd.Series(np.array(values, dtype=float), index=idx[: len(values)])


def _path(p0, p1, p2, n=90):
    """t−40에 p0, t−20에 p1, t에 p2 (그 사이 선형)."""
    a = np.full(n - 41, p0)
    b = np.linspace(p0, p1, 21)
    c = np.linspace(p1, p2, 21)[1:]
    return np.r_[a, b, c]


def test_score_is_recent_minus_prior_rs_strength():
    q = _s(_path(100, 110, 121))  # QQQ +10%, +10%
    st = _s(_path(10, 11, 14.52))  # 최근 +32%, 직전 +10%
    r = accel.rank({"X": st}, q, D, ["X"])
    assert r["ok"] and r["dates"] == (IDX[-41], IDX[-21], D)
    x = r["ranked"][0]
    rs0, rs1, rs2 = 10 / 100, 11 / 110, 14.52 / 121
    assert x["recent"] == pytest.approx(rs2 / rs1 - 1) and x["prior"] == pytest.approx(rs1 / rs0 - 1)
    assert x["score"] == pytest.approx((rs2 / rs1 - 1) - (rs1 / rs0 - 1))
    assert x["score"] == pytest.approx(0.2)


def test_excludes_short_history_and_under_5_dollars():
    q = _s(_path(100, 100, 100))
    full = _path(10, 10, 12)
    closes = {"OK": _s(full), "SHORT": _s(full).iloc[-59:], "PENNY": _s(_path(4, 4, 4.9)), "HOLE": _s(full)}
    closes["HOLE"].iloc[-21] = np.nan
    r = accel.rank(closes, q, D, list(closes))
    assert [x["ticker"] for x in r["ranked"]] == ["OK"]
    assert "60거래일" in r["excluded"]["SHORT"] and "5달러" in r["excluded"]["PENNY"] and "HOLE" in r["excluded"]
    assert accel.rank({"OK60": _s(full).iloc[-60:]}, q, D, ["OK60"])["ranked"]  # 정확히 60거래일이면 대상


def test_top_10_by_score_and_universe_only():
    q = _s(_path(100, 100, 100))
    closes = {f"T{i:02d}": _s(_path(10, 10, 10 + i)) for i in range(14)}
    r = accel.rank(closes, q, D, [t for t in closes if t != "T13"])
    assert [x["ticker"] for x in r["targets"]] == [f"T{i:02d}" for i in range(12, 2, -1)]


def test_qqq_without_enough_history_is_not_ok():
    assert not accel.rank({"X": _s(_path(10, 11, 12))}, _s(_path(100, 100, 100)).iloc[-50:], D, ["X"])["ok"]


def test_group_scores_top3_bottom3_min2():
    ranked = [{"ticker": t, "score": s} for t, s in
              [("MU", 0.3), ("WDC", 0.1), ("AMAT", 0.1), ("LRCX", 0.1), ("PEP", 0.0), ("KDP", 0.0), ("AEP", -0.1),
               ("EXC", -0.1), ("MSFT", -0.2), ("ADBE", -0.2), ("GOOGL", -0.3), ("META", -0.3), ("FANG", 0.9)]]
    gmap = {"MU": "메모리", "WDC": "메모리", "AMAT": "장비", "LRCX": "장비", "PEP": "식음료", "KDP": "식음료",
            "AEP": "유틸", "EXC": "유틸", "MSFT": "SW", "ADBE": "SW", "GOOGL": "인터넷", "META": "인터넷", "FANG": "에너지"}
    top, bottom = accel.group_scores(ranked, gmap)
    assert [x["group"] for x in top] == ["메모리", "장비", "식음료"] and top[0]["avg_score"] == pytest.approx(0.2)
    assert [x["group"] for x in bottom] == ["인터넷", "SW", "유틸"]


@pytest.mark.parametrize("cut", [-10, -3, -1])
def test_no_lookahead_cut_equals_full(cut):
    rng = np.random.default_rng(1)
    q = _s(100 * np.cumprod(1 + rng.normal(0, 0.01, 90)))
    closes = {t: _s(20 * np.cumprod(1 + rng.normal(0, 0.02, 90))) for t in ["A", "B", "C", "D"]}
    d = IDX[cut]
    a = accel.rank(closes, q, d, list(closes))
    b = accel.rank({t: s.loc[:d] for t, s in closes.items()}, q.loc[:d], d, list(closes))
    assert a["ranked"] == b["ranked"]


# ── 브리핑 접기 ──────────────────────────────────────────────────────────


def _summary():
    q = _s(_path(100, 110, 121))
    prices = {t: pd.DataFrame({"close": _path(10, 11, v)}, index=IDX) for t, v in
              [("MU", 15.0), ("WDC", 14.0), ("PEP", 12.0), ("KDP", 11.0), ("NVDA", 13.0)]}
    cfg = copy.deepcopy(daily.load_config())
    view = daily.build_weekly_rs_view(prices, D, cfg, {"MU": "마이크론"}, qqq_fetch=lambda: q)
    s = daily._empty_run_summary("live", D.date(), [])
    s["as_of"] = D
    s["weekly_rs"] = view
    return s, cfg, view


def test_view_keeps_weekly_tables_and_adds_accel():
    s, cfg, view = _summary()
    assert view["status"] == "ok" and view["top"]  # 기존 5거래일 표 그대로
    assert view["accel"]["ok"] and view["accel"]["targets"][0]["ticker"] == "MU"
    assert view["names"]["MU"] == "마이크론"


def test_report_shows_two_accel_tables_inside_weekly_fold(tmp_path):
    s, cfg, _ = _summary()
    html = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    i_fold, i_next = html.index('id="fold-weekly-rs"'), html.index('id="fold-fred"')  # 짧은 판: 강한 종목 → 돈 몰림 → 시장 지표
    i_a, i_b = html.index("QQQ 이기는 힘이 세지는 종목 상위 10"), html.index("힘이 세지는 업종")
    assert i_fold < i_a < i_b < i_next
    assert html.index("이번 주 강한 업종") < i_a  # 기존 표 2개 아래
    assert "관찰용 · 매수 신호 아님" in html and "마이크론" in html and "메모리" in html


def test_accel_not_in_caption_or_public_report(tmp_path):
    s, cfg, _ = _summary()
    text = briefing.build_briefing_text(s, cfg)
    assert "힘이 세지는" not in text and "힘이 세지는" not in telegram.report_caption(text)
