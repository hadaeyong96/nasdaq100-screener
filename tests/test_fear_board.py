"""「공포 매수판」 (thermometer/fear_board.py, 2026-10-31) — 네트워크 없이 합성 자료로."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from notify import briefing
from tests.test_letter_summary import _body, _sec, rich, temp  # noqa: F401 — 픽스처 재사용
from tests.test_thermometer import _cut, make_prices
from thermometer.core import load_config
from thermometer.fear_board import board, caption_suffix, get_fear_board, qqq_frame, zone_of

CFG = load_config()
CALM = {"fear_greed": 50, "rsi": 50, "close": 100, "bb_lower": 90, "sma_long": 95, "sma_short": 99, "sma_short_lag": 98, "vix": 18}
LINES = {"평소": "적립만 그대로입니다.", "눌림": "눌림 구간입니다. 저축일에 그대로 삽니다.",
         "공포": "공포 구간입니다. 여윳돈이 있으면 추가 매수해도 좋은 구간입니다. 가진 주식은 팔지 않습니다."}


def _ok(vals, key):
    return {it["key"]: it["ok"] for it in board({**CALM, **vals}, CFG)["items"]}[key]


# ── 조건별 경계값 ───────────────────────────────────────────────────────────


def test_calm_is_zero():
    b = board(CALM, CFG)
    assert b["count"] == 0 and b["n_avail"] == 5 and b["zone"] == "평소" and [it["key"] for it in b["items"]] == \
        ["fear_greed", "rsi", "bollinger", "trend", "vix"]


@pytest.mark.parametrize("v,ok", [(25, True), (25.01, False), (0, True), (24.9, True)])
def test_fear_greed_boundary(v, ok):
    assert _ok({"fear_greed": v}, "fear_greed") is ok


@pytest.mark.parametrize("v,ok", [(30, True), (30.01, False), (12, True)])
def test_rsi_boundary(v, ok):
    assert _ok({"rsi": v}, "rsi") is ok


@pytest.mark.parametrize("close,lower,ok", [(89.99, 90, True), (90, 90, False), (90.01, 90, False)])
def test_bollinger_boundary_strictly_below(close, lower, ok):
    assert _ok({"close": close, "bb_lower": lower, "sma_long": 200}, "bollinger") is ok


@pytest.mark.parametrize("close,sma_long,sma_short,lag,ok", [
    (95, 95, 97, 98, True),  # 종가 = 60일선(≤), 20일선 < 21일 전
    (95.01, 95, 97, 98, False),  # 종가 > 60일선
    (90, 95, 98, 98, False),  # 20일선 = 21일 전(하락 아님)
    (90, 95, 98.5, 98, False),  # 20일선 상승
    (90, 95, 97.99, 98, True),
])
def test_trend_boundary(close, sma_long, sma_short, lag, ok):
    assert _ok({"close": close, "bb_lower": 0, "sma_long": sma_long, "sma_short": sma_short, "sma_short_lag": lag}, "trend") is ok


@pytest.mark.parametrize("v,ok", [(30, True), (29.99, False), (45, True)])
def test_vix_boundary(v, ok):
    assert _ok({"vix": v}, "vix") is ok


def test_missing_values_are_not_counted():
    b = board({**CALM, "fear_greed": None, "vix": float("nan"), "sma_short_lag": None, "rsi": 20}, CFG)
    by = {it["key"]: it for it in b["items"]}
    assert by["fear_greed"]["ok"] is None and by["vix"]["ok"] is None and by["trend"]["ok"] is None
    assert by["fear_greed"]["value"] is None and by["trend"]["value"] is None
    assert b["count"] == 1 and b["n_avail"] == 2 and len(b["missing"]) == 3


# ── zone ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n,name", [(0, "평소"), (1, "평소"), (2, "눌림"), (3, "공포"), (4, "공포"), (5, "공포")])
def test_zone_boundaries(n, name):
    z = zone_of(n, CFG)
    assert z["name"] == name and z["line"] == LINES[name]


def test_all_five_on():
    b = board({"fear_greed": 10, "rsi": 20, "close": 80, "bb_lower": 85, "sma_long": 95, "sma_short": 90,
               "sma_short_lag": 98, "vix": 40}, CFG)
    assert b["count"] == 5 and b["zone"] == "공포" and b["line"] == LINES["공포"]


# ── 계산(합성 자료) ─────────────────────────────────────────────────────────


def _crash_prices(end="2026-10-27"):
    p = make_prices(end=end)
    q = p["qqq"].copy()
    q.iloc[-8:] = q.iloc[-9] * np.cumprod(np.full(8, 0.99))  # 마지막 8거래일 하락
    q.iloc[-1] = q.iloc[-2] * 0.92  # 마지막 날 급락 → 볼린저 하단 아래
    p["qqq"] = q
    p["vix"] = p["vix"].copy()
    p["vix"].iloc[-1] = [38.0, 40.0]
    p["fear_greed"] = {**p["fear_greed"], q.index[-1].date().isoformat(): 12.0}
    return p


def test_crash_lights_all_five_and_calm_none():
    b = get_fear_board("2026-10-27", prices=_crash_prices(), cfg=CFG)
    assert b["data_date"] == "2026-10-27" and b["count"] == 5 and b["zone"] == "공포"
    calm = get_fear_board("2026-10-27", prices=make_prices(), cfg=CFG)
    assert calm["n_avail"] == 5 and calm["count"] <= 2


def test_rsi_and_band_match_existing_functions():
    """지표는 기존 함수 재사용 — core.indicators.rsi_wilder, 볼린저(모집단 표준편차)와 같은 값."""
    from core.indicators import rsi_wilder

    q = make_prices()["qqq"]
    f = qqq_frame(q, CFG)
    assert f["rsi"].iloc[-1] == pytest.approx(rsi_wilder(q, 14).iloc[-1])
    tail = q.iloc[-20:]
    assert f["bb_lower"].iloc[-1] == pytest.approx(tail.mean() - 2 * tail.std(ddof=0))
    assert f["sma_short_lag"].iloc[-1] == pytest.approx(q.iloc[-41:-21].mean())


def test_no_future_data():
    """그날까지 자른 자료의 결과 == 전체 자료로 그날을 구한 결과 (모든 조건)."""
    p = _crash_prices()
    for ts in p["qqq"].index[[100, 180, 230, -10, -2, -1]]:
        full = get_fear_board(ts, prices=p, cfg=CFG)
        cut = get_fear_board(ts, prices=_cut(p, ts), cfg=CFG)
        assert full == cut


def test_short_history_and_holiday():
    p = make_prices(n=30)
    b = get_fear_board("2026-10-27", prices=p, cfg=CFG)
    assert {it["key"] for it in b["items"] if it["ok"] is None} == {"trend"}  # 60일선·21일 전 20일선이 아직 없음
    sat = get_fear_board("2026-10-25", prices=make_prices(), cfg=CFG)  # 토요일 → 직전 거래일
    assert sat["data_date"] == "2026-10-23"
    empty = get_fear_board("1990-01-01", prices=make_prices(), cfg=CFG)
    assert empty["data_date"] is None and empty["count"] == 0 and empty["n_avail"] == 0


# ── 보고서 ─────────────────────────────────────────────────────────────────


def _fb(zone_count: int, day="2026-09-30"):
    vals = dict(CALM)
    for upd in [{"vix": 40}, {"rsi": 20}, {"fear_greed": 10}, {"bb_lower": 101}, {"sma_long": 100, "sma_short": 97}][:zone_count]:
        vals.update(upd)
    return {"date": day, "data_date": day, **board(vals, CFG)}


@pytest.mark.parametrize("n,zone", [(0, "평소"), (2, "눌림"), (3, "공포"), (5, "공포")])
@pytest.mark.parametrize("edition", ["personal", "group"])
def test_report_table_and_line(rich, temp, cfg, tmp_path, n, zone, edition):  # noqa: F811
    rich["fear_board"] = _fb(n)
    assert rich["fear_board"]["count"] == n
    sec = _sec(_body(rich, cfg, tmp_path, edition), "thermometer")
    assert f"공포 매수판 · {n}/5 · {zone}" in sec and f"▶ {LINES[zone]}" in sec
    assert sec.count("✓") == n and sec.count("✗") == 5 - n
    for name in CFG["fear_board"]["names"].values():
        assert name in sec
    assert sec.index("공포 매수판") > sec.index("오늘 온도") and sec.index("공포 매수판") < sec.index("지수·금리·환율")


def test_report_omits_board_when_missing_or_stale(rich, temp, cfg, tmp_path):  # noqa: F811
    assert "공포 매수판" not in _sec(_body(rich, cfg, tmp_path), "thermometer")
    rich["fear_board"] = _fb(3, day="2026-09-29")  # 기준일과 자료 날짜가 다름
    assert "공포 매수판" not in _sec(_body(rich, cfg, tmp_path), "thermometer")


def test_report_shows_missing_condition(rich, temp, cfg, tmp_path):  # noqa: F811
    rich["fear_board"] = {"date": "2026-09-30", "data_date": "2026-09-30", **board({**CALM, "fear_greed": None}, CFG)}
    assert "자료 없음" in _sec(_body(rich, cfg, tmp_path), "thermometer")


# ── 설명 한 줄 ─────────────────────────────────────────────────────────────


def test_caption_suffix_only_in_fear_zone(rich, temp, cfg):  # noqa: F811
    t = temp["temp"]
    base = f"[오빠의 아침 편지] 10/26(월) · 시장 온도 {t}도"
    for n in (0, 1, 2):
        rich["fear_board"] = _fb(n)
        for ed in ("personal", "group"):
            assert briefing.build_report_caption(rich, cfg, edition=ed) == base
    for n in (3, 4, 5):
        rich["fear_board"] = _fb(n)
        for ed in ("personal", "group"):
            assert briefing.build_report_caption(rich, cfg, edition=ed) == f"{base} · 공포 매수판 {n}/5"


def test_caption_suffix_goes_last(rich, temp, cfg):  # noqa: F811
    s = copy.deepcopy(rich)
    s["savings_day"] = True
    s["fred_missing"] = ["DGS10"]
    s["fear_board"] = _fb(4)
    assert briefing.build_report_caption(s, cfg).endswith(" · 오늘 할 일 있음 · 자료 문제 있음 · 공포 매수판 4/5")
    assert briefing.build_report_caption(s, cfg, edition="group").endswith(" · 자료 문제 있음 · 공포 매수판 4/5")
    s["fear_board"] = _fb(4, day="2026-09-29")  # 오래된 판은 덧붙이지 않는다
    assert "공포 매수판" not in briefing.build_report_caption(s, cfg)


def test_caption_suffix_function():
    assert caption_suffix(None, CFG) is None
    assert caption_suffix({"zone": "눌림", "count": 2}, CFG) is None
    assert caption_suffix({"zone": "공포", "count": 3}, CFG) == "공포 매수판 3/5"


# ── 매일 실행 연결 ─────────────────────────────────────────────────────────


def test_daily_attach_and_failure():
    from engine import daily

    s = {"as_of": pd.Timestamp("2026-10-27")}
    daily._attach_fear_board(s)  # 테스트에서는 오늘 자료 받기가 막혀 있음(conftest) → 실패해도 그 표만 빠진다
    assert s["fear_board"] is None
    daily._attach_fear_board(s, board_fn=lambda d: get_fear_board(d, prices=_crash_prices(), cfg=CFG))
    assert s["fear_board"]["count"] == 5 and s["fear_board"]["data_date"] == "2026-10-27"
    daily._attach_fear_board({}, board_fn=lambda d: 1 / 0)  # 기준일 없음 → 부르지 않음
