"""core/explain.py 단위 테스트 (P3.7). 네트워크 없이 순수 함수만 검증한다."""

from __future__ import annotations

import pytest

from core import explain as expl

_CFG = {
    "assumptions": {
        "a1_to_a2_expiry_days": 10,
        "reentry_cooldown_days": 5,
        "gap_filter_pct": 4.0,
        "whipsaw_max_crosses_20d": 4,
        "swing_low_period": 10,
        "s_grade_macd_norm_min_pct": -0.5,
        "b_grade_macd_norm_max_pct": -2.0,
        "ichimoku_shift": 26,
    },
    "risk": {"b_budget_pct": 1.0},
}


# ── 완전성: 엔진이 낼 수 있는 모든 사유 코드에 설명이 있는지 ─────────────────


BUY_STAGES = ("A1", "A2", "A3", "B")
SELL_KINDS = ("STOP", "A1_EXPIRE", "E3", "E1", "E2")
FILTER_REASON_SAMPLES = [
    ["골든크로스 당일 RSI 70 이상"],
    ["최근 20거래일 MACD 교차 5회 이상 (휩소)"],
    ["종가가 구름 안에 있음"],
    ["앞구름 음운"],
    ["당일 시가 갭 5.0% (필터 4.0% 이상)"],
    ["실적 발표 3거래일 이내"],
    ["동시 보유 종목 수 한도 초과"],
]
WARN_KINDS = ("KIJUN_BREACH", "RSI_RELIEF", "TARGET_REACHED", "STOP_NEAR", "STOP_CHANGED", "STOP_NEEDED")
WATCH_KINDS = ("WAIT_A2", "WAIT_A3")


def _full_buy_ctx(stage: str) -> dict:
    base = {
        "kr": "테스트종목", "score": 55, "grade": "S" if stage in ("A2", "B") else None,
        "vol_ratio": 1.6, "cloud_thickness_pct": 2.1, "bb_width_pct": 0.15,
        "stop": 90.0, "stop_pct": -5.0, "qty": 10, "risk_capped": False, "limited": False,
        "limit": 100.0,
    }
    if stage == "A1":
        base.update({"rsi_prev": 25.1, "rsi_now": 31.2, "a2_expiry_date": "10/06"})
    elif stage == "A2":
        base.update({"rsi_now": 55.0, "macd_norm": -0.3, "a1_date": "2026-01-01"})
    elif stage == "A3":
        base.update({"cloud_ok": True, "future_yang_ok": True, "chikou_ok": True, "momentum_ok": True, "gap_pct": 1.2})
    else:
        base.update({"rsi_now": 58.0, "macd_norm": 0.1})
    return base


@pytest.mark.parametrize("stage", BUY_STAGES)
def test_every_buy_stage_has_explanation(stage):
    out = expl.explain_buy(stage, _full_buy_ctx(stage), _CFG)
    assert out["title"] and out["checks"] and out["body"]


@pytest.mark.parametrize("kind", SELL_KINDS)
def test_every_sell_kind_has_explanation(kind):
    ctx = {"kr": "테스트종목", "close": 95.0, "stop": 100.0, "macd": -0.5, "signal": 0.1, "cloud_bot": 96.0, "chikou_broken": True}
    out = expl.explain_sell(kind, ctx, _CFG)
    assert out["title"] and out["checks"] and out["body"]


@pytest.mark.parametrize("reasons", FILTER_REASON_SAMPLES)
def test_every_filter_reason_has_explanation(reasons):
    ctx = {"kr": "테스트종목", "score": 30, "rsi_now": 72.4, "gc_count_20d": 5, "gap_pct": 5.0, "earnings_date": "2026-09-24", "max_concurrent": 8}
    out = expl.explain_filtered(reasons, ctx, _CFG)
    assert out["title"] and out["checks"]


@pytest.mark.parametrize("kind", WARN_KINDS)
def test_every_warn_kind_has_explanation(kind):
    ctx = {
        "kr": "테스트종목", "close": 95.0, "kijun": 96.0, "rsi_prev": 71.0, "rsi_now": 68.0,
        "avg_entry": 100.0, "stop": 90.0, "stop_near_pct": 3, "old_stop": 90.0, "new_stop": 95.0,
    }
    out = expl.explain_warn(kind, ctx, _CFG)
    assert out["title"] and out["checks"]


@pytest.mark.parametrize("kind", WATCH_KINDS)
def test_every_watch_kind_has_explanation(kind):
    ctx = {
        "kr": "테스트종목", "macd_diff": -0.42, "expiry_date": "10/06",
        "cloud_ok": True, "future_yang_ok": True, "chikou_ok": False, "rsi_now": 47.2,
    }
    out = expl.explain_watch(kind, ctx, _CFG)
    assert out["title"] and out["checks"]


# ── 실제 지표 값이 설명에 들어가는지 ─────────────────────────────────────


def test_a1_explanation_contains_real_rsi_values():
    ctx = _full_buy_ctx("A1")
    out = expl.explain_buy("A1", ctx, _CFG)
    joined = " ".join(c["text"] for c in out["checks"])
    assert "25.1" in joined and "31.2" in joined
    assert "25.1" in out["body"] and "31.2" in out["body"]
    assert "$90.00" in out["next"]  # 손절가


def test_stop_sell_explanation_contains_real_close_and_stop():
    ctx = {"kr": "아마존", "close": 217.90, "stop": 218.70}
    out = expl.explain_sell("STOP", ctx, _CFG)
    text = " ".join(c["text"] for c in out["checks"])
    assert "$217.90" in text and "$218.70" in text


def test_filtered_overheat_explanation_contains_real_rsi():
    ctx = {"kr": "AMD", "score": 40, "rsi_now": 72.4}
    out = expl.explain_filtered(["골든크로스 당일 RSI 70 이상"], ctx, _CFG)
    assert "72.4" in " ".join(c["text"] for c in out["checks"])
    assert "과열로 제외" in out["title"]


def test_filtered_whipsaw_explanation_contains_real_cross_count():
    ctx = {"kr": "램리서치", "score": 30, "gc_count_20d": 5}
    out = expl.explain_filtered(["최근 20거래일 MACD 교차 5회 이상 (휩소)"], ctx, _CFG)
    assert "5회" in " ".join(c["text"] for c in out["checks"])
    assert "횡보장으로 제외" in out["title"]


def test_watch_wait_a2_explanation_contains_real_macd_diff():
    ctx = {"kr": "펩시코", "macd_diff": -0.42, "expiry_date": "10/06"}
    out = expl.explain_watch("WAIT_A2", ctx, _CFG)
    assert "-0.42" in " ".join(c["text"] for c in out["checks"])
    assert "10/06" in out["body"]


# ── config 값을 바꾸면 설명 숫자도 바뀌는지 ──────────────────────────────


def test_cooldown_days_change_is_reflected_in_stop_explanation():
    ctx = {"kr": "아마존", "close": 217.90, "stop": 218.70}
    cfg_5 = _CFG
    cfg_10 = {**_CFG, "assumptions": {**_CFG["assumptions"], "reentry_cooldown_days": 10}}
    out5 = expl.explain_sell("STOP", ctx, cfg_5)
    out10 = expl.explain_sell("STOP", ctx, cfg_10)
    assert "5거래일" in out5["next"]
    assert "10거래일" in out10["next"]
    assert out5["next"] != out10["next"]


def test_a1_to_a2_expiry_days_change_is_reflected_in_a1_explanation():
    ctx = {**_full_buy_ctx("A1"), "a2_expiry_date": None}  # 날짜를 못 구하면 거래일수로 대체 표기
    cfg_10 = _CFG
    cfg_15 = {**_CFG, "assumptions": {**_CFG["assumptions"], "a1_to_a2_expiry_days": 15}}
    out10 = expl.explain_buy("A1", ctx, cfg_10)
    out15 = expl.explain_buy("A1", ctx, cfg_15)
    assert "10거래일" in out10["next"]
    assert "15거래일" in out15["next"]


def test_whipsaw_threshold_change_is_reflected_in_filtered_explanation():
    ctx = {"kr": "램리서치", "score": 30, "gc_count_20d": 5}
    cfg_4 = _CFG
    cfg_6 = {**_CFG, "assumptions": {**_CFG["assumptions"], "whipsaw_max_crosses_20d": 6}}
    out4 = expl.explain_filtered(["최근 20거래일 MACD 교차 5회 이상 (휩소)"], ctx, cfg_4)
    out6 = expl.explain_filtered(["최근 20거래일 MACD 교차 5회 이상 (휩소)"], ctx, cfg_6)
    text4 = " ".join(c["text"] for c in out4["checks"])
    text6 = " ".join(c["text"] for c in out6["checks"])
    assert "기준(4회)" in text4
    assert "기준(6회)" in text6
