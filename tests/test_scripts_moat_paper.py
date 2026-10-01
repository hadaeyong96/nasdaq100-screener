"""scripts/moat_paper.py 테스트 — 네트워크 없이, 순수 로직 부분만 돈다(가격 조회는
monkeypatch 없이 직접 호출하지 않는 함수만 고른다 — finalize_if_ready·build_entries·
try_fill_pending의 병합 로직 등)."""

from __future__ import annotations

import pytest

from scripts import moat_paper as msp


def _cfg():
    return {
        "moat_backtest": {"sector_cap_pct": 30, "random_trials": 1000, "random_seed": 20261001},
        "backtest": {"costs": {"slippage_pct": 0.05}},
    }


# ── build_entries ────────────────────────────────────────────────────────────


def test_build_entries_marks_filled_and_pending():
    entries = msp.build_entries(["A", "B"], {"A": 100.0}, slippage_pct=0.05)
    assert entries["A"]["status"] == "체결"
    assert entries["A"]["entry_price"] == pytest.approx(100.05)
    assert entries["B"] == {"status": "진입 대기"}


# ── finalize_if_ready ────────────────────────────────────────────────────────


def _ready_start():
    return {
        "wide_tickers": ["A", "B"],
        "sectors": {"A": "S1", "B": "S2"},
        "shares_outstanding": {"A": {"value": 1000.0}, "B": {"value": 2000.0}},
        "entries": {
            "A": {"status": "체결", "entry_price": 100.0},
            "B": {"status": "체결", "entry_price": 50.0},
            "QQQM": {"status": "체결", "entry_price": 200.0},
            "QQEW": {"status": "체결", "entry_price": 30.0},
        },
        "p1_weights": None, "p2_weights": None, "p2_market_caps": None,
        "status": "진입 대기",
    }


def test_finalize_if_ready_computes_weights_when_all_filled():
    start = _ready_start()
    msp.finalize_if_ready(start, _cfg())
    assert start["status"] == "체결 완료"
    assert start["p1_weights"] == pytest.approx({"A": 0.5, "B": 0.5})
    assert sum(start["p2_weights"].values()) == pytest.approx(1.0)
    assert "entered_at" in start


def test_finalize_if_ready_stays_pending_when_missing_qqew():
    start = _ready_start()
    start["entries"]["QQEW"] = {"status": "진입 대기"}
    msp.finalize_if_ready(start, _cfg())
    assert start["status"] == "진입 대기"
    assert start["p1_weights"] is None


# ── try_fill_pending: 덮어쓰기 거부(이미 체결된 값은 절대 안 바뀜) ─────────────


def test_try_fill_pending_does_not_overwrite_already_filled(monkeypatch):
    start = _ready_start()
    start["entries"]["B"] = {"status": "진입 대기"}
    start["status"] = "진입 대기"

    def fake_fetch_entry_opens(tickers, cfg):
        # A는 이미 체결 상태라 pending 목록에 안 들어가야 하므로 여기 안 옴.
        assert "A" not in tickers
        return {"B": 999.0}, []

    monkeypatch.setattr(msp, "fetch_entry_opens", fake_fetch_entry_opens)
    out = msp.try_fill_pending(start, _cfg())
    assert out["entries"]["A"] == {"status": "체결", "entry_price": 100.0}  # 손 안 댐
    assert out["entries"]["B"]["status"] == "체결"
    assert out["entries"]["B"]["entry_price"] == pytest.approx(999.0 * 1.0005)
    assert out["status"] == "체결 완료"


def test_try_fill_pending_noop_when_nothing_pending(monkeypatch):
    start = _ready_start()
    msp.finalize_if_ready(start, _cfg())  # 이미 체결 완료로 고정
    frozen_p1 = dict(start["p1_weights"])

    def boom(*args, **kwargs):
        raise AssertionError("체결 완료 상태에서는 가격을 다시 조회하면 안 된다")

    monkeypatch.setattr(msp, "fetch_entry_opens", boom)
    out = msp.try_fill_pending(start, _cfg())
    assert out["p1_weights"] == frozen_p1


# ── 보고서 렌더링 스모크 테스트 ────────────────────────────────────────────────


def test_render_pending_report_smoke():
    start = _ready_start()
    start["entries"]["B"] = {"status": "진입 대기"}
    start["judgment_date"] = "2026-09-30"
    start["entry_date"] = "2026-10-01"
    start["grades"] = {"A": "넓음", "B": "넓음"}
    text = msp.render_pending_report(start, "2026-10", {"A": "넓음", "B": "넓음"})
    assert "진입 대기" in text
    assert "B" in text


def test_render_evaluated_report_smoke():
    start = _ready_start()
    start["entry_date"] = "2026-10-01"
    msp.finalize_if_ready(start, _cfg())
    start["grades"] = {"A": "넓음", "B": "넓음"}
    text = msp.render_evaluated_report(start, "2026-11", {"A": "넓음", "B": "좁음"}, 1.05, 1.03, 1.02, 1.01, [1.0, 1.1, 0.9], 60.0)
    assert "P1" in text and "P2" in text
    assert "B" in text  # 등급이 바뀐 종목 목록에 포함
