"""전략 6·7·9번(추천 따라하기 2개·한 달 가속) 시작값 고정 (2026-10-17) — 설정값과 DB 기억값의 관계. 네트워크 없음.

1~5번 장부는 매 실행 시작일부터 다시 계산하므로 설정값이 DB 기억값보다 우선한다(다르면 경고).
6·7·9번 장부는 DB에 이어 가는 상태 장부라, 설정 시작값은 장부를 처음 만들 때만 쓰고 그 뒤에는 DB 상태를 쓴다.
설정값이 DB 저장값과 다르면 경고 한 줄만 남긴다(어느 값을 쓰는지는 그대로). 아래 테스트는 이 동작을 고정해 둔다.
"""

from __future__ import annotations

import copy

import pytest

from engine.daily import load_config
from store import virtual_books as vbstore
from tests import test_accel_book as ta
from tests import test_picks_engine as tp

PIN_DATE, PIN_FX = "2026-10-16", 1340.949951171875


def _book(path, name):
    conn = vbstore.connect(path / "vb.db")
    try:
        return vbstore.load_picks_book(conn, name)
    finally:
        conn.close()


def test_config_pins_picks_and_accel_but_not_momentum():
    vb = load_config()["virtual_books"]
    for key in ("picks", "accel"):
        assert vb[key]["start_date"] == PIN_DATE and vb[key]["start_fx"] == PIN_FX
    assert vb["momentum"]["start_date"] is None and vb["momentum"]["start_fx"] is None
    assert vb["start_date"] == "2026-10-14" and vb["start_fx"] == 1340.31005859375  # 1~5번 그대로


@pytest.fixture
def picks_cfg(monkeypatch):
    return tp.cfg.__wrapped__(monkeypatch)


@pytest.fixture
def accel_cfg(monkeypatch):
    return ta.cfg.__wrapped__(monkeypatch)


def test_picks_new_book_uses_config_start_fx(picks_cfg, tmp_path):
    m = tp.Market()
    picks_cfg["virtual_books"]["picks"].update({"start_date": tp.IDX[tp.S].date().isoformat(), "start_fx": PIN_FX})
    r = tp._run(picks_cfg, m, tmp_path)  # fx_fn은 1400을 주지만 설정값을 써야 한다
    assert not r["errors"], r["errors"]
    for name in (tp.B1, tp.B2):
        b = _book(tmp_path, name)
        assert b["start_fx"] == PIN_FX and b["fx_date"] == "설정값"
        assert b["start_date"].date().isoformat() == tp.IDX[tp.S].date().isoformat()


def test_picks_existing_db_state_is_kept_when_config_differs(picks_cfg, tmp_path):
    m = tp.Market()
    tp._run(picks_cfg, m, tmp_path)  # 설정 비움 → 기준일·fx_fn 1400으로 시작
    before = _book(tmp_path, tp.B1)
    assert before["start_fx"] == 1400.0
    picks_cfg["virtual_books"]["picks"].update({"start_date": tp.IDX[tp.S - 3].date().isoformat(), "start_fx": PIN_FX})
    m.upto += 1
    r = tp._run(picks_cfg, m, tmp_path)
    assert not r["errors"], r["errors"]
    after = _book(tmp_path, tp.B1)
    # 동작: DB 상태 우선(설정값으로 바꾸지 않음) — 1~5번과 다르다. 다르면 경고 한 줄씩(2026-10-17 추가)
    assert after["start_fx"] == 1400.0 and after["start_date"] == before["start_date"]
    warns = [w for w in r["warnings"] if "설정 시작값이 DB 저장값과 다릅니다" in w]
    assert len(warns) == 2 and any(w.startswith(tp.B1) for w in warns) and any(w.startswith(tp.B2) for w in warns)
    assert all("시작일 설정" in w and "시작 환율 설정" in w and "DB 값을 그대로 씁니다" in w for w in warns)


def test_picks_no_mismatch_warning_when_config_matches_db(picks_cfg, tmp_path):
    m = tp.Market()
    picks_cfg["virtual_books"]["picks"].update({"start_date": tp.IDX[tp.S].date().isoformat(), "start_fx": PIN_FX})
    tp._run(picks_cfg, m, tmp_path)
    m.upto += 1
    r = tp._run(picks_cfg, m, tmp_path)
    assert not any("DB 저장값과 다릅니다" in w for w in r["warnings"])


def test_accel_mismatch_warns_but_keeps_db_values(accel_cfg, tmp_path):
    m = ta.Market(ta.ARMED)
    ta._run(accel_cfg, m, tmp_path)  # 설정 비움 → fx_fn 1400으로 시작
    before = _book(tmp_path, ta.NAME)
    accel_cfg["virtual_books"]["accel"].update({"start_date": before["start_date"].date().isoformat(), "start_fx": PIN_FX})
    m.upto += 1
    r = ta._run(accel_cfg, m, tmp_path)
    assert not r["errors"], r["errors"]
    warns = [w for w in r["warnings"] if "DB 저장값과 다릅니다" in w]
    assert len(warns) == 1 and warns[0].startswith(ta.NAME) and "시작 환율 설정" in warns[0] and "시작일 설정" not in warns[0]
    assert _book(tmp_path, ta.NAME)["start_fx"] == 1400.0


def test_accel_new_book_uses_config_start_fx(accel_cfg, tmp_path):
    m = ta.Market(ta.ARMED)
    accel_cfg["virtual_books"]["accel"].update({"start_fx": PIN_FX})
    r = ta._run(accel_cfg, m, tmp_path)
    assert not r["errors"], r["errors"]
    b = _book(tmp_path, ta.NAME)
    assert b["start_fx"] == PIN_FX and b["fx_date"] == "설정값"


def test_books_1_to_5_config_start_wins_over_db_with_warning(picks_cfg, tmp_path):
    cfg = copy.deepcopy(picks_cfg)
    cfg["virtual_books"]["picks"]["enabled"] = False
    m = tp.Market()
    tp._run(cfg, m, tmp_path)  # DB에 시작일 = 기준일 기억
    cfg["virtual_books"]["start_date"] = tp.IDX[tp.S - 2].date().isoformat()
    cfg["virtual_books"]["start_fx"] = PIN_FX
    r = tp._run(cfg, m, tmp_path)
    assert any("설정 시작일" in w and "설정값을 씁니다" in w for w in r["warnings"])
    conn = vbstore.connect(tmp_path / "vb.db")
    try:
        assert vbstore.get_meta(conn, "start_date") == tp.IDX[tp.S - 2].date().isoformat()
        assert float(vbstore.get_meta(conn, "start_fx")) == PIN_FX
    finally:
        conn.close()
