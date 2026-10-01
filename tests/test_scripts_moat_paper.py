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


# ── 시작 파일 보호 쓰기(write_start_file_guarded) ────────────────────────────


def _write_json(path, obj):
    import json
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def test_write_start_file_guarded_fills_blanks(tmp_path):
    path = tmp_path / "start.json"
    start = _ready_start()
    start["entries"]["B"] = {"status": "진입 대기"}
    _write_json(path, start)
    new = _ready_start()
    msp.finalize_if_ready(new, _cfg())
    changed = msp.write_start_file_guarded(path, new)
    assert "entries.B" in changed and "p1_weights" in changed and "status" in changed
    assert "entries.A" not in changed


def test_write_start_file_guarded_rolls_back_on_protected_change(tmp_path):
    path = tmp_path / "start.json"
    _write_json(path, _ready_start())
    original = path.read_bytes()
    bad = _ready_start()
    bad["entries"]["A"] = {"status": "체결", "entry_price": 123.0}  # 이미 체결된 값 변경
    with pytest.raises(msp.StartFileProtectedError):
        msp.write_start_file_guarded(path, bad)
    assert path.read_bytes() == original  # 되돌려짐


def test_write_start_file_guarded_refuses_when_finalized(tmp_path):
    path = tmp_path / "start.json"
    start = _ready_start()
    msp.finalize_if_ready(start, _cfg())
    _write_json(path, start)
    original = path.read_bytes()
    with pytest.raises(msp.StartFileProtectedError):
        msp.write_start_file_guarded(path, start)
    assert path.read_bytes() == original


def test_main_does_not_write_finalized_start_file(tmp_path, monkeypatch):
    # 체결 완료 상태면 main()은 시작 파일을 읽기만 한다.
    path = tmp_path / "start.json"
    start = _ready_start()
    msp.finalize_if_ready(start, _cfg())
    start["rules_doc_hash"] = msp.rules_doc_hash()
    _write_json(path, start)
    original = path.read_bytes()
    monkeypatch.setattr(msp, "START_PATH", path)
    monkeypatch.setattr(msp, "PAPER_DIR", tmp_path)
    monkeypatch.setattr(msp, "LOG_PATH", tmp_path / "log.txt")
    monkeypatch.setattr(msp, "load_config", lambda: {"moat": {}})
    monkeypatch.setattr(msp, "get_locked_hash", lambda: "x")
    monkeypatch.setattr(msp.moat, "check_lock", lambda cfg, h: None)
    monkeypatch.setattr(msp.edgar, "get_user_agent", lambda: "ua")
    monkeypatch.setattr(msp, "write_monthly_report_and_record", lambda *a, **k: None)

    def boom(*a, **k):
        raise AssertionError("체결 완료 상태에서 시작 파일을 쓰면 안 된다")

    monkeypatch.setattr(msp, "write_start_file_guarded", boom)
    monkeypatch.setattr(msp, "try_fill_pending", boom)
    msp.main()
    assert path.read_bytes() == original


# ── 공유 CIK 유통주식수 보정 ────────────────────────────────────────────────


def test_effective_shares_prefers_correction():
    start = {"shares_outstanding": {"G": {"value": 200.0}}, "shares_outstanding_corrections": {"G": {"old": 200.0, "value": 100.0}}}
    assert msp.effective_shares(start, "G") == 100.0
    assert msp.effective_shares({"shares_outstanding": {"G": {"value": 200.0}}}, "G") == 200.0


def test_build_shares_corrections_fixes_double_count(monkeypatch):
    start = {
        "cik": {"GOOG": 1, "GOOGL": 1, "AAPL": 2},
        "shares_outstanding": {"GOOG": {"value": 200.0}, "GOOGL": {"value": 200.0}, "AAPL": {"value": 50.0}},
    }
    fetched = []
    monkeypatch.setattr(msp.edgar, "fetch_company_facts", lambda cik, user_agent: fetched.append(cik) or {})
    monkeypatch.setattr(msp, "raw_shares_for_ticker", lambda t, facts: (100.0, "diluted_shares_fallback"))
    out = msp.build_shares_corrections(start, "ua")
    assert fetched == [1]  # 티커가 하나뿐인 CIK는 다시 조회하지 않음
    assert out["GOOG"]["old"] == 200.0 and out["GOOG"]["value"] == 100.0
    assert set(out) == {"GOOG", "GOOGL"}


# ── 규칙 문서 해시 알려진 변경 ──────────────────────────────────────────────


def test_is_known_hash_change():
    notes = "| 2026-10-02 | `aaaa` → `bbbb` | 줄바꿈 | 없음 |"
    assert msp.is_known_hash_change("aaaa", "bbbb", notes)
    assert not msp.is_known_hash_change("bbbb", "aaaa", notes)  # 순서 반대
    assert not msp.is_known_hash_change("aaaa", "cccc", notes)
    assert not msp.is_known_hash_change(None, "bbbb", notes)


def test_repo_hash_notes_covers_start_file_pair():
    import json
    start = json.loads(msp.START_PATH.read_text(encoding="utf-8"))
    notes = msp.HASH_NOTES_PATH.read_text(encoding="utf-8")
    assert msp.is_known_hash_change(start["rules_doc_hash"], "51ff52148b926add", notes)


# ── 규칙 문서 해시 줄바꿈 통일 ──────────────────────────────────────────────


def test_rules_doc_hash_same_for_lf_crlf_cr(tmp_path):
    text = "# 규칙\n\n| 진입 | 시가 |\n끝\n"
    lf, crlf, cr = tmp_path / "lf.md", tmp_path / "crlf.md", tmp_path / "cr.md"
    lf.write_bytes(text.encode("utf-8"))
    crlf.write_bytes(text.replace("\n", "\r\n").encode("utf-8"))
    cr.write_bytes(text.replace("\n", "\r").encode("utf-8"))
    assert msp.rules_doc_hash(lf) == msp.rules_doc_hash(crlf) == msp.rules_doc_hash(cr)


def test_rules_doc_hash_differs_when_content_differs(tmp_path):
    a, b = tmp_path / "a.md", tmp_path / "b.md"
    a.write_bytes(b"moat.rebalance_month\n")
    b.write_bytes(b"moat_backtest.rebalance_month\n")
    assert msp.rules_doc_hash(a) != msp.rules_doc_hash(b)


def test_repo_rules_doc_hash_matches_start_file():
    import json
    start = json.loads(msp.START_PATH.read_text(encoding="utf-8"))
    assert msp.rules_doc_hash() == start["rules_doc_hash"]
