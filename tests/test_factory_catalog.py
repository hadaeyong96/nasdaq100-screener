"""factory/catalog.yaml·docs/factory/trial_ledger.csv 형식 점검 (네트워크 없음)."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"기본값", "관찰 중", "탈락 보관", "확정 습관", "대기 후보"}


def test_catalog_shape():
    items = yaml.safe_load((ROOT / "factory" / "catalog.yaml").read_text(encoding="utf-8"))["strategies"]
    ids = [x["id"] for x in items]
    assert len(ids) == len(set(ids))
    for x in items:
        assert {"id", "name", "status", "zone", "result", "evidence"} <= set(x)
        assert x["status"] in STATUSES
    assert sum(x["status"] == "관찰 중" for x in items) == 9
    assert sum(x["status"] == "기본값" for x in items) == 1
    cands = [x for x in items if x["status"] == "대기 후보"]
    assert [x["rank"] for x in cands] == list(range(1, 10))
    assert [x["id"] for x in cands] == ["explore_a", "share_count_down", "lowvol_strength", "cheap_quality_family",
                                        "macro_timing_family", "revenue_growth_r1", "winner_anatomy", "event_radar",
                                        "crash_leaders"]  # 2026-10-17 폭락 뒤 주도주(충격 웅덩이 탈락으로 맨 끝)
    for x in cands:
        assert x["source"] and ("planned_trials" in x)
        assert x["planned_trials"] is None or (isinstance(x["planned_trials"], int) and x["planned_trials"] >= 1)
    assert len(next(x for x in cands if x["id"] == "cheap_quality_family")["variants"]) == 4


def test_rejected_ideas_not_in_ledger():
    cat = yaml.safe_load((ROOT / "factory" / "catalog.yaml").read_text(encoding="utf-8"))
    rej = cat["rejected_ideas"]
    assert rej and all({"id", "name", "reason", "source"} <= set(r) for r in rej)
    ledger = (ROOT / "docs" / "factory" / "trial_ledger.csv").read_text(encoding="utf-8-sig")
    assert all(r["name"] not in ledger for r in rej)


def test_trial_ledger_columns_and_order():
    d = pd.read_csv(ROOT / "docs" / "factory" / "trial_ledger.csv", encoding="utf-8-sig", dtype=str)
    assert list(d.columns) == ["date", "hypothesis", "zone", "params", "result", "prereg_tag"]
    assert d["date"].is_monotonic_increasing and d["result"].notna().all()
