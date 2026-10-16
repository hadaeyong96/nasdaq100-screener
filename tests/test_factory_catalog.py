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
    ranks = [x["rank"] for x in items if x["status"] == "대기 후보"]
    assert ranks == sorted(ranks) == list(range(1, len(ranks) + 1))


def test_trial_ledger_columns_and_order():
    d = pd.read_csv(ROOT / "docs" / "factory" / "trial_ledger.csv", encoding="utf-8-sig", dtype=str)
    assert list(d.columns) == ["date", "hypothesis", "zone", "params", "result", "prereg_tag"]
    assert d["date"].is_monotonic_increasing and d["result"].notna().all()
