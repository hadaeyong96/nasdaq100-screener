"""core/superinvestor 고르기 규칙 (docs/prereg/superinvestor.md) — 합성 데이터, 네트워크 없음."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core import superinvestor as si


def _book(rows):
    df = pd.DataFrame(rows, columns=["cik", "cusip", "value"])
    df["weight"] = df["value"] / df.groupby("cik")["value"].transform("sum")
    return df


C2T = {f"C{i}": f"T{i}" for i in range(10)}
MEM = {f"T{i}" for i in range(10)}


def test_vote_needs_top5_and_increase():
    now = _book([(1, "C0", 50), (1, "C1", 20), (1, "C2", 10), (1, "C3", 8), (1, "C4", 7), (1, "C5", 5)])
    old = _book([(1, "C0", 60), (1, "C1", 10), (1, "C3", 30)])
    v = si.votes(now, old, C2T, MEM)
    # C0 비중 0.5 < 0.6 → 표 없음, C1 늘어남, C2·C4 새로 삼, C3 줄어듦, C5는 6위라 표 없음
    assert sorted(v.index) == ["T1", "T2", "T4"]
    assert (v["votes"] == 1).all()
    v2 = si.votes(now, old, C2T, MEM, require_increase=False)
    assert sorted(v2.index) == ["T0", "T1", "T2", "T3", "T4"]
    v3 = si.votes(now, old, C2T, MEM, k=3)
    assert sorted(v3.index) == ["T1", "T2"]


def test_only_zone_members_and_one_vote_per_manager():
    c2t = {**C2T, "C9b": "T9"}
    now = _book([(1, "C9", 30), (1, "C9b", 30), (1, "C8", 20), (1, "X1", 20), (2, "C9", 50), (2, "C1", 50)])
    old = _book([(9, "C0", 1)])
    v = si.votes(now, old, c2t, {"T9", "T1"})
    assert v.loc["T9", "votes"] == 2 and abs(v.loc["T9", "weight_sum"] - (0.6 + 0.5)) < 1e-12
    assert "T8" not in v.index and "X1" not in v.index


def test_rank_order_tiebreak_weight_then_ticker():
    t = pd.DataFrame({"votes": [2, 2, 3, 2, 0], "weight_sum": [0.3, 0.5, 0.1, 0.3, 0.9]}, index=["B", "C", "A", "D", "E"])
    assert si.rank_order(t) == ["A", "C", "B", "D"]


def test_select_keep_and_sector_cap():
    order = [f"S{i}" for i in range(40)]
    sector = {t: ("X" if i < 10 else f"G{i % 6}") for i, t in enumerate(order)}
    p = si.Params(n=6, keep_rank=30, max_sector=2)
    sel = si.select(order, sector, held=["S25", "S35"], p=p)
    assert sel[0] == "S25" and "S35" not in sel  # 30위 안 유지, 밖은 탈락
    assert sum(sector[t] == "X" for t in sel) <= 2 and len(sel) == 6


def test_select_random_reproducible_and_capped():
    pool = [f"P{i}" for i in range(50)]
    sector = {t: f"G{i % 3}" for i, t in enumerate(pool)}
    a = si.select_random(pool, sector, np.random.default_rng(20261030))
    b = si.select_random(pool, sector, np.random.default_rng(20261030))
    assert a == b and len(a) == 12  # 섹터 3개 × 최대 4
    assert max(pd.Series([sector[t] for t in a]).value_counts()) <= 4


def test_consensus_counts_holders():
    now = _book([(1, "C0", 10), (1, "C1", 90), (2, "C0", 5), (2, "C2", 95)])
    c = si.consensus(now, C2T, MEM)
    assert c.loc["T0", "votes"] == 2 and si.rank_order(c)[0] == "T0"
