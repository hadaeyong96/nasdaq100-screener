"""core/qqq_minus 규칙 (docs/prereg/qqq_minus.md) — 합성 데이터, 네트워크 없음."""

from __future__ import annotations

import numpy as np

from core import qqq_minus as qm


def test_cap_weights_redistribute_pro_rata():
    caps = {"A": 600.0, "B": 300.0, "C": 100.0}
    w = qm.cap_weights(caps)
    assert abs(sum(w.values()) - 1) < 1e-12 and abs(w["A"] - 0.6) < 1e-12
    w2 = qm.cap_weights(caps, {"C"})
    assert set(w2) == {"A", "B"} and abs(w2["A"] / w2["B"] - 2.0) < 1e-12 and abs(sum(w2.values()) - 1) < 1e-12
    assert qm.cap_weights(caps, {"A", "B", "C"}) == {}


def test_drop_reasons():
    assert qm.drop_reasons(False, 0.0) == {"A"}
    assert qm.drop_reasons(True, 0.0501) == {"B"}
    assert qm.drop_reasons(True, 0.05) == set()  # +5% 초과만
    assert qm.drop_reasons(None, None) == set()  # 모르면 빼지 않음
    assert qm.drop_reasons(True, 0.0, below_sma=True, rev_g=-0.01) == {"C"}
    assert qm.drop_reasons(True, 0.0, below_sma=True, rev_g=0.0) == set()
    assert qm.drop_reasons(True, 0.0, below_sma=False, rev_g=-0.5) == set()


def test_random_drop_count_weight_and_seed():
    w = qm.cap_weights({f"T{i}": float(i + 1) for i in range(100)})
    a = qm.random_drop(w, 5, 0.05, np.random.default_rng(20261030))
    b = qm.random_drop(w, 5, 0.05, np.random.default_rng(20261030))
    assert a == b and len(a) == 5
    s = sum(w[t] for t in a)
    assert 0.04 <= s <= 0.0625
    assert qm.random_drop(w, 0, 0.0, np.random.default_rng(1)) == set()
