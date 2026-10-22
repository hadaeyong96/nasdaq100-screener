"""봉인 ⓑ(2022~2026) 보호 — 이번 시험(변동성 맞춤·갈아타기·2배 균형, 2026-10-22)의 모든 데이터 읽기를 2021-12-31에서 자른다.

install()을 부르면 scripts.shock_dip_gate01·gate2·gate3의 load_yf·load_fred를 잘린 판으로 바꾼다(캐시 파일은 그대로 두고
읽은 값만 자른다). 이 모듈의 load_yf·load_fred도 같은 잘린 판이다. 잘린 뒤 값이 남아 있으면 SealedDataError.
"""

from __future__ import annotations

import pandas as pd

CUTOFF = pd.Timestamp("2021-12-31")


class SealedDataError(RuntimeError):
    """봉인 구간 값이 계산에 들어가려 할 때."""


def cut(obj):
    """Series·DataFrame을 CUTOFF 이하로 자른다."""
    if obj is None or len(obj) == 0:
        return obj
    out = obj[obj.index <= CUTOFF]
    if len(out) and out.index.max() > CUTOFF:
        raise SealedDataError("봉인 ⓑ 값이 남았습니다")
    return out


def _wrap(fn):
    def g(*a, **k):
        return cut(fn(*a, **k))
    g.__wrapped__ = fn
    return g


_installed = False


def install() -> None:
    """공용 로더를 잘린 판으로 바꾼다(한 번만)."""
    global _installed, load_yf, load_fred
    if _installed:
        return
    from scripts import shock_dip_gate01 as g1
    from scripts import shock_dip_gate2 as g2
    from scripts import shock_dip_gate3 as g3

    yf_raw, fred_raw = g1.load_yf, g1.load_fred
    load_yf, load_fred = _wrap(yf_raw), _wrap(fred_raw)
    for m in (g1, g2, g3):
        if hasattr(m, "load_yf"):
            m.load_yf = load_yf
        if hasattr(m, "load_fred"):
            m.load_fred = load_fred
    _installed = True


def load_yf(ticker: str):  # install() 뒤 잘린 판으로 바뀐다
    install()
    return load_yf(ticker)


def load_fred(series_id: str):
    install()
    return load_fred(series_id)
