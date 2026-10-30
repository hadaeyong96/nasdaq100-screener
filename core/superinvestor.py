"""슈퍼투자자 최고 확신 종목 전략(superinvestor) — 고르기 순수 함수 (docs/prereg/superinvestor.md).

네트워크·파일 없음. 입력은 분기 기준일에 알려진 13F 보유(조건 통과 매니저만)와 CUSIP→티커 연결, 구역 ③ 대상.
- votes: 매니저마다 비중 상위 k종목 중 직전 분기보다 비중이 늘었거나 새로 산 종목이면 1표(require_increase=False면 상위 k 보유만으로 1표)
- consensus: 매니저 보유 수(비중 무관) — 참고 판
- rank_order: 표 많은 순 → 표 준 매니저들의 비중 합 큰 순 → 티커 순
- select: 보유 종목이 순위 keep_rank 안이면 유지, 나머지는 순위대로 n종목까지, 한 섹터 최대 max_sector
- select_random: 대상에서 무작위 순서로 같은 섹터 제한
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    n: int = 15
    keep_rank: int = 30
    max_sector: int = 4
    top_k: int = 5


def top_holdings(now: pd.DataFrame, k: int) -> pd.DataFrame:
    """매니저(cik)마다 비중 상위 k줄. 입력 열: cik, cusip, value, weight. 같은 비중은 value 큰 순 → cusip 순."""
    s = now.sort_values(["cik", "weight", "value", "cusip"], ascending=[True, False, False, True])
    return s.groupby("cik", sort=False).head(k)


def votes(now: pd.DataFrame, old: pd.DataFrame, c2t: dict[str, str], members: set[str], k: int = 5,
          require_increase: bool = True) -> pd.DataFrame:
    """종목별 표 → DataFrame(index=ticker, votes, weight_sum). 구역 ③ 대상(members)만.

    한 매니저가 같은 티커의 CUSIP 여럿을 상위 k에 넣으면 1표(비중은 합). "늘었다" = 이번 비중 > 직전 분기 비중(없으면 0 = 새로 삼).
    늘었는지는 CUSIP별이 아니라 티커별 비중 합으로 본다."""
    top = top_holdings(now, k).copy()
    top["ticker"] = top["cusip"].map(c2t)
    top = top[top["ticker"].isin(members)]
    if top.empty:
        return pd.DataFrame(columns=["votes", "weight_sum"]).astype(float)
    cur = top.groupby(["cik", "ticker"], as_index=False)["weight"].sum()
    if require_increase:
        o = old.copy()
        o["ticker"] = o["cusip"].map(c2t)
        prev = o.dropna(subset=["ticker"]).groupby(["cik", "ticker"])["weight"].sum()
        pw = np.array([prev.get((c, t), 0.0) for c, t in zip(cur["cik"], cur["ticker"])])
        cur = cur[cur["weight"].to_numpy() > pw]
    out = cur.groupby("ticker").agg(votes=("cik", "nunique"), weight_sum=("weight", "sum"))
    return out.astype(float)


def consensus(now: pd.DataFrame, c2t: dict[str, str], members: set[str]) -> pd.DataFrame:
    """합의 판(참고): 종목별 보유 매니저 수(votes)와 그 매니저들의 비중 합(weight_sum)."""
    h = now.copy()
    h["ticker"] = h["cusip"].map(c2t)
    h = h[h["ticker"].isin(members)]
    out = h.groupby(["cik", "ticker"], as_index=False)["weight"].sum().groupby("ticker").agg(
        votes=("cik", "nunique"), weight_sum=("weight", "sum"))
    return out.astype(float)


def rank_order(table: pd.DataFrame) -> list[str]:
    """표 많은 순 → 비중 합 큰 순 → 티커 순. 표 0은 순위에 없다."""
    t = table[table["votes"] > 0].reset_index().rename(columns={"index": "ticker"})
    if "ticker" not in t.columns:
        t = t.rename(columns={t.columns[0]: "ticker"})
    t = t.sort_values(["votes", "weight_sum", "ticker"], ascending=[False, False, True])
    return list(t["ticker"])


def _sec(sector: dict[str, str | None], t: str) -> str:
    return sector.get(t) or "미상"


def select(order: list[str], sector: dict[str, str | None], held: list[str], p: Params = Params()) -> list[str]:
    """① 유지: 보유 종목 중 순위 keep_rank 안(순위 순, 섹터 최대 max_sector) ② 나머지: 순위대로 n종목까지(섹터 최대 max_sector).
    order = 거래 가능한 대상만 담긴 순위 목록. 고를 종목이 모자라면 n개보다 적게 돌려준다."""
    rank = {t: i + 1 for i, t in enumerate(order)}
    out: list[str] = []
    cnt: dict[str, int] = {}

    def add(t: str) -> None:
        s = _sec(sector, t)
        if t in out or len(out) >= p.n or cnt.get(s, 0) >= p.max_sector:
            return
        out.append(t)
        cnt[s] = cnt.get(s, 0) + 1

    for t in sorted((t for t in held if rank.get(t, 10**9) <= p.keep_rank), key=lambda x: rank[x]):
        add(t)
    for t in order:
        add(t)
    return out


def select_random(pool: list[str], sector: dict[str, str | None], rng: np.random.Generator, p: Params = Params()) -> list[str]:
    """무작위 대조: 대상(pool)을 무작위 순서로 섞어 섹터 최대 max_sector를 지키며 n종목."""
    order = [pool[i] for i in rng.permutation(len(pool))]
    return select(order, sector, [], p)
