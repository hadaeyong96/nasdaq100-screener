"""실적 동반 시총 성장 전략(mcap-growth) — 검증 구역 ③ 순수 함수 (docs/prereg/mcap_growth.md).

네트워크·파일·현재 시각에 접근하지 않는다. 판단일 x의 값은 x 전에 접수(filed < x)된 EDGAR 사실만 쓴다(미래 데이터 없음).

- 주식 수: dei:EntityCommonStockSharesOutstanding(표지 주식 수, 같은 접수·같은 기준일 값은 합 = 여러 종류 주식), 없으면
  us-gaap:WeightedAverageNumberOfSharesOutstandingBasic(가장 최근 기간 끝, 3개월 기간 우선). 400일보다 오래된 값은 쓰지 않는다.
  분할: 기준일(end) 뒤 ~ 쓰는 날 사이 분할 비율 곱을 곱한다(E1 방식, 주식 수 × 분할).
- 매출: 직전 4분기 합(TTM) = 최근 연간 + 올해 누적 − 전년 같은 누적(연간 보고면 연간 그대로). 전년 대비 = TTM(E) ÷ TTM(E − 1년) − 1.
- 점수: 시총[t−s] ÷ 시총[t−L] − 1 (기본 s = 21, L = 252거래일).
- 고르기: 실적·희석 조건 통과 종목 중 섹터별 점수 1등 → 전체 점수 순, 15종목, 섹터 최대 3, 보유 완충(점수 순위 30 안 + 조건 통과).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SHARES_DEI = "EntityCommonStockSharesOutstanding"
SHARES_WA = "WeightedAverageNumberOfSharesOutstandingBasic"
REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet"]  # core/value_score.TAGS["rev"]와 같은 순서
SHARES_MAX_AGE = 400  # 일
REV_MAX_AGE = 200  # 일 — 최근 기간 끝이 판단일보다 이만큼 넘게 오래되면 매출 없음
KINDS = {"Q": (80, 100), "H": (170, 195), "N": (260, 285), "A": (350, 380)}
TOL = 10  # 기간 끝 맞추기 허용(일)


@dataclass(frozen=True)
class Params:
    n: int = 15
    sector_cap: int = 3
    keep_rank: int = 30
    skip: int = 21
    look: int = 252
    dilution: float = 0.05


# ── 주식 수 ──────────────────────────────────────────────────────────────────


def split_factor(splits: pd.Series | None, a, b) -> float:
    """a < 날짜 ≤ b 사이 분할 비율 곱(없으면 1). splits: index 날짜, 값 비율(2 = 1주 → 2주)."""
    if splits is None or not len(splits):
        return 1.0
    a, b = pd.Timestamp(a), pd.Timestamp(b)
    s = splits[(splits.index > a) & (splits.index <= b)]
    return float(np.prod(s.to_numpy(dtype=float))) if len(s) else 1.0


def shares_asof(sh: pd.DataFrame, x, splits: pd.Series | None = None) -> float | None:
    """x 전에 접수된 주식 수(x 기준 분할 조정) 또는 None.

    sh 열: tag, end, filed, val, accn, start(가중평균만). 날짜는 'YYYY-MM-DD' 문자열."""
    if sh is None or sh.empty:
        return None
    x = pd.Timestamp(x)
    xs = x.date().isoformat()
    lo = (x - pd.Timedelta(days=SHARES_MAX_AGE)).date().isoformat()
    s = sh[(sh["filed"] < xs) & (sh["end"] >= lo)]
    dei = s[s["tag"] == SHARES_DEI]
    if len(dei):
        f = dei["filed"].max()
        g = dei[dei["filed"] == f]
        end = g["end"].max()
        g = g[g["end"] == end]
        accn = sorted(g["accn"].dropna().unique())
        if accn:
            g = g[g["accn"] == accn[-1]]
        val = float(g["val"].sum())
        if val > 0:
            return val * split_factor(splits, end, x)
    wa = s[s["tag"] == SHARES_WA]
    if not len(wa):
        return None
    end = wa["end"].max()
    g = wa[wa["end"] == end].copy()
    dur = (pd.to_datetime(g["end"]) - pd.to_datetime(g["start"])).dt.days
    q = g[(dur >= 80) & (dur <= 100)]
    g = q if len(q) else g
    g = g[g["filed"] == g["filed"].max()]
    val = float(g["val"].iloc[-1])
    return val * split_factor(splits, end, x) if val > 0 else None


# ── 매출 ─────────────────────────────────────────────────────────────────────


def revenue_periods(rev: pd.DataFrame, x) -> pd.DataFrame:
    """x 전 접수 매출 사실 → 기간(start, end)마다 값 하나(태그 우선순위 첫 태그, 같은 태그는 마지막 접수본).
    출력 열: start, end(Timestamp), kind(Q·H·N·A), val."""
    if rev is None or rev.empty:
        return pd.DataFrame(columns=["start", "end", "kind", "val"])
    xs = pd.Timestamp(x).date().isoformat()
    r = rev[(rev["filed"] < xs) & rev["start"].notna()].copy()
    if r.empty:
        return pd.DataFrame(columns=["start", "end", "kind", "val"])
    r["prio"] = r["tag"].map({t: i for i, t in enumerate(REV_TAGS)})
    r = r.sort_values(["start", "end", "prio", "filed"])
    r = r[r["prio"] == r.groupby(["start", "end"])["prio"].transform("min")]
    out = r.drop_duplicates(["start", "end"], keep="last")[["start", "end", "val"]].reset_index(drop=True)
    out["start"], out["end"] = pd.to_datetime(out["start"]), pd.to_datetime(out["end"])
    dur = (out["end"] - out["start"]).dt.days
    out["kind"] = None
    for k, (a, b) in KINDS.items():
        out.loc[(dur >= a) & (dur <= b), "kind"] = k
    return out[out["kind"].notna()].reset_index(drop=True)


def _near(p: pd.DataFrame, kind, end, tol=TOL) -> pd.Series | None:
    k = p[(p["kind"].isin(kind if isinstance(kind, (list, tuple, set)) else [kind]))
          & ((p["end"] - end).abs() <= pd.Timedelta(days=tol))]
    if not len(k):
        return None
    return k.loc[(k["end"] - end).abs().idxmin()]


def ttm_at(p: pd.DataFrame, end) -> float | None:
    """기간 끝 end 무렵의 직전 4분기 합. 연간이면 그대로, 아니면 연간(누적 시작 전날 무렵 끝) + 누적 − 전년 같은 누적."""
    end = pd.Timestamp(end)
    a = _near(p, "A", end)
    if a is not None:
        return float(a["val"])
    for k in ("N", "H", "Q"):
        y = _near(p, k, end)
        if y is None:
            continue
        a0 = _near(p, "A", y["start"] - pd.Timedelta(days=1))
        y0 = _near(p, k, end - pd.Timedelta(days=364))
        if a0 is not None and y0 is not None:
            return float(a0["val"] + y["val"] - y0["val"])
    return None


def revenue_growth(rev: pd.DataFrame, x) -> float | None:
    """x에 알려진 최근 기간 끝 E의 TTM ÷ E − 1년 TTM − 1. 못 구하면 None(E가 200일 넘게 오래됐거나 전년 TTM ≤ 0 포함)."""
    p = revenue_periods(rev, x)
    if p.empty:
        return None
    e = p["end"].max()
    if e < pd.Timestamp(x) - pd.Timedelta(days=REV_MAX_AGE):
        return None
    now = ttm_at(p, e)
    prev = ttm_at(p, e - pd.Timedelta(days=364))
    if now is None or prev is None or prev <= 0:
        return None
    return now / prev - 1


# ── 점수·고르기 ───────────────────────────────────────────────────────────────


def score(mcap_skip: float | None, mcap_look: float | None) -> float | None:
    """시총[t−s] ÷ 시총[t−L] − 1."""
    if mcap_skip is None or mcap_look is None or not np.isfinite(mcap_skip) or not np.isfinite(mcap_look) or mcap_look <= 0:
        return None
    return mcap_skip / mcap_look - 1


def passes(rev_g: float | None, sector_median: float | None, share_g: float | None, p: Params = Params(),
           use_rev: bool = True) -> bool:
    """c) 매출 성장 > 같은 섹터 중앙값(use_rev=False면 생략) 그리고 d) 12개월 주식 수 증가율 ≤ 5%. 값이 없으면 통과 못 함."""
    if share_g is None or not np.isfinite(share_g) or share_g > p.dilution:
        return False
    if not use_rev:
        return True
    return rev_g is not None and sector_median is not None and np.isfinite(rev_g) and rev_g > sector_median


def sector_medians(rev_g: dict[str, float | None], sector: dict[str, str]) -> dict[str, float]:
    """섹터별 매출 성장률 중앙값(그 섹터 대상 중 값 있는 종목)."""
    by: dict[str, list[float]] = {}
    for t, g in rev_g.items():
        if g is not None and np.isfinite(g) and sector.get(t):
            by.setdefault(sector[t], []).append(g)
    return {s: float(np.median(v)) for s, v in by.items()}


def rank_order(scores: dict[str, float]) -> list[str]:
    """점수 큰 순(같으면 티커 순)."""
    return [t for t, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def select(scores: dict[str, float], ok: dict[str, bool], sector: dict[str, str], held: list[str], p: Params = Params()) -> list[str]:
    """그 분기 15종목. scores = 대상(점수 있음) 전체, ok = c)·d) 통과, held = 지금 보유.

    1) 완충: 보유 중 점수 순위(대상 전체) keep_rank 안 + 통과 → 유지(점수 순, 섹터 최대 3)
    2) 섹터 1등: 통과 종목 중 섹터마다 점수 1등(점수 큰 섹터부터, 빈자리·섹터 한도 안에서)
    3) 나머지: 통과 종목 전체 점수 순(섹터 최대 3)으로 n까지."""
    order = rank_order(scores)
    rank = {t: i + 1 for i, t in enumerate(order)}
    cnt: dict[str, int] = {}
    out: list[str] = []

    def add(t):
        s = sector.get(t)
        if t in out or len(out) >= p.n or cnt.get(s, 0) >= p.sector_cap:
            return
        out.append(t)
        cnt[s] = cnt.get(s, 0) + 1

    for t in sorted((h for h in held if h in rank and rank[h] <= p.keep_rank and ok.get(h)), key=lambda h: rank[h]):
        add(t)
    passing = [t for t in order if ok.get(t)]
    seen = set()
    for t in passing:
        s = sector.get(t)
        if s in seen:
            continue
        seen.add(s)
        add(t)
    for t in passing:
        add(t)
    return out


def select_random(pool: list[str], sector: dict[str, str], rng: np.random.Generator, p: Params = Params()) -> list[str]:
    """무작위 대조: 같은 대상(pool)에서 섹터 최대 3만 지켜 무작위 n종목(무작위 순서로 한도 안에서 채움)."""
    out, cnt = [], {}
    for k in rng.permutation(len(pool)):
        t = pool[int(k)]
        s = sector.get(t)
        if cnt.get(s, 0) >= p.sector_cap:
            continue
        out.append(t)
        cnt[s] = cnt.get(s, 0) + 1
        if len(out) >= p.n:
            break
    return out


def mcap_ranks(mcap: dict[str, float]) -> dict[str, int]:
    """시총 순위(1 = 가장 큼, 같으면 티커 순)."""
    return {t: i + 1 for i, t in enumerate(rank_order(mcap))}
