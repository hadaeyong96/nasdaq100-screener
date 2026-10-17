"""가치평가 전략(value-score) 점수 70점 — 순수 함수 (docs/prereg/value_score.md 3장).

배점 합은 75점(① 20 + ② 10 + ③ 10 + ④~⑩ 5×7)이고, 총점은 70점 만점으로 환산한다(사용자 결정 2026-10-18).

네트워크·파일·DB·현재 시각에 접근하지 않는다. 미래 데이터 금지:
- 재무는 10-K·10-K/A 연간 값만, 접수일(filed) < 판단일인 사실만 쓴다(접수 다음 거래일부터 사용).
  같은 기간·같은 태그는 판단일 전 마지막 접수본(수정 공시는 접수일 기준).
- 주당 값(EPS)·주식 수는 E1 방식으로 접수일 다음 날 ~ 판단일 분할만 반영한다.
- 가격은 판단일 그날 실제 종가(분할 전)만 받는다. 음수 shift를 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

# 키 → 태그 우선순위 (기간마다 값이 있는 첫 태그)
TAGS: dict[str, list[str]] = {
    "rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet", "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet"],
    "ni": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted"],
    "opinc": ["OperatingIncomeLoss"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "gp": ["GrossProfit"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold",
             "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization"],
    "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
           "DepreciationAmortizationAndAccretionNet", "Depreciation"],
    "shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
             "Cash"],
    "assets": ["Assets"],
    "ltd_total": ["LongTermDebt"],
    "ltd_nc": ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"],
    "ltd_cur": ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"],
    "debt_cur": ["DebtCurrent"],
    "stb": ["ShortTermBorrowings", "CommercialPaper"],
}
FLOW_KEYS = {"rev", "ni", "eps", "opinc", "ocf", "capex", "gp", "cogs", "da", "shares"}
UNITS = {"eps": "USD/shares", "shares": "shares"}  # 나머지는 USD
ALL_TAGS = sorted({t for v in TAGS.values() for t in v})
ANNUAL_DAYS = (350, 380)
END_TOL_DAYS = 7  # 같은 회계연도 끝으로 보는 날짜 차 (52·53주 회계연도, 대차대조표 날짜)
YEAR_GAP = (350, 380)  # 이웃 회계연도 끝 간격 — 벗어나면 그 앞 해는 없는 것으로 본다

ITEMS = ["peg", "rev_growth", "roic", "eps_up", "gpa", "ocf_ni", "fcf", "net_debt", "shares", "fcf_yield"]
MAX_PTS = {"peg": 20.0, "rev_growth": 10.0, "roic": 10.0, "eps_up": 5.0, "gpa": 5.0, "ocf_ni": 5.0, "fcf": 5.0,
           "net_debt": 5.0, "shares": 5.0, "fcf_yield": 5.0}
CROSS = ("gpa", "fcf_yield")  # 그날 대상 안 순위로 매기는 항목
TOTAL = 70.0
MAX_NA = 2
TAX_ADJ = 0.79


# ── 연간 표 ───────────────────────────────────────────────────────────────


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def prepare_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """사실 표에 기간 일수(days, 순간 값은 NaN) 열을 붙인다. 입력 열: tag·start·end·val·filed·unit."""
    f = facts.copy()
    st = pd.to_datetime(f["start"], errors="coerce")
    f["days"] = (pd.to_datetime(f["end"]) - st).dt.days
    return f


def key_series(facts: pd.DataFrame, key: str, by_tag: dict | None = None) -> dict[str, tuple[float, str]]:
    """한 키의 {기간 끝: (값, 접수일)} — 태그 우선순위대로 기간마다 값이 있는 첫 태그, 같은 태그·기간은 마지막 접수본.

    입력: facts(prepare_facts 결과, 이미 판단일 전 접수본만), key(TAGS 키), by_tag(태그별로 나눈 facts, 있으면 재사용)
    """
    unit = UNITS.get(key, "USD")
    out: dict[str, tuple[float, str]] = {}
    for tag in TAGS[key]:
        f = by_tag.get(tag) if by_tag is not None else facts[facts["tag"] == tag]
        if f is None or f.empty:
            continue
        f = f[f["unit"] == unit]
        if key in FLOW_KEYS:
            f = f[(f["days"] >= ANNUAL_DAYS[0]) & (f["days"] <= ANNUAL_DAYS[1])]
        else:
            f = f[f["days"].isna()]
        if f.empty:
            continue
        last = f.sort_values("filed", kind="stable").groupby("end").tail(1)
        for e, v, fd in zip(last["end"], last["val"], last["filed"]):
            if e not in out:
                out[e] = (float(v), str(fd))
    return out


def _near(series: dict[str, tuple[float, str]], end: str) -> tuple[float, str] | None:
    """end와 END_TOL_DAYS 안인 기간 끝 중 가장 가까운 값."""
    best, bd = None, None
    for e, v in series.items():
        dd = abs(_days(e, end))
        if dd <= END_TOL_DAYS and (bd is None or dd < bd):
            best, bd = v, dd
    return best


def annual_table(facts: pd.DataFrame, asof) -> pd.DataFrame:
    """판단일 asof 전에 접수된 10-K 사실로 만든 연간 표. 행 = 최근 회계연도부터 이어진 해(F0, F1, ...).

    입력: facts(한 종목의 사실, 열 tag·start·end·val·filed·form·unit), asof(판단일)
    출력: DataFrame(index=회계연도 끝 문자열, 최근이 먼저; 열 = TAGS 키 값, eps_filed·shares_filed = 그 값의 접수일).
         회계연도 끝은 순이익·매출·EPS 연간 사실의 기간 끝. 이웃 해 간격이 350~380일을 벗어나면 거기서 끊는다.
    """
    cut = pd.Timestamp(asof).date().isoformat()
    f = facts[facts["filed"] < cut]
    if f.empty:
        return pd.DataFrame()
    if "days" not in f:
        f = prepare_facts(f)
    by_tag = dict(tuple(f.groupby("tag")))
    ser = {k: key_series(f, k, by_tag) for k in TAGS}
    ends = sorted(set(ser["ni"]) | set(ser["rev"]) | set(ser["eps"]), reverse=True)
    chain: list[str] = []
    for e in ends:
        if not chain:
            chain.append(e)
            continue
        gap = _days(e, chain[-1])
        if gap < YEAR_GAP[0]:
            continue  # 같은 해의 다른 표기(52·53주) — 먼저 잡은 끝을 쓴다
        if gap > YEAR_GAP[1]:
            break
        chain.append(e)
    if not chain:
        return pd.DataFrame()
    rows = []
    for e in chain:
        row = {"end": e}
        for k in TAGS:
            v = _near(ser[k], e)
            row[k] = v[0] if v else np.nan
            if k in ("eps", "shares"):
                row[f"{k}_filed"] = v[1] if v else None
        rows.append(row)
    return pd.DataFrame(rows).set_index("end")


def split_factor(splits: pd.Series | None, filed: str | None, asof) -> float:
    """접수일 다음 날 ~ 판단일 분할 비율의 곱 (E1 방식). splits: 날짜 → 비율(예 2.0)."""
    if splits is None or not len(splits) or filed is None:
        return 1.0
    a = pd.Timestamp(asof)
    s = splits[(splits.index > pd.Timestamp(filed)) & (splits.index <= a)]
    return float(np.prod(s.to_numpy(dtype=float))) if len(s) else 1.0


def debt_of(row: pd.Series) -> float:
    """이자부채 = 장기부채(유동 포함) + 단기차입. 태그가 하나도 없으면 0 (구현 해석)."""
    st = row["stb"] if pd.notna(row["stb"]) else 0.0
    if pd.notna(row["ltd_total"]):
        return float(row["ltd_total"]) + st
    nc = row["ltd_nc"] if pd.notna(row["ltd_nc"]) else 0.0
    if pd.notna(row["debt_cur"]):
        return float(nc + row["debt_cur"])
    cur = row["ltd_cur"] if pd.notna(row["ltd_cur"]) else 0.0
    return float(nc + cur + st)


# ── 종목 하나의 항목 값 ────────────────────────────────────────────────────


@dataclass
class Metrics:
    """종목 하나의 그날 값. pts[항목] = 점수(None = N/A), raw = 순위 항목(gpa·fcf_yield)의 원래 값."""

    years: int
    profitable: bool  # 최근 3개 회계연도 순이익 모두 > 0 (값이 모두 있어야 True)
    pts: dict
    raw: dict
    peg: float  # 동점 정렬용 (계산 못 하면 inf)
    eps_growth: float | None = None


def _cagr3(a, b) -> float | None:
    """3년 연평균 성장률 (a = 최근, b = 3년 전). 값이 없거나 b ≤ 0이면 None."""
    if pd.isna(a) or pd.isna(b) or b <= 0 or a <= 0:
        return None
    return (a / b) ** (1.0 / 3.0) - 1.0


def metrics(table: pd.DataFrame, price: float, splits: pd.Series | None, asof) -> Metrics:
    """연간 표 + 그날 실제 종가 → 항목별 점수(순위 항목 제외)와 원래 값.

    입력: table(annual_table), price(판단일 실제 종가, 분할 전), splits(그 종목 분할, 판단일까지만 씀), asof
    """
    n = len(table)
    pts = {k: None for k in ITEMS}
    raw = {"gpa": None, "fcf_yield": None}
    if n == 0:
        return Metrics(0, False, pts, raw, float("inf"))
    t = table
    ni = t["ni"].to_numpy(dtype=float)
    profitable = n >= 3 and not np.isnan(ni[:3]).any() and bool((ni[:3] > 0).all())

    def eps(i):
        if i >= n or pd.isna(t["eps"].iloc[i]):
            return np.nan
        return float(t["eps"].iloc[i]) / split_factor(splits, t["eps_filed"].iloc[i], asof)

    def shares(i):
        if i >= n or pd.isna(t["shares"].iloc[i]):
            return np.nan
        return float(t["shares"].iloc[i]) * split_factor(splits, t["shares_filed"].iloc[i], asof)

    def col(k, i):
        return float(t[k].iloc[i]) if i < n and pd.notna(t[k].iloc[i]) else np.nan

    # ① PEG 20
    peg_val, g_eps = float("inf"), None
    e0, e3 = eps(0), eps(3)
    if not np.isnan(e0):
        if e0 <= 0:
            pts["peg"] = 0.0
        elif not np.isnan(e3) and e3 > 0:
            g_eps = (e0 / e3) ** (1.0 / 3.0) - 1.0
            if g_eps <= 0:
                pts["peg"] = 0.0
            else:
                g = min(g_eps, 0.5)
                peg_val = (price / e0) / (g * 100.0)
                pts["peg"] = 20.0 if peg_val <= 1.0 else 13.3 if peg_val <= 1.5 else 6.7 if peg_val <= 2.0 else 0.0
    # ② 매출 3년 성장 10
    g = _cagr3(col("rev", 0), col("rev", 3))
    if g is not None:
        pts["rev_growth"] = 10.0 if 0.10 <= g <= 0.35 else 5.0 if (0.05 <= g < 0.10 or 0.35 < g <= 0.50) else 0.0
    # ③ ROIC 3년 평균 10
    if n >= 3:
        rs = []
        for i in range(3):
            op, eq, ca = col("opinc", i), col("equity", i), col("cash", i)
            if np.isnan(op) or np.isnan(eq) or np.isnan(ca):
                rs = None
                break
            den = eq + debt_of(t.iloc[i]) - ca
            if den <= 0:
                rs = None
                break
            rs.append(op * TAX_ADJ / den)
        if rs:
            r = float(np.mean(rs))
            pts["roic"] = 10.0 if r >= 0.15 else 5.0 if r >= 0.10 else 0.0
    # ④ 희석 EPS 전년 대비 증가 횟수 5
    comps = []
    for i in range(4):
        a, b = eps(i), eps(i + 1)
        if np.isnan(a) or np.isnan(b):
            break
        comps.append(a > b)
    if len(comps) == 4:
        k = sum(comps)
        pts["eps_up"] = 5.0 if k == 4 else 2.5 if k == 3 else 0.0
    elif len(comps) == 3:
        k = sum(comps)
        pts["eps_up"] = 5.0 if k == 3 else 2.5 if k == 2 else 0.0
    # ⑤ 매출총이익 ÷ 총자산 (순위)
    gp = col("gp", 0)
    if np.isnan(gp) and not np.isnan(col("rev", 0)) and not np.isnan(col("cogs", 0)):
        gp = col("rev", 0) - col("cogs", 0)
    ta = col("assets", 0)
    if not np.isnan(gp) and not np.isnan(ta) and ta > 0:
        raw["gpa"] = gp / ta
    # ⑥ 3년 영업현금흐름 합 ÷ 3년 순이익 합 5
    if n >= 3:
        ocf3 = [col("ocf", i) for i in range(3)]
        ni3 = [col("ni", i) for i in range(3)]
        if not np.isnan(ocf3).any() and not np.isnan(ni3).any() and sum(ni3) > 0:
            q = sum(ocf3) / sum(ni3)
            pts["ocf_ni"] = 5.0 if q >= 1.0 else 2.5 if q >= 0.8 else 0.0
    # ⑦ FCF 5
    fcf = [col("ocf", i) - col("capex", i) for i in range(min(n, 3))]
    if n >= 3 and not np.isnan(fcf).any():
        allpos = all(x > 0 for x in fcf)
        rev0 = col("rev", 0)
        margin_ok = not np.isnan(rev0) and rev0 > 0 and fcf[0] / rev0 >= 0.10
        pts["fcf"] = 5.0 if (allpos and margin_ok) else 2.5 if allpos else 0.0
    # ⑧ (이자부채 − 현금) ÷ EBITDA 5
    op0, da0, ca0 = col("opinc", 0), col("da", 0), col("cash", 0)
    if not np.isnan(op0) and not np.isnan(da0) and not np.isnan(ca0):
        ebitda = op0 + da0
        if ebitda <= 0:
            pts["net_debt"] = 0.0
        else:
            x = (debt_of(t.iloc[0]) - ca0) / ebitda
            pts["net_debt"] = 5.0 if x < 1.0 else 2.5 if x < 2.0 else 0.0
    # ⑨ 3년 희석 가중평균 주식 수 변화 5
    s0, s3 = shares(0), shares(3)
    if not np.isnan(s0) and not np.isnan(s3) and s3 > 0:
        ch = s0 / s3 - 1.0
        pts["shares"] = 5.0 if ch <= 0.02 else 2.5 if ch <= 0.06 else 0.0
    # ⑩ FCF 수익률 (순위)
    if fcf and not np.isnan(fcf[0]) and not np.isnan(s0) and s0 > 0 and price > 0:
        raw["fcf_yield"] = fcf[0] / (price * s0)
    return Metrics(n, profitable, pts, raw, peg_val, g_eps)


# ── 그날 점수 ─────────────────────────────────────────────────────────────


def tercile_points(x: float | None, pool: np.ndarray) -> float | None:
    """그날 대상 값 pool 안에서 x의 자리: 상위 1/3 → 5, 중간 → 2.5, 하위 → 0. x 없으면 None.

    자리 = pool 중 x보다 큰 값의 개수(같은 값은 같은 자리). 그 수가 n/3 미만이면 상위, 2n/3 미만이면 중간.
    """
    if x is None or not len(pool):
        return None
    above = int((pool > x).sum())
    n = len(pool)
    return 5.0 if above < n / 3.0 else 2.5 if above < 2.0 * n / 3.0 else 0.0


def score_day(ms: dict[str, Metrics], universe: list[str], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """그날 점수표. 순위 항목은 universe(그날 대상) 값으로 자리를 정하고, universe 밖 종목(보유 중 이탈)도 같은 기준으로 매긴다.

    입력: ms({티커: Metrics} — universe와 보유 종목), universe(그날 대상 티커), weights(항목별 배점, 없으면 MAX_PTS;
         0이면 그 항목을 쓰지 않음 — 참고 보고용 배점판)
    출력: DataFrame(index=티커; 열 = 항목 점수, n_na, total, peg_value(동점 정렬용 PEG 값, 없으면 inf)). N/A가 2개 넘으면 total = NaN(순위 제외).
         total = 받은 점수 합 × 70 ÷ (N/A 아닌 항목 배점 합) — N/A는 0점이 아니라 빼고 비례 환산 (구현 해석).
         기본 배점 합은 75점이라 이 식이 70점 만점 환산도 겸한다 [사용자 결정 2026-10-18: 70점으로 환산, 문턱 49·35·56 그대로].
    """
    w = dict(MAX_PTS if weights is None else weights)
    items = [k for k in ITEMS if w.get(k, 0) > 0]
    pools = {k: np.array([ms[t].raw[k] for t in universe if t in ms and ms[t].raw[k] is not None], dtype=float)
             for k in CROSS}
    rows = {}
    for t, m in ms.items():
        r = {}
        for k in items:
            if k in CROSS:
                p = tercile_points(m.raw[k], pools[k])
            else:
                p = m.pts[k]
            r[k] = None if p is None else p * w[k] / MAX_PTS[k]
        na = sum(1 for k in items if r[k] is None)
        got = sum(r[k] for k in items if r[k] is not None)
        avail = sum(w[k] for k in items if r[k] is not None)
        r["n_na"] = na
        r["total"] = got * TOTAL / avail if (na <= MAX_NA and avail > 0) else np.nan
        r["peg_value"] = m.peg
        rows[t] = r
    return pd.DataFrame.from_dict(rows, orient="index")


def total_from_points(pts: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """항목 점수표(score_day의 항목 열, 기본 배점 기준)에서 다른 배점판의 총점을 다시 계산한다 (참고 보고용).

    배점 0인 항목은 빼고, N/A(빈 칸)가 2개 넘으면 NaN, 아니면 받은 점수 합 × 70 ÷ (N/A 아닌 항목 배점 합).
    """
    items = [k for k in ITEMS if weights.get(k, 0) > 0]
    got = sum(pts[k].fillna(0.0) * weights[k] / MAX_PTS[k] for k in items)
    avail = sum(pts[k].notna() * weights[k] for k in items)
    na = sum(pts[k].isna().astype(int) for k in items)
    out = got * TOTAL / avail.replace(0, np.nan)
    return out.where(na <= MAX_NA)


# ── 고르기 규칙 ───────────────────────────────────────────────────────────


def rank_order(df: pd.DataFrame) -> list[str]:
    """점수 높은 순, 같으면 PEG 낮은 순(계산 못 하면 맨 뒤), 그다음 티커 순."""
    d = df[df["total"].notna()].copy()
    d["_t"] = d.index
    return list(d.sort_values(["total", "peg_value", "_t"], ascending=[False, True, True])["_t"])


def pick(order: list[str], n: int, sic2: dict[str, str | None], held: list[str], cap: int = 3) -> list[str]:
    """order 순서대로 n개를 고른다. 같은 SIC 2자리 업종은 보유 포함 최대 cap개. 이미 보유한 종목은 건너뛴다."""
    cnt: dict = {}
    for h in held:
        s = sic2.get(h)
        cnt[s] = cnt.get(s, 0) + 1
    out = []
    for t in order:
        if len(out) >= n:
            break
        if t in held or t in out:
            continue
        s = sic2.get(t)
        if s is not None and cnt.get(s, 0) >= cap:
            continue
        out.append(t)
        cnt[s] = cnt.get(s, 0) + 1
    return out


def swap_choice(cands: list[str], cand_score: dict[str, float], held_score: dict[str, float],
                sic2: dict[str, str | None], min_cand: float = 56.0, gap: float = 10.0, cap: int = 3) -> tuple[str, str] | None:
    """12월 갈아타기 1건: (팔 보유 종목, 살 후보) 또는 None.

    cands: 비보유 후보를 점수 순(rank_order)으로. held_score: 점수가 있는 보유 종목만.
    1차 — 같은 업종 보유가 있는 후보: 그 업종 최저 점수 보유와 비교(점수 차 ≥ gap).
    2차 — 모든 후보: 전체 최저 점수 보유와 비교, 업종 상한(보유 cap개, 파는 종목 업종은 하나 빠짐) 지킴.
    후보는 min_cand점 이상. 같은 점수 보유는 티커 순으로 앞선 것을 판다.
    """
    if not held_score:
        return None
    ok = [c for c in cands if cand_score.get(c) is not None and cand_score[c] >= min_cand and c not in held_score]
    by_sec: dict = {}
    for h, s in held_score.items():
        by_sec.setdefault(sic2.get(h), []).append((s, h))
    for c in ok:
        sec = sic2.get(c)
        if sec is not None and sec in by_sec:
            s, h = min(by_sec[sec])
            if cand_score[c] - s >= gap:
                return h, c
    s_low, h_low = min((s, h) for h, s in held_score.items())
    cnt: dict = {}
    for h in held_score:
        cnt[sic2.get(h)] = cnt.get(sic2.get(h), 0) + 1
    for c in ok:
        if cand_score[c] - s_low < gap:
            break  # 점수 순이라 뒤 후보는 더 낮다
        sec = sic2.get(c)
        n_sec = cnt.get(sec, 0) - (1 if sic2.get(h_low) == sec else 0)
        if sec is not None and n_sec >= cap:
            continue
        return h_low, c
    return None


def swap_qty(remaining_krw: float, per_share_gain_krw: float, held_qty: float) -> float:
    """갈아타기 수량: 남은 공제 ÷ 1주당 이익, 최대 보유 수량의 절반. 손실(1주당 이익 ≤ 0)이거나 공제로 절반을 다 덮으면 절반.
    남은 공제 ≤ 0이면 0(건너뜀). 소수 주 허용."""
    if remaining_krw <= 0:
        return 0.0
    half = held_qty / 2.0
    if per_share_gain_krw <= 0:
        return half
    return min(remaining_krw / per_share_gain_krw, half)
