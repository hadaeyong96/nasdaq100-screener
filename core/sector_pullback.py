"""탐색 B 「강한 업종 속 강한 종목, 첫 눌림 매수」 — 순수 함수 (configs/sector_pullback_preregistration.yaml, 판정 아님).

네트워크·파일·현재 시각에 접근하지 않는다. 미래 데이터 금지: 음수 shift를 쓰지 않고, i일 값은 i일까지의 종가로만
만든다(눌림 창은 돌파일부터 앞으로 하루씩 보며 그날까지의 값만 쓴다). 날짜 축은 QQQ 거래일, 회사 축은 정수 열.

변형: B0 기본형 / B1 눌림 없이 돌파 다음 날 / B2 업종 조건 없음 / B3 업종 강도 63거래일.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

VARIANTS = ("B0", "B1", "B2", "B3")
CROSS_WINDOW = 20
PULLBACK_FROM_PEAK = 0.03
SMA20_TOL = 1.01
TOP_BUY, TOP_HOLD = 3, 5
MIN_SECTOR_MEMBERS = 3
MAX_POS = 10
MACD_MIN_OBS = 35  # 26일 EMA + 9일 신호선 준비 전 MACD는 쓰지 않는다


# ── 같은 회사 여러 주식 묶기 ─────────────────────────────────────────────


def pick_representative(shares: dict[str, pd.DataFrame]) -> str:
    """가격 기록(행 수)이 가장 긴 주식, 같으면 티커 알파벳 순. 입력 {티커: DataFrame(close, div)} / 출력 티커."""
    return sorted(shares, key=lambda t: (-int(shares[t]["close"].notna().sum()), t))[0]


def combine_shares(shares: dict[str, pd.DataFrame], first_dates: dict[str, pd.Timestamp] | None = None) -> tuple[str, pd.DataFrame]:
    """회사 한 곳의 주식들 → (대표 티커, DataFrame(close, div)).

    실제 상장일(first_dates) 전 행은 버린다. 대표 주식 가격이 있는 날은 대표 가격, 빈 날만 다른 주식(알파벳 순)의
    일간 수익률로 이어 붙인다(가격 수준 차이로 생기는 가짜 점프 방지). 분배금은 그날 쓴 주식의 분배금 ÷ 그 주식 가격 ×
    이어 붙인 가격(배당 수익률 유지). 끝에 대표 주식의 마지막 가격 수준에 맞춘다.
    """
    first_dates = first_dates or {}
    clean = {}
    for t, df in shares.items():
        d = df.dropna(subset=["close"])
        if t in first_dates:
            d = d[d.index >= pd.Timestamp(first_dates[t])]
        if len(d):
            clean[t] = d
    if not clean:
        return "", pd.DataFrame(columns=["close", "div"])
    rep = pick_representative(clean)
    if len(clean) == 1:
        return rep, clean[rep][["close", "div"]].copy()
    order = [rep] + sorted(t for t in clean if t != rep)
    idx = sorted(set().union(*(set(d.index) for d in clean.values())))
    px = {t: clean[t]["close"].to_dict() for t in order}
    dv = {t: clean[t]["div"].fillna(0.0).to_dict() for t in order}
    last_seen: dict[str, pd.Timestamp] = {}
    val_at: dict[pd.Timestamp, float] = {}
    out_p, out_d, prev_val = [], [], None
    for d in idx:
        src = next(t for t in order if d in px[t])
        if prev_val is None:
            val = px[src][d]
        elif src in last_seen:
            # 그 주식이 마지막으로 값이 있던 날의 이어 붙인 값에 그 주식의 그동안 수익률을 곱한다
            # (대표 주식이 빈 날 뒤 돌아오면 그 사이 움직임을 두 번 세지 않고 대표 주식 수익률로 다시 맞춰진다)
            ls = last_seen[src]
            val = val_at[ls] * px[src][d] / px[src][ls]
        else:
            val = prev_val  # 처음 쓰는 주식이고 이을 날이 없음: 그날 수익 0
        val_at[d] = val
        out_p.append(val)
        out_d.append(dv[src].get(d, 0.0) * val / px[src][d])
        for t in order:
            if d in px[t]:
                last_seen[t] = d
        prev_val = val
    out = pd.DataFrame({"close": out_p, "div": out_d}, index=pd.DatetimeIndex(idx))
    rep_last = clean[rep].index.max()
    k = float(clean[rep].loc[rep_last, "close"]) / float(out.loc[rep_last, "close"])
    out["close"] *= k
    out["div"] *= k
    return rep, out


# ── 지표 행렬 ────────────────────────────────────────────────────────────


@dataclass
class Market:
    """날짜(QQQ 거래일) × 회사 행렬 묶음. 모든 값은 그날 종가까지로만 만든다."""

    days: pd.DatetimeIndex
    companies: list[str]
    close: np.ndarray            # D×C, 없으면 NaN
    div: np.ndarray              # D×C 주당 분배금
    member: np.ndarray           # D×C bool, 그날 대상
    sector: np.ndarray           # C, 업종 번호(-1 없음)
    sectors: list[str]
    qclose: np.ndarray           # D
    qdiv: np.ndarray             # D
    ind: dict = field(default_factory=dict)  # 지표 행렬들 (build_indicators)

    @property
    def last_idx(self) -> np.ndarray:
        """회사별 마지막으로 종가가 있는 날 번호 (없으면 -1)."""
        ok = ~np.isnan(self.close)
        rev = ok[::-1].argmax(axis=0)
        return np.where(ok.any(axis=0), len(self.days) - 1 - rev, -1)


def nday_return(close: np.ndarray, n: int) -> np.ndarray:
    """N거래일 수익률 = 오늘 ÷ N거래일 전(날짜 축 기준) − 1. 두 날 중 하나라도 NaN이면 NaN. 1차원·2차원 모두."""
    out = np.full(close.shape, np.nan)
    out[n:] = close[n:] / close[:-n] - 1.0
    return out


def own_series_indicators(close_col: np.ndarray) -> dict[str, np.ndarray]:
    """회사 한 곳의 종가 열(NaN = 그날 없음) → 그 회사 자기 종가로 만든 sma20·sma50·macd (같은 길이, 없는 날 NaN)."""
    ok = ~np.isnan(close_col)
    out = {k: np.full(close_col.shape, np.nan) for k in ("sma20", "sma50", "macd")}
    if ok.sum() == 0:
        return out
    s = pd.Series(close_col[ok])
    out["sma20"][ok] = s.rolling(20).mean().to_numpy()
    out["sma50"][ok] = s.rolling(50).mean().to_numpy()
    m = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).to_numpy()
    m[: MACD_MIN_OBS - 1] = np.nan
    out["macd"][ok] = m
    return out


def build_indicators(mk: Market) -> None:
    """mk.ind에 ret126·ret63·sma20·sma50·macd(D×C), q126·q63(D)를 채운다."""
    D, C = mk.close.shape
    ind = {"sma20": np.full((D, C), np.nan), "sma50": np.full((D, C), np.nan), "macd": np.full((D, C), np.nan)}
    for j in range(C):
        o = own_series_indicators(mk.close[:, j])
        for k in ind:
            ind[k][:, j] = o[k]
    ind["ret126"], ind["ret63"] = nday_return(mk.close, 126), nday_return(mk.close, 63)
    ind["q126"], ind["q63"] = nday_return(mk.qclose, 126), nday_return(mk.qclose, 63)
    mk.ind = ind


def week_end_mask(days: pd.DatetimeIndex) -> np.ndarray:
    """매주 마지막 거래일(ISO 주 기준) True. 마지막 날은 그 주가 끝났는지 모르므로 다음 날이 없으면 True로 둔다."""
    iso = days.isocalendar()
    key = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    m = np.ones(len(days), dtype=bool)
    m[:-1] = key[:-1] != key[1:]
    return m


def sector_avg(ret: np.ndarray, member: np.ndarray, sector: np.ndarray, n_sec: int) -> tuple[np.ndarray, np.ndarray]:
    """그날 대상이고 수익률이 있는 회사들의 업종 평균·회사 수. 출력 (D×S 평균(없으면 NaN), D×S 개수)."""
    D = ret.shape[0]
    avg, cnt = np.full((D, n_sec), np.nan), np.zeros((D, n_sec), dtype=int)
    valid = member & ~np.isnan(ret)
    for s in range(n_sec):
        cols = sector == s
        v = valid[:, cols]
        n = v.sum(axis=1)
        tot = np.where(v, ret[:, cols], 0.0).sum(axis=1)
        cnt[:, s] = n
        avg[:, s] = np.where(n > 0, tot / np.maximum(n, 1), np.nan)
    return avg, cnt


def weekly_sector_rank(ret: np.ndarray, qret: np.ndarray, member: np.ndarray, sector: np.ndarray, n_sec: int,
                       judge: np.ndarray, min_members: int = MIN_SECTOR_MEMBERS) -> np.ndarray:
    """매주 마지막 거래일 업종 순위(업종 평균 − QQQ, 큰 순 1부터), 다음 판정일 전날까지 그대로. 순위 없음 = 0.
    입력 ret(D×C N거래일 수익률), qret(D), judge(D bool 판정일) / 출력 D×S int."""
    avg, cnt = sector_avg(ret, member, sector, n_sec)
    D = ret.shape[0]
    rank = np.zeros((D, n_sec), dtype=int)
    cur = np.zeros(n_sec, dtype=int)
    for i in range(D):
        if judge[i]:
            cur = np.zeros(n_sec, dtype=int)
            if not np.isnan(qret[i]):
                ok = [s for s in range(n_sec) if cnt[i, s] >= min_members and not np.isnan(avg[i, s])]
                ok.sort(key=lambda s: (-(avg[i, s] - qret[i]), s))
                for r, s in enumerate(ok, start=1):
                    cur[s] = r
        rank[i] = cur
    return rank


@dataclass
class VariantData:
    """변형 하나의 신호 판정용 행렬."""

    name: str
    rank: np.ndarray | None       # D×S 업종 순위 (B2는 None)
    avg126: np.ndarray            # D×S 업종 126거래일 평균 (그날)


def variant_data(mk: Market, name: str) -> VariantData:
    """변형별 업종 순위(B0·B1 126, B3 63, B2 없음)와 그날 업종 126 평균."""
    n_sec = len(mk.sectors)
    avg126, _ = sector_avg(mk.ind["ret126"], mk.member, mk.sector, n_sec)
    judge = week_end_mask(mk.days)
    rank = None
    if name in ("B0", "B1"):
        rank = weekly_sector_rank(mk.ind["ret126"], mk.ind["q126"], mk.member, mk.sector, n_sec, judge)
    elif name == "B3":
        rank = weekly_sector_rank(mk.ind["ret63"], mk.ind["q63"], mk.member, mk.sector, n_sec, judge)
    return VariantData(name, rank, avg126)


def passes_filters(mk: Market, vd: VariantData, i: int, j: int) -> tuple[bool, float]:
    """i일 회사 j가 변형의 대상·업종·종목 조건을 통과하는지. 출력 (통과, 점수 = 126거래일 QQQ 대비 초과)."""
    if not mk.member[i, j]:
        return False, np.nan
    ex = mk.ind["ret126"][i, j] - mk.ind["q126"][i]
    if np.isnan(ex) or ex <= 0:
        return False, ex
    if vd.name == "B2":
        return True, ex
    s = mk.sector[j]
    if s < 0 or not (1 <= vd.rank[i, s] <= TOP_BUY):
        return False, ex
    a = vd.avg126[i, s]
    if np.isnan(a) or not (mk.ind["ret126"][i, j] > a):
        return False, ex
    return True, ex


def signal_events(mk: Market, vd: VariantData, start: int, end: int) -> list[dict]:
    """변형의 매수 신호 [{sig: 신호일 번호, c: 회사 번호, score}] — 신호일 start..end-1 (체결일 ≤ end).

    MACD 0선 상향 돌파(전날 ≤ 0, 오늘 > 0)마다 한 번. B1은 돌파일 조건으로 신호. 나머지는 돌파일 +1 ~ +20 거래일
    창에서 MACD ≤ 0이면 끝, 가격 조건 4개가 처음 맞는 날(첫 눌림)에 조건을 보고 통과하면 신호, 아니면 그 돌파는 끝.
    """
    close, macd, sma20, sma50 = mk.close, mk.ind["macd"], mk.ind["sma20"], mk.ind["sma50"]
    events = []
    D, C = close.shape
    for j in range(C):
        vi = np.flatnonzero(~np.isnan(macd[:, j]))
        for a in range(1, len(vi)):
            i0, ip = vi[a], vi[a - 1]
            if not (macd[ip, j] <= 0 < macd[i0, j]):
                continue
            if vd.name == "B1":
                if start <= i0 < end:
                    ok, sc = passes_filters(mk, vd, i0, j)
                    if ok:
                        events.append({"sig": int(i0), "c": j, "score": float(sc)})
                continue
            peak = close[i0, j]
            for b in range(a + 1, len(vi)):
                i = vi[b]
                if i - i0 > CROSS_WINDOW:
                    break
                c = close[i, j]
                if macd[i, j] <= 0:
                    break
                peak = max(peak, c)
                if (c <= peak * (1 - PULLBACK_FROM_PEAK) and not np.isnan(sma20[i, j]) and c <= sma20[i, j] * SMA20_TOL
                        and not np.isnan(sma50[i, j]) and c > sma50[i, j]):
                    if start <= i < end:
                        ok, sc = passes_filters(mk, vd, i, j)
                        if ok:
                            events.append({"sig": int(i), "c": j, "score": float(sc)})
                    break
    events.sort(key=lambda e: (e["sig"], -e["score"], e["c"]))
    return events


def exit_flags(mk: Market, vd: VariantData) -> np.ndarray:
    """D×C bool 매도 조건 (그날 종가): 종가 < 50일선 / MACD < 0 / (B2 빼고) 업종이 상위 5개 밖. 종가 없는 날 False."""
    c, s50, m = mk.close, mk.ind["sma50"], mk.ind["macd"]
    with np.errstate(invalid="ignore"):
        f = (c < s50) | (m < 0)
    if vd.rank is not None:
        sec = mk.sector
        r = np.where(sec >= 0, vd.rank[:, np.maximum(sec, 0)], 0)
        f |= ~((r >= 1) & (r <= TOP_HOLD))
    f &= ~np.isnan(c)
    return f


# ── 장부 ─────────────────────────────────────────────────────────────────


@dataclass
class Costs:
    one_way: float = 0.001
    tax: float = 0.22
    withhold: float = 0.15


def year_end_mask(days: pd.DatetimeIndex) -> np.ndarray:
    """그해 마지막 거래일 True."""
    y = days.year.to_numpy()
    m = np.ones(len(days), dtype=bool)
    m[:-1] = y[:-1] != y[1:]
    return m


def run_book(mk: Market, start: int, end: int, costs: Costs, *, events: list[dict] | None = None,
             exits: np.ndarray | None = None, schedule: list[dict] | None = None, initial: float = 100_000.0) -> dict:
    """장부 하나 (쉬는 돈 QQQ). 신호 모드(events + exits) 또는 일정 모드(schedule: [{c, buy, sell}]).

    신호 모드: 신호일 다음 날 종가 체결(점수 큰 순, 자리 10), 매도 조건이 선 다음 날 종가 매도.
    일정 모드: 정해진 날 사고판다(무작위 대조·상위 5건 제외용). 두 모드 모두 가격이 끝난 회사는 마지막 종가에 판다.
    출력: {curve(start..end 일별 총자산, 끝 청산 전), final(끝 청산·세금 뒤), trades, no_slot, no_fill, taxes, qqq_pnl}
    """
    close, div, qc, qd = mk.close, mk.div, mk.qclose, mk.qdiv
    last_idx = mk.last_idx
    c1 = costs.one_way
    ye = year_end_mask(mk.days)
    by_sig: dict[int, list[dict]] = {}
    for e in events or []:
        by_sig.setdefault(e["sig"] + 1, []).append(e)
    sched_buy: dict[int, list[dict]] = {}
    for s in schedule or []:
        sched_buy.setdefault(s["buy"], []).append(s)

    cash, qsh, qbasis = 0.0, 0.0, 0.0
    pos: dict[int, dict] = {}
    realized, taxes = 0.0, 0.0
    trades, no_slot, no_fill = [], [], []
    curve = np.empty(end - start + 1)

    def qqq_buy(amount: float, i: int) -> None:
        nonlocal cash, qsh, qbasis
        if amount <= 0:
            return
        qsh += amount / (qc[i] * (1 + c1))
        qbasis += amount
        cash -= amount

    def qqq_sell_for(need: float, i: int) -> None:
        """현금이 need만큼 되도록 QQQ를 판다(있는 만큼)."""
        nonlocal cash, qsh, qbasis, realized
        if need <= 0 or qsh <= 0:
            return
        sh = min(qsh, need / (qc[i] * (1 - c1)))
        proceeds = sh * qc[i] * (1 - c1)
        b = qbasis * sh / qsh
        realized += proceeds - b
        qbasis -= b
        qsh -= sh
        cash += proceeds

    def sell(j: int, i: int, px: float, why: str) -> None:
        nonlocal cash, realized
        p = pos.pop(j)
        proceeds = p["sh"] * px * (1 - c1)
        realized += proceeds - p["basis"]
        cash += proceeds
        trades.append({"c": j, "buy": p["buy"], "sell": i, "basis": p["basis"], "proceeds": proceeds,
                       "divs": p["divs"], "pnl": proceeds + p["divs"] - p["basis"], "why": why})

    def equity(i: int) -> float:
        v = cash + qsh * qc[i]
        for j, p in pos.items():
            px = close[i, j]
            v += p["sh"] * (p["px"] if np.isnan(px) else px)
        return v

    def buy(j: int, i: int, amount: float) -> None:
        nonlocal cash
        if cash < amount:
            qqq_sell_for(amount - cash, i)
        amount = min(amount, cash)
        if amount <= 0:
            return
        cash -= amount
        pos[j] = {"sh": amount / (close[i, j] * (1 + c1)), "basis": amount, "buy": i, "px": close[i, j],
                  "divs": 0.0, "pending": False, "sell_at": None}

    cash = initial
    qqq_buy(cash, start)
    for i in range(start, end + 1):
        if i > start:
            # 분배금 (그날 보유분, 85% 현금)
            if qd[i] > 0 and qsh > 0:
                cash += qsh * qd[i] * (1 - costs.withhold)
            for j, p in pos.items():
                if div[i, j] > 0:
                    amt = p["sh"] * div[i, j] * (1 - costs.withhold)
                    cash += amt
                    p["divs"] += amt
            # 매도
            for j in list(pos):
                p = pos[j]
                due = p["pending"] if schedule is None else (p["sell_at"] is not None and i >= p["sell_at"])
                if i > last_idx[j]:
                    sell(j, i, p["px"], "가격 끝(마지막 종가)")
                elif due and not np.isnan(close[i, j]):
                    sell(j, i, close[i, j], "매도 조건" if schedule is None else "일정")
            # 매수
            if schedule is None:
                todays = sorted(by_sig.get(i, []), key=lambda e: (-e["score"], e["c"]))
                eq = equity(i)
                for e in todays:
                    j = e["c"]
                    if j in pos:
                        continue
                    if len(pos) >= MAX_POS:
                        no_slot.append(e)
                        continue
                    if np.isnan(close[i, j]):
                        no_fill.append(e)
                        continue
                    buy(j, i, eq / MAX_POS)
            else:
                eq = equity(i)
                for s in sched_buy.get(i, []):
                    j = s["c"]
                    if j in pos or np.isnan(close[i, j]):
                        no_fill.append(s)
                        continue
                    buy(j, i, eq / MAX_POS)
                    pos[j]["sell_at"] = s["sell"]
        # 가격 기록·매도 조건 (체결 다음 날부터)
        for j, p in pos.items():
            if not np.isnan(close[i, j]):
                p["px"] = close[i, j]
            if schedule is None and exits is not None and i > p["buy"] and exits[i, j]:
                p["pending"] = True
        # 연말 세금 (마지막 날은 끝 청산에서)
        if ye[i] and i < end:
            t = max(realized, 0.0) * costs.tax
            realized = 0.0
            taxes += t
            if cash < t:
                qqq_sell_for(t - cash, i)
            cash -= t
        if cash > 0:
            qqq_buy(cash, i)
        curve[i - start] = equity(i)

    # 끝: 전부 청산 → 그해 실현 순이익 × 세율
    for j in list(pos):
        px = close[end, j]
        sell(j, end, pos[j]["px"] if np.isnan(px) else px, "끝 청산")
    if qsh > 0:
        proceeds = qsh * qc[end] * (1 - c1)
        realized += proceeds - qbasis
        cash += proceeds
        qsh, qbasis = 0.0, 0.0
    t = max(realized, 0.0) * costs.tax
    taxes += t
    final = cash - t
    pre_tax_total = final + taxes - initial
    qqq_pnl = pre_tax_total - sum(tr["pnl"] for tr in trades)
    return {"curve": curve, "final": final, "trades": trades, "no_slot": no_slot, "no_fill": no_fill,
            "taxes": taxes, "qqq_pnl": qqq_pnl}


def run_qqq(mk: Market, start: int, end: int, costs: Costs, initial: float = 100_000.0) -> dict:
    """QQQ 그냥 보유: start 종가에 사고 분배금 85%를 QQQ로 다시 산다. 끝 청산·세금."""
    qc, qd, c1 = mk.qclose, mk.qdiv, costs.one_way
    sh = initial / (qc[start] * (1 + c1))
    basis = initial
    curve = np.empty(end - start + 1)
    for i in range(start, end + 1):
        if i > start and qd[i] > 0:
            amt = sh * qd[i] * (1 - costs.withhold)
            sh += amt / (qc[i] * (1 + c1))
            basis += amt
        curve[i - start] = sh * qc[i]
    proceeds = sh * qc[end] * (1 - c1)
    t = max(proceeds - basis, 0.0) * costs.tax
    return {"curve": curve, "final": proceeds - t, "taxes": t}


def run_equal_weight(mk: Market, start: int, end: int, costs: Costs, initial: float = 100_000.0) -> dict:
    """동일 비중: 매달 첫 거래일 종가에 그날 대상(종가 있음) 전체를 같은 비중으로. 분배금은 다음 재조정까지 현금.
    연말 세금은 현금에서(모자라면 다음 재조정 때 맞춰짐). 가격이 끝난 회사는 다음 재조정에 마지막 종가로 판다."""
    close, div, c1 = mk.close, mk.div, costs.one_way
    m = mk.days.month.to_numpy()
    first_of_month = np.ones(len(mk.days), dtype=bool)
    first_of_month[1:] = m[1:] != m[:-1]
    ye = year_end_mask(mk.days)
    cash, realized, taxes = initial, 0.0, 0.0
    sh = np.zeros(close.shape[1])
    basis = np.zeros(close.shape[1])
    lastpx = np.full(close.shape[1], np.nan)
    curve = np.empty(end - start + 1)
    for i in range(start, end + 1):
        ok = ~np.isnan(close[i])
        lastpx[ok] = close[i, ok]
        if i > start:
            cash += float((sh * np.where(np.isnan(div[i]), 0.0, div[i])).sum()) * (1 - costs.withhold)
        if i == start or first_of_month[i]:
            px = np.where(np.isnan(lastpx), 0.0, lastpx)
            value = cash + float((sh * px).sum())
            tgt_set = mk.member[i] & ok
            n = int(tgt_set.sum())
            tgt_val = np.where(tgt_set, value / max(n, 1), 0.0)
            cur_val = sh * px
            # 매도 먼저 (비중 초과분·대상 밖)
            for j in np.flatnonzero(cur_val > tgt_val + 1e-9):
                q = sh[j] if not tgt_set[j] else (cur_val[j] - tgt_val[j]) / px[j]
                proceeds = q * px[j] * (1 - c1)
                b = basis[j] * q / sh[j]
                realized += proceeds - b
                basis[j] -= b
                sh[j] -= q
                cash += proceeds
            for j in np.flatnonzero(tgt_set & (sh * px < tgt_val - 1e-9)):
                amt = min(tgt_val[j] - sh[j] * px[j], cash)
                if amt <= 0:
                    continue
                sh[j] += amt / (px[j] * (1 + c1))
                basis[j] += amt
                cash -= amt
        if ye[i] and i < end:
            t = max(realized, 0.0) * costs.tax
            realized, taxes, cash = 0.0, taxes + t, cash - t
        curve[i - start] = cash + float((sh * np.where(np.isnan(lastpx), 0.0, lastpx)).sum())
    px = np.where(np.isnan(lastpx), 0.0, lastpx)
    proceeds = sh * px * (1 - c1)
    realized += float((proceeds - basis).sum())
    cash += float(proceeds.sum())
    t = max(realized, 0.0) * costs.tax
    return {"curve": curve, "final": cash - t, "taxes": taxes + t}


# ── 무작위·가짜 ──────────────────────────────────────────────────────────


def random_schedule(mk: Market, trades: list[dict], rng: np.random.Generator) -> list[dict]:
    """실제 매매의 매수·매도일 그대로, 회사만 그날 대상(종가 있음, 신호 회사·그때 보유 회사 제외) 중 무작위."""
    out, active = [], []
    for tr in sorted(trades, key=lambda t: (t["buy"], t["c"])):
        b = tr["buy"]
        active = [(j, s) for j, s in active if s > b]
        held = {j for j, _ in active}
        pool = np.flatnonzero(mk.member[b] & ~np.isnan(mk.close[b]))
        pool = [j for j in pool if j != tr["c"] and j not in held]
        if not pool:
            continue
        j = int(pool[rng.integers(len(pool))])
        out.append({"c": j, "buy": b, "sell": tr["sell"]})
        active.append((j, tr["sell"]))
    return out


def fake_events(mk: Market, n: int, start: int, end: int, rng: np.random.Generator) -> list[dict]:
    """가짜 신호 n개: 신호일은 start..end-1 균등, 회사는 그날 대상(종가 있음) 균등, 점수는 무작위(자리 다툼 순서용)."""
    out = []
    for _ in range(n):
        for _try in range(50):
            i = int(rng.integers(start, end))
            pool = np.flatnonzero(mk.member[i] & ~np.isnan(mk.close[i]))
            if len(pool):
                out.append({"sig": i, "c": int(pool[rng.integers(len(pool))]), "score": float(rng.random())})
                break
    out.sort(key=lambda e: (e["sig"], -e["score"], e["c"]))
    return out


# ── 성적 ─────────────────────────────────────────────────────────────────


def cagr(v0: float, v1: float, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    """연복리(%)."""
    return ((v1 / v0) ** (365.25 / (d1 - d0).days) - 1.0) * 100.0


def mdd(curve: np.ndarray) -> float:
    """최대 낙폭(%)."""
    c = np.asarray(curve, dtype=float)
    return float((c / np.maximum.accumulate(c) - 1.0).min() * 100.0)


def regime(qclose: np.ndarray) -> np.ndarray:
    """그날 종가까지의 QQQ로 국면: 2 폭락장(252일 고점 대비 −20% 이하 또는 20일 −10% 이하) > 1 하락장(200일선 아래) > 0 상승장.
    준비 기간(252일 미만)은 −1."""
    q = pd.Series(qclose)
    hi = q.rolling(252).max()
    r20 = q / q.shift(20) - 1.0
    sma = q.rolling(200).mean()
    crash = (q / hi - 1.0 <= -0.20) | (r20 <= -0.10)
    out = np.where(crash, 2, np.where(q < sma, 1, 0))
    out[hi.isna().to_numpy()] = -1
    return out
