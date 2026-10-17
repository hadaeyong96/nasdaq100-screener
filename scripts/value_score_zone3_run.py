"""가치평가 전략(value-score) 검증 구역 ③ 판정 실행 — 사전 등록 태그 뒤에만 수익을 계산한다.

    python scripts/value_score_zone3_run.py            # 판정 + 참고 보고 전부
    python scripts/value_score_zone3_run.py --quick    # 무작위 20회만 (실행기 점검용, 판정 아님)

시작할 때 (1) 태그 value-score-prereg가 있는지, (2) 잠근 파일(docs/prereg/value_score.md, core/value_score.py,
scripts/value_score_zone3_check.py, scripts/value_score_zone3_data.py)이 태그 내용과 같은지 확인하고 아니면 멈춘다.
가격·재무는 value_score_zone3_data(구역 가드)로만 읽는다. 규칙·해석은 docs/prereg/value_score.md 1장·4장 그대로다.
결과 숫자는 data/cache/value_score/result.json(git 제외)에 남기고 사람이 docs/results/value_score_zone3.md에 정리한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rs26_zone3_check as z  # noqa: E402
import value_score_zone3_data as vd  # noqa: E402
from core import tax_harvest as th  # noqa: E402
from core import value_score as vs  # noqa: E402
from factory import regimes as rg  # noqa: E402
from factory import scorecard as fsc  # noqa: E402

TAG = "value-score-prereg"
LOCKED = ["docs/prereg/value_score.md", "core/value_score.py", "scripts/value_score_zone3_check.py",
          "scripts/value_score_zone3_data.py"]
START = pd.Timestamp("2012-04-02")
END = pd.Timestamp("2021-12-31")
INITIAL_KRW = 100_000_000.0
MONTHLY_KRW = 1_000_000.0
DEDUCT = 2_500_000.0
TAX = 0.22
KEEP_DIV = 0.85
SLOTS = 10
MIN_PICK, MIN_KEEP, MIN_SWAP, SWAP_GAP = 49.0, 35.0, 56.0, 10.0
SEED = 20261017
N_RANDOM = 1000
FX_CSV = ROOT / "data" / "cache" / "shock_dip" / "fred_DEXKOUS.csv"
OUT = vd.c.OUT / "result.json"
QQQ = "__QQQ__"


@dataclass(frozen=True)
class Cost:
    comm: float = 0.0007
    slip: float = 0.001
    fx: float = 0.001
    tax: bool = True


FULL = Cost()
NO_TAX = Cost(tax=False)
ZERO = Cost(0.0, 0.0, 0.0, False)


# ── 사전 등록 확인 ─────────────────────────────────────────────────────


def verify_prereg() -> str:
    """태그가 있고 잠근 파일이 태그 내용과 같은지. 출력: 태그 커밋. 아니면 SystemExit."""
    r = subprocess.run(["git", "rev-parse", "--verify", f"{TAG}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"태그 {TAG}가 없습니다 — 사전 등록 전에는 수익을 계산하지 않습니다")
    for f in LOCKED:
        if subprocess.run(["git", "diff", "--quiet", TAG, "--", f], cwd=ROOT).returncode != 0:
            raise SystemExit(f"{f}가 태그 {TAG} 뒤에 바뀌었습니다 — 멈춥니다")
    return r.stdout.strip()


# ── 시장 ────────────────────────────────────────────────────────────────


def load_fx(days: pd.DatetimeIndex) -> np.ndarray:
    """FRED DEXKOUS(2021-12-31까지 읽음), 빈 날은 그 전 가장 가까운 값."""
    rows = []
    with open(FX_CSV, encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            d, v = line.strip().split(",")[:2]
            ts = pd.Timestamp(d)
            if ts > END:
                break
            if ts >= pd.Timestamp("2011-01-01") and v not in ("", "."):
                rows.append((ts, float(v)))
    s = pd.Series(dict(rows)).sort_index()
    out = s.reindex(s.index.union(days)).ffill().reindex(days)
    if out.isna().any():
        raise RuntimeError("환율 빈 날을 채우지 못했습니다")
    return out.to_numpy()


class Market:
    """(거래일 × 종목) 배열과 달력. 열 마지막은 QQQ."""

    def __init__(self, data: vd.Data):
        days = vd.nyse_days()
        days = days[(days >= START) & (days <= END)]
        qc = pd.read_csv(z.QQQ_CSV, index_col=0, parse_dates=True)
        if qc.index.max() > END:
            raise z.ZoneGuardError("QQQ 파일에 2021-12-31 뒤 줄이 있습니다")
        cl = {t: data.close[t] for t in data.close}
        cl[QQQ] = qc["close"].astype(float)
        dv = {t: data.div[t] for t in data.div}
        dv[QQQ] = qc["div"].fillna(0.0).astype(float)
        self.setup(days, cl, dv, load_fx(days), vd.month_ends(), qc["close"].astype(float))

    @classmethod
    def from_frames(cls, days, close: dict, div: dict, fx: np.ndarray, mes, qqq_close_full) -> "Market":
        """테스트용: 합성 가격으로 만든다."""
        m = cls.__new__(cls)
        m.setup(days, close, div, fx, mes, qqq_close_full)
        return m

    def setup(self, days, cl: dict, dv: dict, fx: np.ndarray, mes, qqq_close_full) -> None:
        self.days = pd.DatetimeIndex(days)
        self.di = {d: i for i, d in enumerate(self.days)}
        self.tickers = sorted(t for t in cl if t != QQQ) + [QQQ]
        self.ti = {t: j for j, t in enumerate(self.tickers)}
        self.qqq_close_full = qqq_close_full
        self.close = pd.DataFrame(cl).reindex(columns=self.tickers).reindex(self.days).to_numpy()
        self.ffill = pd.DataFrame(cl).reindex(columns=self.tickers).ffill().reindex(self.days).to_numpy()
        self.div = pd.DataFrame(dv).reindex(columns=self.tickers).reindex(self.days).fillna(0.0).to_numpy()
        # 마지막 종가 위치 (그 뒤 종가 없음 = 상장폐지·인수)
        has = ~np.isnan(self.close)
        self.last_idx = np.where(has.any(axis=0), len(self.days) - 1 - np.argmax(has[::-1], axis=0), -1)
        self.fx = fx
        d = self.days
        s = pd.Series(np.arange(len(d)), index=d)
        ym = [d.year, d.month]
        self.month_first = set(s.groupby(ym).min()) - {0}
        self.month_last = set(s.groupby(ym).max())
        self.may_first = {i for i in self.month_first if d[i].month == 5}
        self.quarter_end = {i for i in self.month_last if d[i].month in (3, 6, 9, 12)}
        self.dec_exec = {int(self.di[th.deadline(d, y)]): y for y in sorted(set(d.year)) if (d.year == y).sum() >= 3
                         and d[d.year == y][-1].month == 12}
        self.year_last = set(s.groupby(d.year).max())
        mes = sorted(mes)
        self.me_of = {}  # 거래일 → 그날 쓸 점수의 월말(그날 이전 마지막 월말)
        k = -1
        for i, day in enumerate(d):
            while k + 1 < len(mes) and mes[k + 1] < day:
                k += 1
            self.me_of[i] = mes[k] if k >= 0 else None
        self.nov_end = {y: max(x for x in mes if x.year == y and x.month == 11) for y in set(d.year)
                        if any(x.year == y and x.month == 11 for x in mes)}


# ── 점수판 ───────────────────────────────────────────────────────────────


class Board:
    """월말마다 {티커: (총점, PEG 값, 대상 여부)} + 대상 중 점수 있는 순위."""

    def __init__(self, scores: pd.DataFrame, total: pd.Series, sic2: dict):
        self.sic2 = sic2
        s = scores.assign(total=total.to_numpy())
        self.tot, self.order, self.pool = {}, {}, {}
        for me, g in s.groupby("date"):
            g = g.set_index("ticker")
            self.tot[me] = {t: (None if pd.isna(v) else float(v)) for t, v in g["total"].items()}
            u = g[g["universe"] & g["total"].notna()]
            self.order[me] = vs.rank_order(u[["total", "peg_value"]])
            self.pool[me] = sorted(u.index)

    def score(self, me, t):
        return self.tot.get(me, {}).get(t)


def peg_only_total(scores: pd.DataFrame) -> pd.Series:
    """참고 (b): PEG 점수 = 70 × (그날 대상 중 PEG 값이 더 큰 종목 비율). PEG 계산 못 한·0점 종목은 0점."""
    out = pd.Series(0.0, index=scores.index)
    for me, g in scores.groupby("date"):
        u = g.loc[g["universe"], "peg_value"].to_numpy(dtype=float)
        n = len(u)
        pv = g["peg_value"].to_numpy(dtype=float)
        val = np.array([70.0 * (u > x).sum() / n if (n and np.isfinite(x)) else 0.0 for x in pv])
        out.loc[g.index] = val
    return out


# ── 장부 ────────────────────────────────────────────────────────────────


@dataclass
class Book:
    m: Market
    cost: Cost
    record: bool = False
    hold: dict = field(default_factory=dict)  # 열 → [수량, 원화 취득가, 첫 매수 위치]
    cash: float = 0.0
    realized: dict = field(default_factory=dict)
    pending_tax: tuple | None = None
    taxes: dict = field(default_factory=dict)
    flows: list = field(default_factory=list)
    units_pre: float = 0.0
    units_liq: float = 0.0
    sells_usd: dict = field(default_factory=dict)
    pnl: dict = field(default_factory=dict)  # 열 → 원화 손익 기여(실현 + 분배금)
    log: list = field(default_factory=list)
    curve_pre: list = field(default_factory=list)
    curve_liq: list = field(default_factory=list)
    value_usd_daily: list = field(default_factory=list)

    # 가격·환율
    def px(self, i, j):
        return float(self.m.ffill[i, j])

    def net_sell(self, i, j):
        return self.px(i, j) * (1 - self.cost.slip) * (1 - self.cost.comm)

    def krw_to_usd(self, i, krw):
        return krw / self.m.fx[i] * (1 - self.cost.fx)

    # 매매
    def buy(self, i, j, usd, why=""):
        if usd <= 0:
            return
        unit = self.px(i, j) * (1 + self.cost.slip) * (1 + self.cost.comm)
        q = usd / unit
        h = self.hold.get(j)
        if h is None:
            self.hold[j] = [q, usd * self.m.fx[i], i]
        else:
            h[0] += q
            h[1] += usd * self.m.fx[i]
        self.cash -= usd
        if self.record:
            self.log.append((i, "buy", j, q, usd, why))

    def sell(self, i, j, q, why=""):
        h = self.hold[j]
        q = min(q, h[0])
        if q <= 0:
            return 0.0
        usd = q * self.net_sell(i, j)
        avg = h[1] / h[0]
        gain = usd * self.m.fx[i] - avg * q
        y = self.m.days[i].year
        self.realized[y] = self.realized.get(y, 0.0) + gain
        self.pnl[j] = self.pnl.get(j, 0.0) + gain
        self.sells_usd[y] = self.sells_usd.get(y, 0.0) + usd
        self.cash += usd
        h[0] -= q
        h[1] -= avg * q
        if h[0] <= 1e-12:
            del self.hold[j]
        if self.record:
            self.log.append((i, "sell", j, q, usd, why))
        return usd

    def per_share_gain(self, i, j):
        h = self.hold[j]
        return self.net_sell(i, j) * self.m.fx[i] - h[1] / h[0]

    # 평가
    def value_usd(self, i):
        return self.cash + sum(h[0] * self.px(i, j) for j, h in self.hold.items())

    def value_pre(self, i):
        return self.value_usd(i) * self.m.fx[i]

    def value_liq(self, i):
        """그날 전부 팔았다면 세후 원화 (공용 규칙 2)."""
        fx = self.m.fx[i]
        usd, gain = self.cash, 0.0
        for j, h in self.hold.items():
            u = h[0] * self.net_sell(i, j)
            usd += u
            gain += u * fx - h[1]
        out = usd * fx * (1 - self.cost.fx)
        if self.cost.tax:
            y = self.m.days[i].year
            out -= max(self.realized.get(y, 0.0) + gain - DEDUCT, 0.0) * TAX
            if self.pending_tax is not None:
                out -= self.pending_tax[1]
        return out

    # 하루 공통 처리
    def start(self, i):
        self.cash = self.krw_to_usd(i, INITIAL_KRW)
        self.flows.append((self.m.days[i], -INITIAL_KRW))
        self.units_pre = self.units_liq = INITIAL_KRW

    def dividends(self, i):
        for j, h in self.hold.items():
            dv = self.m.div[i, j]
            if dv > 0:
                usd = h[0] * dv * KEEP_DIV
                self.cash += usd
                self.pnl[j] = self.pnl.get(j, 0.0) + usd * self.m.fx[i]

    def delist(self, i):
        gone = [j for j in self.hold if self.m.last_idx[j] < i]
        for j in gone:
            self.sell(i, j, self.hold[j][0], "가격 끝(마지막 종가)")
        return len(gone)

    def raise_cash(self, i, need_usd):
        """현금이 need_usd보다 모자라면 평가액 큰 종목부터 판다."""
        while self.cash < need_usd - 1e-9 and self.hold:
            j = max(self.hold, key=lambda k: self.hold[k][0] * self.px(i, k))
            short = need_usd - self.cash
            q = min(self.hold[j][0], short / self.net_sell(i, j))
            self.sell(i, j, q, "세금 마련")

    def pay_tax(self, i):
        if self.pending_tax is None:
            return
        y, krw = self.pending_tax
        usd = krw / (self.m.fx[i] * (1 - self.cost.fx))
        self.raise_cash(i, usd)
        self.cash -= usd
        self.pending_tax = None

    def close_year(self, i):
        y = self.m.days[i].year
        tax = max(self.realized.get(y, 0.0) - DEDUCT, 0.0) * TAX if self.cost.tax else 0.0
        self.taxes[y] = tax
        if tax > 0 and i != len(self.m.days) - 1:
            self.pending_tax = (y, tax)

    def contribute(self, i, vacancy_cash: bool, amount_only: bool = False):
        """납입: 단위를 납입 직전 가치로 사고, 빈자리면 현금, 아니면 평가액 가장 작은 보유 종목에 남은 현금 전부
        (amount_only면 이번 납입분만 — 동일 비중: 상장폐지 돈은 재조정까지 현금)."""
        vp, vl = self.value_pre(i), self.value_liq(i)
        self.units_pre += MONTHLY_KRW / (vp / self.units_pre)
        self.units_liq += MONTHLY_KRW / (vl / self.units_liq)
        usd = self.krw_to_usd(i, MONTHLY_KRW)
        self.cash += usd
        self.flows.append((self.m.days[i], -MONTHLY_KRW))
        if vacancy_cash or not self.hold:
            return
        j = min(self.hold, key=lambda k: (self.hold[k][0] * self.px(i, k), self.m.tickers[k]))
        self.buy(i, j, usd if amount_only else self.cash, "납입")

    def harvest(self, i):
        """공제 소진: 1주당 이익 + 인 종목을 평가 이익 큰 순으로 팔고 같은 날 같은 수량 다시 산다."""
        y = self.m.days[i].year
        left = DEDUCT - self.realized.get(y, 0.0)
        order = sorted(self.hold, key=lambda k: -(self.hold[k][0] * self.net_sell(i, k) * self.m.fx[i] - self.hold[k][1]))
        n = 0
        for j in order:
            if left <= 1.0:
                break
            g = self.per_share_gain(i, j)
            if g <= 0:
                continue
            q = min(left / g, self.hold[j][0])
            self.sell(i, j, q, "공제 소진")
            unit = self.px(i, j) * (1 + self.cost.slip) * (1 + self.cost.comm)
            self.buy(i, j, min(q * unit, max(self.cash, 0.0)), "공제 소진 다시 사기")
            left -= q * g
            n += 1
        return n

    def snapshot(self, i):
        if not self.record:
            return
        self.curve_pre.append(self.value_pre(i) / self.units_pre)
        self.curve_liq.append(self.value_liq(i) / self.units_liq)
        self.value_usd_daily.append(self.value_usd(i) - self.cash)

    def finish(self, i):
        for j in list(self.hold):
            self.sell(i, j, self.hold[j][0], "끝 청산")
        y = self.m.days[i].year
        tax = max(self.realized.get(y, 0.0) - DEDUCT, 0.0) * TAX if self.cost.tax else 0.0
        self.taxes[y] = tax
        final = self.cash * self.m.fx[i] * (1 - self.cost.fx) - tax
        if self.pending_tax is not None:  # 납부 전 전년 세금(없어야 정상)
            final -= self.pending_tax[1]
            self.pending_tax = None
        self.flows.append((self.m.days[i], final))
        return final


# ── 전략들 ───────────────────────────────────────────────────────────────


@dataclass
class Rules:
    swap: str = "rule"  # rule | off | half
    random_seed: int | None = None
    events: dict | None = None  # 무작위용: 전략 사건(첫 선정 수, 수시 매도 날짜별 수, 갈아타기 해)


def run_value(m: Market, board: Board, cost: Cost, rules: Rules = Rules(), record: bool = False) -> dict:
    """전략(또는 같은 날짜·같은 개수 무작위) 장부 하나."""
    rng = np.random.default_rng(rules.random_seed) if rules.random_seed is not None else None
    ev = rules.events
    b = Book(m, cost, record)
    sic2 = {m.ti[t]: board.sic2.get(t) for t in board.sic2 if t in m.ti}
    flags: dict = {}
    sell_next: list = []
    stats = {"init": 0, "sell_events": {}, "swap_years": [], "swaps": [], "fills": 0, "unknown": 0, "checks": 0,
             "delist": 0, "harvest_trades": 0}
    n_days = len(m.days)
    for i in range(n_days):
        d = m.days[i]
        if i == 0:
            b.start(i)
            me = m.me_of[i]
            alive = lambda t: not np.isnan(m.close[i, m.ti[t]])  # noqa: E731
            if rng is None:
                order = [t for t in board.order[me] if board.score(me, t) >= MIN_PICK and alive(t)]
                picks = vs.pick(order, SLOTS, board.sic2, [])
            else:
                order = [t for t in rng.permutation(board.pool[me]) if alive(t)]
                picks = vs.pick(order, ev["init"], board.sic2, [])
            each = b.cash / SLOTS
            for t in picks:
                b.buy(i, m.ti[t], each, "첫 선정")
            stats["init"] = len(picks)
        else:
            b.dividends(i)
            stats["delist"] += b.delist(i)
            if i in m.may_first:
                b.pay_tax(i)
            # 수시 매도 (분기 말 다음 거래일)
            if sell_next:
                sold = 0
                for j in sell_next:
                    if j in b.hold and (d - m.days[b.hold[j][2]]).days >= 365:
                        b.sell(i, j, b.hold[j][0], "분기 35점 미만 2번")
                        sold += 1
                        flags.pop(j, None)
                if sold:
                    stats["sell_events"][i] = sold
                sell_next = []
            if rng is not None and ev and i in ev["sell_events"]:
                elig = [j for j in b.hold if (d - m.days[b.hold[j][2]]).days >= 365]
                k = min(ev["sell_events"][i], len(elig))
                for j in rng.choice(sorted(elig), size=k, replace=False) if k else []:
                    b.sell(i, int(j), b.hold[int(j)][0], "무작위 수시 매도")
            if i in m.month_first:
                b.contribute(i, vacancy_cash=len(b.hold) < SLOTS)
            if i in m.dec_exec:
                y = m.dec_exec[i]
                me = m.nov_end[y]
                _december(b, i, y, me, board, sic2, rules, rng, ev, stats)
        # 분기 말 판정 (그날 종가 점수 → 다음 거래일 매도)
        if i in m.quarter_end and rng is None and i < n_days - 1:
            me = d
            for j in list(b.hold):
                sc = board.score(me, m.tickers[j])
                stats["checks"] += 1
                if sc is None:
                    stats["unknown"] += 1
                    flags[j] = ("unknown", b.hold[j][2])
                    continue
                cur = "below" if sc < MIN_KEEP else "ok"
                if cur == "below" and flags.get(j) == ("below", b.hold[j][2]):
                    sell_next.append(j)
                flags[j] = (cur, b.hold[j][2])  # 첫 매수 위치와 묶어 판 뒤 다시 산 종목에 옛 표시가 남지 않게
        if i in m.year_last and i != n_days - 1:
            b.close_year(i)
        b.snapshot(i)
    final = b.finish(n_days - 1)
    return {"book": b, "final": final, "stats": stats}


def _december(b: Book, i, y, me, board: Board, sic2, rules: Rules, rng, ev, stats) -> None:
    m = b.m
    held_t = [m.tickers[j] for j in b.hold]
    # ① 갈아타기
    choice = None
    if rules.swap != "off":
        if rng is None:
            held_score = {t: board.score(me, t) for t in held_t if board.score(me, t) is not None}
            cands = [t for t in board.order[me] if t not in held_t]
            cand_score = {t: board.score(me, t) for t in cands}
            choice = vs.swap_choice(cands, cand_score, held_score, board.sic2, MIN_SWAP, SWAP_GAP)
        elif ev and y in ev["swap_years"] and held_t:
            h = str(rng.choice(sorted(held_t)))
            order = [t for t in rng.permutation(board.pool[me]) if t not in held_t]
            c = vs.pick(order, 1, board.sic2, [t for t in held_t if t != h])
            choice = (h, c[0]) if c else None
    if choice is not None:
        h, c = choice
        jh, jc = m.ti[h], m.ti[c]
        left = DEDUCT - b.realized.get(y, 0.0)
        qty = b.hold[jh][0]
        if rules.swap == "half":
            q = qty / 2.0
        else:
            q = vs.swap_qty(left, b.per_share_gain(i, jh), qty)
        if q > 0 and not np.isnan(m.close[i, jc]):
            usd = b.sell(i, jh, q, f"갈아타기 → {c}")
            b.buy(i, jc, usd, f"갈아타기 ← {h}")
            stats["swap_years"].append(y)
            stats["swaps"].append((str(m.days[i].date()), h, c, round(q / qty, 4)))
    # ② 빈자리 채우기
    vac = SLOTS - len(b.hold)
    if vac > 0:
        held_t = [m.tickers[j] for j in b.hold]
        if rng is None:
            order = [t for t in board.order[me] if board.score(me, t) >= MIN_PICK]
        else:
            order = list(rng.permutation(board.pool[me]))
        order = [t for t in order if not np.isnan(m.close[i, m.ti[t]])]
        picks = vs.pick(order, vac, board.sic2, held_t)
        if picks:
            each = b.cash / len(picks)
            for t in picks:
                b.buy(i, m.ti[t], each, "빈자리 채우기")
            stats["fills"] += len(picks)
    # ③ 공제 소진
    if b.cost.tax or b.cost.comm > 0:
        stats["harvest_trades"] += b.harvest(i)


def run_qqq(m: Market, cost: Cost, record: bool = False) -> dict:
    b = Book(m, cost, record)
    q = m.ti[QQQ]
    for i in range(len(m.days)):
        if i == 0:
            b.start(i)
            b.buy(i, q, b.cash, "시작")
        else:
            b.dividends(i)
            if i in m.may_first:
                b.pay_tax(i)
            if i in m.month_first:
                b.contribute(i, vacancy_cash=False)
            if i in m.dec_exec:
                b.harvest(i)
        if i in m.year_last and i != len(m.days) - 1:
            b.close_year(i)
        b.snapshot(i)
    return {"book": b, "final": b.finish(len(m.days) - 1)}


def run_ew(m: Market, data: vd.Data, cost: Cost, record: bool = False) -> dict:
    """구역 ③ 동일 비중: 명단(필터 없음) 전 종목, 12월 실행일 재조정."""
    b = Book(m, cost, record)

    def members(i, me):
        return [m.ti[t] for t in sorted(data.mem[me]) if t in m.ti and not np.isnan(m.close[i, m.ti[t]])]

    def rebalance(i, cols):
        cols = set(cols)
        for j in [j for j in b.hold if j not in cols]:
            b.sell(i, j, b.hold[j][0], "재조정(명단 밖)")
        total = b.value_usd(i)
        target = total / len(cols) * (1 - cost.comm - cost.slip)
        for j in list(b.hold):
            v = b.hold[j][0] * b.px(i, j)
            if v > target:
                b.sell(i, j, (v - target) / b.px(i, j), "재조정")
        need = {j: target - (b.hold[j][0] * b.px(i, j) if j in b.hold else 0.0) for j in cols}
        need = {j: x for j, x in need.items() if x > 0}
        tot = sum(need.values())
        cash = b.cash
        for j, x in need.items():
            b.buy(i, j, cash * x / tot, "재조정")

    for i in range(len(m.days)):
        if i == 0:
            b.start(i)
            rebalance(i, members(i, m.me_of[i]))
        else:
            b.dividends(i)
            b.delist(i)
            if i in m.may_first:
                b.pay_tax(i)
            if i in m.month_first:
                b.contribute(i, vacancy_cash=False, amount_only=True)
            if i in m.dec_exec:
                rebalance(i, members(i, m.nov_end[m.dec_exec[i]]))
                b.harvest(i)
        if i in m.year_last and i != len(m.days) - 1:
            b.close_year(i)
        b.snapshot(i)
    return {"book": b, "final": b.finish(len(m.days) - 1)}


# ── 지표 ────────────────────────────────────────────────────────────────


def xirr(flows: list[tuple[pd.Timestamp, float]]) -> float:
    """원화 현금흐름 내부수익률(%/년). 이분법."""
    t0 = flows[0][0]
    ts = np.array([(d - t0).total_seconds() / 86400.0 / 365.25 for d, _ in flows])
    cf = np.array([v for _, v in flows])

    def npv(r):
        return float((cf / (1.0 + r) ** ts).sum())

    lo, hi = -0.99, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * 100.0


def cagr_slice(c: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float:
    x = c[c.index <= a]
    y = c[c.index <= b]
    return ((y.iloc[-1] / x.iloc[-1]) ** (365.25 / (y.index[-1] - x.index[-1]).days) - 1) * 100.0


def mdd(c: pd.Series) -> float:
    return float((c / c.cummax() - 1.0).min() * 100.0)


def yearly(c: pd.Series) -> dict[int, float]:
    out, prev = {}, float(c.iloc[0])
    for y in sorted(set(c.index.year)):
        last = float(c[c.index.year == y].iloc[-1])
        out[y] = (last / prev - 1) * 100.0
        prev = last
    return out


def regime_mdd(c: pd.Series, reg: pd.Series) -> tuple[float, str]:
    """하락장·폭락장이 이어진 구간마다 구간 안 최대 낙폭, 가장 깊은 값과 그 구간."""
    r = reg.reindex(c.index)
    bad = r.isin([rg.DOWN, rg.CRASH])
    key = (bad != bad.shift()).cumsum()
    worst, where = 0.0, ""
    for _, seg in c[bad].groupby(key[bad]):
        v = mdd(seg)
        if v < worst:
            worst, where = v, f"{seg.index[0].date()}~{seg.index[-1].date()}"
    return worst, where


def curves(res: dict, m: Market) -> tuple[pd.Series, pd.Series]:
    b = res["book"]
    return pd.Series(b.curve_pre, index=m.days), pd.Series(b.curve_liq, index=m.days)


# ── 무작위 (프로세스) ───────────────────────────────────────────────────

_POOL: dict = {}


def _init(m, board, ev):
    _POOL.update(m=m, board=board, ev=ev)


def _rand(i: int) -> float:
    r = run_value(_POOL["m"], _POOL["board"], FULL, Rules(random_seed=SEED + i, events=_POOL["ev"]))
    return xirr(r["book"].flows)


# ── 실행 ────────────────────────────────────────────────────────────────


def summarize(name: str, res: dict, m: Market) -> dict:
    b = res["book"]
    out = {"name": name, "xirr": xirr(b.flows), "final_krw": res["final"], "taxes": {int(k): v for k, v in b.taxes.items()}}
    if b.record:
        pre, liq = curves(res, m)
        hv = pd.Series(b.value_usd_daily, index=m.days)
        turnover = {int(y): b.sells_usd.get(y, 0.0) / float(hv[hv.index.year == y].mean()) * 100.0
                    for y in sorted(set(m.days.year)) if float(hv[hv.index.year == y].mean()) > 0}
        out.update({"mdd_pre": mdd(pre), "turnover_pct": turnover})
    if "stats" in res:
        st = res["stats"]
        out["stats"] = {k: (v if k != "sell_events" else {str(m.days[i].date()): n for i, n in v.items()}) for k, v in st.items()}
    return out


def judge(s: dict, q: dict, ew: dict, rand: list[float], m: Market) -> dict:
    sp, sl = curves(s, m)
    qp, ql = curves(q, m)
    xs, xq, xe = xirr(s["book"].flows), xirr(q["book"].flows), xirr(ew["book"].flows)
    r = np.array(rand)
    pct = 100.0 * (float((r < xs).sum()) + 0.5 * float((r == xs).sum())) / len(r)
    mid = pd.Timestamp("2016-12-30")
    sub = {"시작~2016": (cagr_slice(sl, START, mid), cagr_slice(ql, START, mid)),
           "2017~2021": (cagr_slice(sl, mid, END), cagr_slice(ql, mid, END))}
    ys, yq = yearly(sl), yearly(ql)
    beat = sum(1 for y in ys if ys[y] - yq[y] > 0)
    ms, mq = mdd(sp), mdd(qp)
    reg = rg.classify(m.qqq_close_full)
    rs, rs_where = regime_mdd(sp, reg)
    rq, rq_where = regime_mdd(qp, reg)
    st = s["stats"]
    unk = st["unknown"] / st["checks"] * 100.0 if st["checks"] else 0.0
    crit = [
        {"id": 1, "name": "세후 청산 연환산 > QQQ", "ok": xs > xq, "value": xs - xq},
        {"id": 2, "name": "> 구역 ③ 동일 비중", "ok": xs > xe, "value": xs - xe},
        {"id": 3, "name": "무작위 99백분위 이상", "ok": pct >= 99.0, "value": pct},
        {"id": 4, "name": "두 구간 모두 QQQ 초과", "ok": all(a - b > 0 for a, b in sub.values()),
         "value": {k: a - b for k, (a, b) in sub.items()}},
        {"id": 5, "name": "달력 연도 60% 이상 QQQ 초과", "ok": beat >= 0.6 * len(ys), "value": f"{beat}/{len(ys)}"},
        {"id": 6, "name": "최대 낙폭 − QQQ ≥ −15%p", "ok": ms - mq >= -15.0, "value": ms - mq},
        {"id": 7, "name": "하락장·폭락장 구간 최대 낙폭 − QQQ ≥ −10%p", "ok": rs - rq >= -10.0, "value": rs - rq},
        {"id": 8, "name": "판정 불가 보유 비율 ≤ 10%", "ok": unk <= 10.0, "value": unk},
    ]
    if not crit[7]["ok"]:
        verdict = "INCONCLUSIVE"
    elif not crit[0]["ok"]:
        verdict = "FAIL"
    elif all(c["ok"] for c in crit[1:7]):
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "criteria": crit, "xirr": {"strategy": xs, "qqq": xq, "ew": xe},
            "random": {"pct": pct, "p50": float(np.median(r)), "p99": float(np.percentile(r, 99)), "n": len(r)},
            "sub": sub, "years": {int(y): (ys[y], yq[y]) for y in ys}, "mdd": {"strategy": ms, "qqq": mq},
            "regime_mdd": {"strategy": (rs, rs_where), "qqq": (rq, rq_where)}, "unknown_pct": unk}


def main(quick: bool = False) -> None:
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    tag_commit = verify_prereg()
    t0 = time.time()
    data = vd.Data()
    scores = vd.load_scores(data, rebuild=True)
    m = Market(data)
    board = Board(scores, scores["total"], data.sic2)
    print(f"태그 {TAG} {tag_commit[:7]} 확인 · 거래일 {len(m.days)} ({m.days[0].date()} ~ {m.days[-1].date()}) · "
          f"종목 {len(m.tickers) - 1} · 준비 {time.time() - t0:.0f}초", flush=True)
    out = {"tag_commit": tag_commit, "quick": quick}

    # 판정 장부
    s = run_value(m, board, FULL, record=True)
    q = run_qqq(m, FULL, record=True)
    ew = run_ew(m, data, FULL, record=True)
    ev = {"init": s["stats"]["init"], "sell_events": s["stats"]["sell_events"], "swap_years": set(s["stats"]["swap_years"])}
    n_rand = 20 if quick else N_RANDOM
    with ProcessPoolExecutor(initializer=_init, initargs=(m, board, ev)) as ex:
        rand = list(ex.map(_rand, range(n_rand), chunksize=10))
    j = judge(s, q, ew, rand, m)
    out["judge"] = j
    out["strategy"], out["qqq"], out["ew"] = summarize("전략", s, m), summarize("QQQ", q, m), summarize("동일 비중", ew, m)
    out["random_xirr"] = rand
    print(f"판정 {j['verdict']} · XIRR 전략 {j['xirr']['strategy']:.2f}% · QQQ {j['xirr']['qqq']:.2f}% · "
          f"동일 비중 {j['xirr']['ew']:.2f}% · 무작위 {j['random']['pct']:.1f}백분위", flush=True)
    for c in j["criteria"]:
        print(f"  {c['id']} {c['name']}: {'충족' if c['ok'] else '미달'} ({c['value']})")

    # 단계별 (비용 전 → 비용 뒤 → 세후)
    stages = {}
    for name, cost in (("비용 전", ZERO), ("비용 뒤", NO_TAX), ("세후", FULL)):
        stages[name] = {"strategy": xirr(run_value(m, board, cost)["book"].flows), "qqq": xirr(run_qqq(m, cost)["book"].flows),
                        "ew": xirr(run_ew(m, data, cost)["book"].flows)}
    out["stages"] = stages

    # 참고 보고 (a)~(e)
    ref = {}
    v1 = dict(vs.MAX_PTS, peg=0.0, gpa=10.0, fcf_yield=15.0)
    for name, total in (("v1 배점판", vs.total_from_points(scores, v1)), ("PEG만", peg_only_total(scores))):
        r = run_value(m, Board(scores, total, data.sic2), FULL, record=True)
        ref[name] = summarize(name, r, m)
    for name, rule in (("갈아타기 끔", Rules(swap="off")), ("절반 고정", Rules(swap="half"))):
        ref[name] = summarize(name, run_value(m, board, FULL, rule, record=True), m)
    loo = {}
    for k in vs.ITEMS:
        w = dict(vs.MAX_PTS, **{k: 0.0})
        loo[k] = xirr(run_value(m, Board(scores, vs.total_from_points(scores, w), data.sic2), FULL)["book"].flows)
    ref["항목 하나씩 뺀 판"] = loo
    # (d) 대상·N/A·PEG 0점
    u = scores[scores["universe"]]
    comp = u[u["total"].notna()]
    ref["대상"] = {"평균 대상 수": float(u.groupby("date").size().mean()),
                 "평균 점수 완전 수": float(comp.groupby("date").size().mean()),
                 "항목별 N/A %": {k: float(u[k].isna().mean() * 100) for k in vs.ITEMS},
                 "PEG 0점 % (점수 완전 중)": float((comp["peg"] == 0).mean() * 100)}
    # (e) 상위 5종목 기여 제외 · 국면별
    b = s["book"]
    pnl = sorted(((m.tickers[j], v) for j, v in b.pnl.items()), key=lambda x: -x[1])
    top5 = pnl[:5]
    fl = list(b.flows)
    fl[-1] = (fl[-1][0], fl[-1][1] - sum(v for _, v in top5))
    ref["상위 5종목 기여"] = {"top5": top5, "xirr_without_top5": xirr(fl), "bottom5": pnl[-5:]}
    sp, _ = curves(s, m)
    qp, _ = curves(q, m)
    ref["국면별"] = fsc.regime_perf(sp, qp, m.qqq_close_full)
    out["reference"] = ref
    # 분사 점검: 보유 중 하루 −15% 이상
    drops = []
    held_days: dict = {}
    for (i, kind, j, qq, usd, why) in b.log:
        held_days.setdefault(j, []).append((i, kind))
    for j, evs in held_days.items():
        if j == m.ti[QQQ]:
            continue
        r = pd.Series(m.close[:, j], index=m.days)
        ret = r / r.ffill().shift(1) - 1
        first = min(i for i, k in evs if k == "buy")
        last = max(i for i, _ in evs)
        if any(k == "buy" for i, k in evs) and j in b.hold:
            last = len(m.days) - 1
        for d, x in ret.iloc[first + 1:last + 1].items():
            if x <= -0.15:
                drops.append((str(d.date()), m.tickers[j], round(float(x) * 100, 1)))
    out["drops_15"] = sorted(drops)
    out["trades"] = [(str(m.days[i].date()), k, m.tickers[j], round(qq, 4), round(usd, 2), why) for (i, k, j, qq, usd, why) in b.log]
    out["seconds"] = time.time() - t0
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"끝 · {out['seconds']:.0f}초 · {OUT}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
