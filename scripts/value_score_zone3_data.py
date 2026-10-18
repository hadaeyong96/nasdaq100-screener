"""가치평가 전략(value-score) 구역 ③ 데이터 — 명단·가격·재무·월말 점수 (0단계 점검과 판정 실행이 같이 쓴다).

가드(docs/prereg/value_score.md 2장):
- 명단: 매달 마지막 거래일 기준 S&P 500 − (나스닥 100 ∪ S&P 100), rs26 0단계와 같은 재료·같은 빼기 규칙.
- 가격: 구역 ③에 한 번이라도 속한 종목만, 2011-06-01 ~ 2021-12-31 줄만 읽는다(넘는 줄에서 멈춤).
- 재무: data/cache/value_score/facts.parquet(접수일 ≤ 2021-12-31 10-K·10-K/A만 저장)만 읽고, 한 줄이라도 넘으면 멈춘다.
- 그날 실제 종가: Tiingo는 raw_close, yfinance는 캐시 종가 × (그날 뒤 ~ 2021-12-31 분할·분사 비율 곱) × K
  (K = 2022년 이후 기록 곱, ksplit.csv — 사용자 결정 2026-10-17).
- EPS·주식 수 조정(E1 방식)에는 흔한 분할 비율(2, 3, 1.5, 1/10 …)인 기록만 쓴다 — yfinance는 분사도 split 열에 적는다.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rs26_zone3_check as z  # noqa: E402
import value_score_zone3_check as c  # noqa: E402
from core import value_score as vs  # noqa: E402
from data.market_calendar import trading_days_between  # noqa: E402

END = pd.Timestamp("2021-12-31")
SCORES = c.OUT / "scores.parquet"
MIN_PRICE = 5.0
NICE = [2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 50, 1.5, 2.5, 1.25, 4 / 3]
NICE_TOL = 0.01


def is_true_split(r: float) -> bool:
    """흔한 분할·역분할 비율과 1% 안이면 분할(분사 조정 비율은 아님)."""
    cands = NICE + [1 / q for q in NICE]
    return any(abs(r / q - 1) < NICE_TOL for q in cands)


def month_ends() -> list[pd.Timestamp]:
    """2012-01 ~ 2021-12 매월 마지막 NYSE 거래일."""
    return z.signal_days()


def nyse_days() -> pd.DatetimeIndex:
    return trading_days_between(date(2011, 6, 1), END.date())


class Data:
    """구역 ③ 명단(월말)·가격·재무·SIC."""

    def __init__(self) -> None:
        self.tickers, gap, renames = c.zone_tickers()
        self.mes = month_ends()
        self._members_init()
        meta = json.loads(c.META.read_text(encoding="utf-8"))
        self.sic = {t: (str(meta[t]["sic"]).zfill(4) if meta.get(t, {}).get("sic") else None) for t in self.tickers}
        self.sic2 = {t: (s[:2] if s else None) for t, s in self.sic.items()}
        self.financial = {t: (s is not None and 6000 <= int(s) <= 6799) for t, s in self.sic.items()}
        allowed = set().union(*self.mem.values())
        ks = pd.read_csv(c.KSPLIT, encoding="utf-8-sig").set_index("ticker")
        self.close, self.raw, self.div, self.splits = {}, {}, {}, {}
        for t in self.tickers:
            if t not in allowed:
                continue
            p = z.price_path(t, gap, renames)
            if p is None or not p.exists():
                continue
            if str(gap.at[t, "price_source"]) == "tiingo":
                df = z.read_prices_guarded(t, p, allowed)
                if df.empty:
                    continue
                self.close[t], self.raw[t], self.div[t] = df["close"], df["raw_close"], df["div"]
                sp = df["split"][(df["split"] > 0) & (df["split"] != 1.0)]
            else:
                if t not in allowed:
                    raise z.ZoneGuardError(t)
                df = c.read_yf_guarded(t, p, allowed)
                if df.empty:
                    continue
                k = float(ks.at[t, "k"])
                after = float(df["factor"].prod()) / df["factor"].cumprod()  # 다음 날 ~ 2021-12-31 비율 곱 (음수 shift 없이)
                self.close[t], self.div[t] = df["close"], df["div"]
                self.raw[t] = df["close"] * after * k
                sp = df["factor"][df["factor"] != 1.0]
            self.splits[t] = sp[[is_true_split(float(x)) for x in sp]] if len(sp) else sp
        f = pd.read_parquet(c.FACTS)
        if (f["filed"] > c.FILED_LAST).any():
            raise z.ZoneGuardError("재무 캐시에 2022년 이후 접수 사실이 있습니다")
        f = vs.prepare_facts(f)
        self.facts = {t: g for t, g in f.groupby("ticker")}
        self.filed = {t: np.array(sorted(g["filed"].unique())) for t, g in self.facts.items()}
        self._tables: dict = {}

    def _members_init(self) -> None:
        fja = z.load_fja()
        _, _, sp100 = z.load_sp100()
        cur = {z.norm(t) for t in pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"]}
        aliases = z.alias_map()
        keep = set(self.tickers)
        self.mem = {}
        for d in self.mes:
            sp = fja.loc[:d].iloc[-1]["tickers"]
            excl = z.ndx_on(d, cur) | set(sp100.get(d.strftime("%Y-%m")) or set())
            m = {t for t in sp if t not in excl and not (aliases.get(t, {t}) & excl)}
            self.mem[d] = m
        self.mem_all = dict(self.mem)  # 가격 없는 종목 포함 (가격 공백 집계용)
        self.mem = {d: {t for t in m if t in keep} for d, m in self.mem.items()}

    def table(self, t: str, d: pd.Timestamp) -> pd.DataFrame:
        """d 전에 접수된 10-K로 만든 연간 표 (마지막 접수일로 메모)."""
        fl = self.filed.get(t)
        if fl is None:
            return pd.DataFrame()
        i = int(np.searchsorted(fl, d.date().isoformat(), side="left"))
        if i == 0:
            return pd.DataFrame()
        key = (t, fl[i - 1])
        if key not in self._tables:
            self._tables[key] = vs.annual_table(self.facts[t], d)
        return self._tables[key]

    def price_on(self, t: str, d: pd.Timestamp) -> float | None:
        s = self.raw.get(t)
        if s is None or d not in s.index or pd.isna(s[d]):
            return None
        return float(s[d])


def compute_scores(data: Data) -> pd.DataFrame:
    """월말마다 그날 가격이 있는 모든 종목의 항목 점수·총점 + 대상 여부(판단일 = 월말 종가).

    출력 열: date, ticker, member, universe(대상: 명단·금융 아님·5달러 이상·연간 3년 이상·최근 3년 흑자),
            years, profitable, price_raw, financial, sic2, 항목 10개 점수(기본 배점, N/A = NaN), n_na, total, peg_value, eps_growth,
            gpa_raw, fcf_yield_raw
    """
    rows = []
    for d in data.mes:
        ms, info = {}, {}
        for t in data.tickers:
            px = data.price_on(t, d)
            if px is None:
                continue
            tb = data.table(t, d)
            m = vs.metrics(tb, px, data.splits.get(t), d)
            ms[t] = m
            member = t in data.mem[d]
            uni = (member and not data.financial[t] and px >= MIN_PRICE and m.years >= 3 and m.profitable)
            info[t] = {"member": member, "universe": uni, "years": m.years, "profitable": m.profitable,
                       "price_raw": px, "financial": data.financial[t], "sic2": data.sic2[t]}
        universe = [t for t in ms if info[t]["universe"]]
        sc = vs.score_day(ms, universe)
        for t in ms:
            r = {"date": d, "ticker": t, **info[t], **{k: sc.at[t, k] for k in vs.ITEMS},
                 "n_na": int(sc.at[t, "n_na"]), "total": sc.at[t, "total"], "peg_value": ms[t].peg,
                 "eps_growth": ms[t].eps_growth, "gpa_raw": ms[t].raw["gpa"], "fcf_yield_raw": ms[t].raw["fcf_yield"]}
            rows.append(r)
    out = pd.DataFrame(rows)
    for k in vs.ITEMS + ["total", "peg_value", "eps_growth", "gpa_raw", "fcf_yield_raw"]:
        out[k] = pd.to_numeric(out[k], errors="coerce")
    return out


def load_scores(data: Data | None = None, rebuild: bool = False) -> pd.DataFrame:
    if SCORES.exists() and not rebuild:
        return pd.read_parquet(SCORES)
    data = data or Data()
    s = compute_scores(data)
    s.to_parquet(SCORES, index=False)
    return s
