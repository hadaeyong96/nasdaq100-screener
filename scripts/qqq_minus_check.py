"""QQQ 실수 빼기 전략(qqq-minus) 0단계 — 데이터 확인만 (수익 계산 없음, docs/validation_zones.md 2장 4번). 탐색 구역(나스닥100, ~2021).

    python scripts/qqq_minus_check.py members   # 매달 첫 거래일 나스닥100 시점별 명단(위키 변경 이력 역산, 2007-03 ~ 2021-12)
    python scripts/qqq_minus_check.py prices    # yfinance 종가·배당·분할(auto_adjust=False, 받을 때 end=2022-01-01) → 실패 종목은 기존 캐시
    python scripts/qqq_minus_check.py facts     # EDGAR companyfacts → 주식 수·매출 사실, 접수일 ≤ 2021-12-31만 저장(10-K·10-Q)
    python scripts/qqq_minus_check.py kfactor   # 2022년 뒤 분할 배수 K를 2021년 이전 13F 분기 말 가격으로 역산(2022년 뒤 자료 안 받음)
    python scripts/qqq_minus_check.py check     # 매달 확보율 표(가격·첫 거래일·주식 수·시가총액) — 수익은 보지 않는다

가드:
- 가격은 받을 때부터 end=2022-01-01(2021-12-31까지)만 요청하고, 저장·읽기 때 2021-12-31 뒤 줄이 있으면 멈춘다.
- EDGAR 사실은 접수일 ≤ 2021-12-31만 꺼내 저장한다(2022년 이후 사실은 버린 줄 수만 센다). 요청은 data.edgar(초당 10회 이하), SEC_USER_AGENT 값은 출력하지 않는다.
- K(2022년 뒤 분할 배수)는 2022년 뒤 분할 기록을 받지 않고, 이미 받은 13F 데이터셋(접수일 ≤ 2021-12-30)의 분기 말 가격 ÷ 우리 종가로 구한다.

출력: data/cache/qqq_minus/ (git 제외) + 화면 요약
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from core import mcap_growth as mg  # noqa: E402

OUT = ROOT / "data" / "cache" / "qqq_minus"
PRICES = OUT / "prices"
MEMBERS = OUT / "members.csv"
PRICE_META = OUT / "price_meta.csv"
FACTS = OUT / "facts.parquet"
CIKS = OUT / "ciks.csv"
KFACTOR = OUT / "kfactor.csv"
COVER = OUT / "coverage.csv"
SEAL = pd.Timestamp("2021-12-31")
FILED_LAST = "2021-12-31"
NDX_FIRST = pd.Timestamp("2007-03-01")  # 위키 변경 이력 첫 기록 2007-02-01 → 그 뒤 첫 달부터
FORMS = ("10-K", "10-K/A", "10-Q", "10-Q/A")
SHARES_CSO = "CommonStockSharesOutstanding"  # 대차대조표 발행 주식(여러 종류 주식 회사의 합계) — dei·가중평균이 없을 때만
MCAP_MIN_USD = 1e9  # 명단에 있던 달 시가총액 중앙값이 이보다 작으면 다른 회사 의심(sw1·S&P 500 점검과 같은 기준)
K_TOL, K_MIN_HIT, K_MIN_RATIO = 0.02, 4, 0.6
MCAP_BOUNDS = (5e8, 5e12)  # 주식 수 후보 거르기: 그날 시가총액 5억 ~ 5조 달러 밖이면 XBRL 단위 실수로 본다


def trading_days(a: date = date(2000, 1, 1), b: date = date(2021, 12, 31)) -> pd.DatetimeIndex:
    from data.market_calendar import trading_days_between

    return trading_days_between(a, b)


def month_firsts(start: str = "2000-01-01") -> list[pd.Timestamp]:
    """매달 첫 NYSE 거래일 (start ~ 2021-12)."""
    d = trading_days()
    s = pd.Series(d, index=d)
    return [x for x in s.groupby([d.year, d.month]).min() if x >= pd.Timestamp(start)]


def renames() -> dict[str, str]:
    import sw1_data_check as sw

    return sw.renames()


# ── 1. 명단 ──────────────────────────────────────────────────────────────────


def step_members() -> None:
    from data import universe_history as uh

    cur = set(pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"])
    rows = [{"date": d.date(), "ticker": t} for d in month_firsts(str(NDX_FIRST.date()))
            for t in sorted(uh.point_in_time_universe(d.date(), cur))]
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(MEMBERS, index=False)
    n = m.groupby("date").size()
    print(f"[members] 달 {len(n)} ({n.index.min()} ~ {n.index.max()}) · 명단 수 {n.min()}~{n.max()} · "
          f"합집합 {m['ticker'].nunique()} (2012~ {m.loc[pd.to_datetime(m['date']) >= '2012-01-01', 'ticker'].nunique()})")


def load_members() -> dict[pd.Timestamp, list[str]]:
    m = pd.read_csv(MEMBERS, parse_dates=["date"])
    return {d: sorted(g["ticker"]) for d, g in m.groupby("date")}


# ── 2. 가격 ──────────────────────────────────────────────────────────────────


def _yf(sym: str) -> pd.DataFrame | None:
    import yfinance as yf

    h = yf.Ticker(sym).history(start="1985-01-01", end="2022-01-01", auto_adjust=False, actions=True)
    if h is None or h.empty or "Close" not in h:
        return None
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    d = pd.DataFrame({"close": h["Close"], "div": h.get("Dividends", 0.0), "split": h.get("Stock Splits", 0.0)})
    d = d[d.index <= SEAL].dropna(subset=["close"])
    return d if len(d) else None


def step_prices() -> None:
    """명단 합집합마다: yfinance(그 티커 → 바뀐 티커) 전체 기간 → 실패면 E1 캐시(2011-06~, 배당 있음) → sw1 load_px(배당 없음).
    첫 거래일은 yfinance 전체 기간에서 받은 것만 '확실', 기존 캐시는 캐시 시작일이면 '그 전 상장'으로 표시."""
    import sw1_data_check as sw

    PRICES.mkdir(parents=True, exist_ok=True)
    tickers = sorted(pd.read_csv(MEMBERS)["ticker"].unique())
    ren = renames()
    rows, failed = [], []
    for t in tickers:
        path = PRICES / f"{t}.csv"
        src = None
        if path.exists():
            d = pd.read_csv(path, index_col=0, parse_dates=True)
            src = json.loads((PRICES / f"{t}.json").read_text(encoding="utf-8"))["source"]
        else:
            d = None
            for sym in [t.replace(".", "-")] + ([ren[t]] if t in ren else []):
                try:
                    d = _yf(sym)
                except Exception:  # noqa: BLE001 — 실패 종목은 모아서 보고
                    d = None
                time.sleep(0.4)
                if d is not None:
                    src = f"yfinance({sym})"
                    break
            if d is None:
                e1 = ROOT / "data" / "cache" / "e1" / "prices" / f"{t}.csv"
                if e1.exists():
                    x = pd.read_csv(e1, index_col=0, parse_dates=True)
                    if len(x):
                        d, src = x[["close", "div", "split"]], "e1 캐시(2011-06~)"
            if d is None:
                x, s2 = sw.load_px(t)
                if x is not None:
                    d = pd.DataFrame({"close": x["close"], "div": 0.0, "split": 0.0})
                    src = f"sw1 {s2}(배당 없음)"
                    tf = {"tiingo": sw.SP500_TIINGO, "tiingo(SW1)": sw.TIINGO_CACHE}.get(s2)
                    if tf is not None and (tf / f"{t}.csv").exists():  # 분할 반영 종가 → 분할 날짜·비율을 Tiingo 원본 split_factor에서 되살린다
                        r = pd.read_csv(tf / f"{t}.csv", index_col=0, parse_dates=True)
                        r.index = pd.DatetimeIndex(r.index).tz_localize(None).normalize()
                        sf = r["split_factor"]
                        d["split"] = sf[(sf != 1.0) & (sf > 0) & (sf.index <= SEAL)].reindex(d.index).fillna(0.0)
            if d is None:
                failed.append(t)
                continue
            d = d[d.index <= SEAL]
            d.to_csv(path)
            (PRICES / f"{t}.json").write_text(json.dumps({"source": src}, ensure_ascii=False), encoding="utf-8")
        if (d.index > SEAL).any():
            raise RuntimeError(f"{t}: 2021-12-31 뒤 가격 줄")
        sure = src.startswith("yfinance")
        rows.append({"ticker": t, "source": src, "first": d.index.min().date(), "last": d.index.max().date(), "first_sure": sure,
                     "has_div": bool((d["div"].fillna(0) > 0).any())})
        print(f"[prices] {t} {src} {d.index.min().date()}~{d.index.max().date()}", flush=True)
    meta = pd.DataFrame(rows)
    meta.to_csv(PRICE_META, index=False, encoding="utf-8-sig")
    print(f"[prices] {len(meta)}/{len(tickers)} · 출처 {meta['source'].str.split('(').str[0].value_counts().to_dict()} · 실패 {failed}")


def load_prices() -> dict[str, pd.DataFrame]:
    out = {}
    for f in PRICES.glob("*.csv"):
        d = pd.read_csv(f, index_col=0, parse_dates=True)
        if (d.index > SEAL).any():
            raise RuntimeError(f"{f.stem}: 2021-12-31 뒤 가격 줄")
        d["div"] = d["div"].fillna(0.0)
        d["split"] = d["split"].fillna(0.0)
        out[f.stem] = d
    return out


def splits_of(d: pd.DataFrame) -> pd.Series:
    """분할 열 전부(yfinance는 분사 조정 비율도 여기에 적는다) — 그날 실제 종가 되살리기용."""
    s = d["split"]
    return s[(s > 0) & (s != 1.0)].astype(float)


def true_splits(d: pd.DataFrame) -> pd.Series:
    """흔한 분할 비율만(value_score_zone3_data.is_true_split) — 주식 수 조정용(분사 비율은 주식 수를 바꾸지 않는다)."""
    import value_score_zone3_data as vd

    s = splits_of(d)
    return s[[vd.is_true_split(float(x)) for x in s]] if len(s) else s


# ── 3. EDGAR ─────────────────────────────────────────────────────────────────


# 지주회사 재편·합병으로 CIK가 바뀐 종목(옛 CIK → 새 CIK 순서, 모두 같은 회사의 연속). 실행 때 entityName을 찍어 확인한다.
# AVGO는 sw1 표의 1054374(옛 Broadcom Corp, 2016 인수된 다른 회사)를 쓰지 않는다.
MANUAL_CIKS = {
    "AVGO": [1441634, 1649338, 1730168],   # Avago Technologies → Broadcom Ltd → Broadcom Inc
    "GOOG": [1288776, 1652044], "GOOGL": [1288776, 1652044],  # Google Inc → Alphabet
    "ESRX": [885721, 1532063],             # Express Scripts Inc → Express Scripts Holding(2012 Medco 합병)
    "MYL": [69499, 1623613],               # Mylan Inc → Mylan N.V.(2015)
    "MRVL": [1058057, 1835632],            # Marvell Technology Group → Marvell Technology Inc(2021)
    "LBTYA": [1316631, 1570585], "LBTYK": [1316631, 1570585],  # Liberty Global Inc → Liberty Global plc(2013)
}


def resolve_ciks(tickers: list[str]) -> dict[str, list[int]]:
    """MANUAL_CIKS → sw1 cik.csv → 가치평가 sec_meta(CIK 여럿) → SEC 현재 티커 목록(그 티커, 바뀐 티커)."""
    from data import edgar

    import value_score_zone3_check as c

    out: dict[str, list[int]] = {}
    sw = pd.read_csv(ROOT / "data" / "cache" / "sw1" / "cik.csv").dropna(subset=["cik"])
    out.update({t: [int(v)] for t, v in zip(sw["ticker"], sw["cik"])})
    meta = json.loads(c.META.read_text(encoding="utf-8"))
    for t in tickers:
        if t not in out and meta.get(t, {}).get("ciks"):
            out[t] = [int(x) for x in meta[t]["ciks"]]
    cmap = edgar.fetch_ticker_to_cik()
    ren = renames()
    for t in tickers:
        if t in out:
            continue
        for sym in (t, ren.get(t)):
            if sym and cmap.get(sym.upper().replace("-", ".")):
                out[t] = [int(cmap[sym.upper().replace("-", ".")])]
                break
    out.update(MANUAL_CIKS)
    return {t: out[t] for t in tickers if t in out}


def step_facts() -> None:
    from data import edgar

    tickers = sorted(pd.read_csv(MEMBERS)["ticker"].unique())
    ciks = resolve_ciks(tickers)
    pd.DataFrame([{"ticker": t, "ciks": ";".join(map(str, ciks[t])) if t in ciks else None} for t in tickers]).to_csv(CIKS, index=False)
    rows, failed, late, fetched, names = [], [], 0, 0, {}
    for t, cik in ((t, c) for t, cs in ciks.items() for c in cs):
        cached = (edgar.FACTS_CACHE_DIR / f"CIK{cik:010d}.json").exists()
        try:
            cf = edgar.fetch_company_facts(cik)
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{t}/{cik}({type(exc).__name__})")
            continue
        fetched += 0 if cached else 1
        names.setdefault(t, cf.get("entityName"))
        if t in MANUAL_CIKS:
            print(f"[facts] 수동 CIK {t} {cik} = {cf.get('entityName')}")
        for tax, tags in (("dei", [mg.SHARES_DEI]), ("us-gaap", [mg.SHARES_WA, SHARES_CSO] + mg.REV_TAGS)):
            for tag in tags:
                for e in edgar.extract_fact_entries(cf, tax, tag):
                    if e.get("form") not in FORMS or not e.get("filed") or not e.get("end"):
                        continue
                    if e["filed"] > FILED_LAST:
                        late += 1
                        continue
                    rows.append({"ticker": t, "cik": cik, "tag": tag, "start": e.get("start"), "end": e["end"], "val": float(e["val"]),
                                 "filed": e["filed"], "form": e["form"], "accn": e.get("accn"), "unit": e["unit"]})
    df = pd.DataFrame(rows)
    if (df["filed"] > FILED_LAST).any():
        raise RuntimeError("2022년 이후 접수 사실이 섞였습니다")
    df.to_parquet(FACTS, index=False)
    (OUT / "entity_names.json").write_text(json.dumps(names, ensure_ascii=False), encoding="utf-8")
    print(f"[facts] 티커 {len(tickers)} · CIK {len(ciks)} · 새로 받은 companyfacts {fetched} · 사실 {len(df):,} · 2022년 이후 접수라 버린 사실 {late:,}")
    print(f"[facts] 주식 수 있는 티커 {df.loc[df['tag'].isin([mg.SHARES_DEI, mg.SHARES_WA, SHARES_CSO]), 'ticker'].nunique()} · "
          f"매출 {df.loc[df['tag'].isin(mg.REV_TAGS), 'ticker'].nunique()} · 실패 {failed} · CIK 없음 {[t for t in tickers if t not in ciks]}")


def load_facts() -> dict[str, pd.DataFrame]:
    f = pd.read_parquet(FACTS)
    if (f["filed"] > FILED_LAST).any():
        raise RuntimeError("사실 캐시에 2022년 이후 접수 사실")
    return {t: g for t, g in f.groupby("ticker")}


# ── 4. K (2022년 뒤 분할 배수) ────────────────────────────────────────────────


def step_kfactor() -> None:
    """2021-12-31까지 가격이 있는 종목: 13F 분기 말 가격(매니저 5명 이상 중앙값) ÷ (우리 종가 × 그날 뒤 ~ 2021-12-31 분할 열 곱) = K(상수)인 CUSIP을 찾는다.
    yfinance는 2022년 뒤 분사도 분할 열·종가에 반영하므로 K는 흔한 분할 배수가 아닐 수 있다 →
    회사명이 맞는 CUSIP(13F 이름 ↔ EDGAR 이름·sw1 이름)마다 비율의 중앙값을 K로 보고, ±2% 안 분기 4개 이상·같이 있는 분기의 60% 이상이면 후보.
    가장 많이 맞은 CUSIP의 K. 못 찾으면 K 없음(시가총액 못 구함). 2021-12-31 전에 가격이 끝난 종목은 2022년 뒤 분할·분사가 없으므로 K = 1."""
    import superinvestor_check as sc

    px = load_prices()
    pt = sc.price_table()
    pt = pt[pt["n"] >= 5]
    names = json.loads((OUT / "entity_names.json").read_text(encoding="utf-8"))
    sub_dir = ROOT / "data" / "cache" / "edgar" / "submissions"  # 이미 받아 둔 것만(옛 이름, 2022년 전 시작) — 새로 받지 않는다
    former: dict[str, list[str]] = {}
    for t, cs in pd.read_csv(CIKS).dropna().set_index("ticker")["ciks"].astype(str).items():
        for cik in cs.split(";"):
            p = sub_dir / f"CIK{int(cik):010d}.json"
            if p.exists():
                j = json.loads(p.read_text(encoding="utf-8"))
                former.setdefault(t, []).extend(x["name"] for x in j.get("formerNames") or [] if str(x.get("from", ""))[:4] < "2022")
    off = pd.read_csv(sc.OFFICIAL / "names.csv", dtype=str).set_index("cusip")["name"].to_dict()
    issuer = pt.sort_values("period").groupby("cusip")["issuer"].last().to_dict()
    sw = pd.read_csv(ROOT / "data" / "cache" / "sw1" / "cik.csv").set_index("ticker")["edgar_name"].dropna().to_dict()
    groups = {pd.Timestamp(p): g for p, g in pt.groupby("period")}
    rows = []
    for t, d in sorted(px.items()):
        if d.index.max() < pd.Timestamp("2021-12-30"):
            rows.append({"ticker": t, "k": 1.0, "how": "2021년 전 가격 끝(2022년 뒤 분할 없음)"})
            continue
        sp = splits_of(d)
        tn = [x for x in (names.get(t), sw.get(t)) if x] + former.get(t, [])
        name_ok: dict[str, bool] = {}
        ratios: dict[str, list[float]] = {}
        for p, g in groups.items():
            day = d.index[d.index <= p]
            if not len(day) or (p - day[-1]).days > 5:
                continue
            base = float(d.at[day[-1], "close"]) * mg.split_factor(sp, day[-1], SEAL)
            for c, v in zip(g["cusip"], g["px"]):
                if c not in name_ok:
                    name_ok[c] = any(sc.name_match(off.get(c) or issuer.get(c), x) for x in tn)
                if name_ok[c]:
                    ratios.setdefault(c, []).append(v / base)
        cand = []
        for c, rs in ratios.items():
            r = np.array(rs)
            k = float(np.median(r))
            n = int((np.abs(r / k - 1) < K_TOL).sum())
            if n >= K_MIN_HIT and n / len(r) >= K_MIN_RATIO:
                cand.append((n, n / len(r), c, k, off.get(c) or issuer.get(c)))
        if not cand:
            rows.append({"ticker": t, "k": None, "how": "13F 가격 짝 없음"})
            continue
        n, ratio, c, k, nm = max(cand)
        rows.append({"ticker": t, "k": k, "how": f"13F {c} {nm} {n}분기({ratio:.0%})"})
    out = pd.DataFrame(rows)
    out.to_csv(KFACTOR, index=False, encoding="utf-8-sig")
    print(f"[kfactor] 종목 {len(out)} · K=1 {int((out['k'] == 1).sum())} · K≠1 {int(((out['k'] != 1) & out['k'].notna()).sum())} · 없음 {int(out['k'].isna().sum())}")
    print(out[(out["k"] != 1)].to_string(index=False))


def load_k() -> dict[str, float | None]:
    k = pd.read_csv(KFACTOR, encoding="utf-8-sig")
    return {t: (None if pd.isna(v) else float(v)) for t, v in zip(k["ticker"], k["k"])}


# ── 5. 확보율 ─────────────────────────────────────────────────────────────────


class Inputs:
    """매달 첫 거래일마다 종목별 그날 실제 종가·첫 거래일·주식 수·시가총액(수익 없음)."""

    def __init__(self) -> None:
        self.members = load_members()
        self.px = load_prices()
        self.facts = load_facts()
        self.k = load_k()
        self.meta = pd.read_csv(PRICE_META, encoding="utf-8-sig").set_index("ticker")
        self.cik = pd.read_csv(CIKS).dropna().set_index("ticker")["ciks"].astype(str).to_dict()  # "옛;새" — 같은 문자열 = 같은 회사
        self.days = trading_days(date(1985, 1, 1))
        self.di = {d: i for i, d in enumerate(self.days)}

    def raw_close(self, t: str, d: pd.Timestamp) -> float | None:
        """그날 실제 종가 = 캐시 종가(받은 날까지 분할 반영) × (그날 뒤 ~ 2021-12-31 분할 곱) × K."""
        x = self.px.get(t)
        k = self.k.get(t)
        if x is None or k is None or d not in x.index or pd.isna(x.at[d, "close"]):
            return None
        return float(x.at[d, "close"]) * mg.split_factor(splits_of(x), d, SEAL) * k

    def first_day(self, t: str) -> tuple[pd.Timestamp | None, bool]:
        if t not in self.meta.index:
            return None, False
        return pd.Timestamp(self.meta.at[t, "first"]), bool(self.meta.at[t, "first_sure"])

    def age_ok(self, t: str, d: pd.Timestamp, n: int = 504) -> bool | None:
        """규칙 A: 첫 거래일로부터 n거래일 이상이면 True, 미만 False, 모르면 None(첫 거래일이 캐시 시작일이라 그 전 상장일 수 있는데 n일이 안 지남)."""
        f, sure = self.first_day(t)
        if f is None:
            return None
        age = self.di[d] - self.di[self.days[self.days >= f][0]]
        if age >= n:
            return True
        return False if sure else None

    def share_cands(self, t: str, x) -> list[float]:
        """x 전 접수 주식 수 후보(400일 한도) 우선순위 순: dei(가장 늦은 접수 → 가장 늦은 기준일, 같은 접수번호 값 합 = 여러 종류 주식)
        → 가중평균 기본(가장 늦은 기준일, 3개월 기간 우선, 가장 늦은 접수) → CommonStockSharesOutstanding(가장 늦은 접수 → 가장 늦은 기준일).
        분할 조정은 **접수일 뒤 ~ x** 분할만 곱한다 — 분할 뒤 접수된 보고는 이미 분할 기준으로 소급 수정돼 있다(기준일부터 곱하면 두 번 조정)."""
        f = self.facts.get(t)
        if f is None:
            return []
        sp = true_splits(self.px[t]) if t in self.px else None
        x = pd.Timestamp(x)
        xs, lo = x.date().isoformat(), (x - pd.Timedelta(days=mg.SHARES_MAX_AGE)).date().isoformat()
        s = f[(f["filed"] < xs) & (f["end"] >= lo)]
        out = []
        dei = s[s["tag"] == mg.SHARES_DEI]
        if len(dei):
            g = dei[dei["filed"] == dei["filed"].max()]
            g = g[g["end"] == g["end"].max()]
            accn = sorted(g["accn"].dropna().unique())
            g = g[g["accn"] == accn[-1]] if accn else g
            out.append((float(g["val"].sum()), g["filed"].iloc[-1]))
        wa = s[s["tag"] == mg.SHARES_WA]
        if len(wa):
            g = wa[wa["end"] == wa["end"].max()].copy()
            dur = (pd.to_datetime(g["end"]) - pd.to_datetime(g["start"])).dt.days
            q3 = g[(dur >= 80) & (dur <= 100)]
            g = q3 if len(q3) else g
            g = g[g["filed"] == g["filed"].max()]
            out.append((float(g["val"].iloc[-1]), g["filed"].iloc[-1]))
        c = s[s["tag"] == SHARES_CSO]
        if len(c):
            c = c[c["filed"] == c["filed"].max()]
            c = c[c["end"] == c["end"].max()]
            out.append((float(c["val"].iloc[-1]), c["filed"].iloc[-1]))
        return [v * mg.split_factor(sp, filed, x) for v, filed in out if v > 0]

    def last_day(self, x) -> pd.Timestamp:
        return self.days[self.days <= pd.Timestamp(x)][-1]

    def shares(self, t: str, x) -> float | None:
        """주식 수(XBRL 단위 실수 거르기): 후보 중 그날(그 전 마지막 거래일) 실제 종가로 본 시가총액이 MCAP_BOUNDS 안인 것만.
        셋 남으면 중앙값, 하나·둘이면 우선순위 첫 값. 종가가 없으면 None."""
        p = self.raw_close(t, self.last_day(x))
        if p is None:
            return None
        ok = [v for v in self.share_cands(t, x) if MCAP_BOUNDS[0] <= v * p <= MCAP_BOUNDS[1]]
        if not ok:
            return None
        return float(np.median(ok)) if len(ok) == 3 else ok[0]

    def share_growth(self, t: str, d: pd.Timestamp) -> float | None:
        """규칙 B: 직전 보고 주식 수 ÷ 12개월 전(달력 1년 전 날짜까지 접수) 보고 주식 수(오늘 기준 분할 조정) − 1."""
        now = self.shares(t, d)
        back = d - pd.DateOffset(years=1)
        old = self.shares(t, back)
        if now is None or old is None:
            return None
        sp = true_splits(self.px[t]) if t in self.px else None
        return now / (old * mg.split_factor(sp, back, d)) - 1

    def mcaps(self, d: pd.Timestamp) -> dict[str, float]:
        """그날 명단 종목의 시가총액(달러). 같은 CIK 종목(여러 종류 주식)이 명단에 둘 이상이면 회사 시가총액을 그 수로 나눈다(종류별 주식 수 없음)."""
        mem = self.members[d]
        out = {}
        by_cik: dict[str, int] = {}
        for t in mem:
            if t in self.cik:
                by_cik[self.cik[t]] = by_cik.get(self.cik[t], 0) + 1
        for t in mem:
            p, s = self.raw_close(t, d), self.shares(t, d)
            if p is not None and s is not None:
                out[t] = p * s / by_cik.get(self.cik.get(t), 1)
        return out


def step_check() -> None:
    inp = Inputs()
    rows = []
    for d, mem in sorted(inp.members.items()):
        n = len(mem)
        has_px = [t for t in mem if t in inp.px and d in inp.px[t].index]
        has_k = [t for t in has_px if inp.k.get(t) is not None]
        caps = inp.mcaps(d)
        age = [inp.age_ok(t, d) for t in mem]
        shg = [inp.share_growth(t, d) for t in mem]
        rows.append({"date": d.date(), "members": n, "price": len(has_px) / n, "k": len(has_k) / n,
                     "shares": sum(1 for t in mem if inp.shares(t, d) is not None) / n, "mcap": len(caps) / n, "n_mcap": len(caps),
                     "first_known": sum(a is not None for a in age) / n, "young": sum(a is False for a in age),
                     "share_g_known": sum(g is not None for g in shg) / n, "diluted": sum(1 for g in shg if g is not None and g > 0.05)})
        print(f"[check] {d.date()} 명단 {n} · 시총 {len(caps)}", flush=True)
    m = pd.DataFrame(rows)
    m.to_csv(COVER, index=False, encoding="utf-8-sig")
    m["year"] = pd.to_datetime(m["date"]).dt.year
    pd.set_option("display.width", 220)
    agg = m.groupby("year").agg(months=("date", "size"), members=("members", "mean"), price=("price", "mean"), k=("k", "mean"),
                                shares=("shares", "mean"), mcap_min=("mcap", "min"), mcap=("mcap", "mean"), first=("first_known", "mean"),
                                young=("young", "mean"), share_g=("share_g_known", "mean"), diluted=("diluted", "mean"))
    print(agg.round(3).to_string())


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    step = sys.argv[1] if len(sys.argv) > 1 else "check"
    {"members": step_members, "prices": step_prices, "facts": step_facts, "kfactor": step_kfactor, "check": step_check}[step]()
