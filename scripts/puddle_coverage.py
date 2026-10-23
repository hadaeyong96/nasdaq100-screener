"""상승장 대형주 웅덩이 — 1단계 확보율 표 (수익 계산 없음, 2026-10-23).

명단: 2000-01 ~ 2007-01은 나스닥 공식 구성 종목 페이지의 Wayback 사본(scripts.puddle_data), 2007-02 ~ 2021-12는 기존 위키백과 역산.
확보율(달별, 그달 첫 거래일):
- 사본에 지수 비중이 있는 달: 「비중 상위 10」(같은 회사는 비중 합쳐 하나로) 중 그날 20일 평균 거래대금을 낼 수 있는(가격 있는) 비율.
  비중(수정 시가총액 가중)은 이 전략의 거래대금 상위 10이 대신하려는 시가총액 순위에 가장 가깝다.
- 비중이 없는 달: 구성 종목 중 가격 있는 비율(회사 기준)과 못 구한 티커.
출력: docs/research/puddle_megacap_data.md, data/cache/puddle/coverage.json, data/cache/puddle/membership.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import puddle_data as pdd  # noqa: E402

MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}
# 같은 회사 여러 종류 주식 → 대표 키 (GOOG·GOOGL처럼 한 종목으로)
COMPANY = {"GOOG": "GOOGL", "FOX": "FOXA", "DISCK": "DISCA", "DISCB": "DISCA", "LBTYK": "LBTYA", "LBTYB": "LBTYA",
           "LMCK": "LMCA", "LMCB": "LMCA", "NWS": "NWSA", "LILAK": "LILA", "FWONK": "FWONA", "VIA": "VIAB",
           "CMCSK": "CMCSA", "LINTA": "LINTA", "ADLAE": "ADLAC", "GMSTE": "GMST", "MOLXE": "MOLX", "MOLXA": "MOLX",
           "BATRK": "BATRA", "LSXMK": "LSXMA", "LSXMB": "LSXMA", "QRTEB": "QRTEA", "ZG": "Z", "UAA": "UA"}


def company_of(t: str) -> str:
    return COMPANY.get(t, t)


def parse_asof(text: str | None, snap_ts: str) -> pd.Timestamp:
    """"Dec. 11, 00" → 2000-12-11. 못 읽으면 사본 날짜."""
    if text:
        m = re.match(r"([A-Za-z]{3})\.? *(\d{1,2}), *(\d{2,4})", text.strip())
        if m and m.group(1) in MONTHS:
            y = int(m.group(3))
            y = (2000 + y if y < 50 else 1900 + y) if y < 100 else y
            return pd.Timestamp(y, MONTHS[m.group(1)], int(m.group(2)))
    return pd.Timestamp(snap_ts[:8])


def snapshot_membership() -> list[dict]:
    """사본 → [{asof, tickers, weights(dict 또는 None), src}] (구성 90~110개인 사본만, 날짜순)."""
    df = pdd.load_snapshot_table()
    out = []
    for _, r in df.iterrows():
        if not (90 <= r["n"] <= 110):
            continue
        ws = r["weights"]
        has_w = sum(w is not None and not (isinstance(w, float) and np.isnan(w)) for w in ws) >= 90
        out.append({"asof": parse_asof(r["asof"], str(r["snap_ts"])), "tickers": list(r["tickers"]),
                    "weights": dict(zip(r["tickers"], ws)) if has_w else None, "src": f"{r['snap_ts']} {r['url']}"})
    return sorted(out, key=lambda x: x["asof"])


def has_price(t: str, d: pd.Timestamp, cache: dict) -> bool:
    """d일에 20일 평균 거래대금을 낼 수 있는가(그날까지 20거래일 값이 있음)."""
    if t not in cache:
        cache[t] = pdd.price_for(t)
    df = cache[t]
    if df is None:
        return False
    s = df.loc[:d]
    return len(s) >= 20 and s.index[-1] >= d - pd.Timedelta(days=7) and (s["close"].tail(20) * s["volume"].tail(20)).notna().all()


def main() -> None:
    q = pdd.fetch_price("QQQ")
    cal = q.index
    firsts = [cal[i] for i in range(len(cal)) if (i == 0 or cal[i].month != cal[i - 1].month) and cal[i] >= pd.Timestamp("2000-01-01")]
    snaps = snapshot_membership()
    wiki = json.loads((pdd.CACHE / "wiki_membership.json").read_text())
    switch = pd.Timestamp("2007-02-01")
    cache: dict = {}
    rows, membership = [], {}
    for d in firsts:
        if d < switch:
            prior = [s for s in snaps if s["asof"] <= d]
            if not prior:
                rows.append({"month": d.strftime("%Y-%m"), "source": "명단 없음", "members": 0, "basis": "-", "cov": np.nan,
                             "missing": []})
                continue
            s = prior[-1]
            mem = [pdd.snapshot_ticker(t) for t in s["tickers"]]
            src = f"사본 {s['asof'].date()}"
            weights = s["weights"]
        else:
            mem = wiki[str(d.date())]
            src, weights = "위키 역산", None
        membership[str(d.date())] = mem
        comps: dict[str, list[str]] = {}
        for t in mem:
            comps.setdefault(company_of(t), []).append(t)
        priced = {c: any(has_price(t, d, cache) for t in ts) for c, ts in comps.items()}
        member_cov = sum(priced.values()) / len(priced)
        if weights:
            wmap = {pdd.snapshot_ticker(t): w for t, w in weights.items()}
            cw = {c: sum(float(wmap.get(t) or 0) for t in ts) for c, ts in comps.items()}
            top = sorted(cw, key=lambda c: -cw[c])[:10]
            cov = sum(priced[c] for c in top) / len(top)
            miss = [c for c in top if not priced[c]]
            basis = "비중 상위 10"
            wsh = sum(cw[c] for c in top if priced[c]) / max(sum(cw[c] for c in top), 1e-9)
        else:
            cov = member_cov
            miss = sorted(c for c in priced if not priced[c])
            basis, wsh = "구성 종목 전체", np.nan
        rows.append({"month": d.strftime("%Y-%m"), "source": src, "members": len(comps), "basis": basis, "cov": cov,
                     "member_cov": member_cov, "weight_share": wsh, "missing": miss})
    (pdd.CACHE / "membership.json").write_text(json.dumps(membership))
    (pdd.CACHE / "coverage.json").write_text(json.dumps(rows, ensure_ascii=False, default=float), encoding="utf-8")
    df = pd.DataFrame(rows)
    df["year"] = df["month"].str[:4].astype(int)
    print(df.groupby("year").agg(cov=("cov", "mean"), member=("member_cov", "mean"), cmin=("cov", "min"),
                                 basis=("basis", "first"), n=("month", "size")).to_string())
    print(df[["month", "source", "basis", "cov", "missing"]].to_string())


if __name__ == "__main__":
    main()
