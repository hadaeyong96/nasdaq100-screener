"""상승장 대형주 웅덩이 갈아타기 — 1단계 데이터 확인 (2026-10-23, research/puddle-megacap).

수익 계산 없음. 봉인 ⓑ: 모든 요청은 2021-12-31(가격 끝 2022-01-01 미포함)까지만 받는다.
1) 2000-01 ~ 2007-01 시점별 나스닥100 명단: 나스닥 공식 페이지(dynamic.nasdaq.com nasdaq100_activity.stm,
   nasdaq.com/indexshares/nasdaq100.stm)의 인터넷 아카이브(Wayback) 사본에서 구성 종목 표를 읽는다(무료 공개 자료).
   2007-02 이후는 기존 data.universe_history(위키백과 변경 기록 역산)를 그대로 쓴다.
2) 가격: yfinance(auto_adjust=False, end=2022-01-01). 캐시 data/cache/puddle/ (gitignore).
3) 환율: FRED DEXKOUS(fredgraph.csv, 키 없음, coed=2021-12-31).
"""

from __future__ import annotations

import html as htmllib
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache" / "puddle"
SNAP_DIR = CACHE / "snapshots"
H = {"User-Agent": "nasdaq100-screener/1.0 (backtest research)"}
END = "2022-01-01"  # yfinance end(미포함) — 봉인 ⓑ 앞
CUTOFF = pd.Timestamp("2021-12-31")
SNAP_URLS = ["dynamic.nasdaq.com/dynamic/nasdaq100_activity.stm", "www.nasdaq-amex.com/asp/nasdaq100_activity.stm",
             "www.nasdaq.com/asp/nasdaq100_activity.stm"]  # 구성 종목 표가 있는 페이지만(indexshares 페이지는 QQQ 소개뿐)
TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _get(url: str, tries: int = 4) -> bytes:
    import requests

    for i in range(tries):
        try:
            r = requests.get(url, headers=H, timeout=90)
            if r.status_code == 200:
                return r.content
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3 * (i + 1))
    raise RuntimeError(f"받기 실패: {url}")


def list_snapshots() -> list[tuple[str, str]]:
    """Wayback CDX → [(timestamp, url)] (1999-06 ~ 2007-03, 상태 200). 달마다 하나(세 주소 중 그달 가장 이른 사본),
    12월·1월은 연례 교체(12월 셋째 금요일) 앞뒤를 보려고 반달마다 하나."""
    p = CACHE / "cdx.json"
    if p.exists():
        return [tuple(x) for x in json.loads(p.read_text())]
    cand = []
    for u in SNAP_URLS:
        try:
            txt = _get(f"http://web.archive.org/cdx/search/cdx?url={u}&from=199906&to=200703&filter=statuscode:200"
                       f"&collapse=timestamp:8", tries=6).decode()
        except RuntimeError as exc:
            print(f"  [CDX 실패, 이 주소는 건너뜀] {u}: {exc}")
            continue
        cand += [(l.split()[1], l.split()[2]) for l in txt.splitlines()]
        time.sleep(5)
    out, seen = [], set()
    for ts, orig in sorted(cand):
        d = pd.Timestamp(ts[:8])
        key = (d.year, d.month, d.day > 15 if d.month in (12, 1) else 0)
        if key not in seen:
            seen.add(key)
            out.append((ts, orig))
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return out


def parse_snapshot(raw: str) -> tuple[str | None, list[tuple[str, str, float | None]]]:
    """구성 종목 표 → (자료 기준일 문구, [(회사명, 티커, 비중% 또는 None)]). 순수 함수."""
    t = re.sub(r"(?is)<script.*?</script>", " ", raw)
    asof = re.search(r"Data as of ([^<|]{4,40}?)(?:Mkt|Market)? ?Close", htmllib.unescape(re.sub(r"<[^>]+>", " ", t)))
    rows = []
    for tr in re.findall(r"(?is)<tr[^>]*>(.*?)</tr>", t):
        cells = [htmllib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", c))).strip()
                 for c in re.findall(r"(?is)<td[^>]*>(.*?)</td>", tr)]
        cells = [c for c in cells if c]
        for i in range(1, len(cells)):
            if TICKER_RE.match(cells[i]) and len(cells[i - 1]) > 3 and not TICKER_RE.match(cells[i - 1]) and \
                    re.search(r"[a-z]", cells[i - 1]):
                w = None
                if i + 1 < len(cells):
                    try:
                        w = float(cells[i + 1])
                    except ValueError:
                        w = None
                rows.append((cells[i - 1], cells[i], w))
                break
    seen, out = set(), []
    for r in rows:
        if r[1] not in seen:
            seen.add(r[1])
            out.append(r)
    return (asof.group(1).strip() if asof else None), out


def fetch_snapshots() -> pd.DataFrame:
    """스냅샷을 받아(캐시) 파싱 → DataFrame(snap_ts, n, asof, tickers, weights)."""
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ts, orig in list_snapshots():
        f = SNAP_DIR / f"{ts}_{re.sub(r'[^a-z0-9]+', '_', orig.lower())[-40:]}.html"
        if not f.exists():
            try:
                f.write_bytes(_get(f"http://web.archive.org/web/{ts}id_/{orig}"))
            except RuntimeError as exc:
                print(f"  [실패] {ts} {orig}: {exc}")
                continue
            time.sleep(4)
    for f in sorted(SNAP_DIR.glob("*.html")):  # 받아 둔 사본 전부(이전 실행에서 받은 것 포함)
        asof, comp = parse_snapshot(f.read_text(encoding="utf-8", errors="replace"))
        rows.append({"snap_ts": f.name[:14], "url": f.name[15:], "n": len(comp), "asof": asof,
                     "tickers": [c[1] for c in comp], "weights": [c[2] for c in comp], "names": [c[0] for c in comp]})
    return pd.DataFrame(rows)


def load_snapshot_table() -> pd.DataFrame:
    p = CACHE / "snapshots.json"
    if p.exists():
        return pd.read_json(p, dtype={"snap_ts": str})
    df = fetch_snapshots()
    df.to_json(p, force_ascii=False)
    return df


if __name__ == "__main__":
    df = load_snapshot_table()
    print(df[["snap_ts", "url", "n", "asof"]].to_string())


# ── 2007-02 이후 명단: 기존 위키백과 변경 기록 역산(data.universe_history) ───────────────────────────


def wiki_membership(first_days: list[pd.Timestamp]) -> dict[pd.Timestamp, set[str]]:
    """달 첫 거래일 → 구성 종목(yfinance 티커). 변경 기록은 오늘 받은 위키텍스트(캐시 data/cache/puddle/wiki_changes.txt),
    현재 구성 종목은 data.universe.get_universe()."""
    from data import universe_history as uh
    from data.universe import get_universe

    p = CACHE / "wiki_changes.txt"
    if not p.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        p.write_text(uh.fetch_changes_wikitext(), encoding="utf-8")
    changes = uh.parse_changes(p.read_text(encoding="utf-8"))
    cur_p = CACHE / "current_universe.json"
    if not cur_p.exists():
        cur_p.write_text(json.dumps(sorted(get_universe()["ticker"])))
    current = set(json.loads(cur_p.read_text()))
    return {d: uh.reconstruct_membership(current, changes, d.date()) for d in first_days}


# ── 가격(yfinance, 봉인 ⓑ 앞까지만) ────────────────────────────────────────────────────


PRICE_DIR = CACHE / "prices"


def fetch_price(ticker: str) -> pd.DataFrame | None:
    """yfinance 일봉(auto_adjust=False) 1998-06-01 ~ 2021-12-31 → DataFrame(close, adj, volume) 또는 None(없음). 캐시."""
    import yfinance as yf

    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    f = PRICE_DIR / f"{ticker}.parquet"
    miss = PRICE_DIR / f"{ticker}.missing"
    if f.exists():
        return pd.read_parquet(f)
    if miss.exists():
        return None
    try:
        raw = yf.Ticker(ticker).history(start="1998-06-01", end=END, auto_adjust=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  [가격 실패] {ticker}: {type(exc).__name__}")
        raw = None
    if raw is None or raw.empty or "Close" not in raw:
        miss.write_text("none")
        return None
    df = pd.DataFrame({"close": raw["Close"], "adj": raw["Adj Close"], "volume": raw["Volume"]})
    df.index = pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None).normalize() for x in df.index])
    df = df[df.index <= CUTOFF]
    df.to_parquet(f)
    return df


def fetch_fx() -> pd.Series:
    """FRED DEXKOUS(원/달러) 1998-01-01 ~ 2021-12-31 (fredgraph.csv, 키 없음). 캐시."""
    import io

    f = CACHE / "fred_DEXKOUS.csv"
    if not f.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        f.write_bytes(_get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS&cosd=1998-01-01&coed=2021-12-31"))
    s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce")
    return s[s.index <= CUTOFF]


# ── 옛 티커 → 지금 yfinance 티커 (같은 회사·같은 주식 이력이 이어진 경우만, 공개 정보로 확인) ──────────────
ALIAS = {
    "CMCSK": "CMCSA",  # 컴캐스트 특별 K주 → A주(같은 회사, 2015 통합) — 회사 하나로 본다
    "JDSU": "VIAV",  # JDS Uniphase → Viavi(2015 이름 변경, 같은 이력)
    "SYMC": "GEN", "NLOK": "GEN",  # Symantec → NortonLifeLock → Gen Digital
    "IDPH": "BIIB", "BIIB": "BIIB",  # IDEC → Biogen Idec(2003, IDEC가 존속 회사) → Biogen
    "ERICY": "ERIC",  # 에릭슨 ADR
    "PMTC": "PTC",  # Parametric Technology → PTC
    "MLHR": "MLKN",  # Herman Miller → MillerKnoll
    "RIMM": "BB",  # Research In Motion → BlackBerry
    "HANS": "MNST",  # Hansen Natural → Monster Beverage(2012 이름 변경, 같은 이력)
    "FI": "FISV",  # Fiserv(2023 티커 FI, yfinance는 FISV로 이력)
    "MYL": "VTRS",  # Mylan → Viatris(Mylan이 존속 회사)
    "KFT": "MDLZ", "KRFT": None,  # Kraft Foods Inc → Mondelez(존속). Kraft Foods Group(KRFT)은 따로 없음
    # 위키 역산 명단 티커가 지금 yfinance에서는 다른 회사(회사 이름 대조, data/cache/puddle/name_check_wiki.json)
    "GENZ": None, "GOLD": None, "VIP": None, "JAVA": None,  # Genzyme·Randgold·VimpelCom·Sun → ETF·다른 회사
    "TEVA": "TEVA", "ERTS": "EA", "GMSTE": "GMST", "ADLAE": "ADLAC", "MOLXE": "MOLX",
}


# 2000~2007 사본 명단의 티커가 지금 yfinance에서는 다른 회사인 경우(회사 이름 대조, data/cache/puddle/name_check.json).
# 사본 명단에서만 쓴다(위키 역산 2007-02 이후의 MNST는 Monster Beverage가 맞다).
SNAPSHOT_ALIAS = {
    "CECO": "PRDO",  # Career Education → Perdoceo(같은 회사). yfinance CECO는 CECO Environmental
    "MNST": "X:MNST",  # Monster Worldwide(뒤 MWW, 2016 인수) — yfinance MNST는 Monster Beverage
    "ATHM": "X:ATHM", "BMCS": "X:BMCS", "CNET": "X:CNET", "CNXT": "X:CNXT", "COMS": "X:COMS", "CPWR": "X:CPWR",
    "GENZ": "X:GENZ", "INKT": "X:INKT", "CNTO": "X:CNTO", "GBLX": "X:GBLX", "SPOT": "X:SPOT", "SSCC": "X:SSCC", "USAI": "X:USAI", "VRTS": "X:VRTS",
}


def snapshot_ticker(t: str) -> str:
    """사본 명단 티커 → 명단에 넣을 티커(다른 회사로 재사용된 티커는 "X:" 표시 = 가격 없음)."""
    return SNAPSHOT_ALIAS.get(t, t)


def yf_ticker(t: str) -> str | None:
    """명단 티커 → 가격 받을 yfinance 티커(없으면 None)."""
    if t.startswith("X:"):
        return None
    if t in ALIAS:
        return ALIAS[t]
    return t.replace(".", "-")


def price_for(t: str) -> pd.DataFrame | None:
    y = yf_ticker(t)
    return fetch_price(y) if y else None
