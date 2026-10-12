"""S&P 400·600 중소형: 가격·실적 발표 시각·생존 편향·신호 건수 조사 — 일회성 점검 (수익 계산 금지).

입력: scripts/smallmid_universe_check.py가 만든 data/cache/smallmid/membership_intervals.csv (CIK별 편입·편출 범위).
범위: 가격 2011-06-01 ~ 2021-12-31. 그 뒤 행이 오면 잘라내지 않고 멈춘다(봉인 확인).

단계:
  --step tickers   CIK별 그 시기 티커: 그 시기 10-K·10-Q의 XBRL 파일 이름(abc-20151231.xml) → 없으면 현재 티커
  --step yf        yfinance(auto_adjust=False) 시가·종가·거래량·분할 → data/cache/smallmid/prices/<cik>.csv
  --step tiingo    yfinance로 못 받은 CIK만 Tiingo. 검증 ① 종목 먼저(2017~2021 발표 많은 순) → 탐색 ② → 나머지.
                   --wait-on-limit이면 한도(429)에서 다음 정시까지 기다렸다 이어 받음. 403·월 한도면 멈추고 tiingo_stop.json.
                   받은 결과(빈 응답·404 포함)는 tiingo_progress.json에 남겨 PC가 꺼져도 다시 실행하면 이어 받는다.
  --step check     같은 회사 확인: 가격 기간 ↔ 편입 기간, 시가총액(EDGAR 주식 수 × 원가격), 편출일 가격 연속성
  --step accepted  실적 발표(8-K 2.02, 없으면 10-Q·10-K)마다 EDGAR 색인 페이지 Accepted(동부시간)
  --step survival  생존 편향 표: 가격 공백 수·비율, 사라진 이유(인수·합병 / 지수 편출만 / 파산 8-K 1.03 / 불명)
  --step signals   허용 조회: 실적급등 신호 건수(반응일 +5%, QQQ 대비 +5%p, 거래량 20일 평균 2배)와 유동성 필터 통과 비율

신호 판정은 반응일까지의 가격만 쓴다(SW1 signal_on). 반응일 뒤 가격·수익은 읽지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import smallmid_universe_check as su  # noqa: E402
import sw1_data_check as sw  # noqa: E402

CACHE = ROOT / "data" / "cache" / "smallmid"
PX = CACHE / "prices"
TI = CACHE / "tiingo"
TICKERS_CSV = CACHE / "tickers.csv"
PRICE_STATUS_CSV = CACHE / "price_status.csv"
TI_PROGRESS = CACHE / "tiingo_progress.json"
TI_STOP = CACHE / "tiingo_stop.json"  # 403·월 한도로 멈춘 위치
CHECK_CSV = CACHE / "price_check.csv"
ACCEPTED_JSON = CACHE / "accepted.json"
EVENTS_CSV = CACHE / "earnings_events.csv"
SURVIVAL_CSV = CACHE / "survival.csv"
SIGNALS_CSV = CACHE / "signals.csv"
QQQ = ROOT / "data" / "cache" / "backtest" / "QQQ.parquet"

PX_START, SEAL = "2011-06-01", pd.Timestamp("2021-12-31")
MCAP_MIN, MCAP_MAX = 1e8, 5e10  # 중소형으로 그럴듯한 편입 기간 시가총액 중앙값 (USD)
LIQ_PRICE, LIQ_DOLLAR_VOL = 5.0, 5e6  # 유동성 필터: 주가 5달러 이상, 20일 평균 거래대금 500만 달러 이상
YF_GAP_SEC = 1.5  # yfinance 요청 시작 간격 (Yahoo 요청 한도)
XBRL_RE = re.compile(r"^([a-z][a-z0-9.\-]{0,9}?)-(\d{8})(?:_htm)?\.(?:xml|htm)$", re.I)


class SealError(RuntimeError):
    """봉인일(2021-12-31) 뒤 가격 행이 들어옴 — 잘라내지 않고 멈춘다."""


def check_seal(df: pd.DataFrame, what: str) -> None:
    """가격 표에 봉인일 뒤 행이 있으면 멈춘다 (순수)."""
    if len(df) and pd.Timestamp(df.index.max()) > SEAL:
        raise SealError(f"{what}: 봉인일 뒤 행 {int((df.index > SEAL).sum())}개 — 잘라내지 않고 멈춤")


# ── 편입 기간 ────────────────────────────────────────────────────────────────

def membership() -> pd.DataFrame:
    """CIK별 편입 기간 요약: index(들), first(첫 처음 보인 날), last(마지막 보인 날), names (파일)."""
    iv = pd.read_csv(su.INTERVALS_CSV)
    iv = iv[iv["cik"].notna()].copy()
    iv["cik"] = iv["cik"].astype(int)
    g = iv.groupby("cik").agg(index=("index", lambda s: ";".join(sorted(set(s)))), first=("entry_latest", "min"),
                              last=("exit_earliest", "max"), names=("names", "first"))
    return g.reset_index()


def intervals_by_cik() -> dict[int, list[tuple[pd.Timestamp, pd.Timestamp, str, int]]]:
    """CIK → [(처음 보인 날, 마지막 보인 날, 지수, 보고 수)] (파일)."""
    iv = pd.read_csv(su.INTERVALS_CSV)
    iv = iv[iv["cik"].notna()]
    out: dict[int, list] = {}
    for r in iv.itertuples():
        out.setdefault(int(r.cik), []).append((pd.Timestamp(r.entry_latest), pd.Timestamp(r.exit_earliest), r.index,
                                                int(r.n_reports)))
    return out


# ── 티커 ────────────────────────────────────────────────────────────────────

def sub_jsons(cik: int) -> list[dict]:
    """submissions 본문 + 추가 파일 (캐시 우선, 없으면 받음)."""
    _, ua = su._edgar()
    return sw.fetch_submissions(cik, ua)


def filings_table(js: list[dict]) -> pd.DataFrame:
    """submissions JSON들 → form, filing_date, accn, primary_doc, items (순수)."""
    rows = []
    for j in js:
        f = j["filings"]["recent"] if "filings" in j else j
        for k in range(len(f["form"])):
            rows.append({"form": f["form"][k], "filing_date": f["filingDate"][k], "accn": f["accessionNumber"][k],
                         "primary_doc": f.get("primaryDocument", [""] * len(f["form"]))[k],
                         "items": f.get("items", [""] * len(f["form"]))[k]})
    return pd.DataFrame(rows).drop_duplicates("accn") if rows else pd.DataFrame(
        columns=["form", "filing_date", "accn", "primary_doc", "items"])


def xbrl_prefix(filename: str) -> str | None:
    """XBRL 파일 이름 'abc-20151231.xml' → 'ABC'. 'd123456d10k.htm' 같은 이름은 None (순수)."""
    m = XBRL_RE.match(filename or "")
    if not m:
        return None
    t = m.group(1).upper()
    return t if re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,6}", t) else None


def step_tickers() -> None:
    """CIK별 그 시기 티커 → tickers.csv. 편입 기간 안의 10-K·10-Q 중 기간 중간에 가장 가까운 것의 XBRL 이름."""
    mem = membership()
    _, ua = su._edgar()
    get = su._edgar()[0]
    rows = []
    for i, r in enumerate(mem.itertuples()):
        js = sub_jsons(r.cik)
        cur = [t for t in (js[0].get("tickers") or []) if t]
        f = filings_table(js)
        q = f[f["form"].isin(["10-K", "10-Q"]) & (f["filing_date"] >= r.first) & (f["filing_date"] <= r.last)]
        if q.empty:  # 편입 기간 안에 없으면 앞뒤 1년
            lo = (pd.Timestamp(r.first) - pd.Timedelta(days=365)).date().isoformat()
            hi = (pd.Timestamp(r.last) + pd.Timedelta(days=365)).date().isoformat()
            q = f[f["form"].isin(["10-K", "10-Q"]) & (f["filing_date"] >= lo) & (f["filing_date"] <= hi)]
        hist, src = None, ""
        for doc in q["primary_doc"]:  # 2019 이후 인라인 XBRL은 본문 이름이 abc-20191231.htm
            hist = xbrl_prefix(doc)
            if hist:
                src = "XBRL 본문 이름"
                break
        if hist is None and len(q):
            mid = pd.Timestamp(r.first) + (pd.Timestamp(r.last) - pd.Timestamp(r.first)) / 2
            qq = q.assign(dist=(pd.to_datetime(q["filing_date"]) - mid).abs()).sort_values("dist")
            for accn in qq["accn"].head(3):
                url = f"https://www.sec.gov/Archives/edgar/data/{r.cik}/{accn.replace('-', '')}/{accn}-index.htm"
                try:
                    html = su.cached_get(url, su.RAW / "index_10k" / f"{accn}.htm")
                except Exception:  # noqa: BLE001 — 실패는 아래 src로 남김
                    continue
                for href in re.findall(r'href="/Archives/[^"]+/([^"/]+)"', html):
                    hist = xbrl_prefix(href)
                    if hist:
                        src = "XBRL 첨부 이름"
                        break
                if hist:
                    break
        rows.append({"cik": r.cik, "names": r.names, "index": r.index, "first": r.first, "last": r.last,
                     "ticker_hist": hist, "ticker_now": ";".join(cur), "src": src or ("현재 티커만" if cur else "없음")})
        if i % 200 == 0:
            print(i, len(mem), flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(TICKERS_CSV, index=False, encoding="utf-8-sig")
    print(t["src"].value_counts().to_dict())


def candidate_tickers(r) -> list[str]:
    """그 시기 티커 → 현재 티커 순서, 중복 없이 (순수). '.'은 yfinance 표기 '-'로."""
    out = []
    for t in [r.get("ticker_hist")] + str(r.get("ticker_now") or "").split(";"):
        if isinstance(t, str) and t and t.lower() != "nan":
            t = t.replace(".", "-").upper()
            if re.search(r"-(P[A-Z]?|W[ST]?|WT[A-Z]?|U|R|RT)$", t):  # 우선주·워런트·유닛·권리 티커는 보통주가 아님
                continue
            if t not in out:
                out.append(t)
    return out


# ── 가격: yfinance ───────────────────────────────────────────────────────────

def overlap_days(px: pd.DataFrame, first: str, last: str, pad_days: int = 92) -> int:
    """가격 행 중 편입 기간 [first − 한 분기, last + 한 분기] 안에 있는 날 수 (순수).

    보고에 한 번만 보인 종목은 first = last라 기간이 하루뿐이어서, 분기 정밀도만큼 넓혀 본다.
    """
    if px.empty:
        return 0
    lo, hi = pd.Timestamp(first) - pd.Timedelta(days=pad_days), pd.Timestamp(last) + pd.Timedelta(days=pad_days)
    return int(((px.index >= lo) & (px.index <= hi)).sum())


def step_yf() -> None:
    """후보 티커를 차례로 yfinance에서 받아, 편입 기간과 겹치는 첫 티커를 쓴다 → prices/<cik>.csv, price_status.csv."""
    import yfinance as yf

    import threading
    from concurrent.futures import ThreadPoolExecutor

    from yfinance.exceptions import YFRateLimitError

    t = pd.read_csv(TICKERS_CSV)
    PX.mkdir(parents=True, exist_ok=True)
    tried = CACHE / "yf_tried"
    tried.mkdir(exist_ok=True)
    lock, last = threading.Lock(), [0.0]
    yf.config.debug.hide_exceptions = False  # 요청 한도·자료 없음을 예외로 받아 구분

    def one(r: dict) -> dict:
        out = PX / f"{r['cik']}.csv"
        if out.exists():
            d = pd.read_csv(out, index_col=0, parse_dates=True)
            return {"cik": r["cik"], "source": d["source"].iloc[0] if "source" in d and len(d) else "yfinance",
                    "ticker": d["ticker"].iloc[0] if len(d) else None}
        got = None
        for tk in candidate_tickers(r):
            raw_path = tried / f"{tk}.csv"
            if raw_path.exists():
                h = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            else:
                h = None
                for attempt in range(8):  # Yahoo 요청 한도: 빈 결과로 캐시하지 않고 기다렸다 다시
                    try:
                        with lock:
                            w = last[0] + YF_GAP_SEC - time.monotonic()
                            if w > 0:
                                time.sleep(w)
                            last[0] = time.monotonic()
                        h = yf.Ticker(tk).history(start=PX_START, end="2022-01-01", auto_adjust=False, actions=True)
                        break
                    except YFRateLimitError:
                        print(f"[yf] 요청 한도 — {tk} {60 * (attempt + 1)}초 기다림", flush=True)
                        time.sleep(60 * (attempt + 1))
                    except Exception:  # noqa: BLE001 — 상장폐지 등 자료 없음은 빈 결과로
                        h = pd.DataFrame()
                        break
                if h is None:
                    raise RuntimeError(f"yfinance 요청 한도가 풀리지 않음: {tk} — 나중에 다시 실행하면 이어 받음")
                if h is not None and len(h):
                    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
                    h = pd.DataFrame({"open": h["Open"], "close": h["Close"], "volume": h["Volume"],
                                      "split": h.get("Stock Splits", 0.0)})
                else:
                    h = pd.DataFrame(columns=["open", "close", "volume", "split"])
                h.to_csv(raw_path)
            check_seal(h, f"yfinance {tk}")
            if overlap_days(h, r["first"], r["last"]) >= 20:
                got = h.assign(ticker=tk)
                break
        if got is not None:
            got.to_csv(out)
            return {"cik": r["cik"], "source": "yfinance", "ticker": got["ticker"].iloc[0]}
        return {"cik": r["cik"], "source": "없음", "ticker": None}

    status = []
    with ThreadPoolExecutor(max_workers=2) as ex:  # 요청 간격은 YF_GAP_SEC로 묶음
        for i, st in enumerate(ex.map(one, t.to_dict("records"))):
            status.append(st)
            if i % 100 == 0:
                print(i, len(t), flush=True)
    s = pd.DataFrame(status)
    s.to_csv(PRICE_STATUS_CSV, index=False, encoding="utf-8-sig")
    print(s["source"].value_counts().to_dict())


# ── 가격: Tiingo ─────────────────────────────────────────────────────────────

def next_hour_wait_seconds(now_epoch: float, margin_sec: int = 120) -> int:
    """다음 정시 + 여유까지 남은 초 (S&P 500 점검과 같은 규칙) (순수)."""
    return int(3600 - (now_epoch % 3600) + margin_sec)


def tiingo_adjust(raw: pd.DataFrame) -> pd.DataFrame:
    """Tiingo raw open·close·volume·splitFactor → 분할 반영 open·close·volume (마지막 날 기준) (순수)."""
    f = raw["split_factor"].fillna(1.0).replace(0, 1.0).astype(float)
    k = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)  # 그날 뒤에 일어난 분할의 곱
    return pd.DataFrame({"open": raw["raw_open"] / k, "close": raw["raw_close"] / k, "volume": raw["volume"] * k,
                         "split": f.where(f != 1.0, 0.0)}, index=raw.index)


def big_overlap(b: tuple | None, d: pd.Timestamp) -> bool:
    """그날이 나스닥100 ∪ S&P 100 소속 기간(앞뒤 한 분기 = 92일) 안인지 (순수). b = (first, last) 또는 None."""
    return bool(b) and b[0] - pd.Timedelta(days=92) <= d <= b[1] + pd.Timedelta(days=92)


def event_day(r: dict) -> pd.Timestamp:
    """받기 순서용 발표 날짜: Accepted(동부시간) 날짜, 없으면 제출일 (순수). 반응일 계산은 signals 단계에서만."""
    a = r.get("accepted_et")
    return pd.Timestamp(str(a)[:10] if isinstance(a, str) and a else r["filing_date"]).normalize()


def tiingo_priority(events: pd.DataFrame, ivs: dict, big_ivs: dict) -> dict[int, tuple[int, int]]:
    """CIK → 정렬 키 (구역 순위, -발표 수) (순수).

    발표가 편입 중(보고 1번만 보인 구간 제외)이고 대형 겹침(앞뒤 한 분기)이 아닐 때만 센다.
    구역 순위: 0 = 검증 ①(2017~2021 발표 1건 이상, 키는 2017~2021 발표 수), 1 = 탐색 ②(2012~2016, 키는 그 수).
    둘 다 없는 CIK는 넣지 않는다(= 나머지).
    """
    n: dict[int, list[int]] = {}
    for r in events.to_dict("records"):
        cik = int(r["cik"])
        d = event_day(r)
        if not 2012 <= d.year <= 2021 or not member_on(ivs.get(cik, []), d)[0] or big_overlap(big_ivs.get(cik), d):
            continue
        n.setdefault(cik, [0, 0])[0 if d.year >= 2017 else 1] += 1
    return {c: (0, -v) if v else (1, -e) for c, (v, e) in n.items()}


def order_todo(need: pd.DataFrame, prog: dict, prio: dict[int, tuple[int, int]]) -> list[str]:
    """받을 티커 순서 (순수): 검증 ① → 탐색 ② → 나머지, 구역 안에서는 발표 수 많은 순, 같으면 원래 순서.

    진행 파일(prog)에 있는 티커는 뺀다. CIK 안에서는 candidate_tickers 순서, 중복 티커는 처음 나온 자리만.
    """
    recs = need.to_dict("records")
    keyed = sorted(range(len(recs)), key=lambda i: (*prio.get(int(recs[i]["cik"]), (2, 0)), i))
    todo = [tk for i in keyed for tk in candidate_tickers(recs[i]) if tk not in prog]
    return list(dict.fromkeys(todo))


def is_hard_stop(status: int, text: str) -> bool:
    """정시까지 기다려도 풀리지 않는 응답인지 (순수): 403 전부, 또는 본문에 월 한도 문구(month·monthly)."""
    return status == 403 or bool(re.search(r"(?i)month", text or ""))


def load_tiingo_priority() -> dict[int, tuple[int, int]]:
    """earnings_events.csv·편입 기간·대형 소속 기간으로 tiingo_priority (파일)."""
    big = pd.read_csv(su.SW1_CIK, parse_dates=["first", "last"]).dropna(subset=["cik"])
    big_ivs = {int(c): (a, b) for c, a, b in zip(big["cik"], big["first"], big["last"])}
    return tiingo_priority(pd.read_csv(EVENTS_CSV), intervals_by_cik(), big_ivs)


def step_tiingo(wait_on_limit: bool, get=None, sleep=time.sleep) -> None:
    """yfinance로 못 받은 CIK의 후보 티커를 Tiingo에서 받음. 결과는 tiingo/<ticker>.csv, 진행은 tiingo_progress.json.

    순서는 order_todo(검증 ① 먼저). 403이나 월 한도 문구가 오면 바로 멈추고 tiingo_stop.json에
    멈춘 시각·응답·마지막 성공 티커를 남긴다. TIINGO_API_KEY는 .env에서 읽고 값은 출력하지 않는다.
    429면 --wait-on-limit일 때 다음 정시까지 기다린다. get·sleep은 테스트용 (기본 requests.get·time.sleep).
    """
    import requests
    from dotenv import load_dotenv

    get = get or requests.get
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    print(f"[tiingo] TIINGO_API_KEY 있음: {bool(key)}", flush=True)
    t = pd.read_csv(TICKERS_CSV)
    st = pd.read_csv(PRICE_STATUS_CSV)
    need = t[t["cik"].isin(st.loc[st["source"] != "yfinance", "cik"])]
    prog = json.loads(TI_PROGRESS.read_text(encoding="utf-8")) if TI_PROGRESS.exists() else {}
    TI.mkdir(parents=True, exist_ok=True)
    attach_tiingo(need)  # 지난 실행에서 받아 두고 못 붙인 것
    prio = load_tiingo_priority()
    todo = order_todo(need, prog, prio)
    zone = {0: 0, 1: 0, 2: 0}
    for r in need.to_dict("records"):
        if any(tk in todo for tk in candidate_tickers(r)):
            zone[prio.get(int(r["cik"]), (2, 0))[0]] += 1
    print(f"[tiingo] 대상 CIK {len(need)} · 받을 티커 {len(todo)} · 이미 받음 {len(prog)} · "
          f"남은 CIK 검증① {zone[0]} / 탐색② {zone[1]} / 나머지 {zone[2]}", flush=True)
    last_ok = None
    i = 0
    while i < len(todo):
        tk = todo[i]
        try:
            resp = get(f"https://api.tiingo.com/tiingo/daily/{tk}/prices",
                       params={"startDate": PX_START, "endDate": "2021-12-31", "token": key, "format": "json"},
                       timeout=30)
        except requests.RequestException as exc:
            print(f"[tiingo] {tk} 네트워크 실패 {type(exc).__name__} — 60초 뒤 다시", flush=True)
            sleep(60)
            continue
        if is_hard_stop(resp.status_code, resp.text):
            # 월 한도·403: 정시까지 기다려도 풀리지 않는다 → 멈추고 위치 기록 (응답 본문은 키를 지운 안내문만)
            msg = (resp.text or "")[:160]
            msg = msg.replace(key, "***") if key else msg
            TI_STOP.write_text(json.dumps({
                "stopped_at": time.strftime("%Y-%m-%d %H:%M:%S"), "status": resp.status_code, "message": msg,
                "last_success": last_ok, "next_ticker": tk, "index_in_run": i, "remaining": len(todo) - i,
                "progress_count": len(prog)}, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[tiingo] 멈춤 ({resp.status_code}): {msg!r} — 마지막 성공 {last_ok} · 다음 {tk} · "
                  f"남음 {len(todo) - i}개 (tiingo_stop.json)", flush=True)
            break
        if resp.status_code == 429:
            if not wait_on_limit:
                print(f"[tiingo] 한도(429) — {tk}부터 남음 {len(todo) - i}개, 다시 실행하면 이어 받음", flush=True)
                break
            w = next_hour_wait_seconds(time.time())
            print(f"[tiingo] 한도(429) — {w}초 기다림 (남음 {len(todo) - i})", flush=True)
            sleep(w)
            continue
        recs = resp.json() if resp.status_code == 200 else None
        if isinstance(recs, list) and recs:
            raw = pd.DataFrame([{"date": x["date"][:10], "raw_open": x["open"], "raw_close": x["close"],
                                 "volume": x["volume"], "split_factor": x.get("splitFactor", 1.0)} for x in recs])
            raw = raw.set_index(pd.to_datetime(raw["date"])).drop(columns="date").sort_index()
            check_seal(raw, f"Tiingo {tk}")
            raw.to_csv(TI / f"{tk}.csv")
            prog[tk] = f"있음 {len(raw)}"
        else:
            prog[tk] = "빈 응답" if resp.status_code == 200 else f"HTTP {resp.status_code}"
        TI_PROGRESS.write_text(json.dumps(prog, ensure_ascii=False), encoding="utf-8")
        last_ok = tk
        i += 1
        sleep(0.5)
        if i % 25 == 0:
            print(f"[tiingo] {i}/{len(todo)}", flush=True)
    attach_tiingo(need)
    print(f"[tiingo] 진행 파일 {len(prog)}개 · 남음 {len([x for x in todo if x not in prog])}", flush=True)


def attach_tiingo(need: pd.DataFrame) -> int:
    """이미 받은 Tiingo 결과를 CIK에 붙임: 편입 기간(앞뒤 한 분기)과 20일 이상 겹치는 첫 후보 → prices/<cik>.csv (파일).

    실행이 중간에 끊겨도(백그라운드 시간 제한·PC 종료) 다음 실행 시작 때 붙도록 시작과 끝에 부른다.
    """
    n = 0
    for r in need.to_dict("records"):
        out = PX / f"{r['cik']}.csv"
        if out.exists():
            continue
        for tk in candidate_tickers(r):
            p = TI / f"{tk}.csv"
            if not p.exists():
                continue
            adj = tiingo_adjust(pd.read_csv(p, index_col=0, parse_dates=True))
            if overlap_days(adj, r["first"], r["last"]) >= 20:
                adj.assign(ticker=tk, source="tiingo").to_csv(out)
                n += 1
                break
    print(f"[tiingo] CIK에 붙임 {n}개", flush=True)
    return n


# ── 같은 회사 확인 ────────────────────────────────────────────────────────────

def raw_close(px: pd.DataFrame) -> pd.Series:
    """분할 반영 종가 → 그날 원래 종가 (그날 뒤 분할의 곱을 다시 곱함). split 열 0은 분할 없음 (순수)."""
    f = px.get("split", pd.Series(0.0, index=px.index)).fillna(0.0).replace(0.0, 1.0).astype(float)
    k = f[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    return px["close"] * k


def max_gap_days(px: pd.DataFrame, first: str, last: str) -> int:
    """편입 기간 안 가격 행 사이 가장 긴 달력일 간격 (순수)."""
    w = px[(px.index >= pd.Timestamp(first)) & (px.index <= pd.Timestamp(last))].index
    return int(pd.Series(w).diff().dt.days.max()) if len(w) > 1 else 0


def step_check() -> None:
    """가격이 있는 CIK마다: 기간 대조(judge_period), 시가총액 중앙값, 편입 기간 안 최대 공백, 편출일 가격 → price_check.csv."""
    t = pd.read_csv(TICKERS_CSV)
    rows = []
    for i, r in enumerate(t.to_dict("records")):
        p = PX / f"{r['cik']}.csv"
        row = {"cik": r["cik"], "names": r["names"], "index": r["index"], "first": r["first"], "last": r["last"]}
        if not p.exists():
            rows.append({**row, "price": "없음"})
            continue
        px = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
        check_seal(px, f"prices/{r['cik']}")
        real = px.index[px["volume"].fillna(0) > 0]
        p_first, p_last = px.index.min(), (real[-1] if len(real) else px.index[-1])
        dc = sw.judge_period(pd.Timestamp(r["first"]), pd.Timestamp(r["last"]), p_first, p_last)
        try:
            mc = sw.shares_median_mcap(int(r["cik"]), pd.DataFrame({"close": raw_close(px)}))
        except Exception:  # noqa: BLE001 — EDGAR 실패는 시가총액 없음으로
            mc = None
        mc_ok = None if mc is None else bool(MCAP_MIN <= mc <= MCAP_MAX)
        gap = max_gap_days(px, r["first"], r["last"])
        at_exit = bool(len(px[(px.index >= pd.Timestamp(r["last"]) - pd.Timedelta(days=10))
                               & (px.index <= pd.Timestamp(r["last"]))]))
        # 시가총액은 EDGAR 표지 주식 수가 종류별·단위별로 섞여 믿기 어려워(Compuware 2백만, SunEdison 10^15 달러)
        # 탈락 기준에서 빼고 '확인 필요'로만 남긴다. 탈락은 날짜 대조(티커 재사용: ACET·DV·ESI)와 공백으로.
        same = dc in ("맞음", "끝이 이름") and gap <= 15
        rows.append({**row, "price": px["source"].iloc[0] if "source" in px else "yfinance", "ticker": px["ticker"].iloc[0],
                     "price_first": p_first.date(), "price_last": p_last.date(), "date_check": dc,
                     "mcap_median_m": round(mc / 1e6) if mc else None, "mcap_ok": mc_ok, "max_gap_days": gap,
                     "price_at_last_seen": at_exit, "same_company": same})
        if i % 200 == 0:
            print(i, len(t), flush=True)
    c = pd.DataFrame(rows)
    c.to_csv(CHECK_CSV, index=False, encoding="utf-8-sig")
    print(c["price"].value_counts().to_dict())
    if "same_company" in c:
        print("같은 회사 통과", int(c["same_company"].astype("boolean").fillna(False).sum()), "/ 가격 있음",
              int((c["price"] != "없음").sum()), "· 시가총액 확인 필요", int((c["mcap_ok"] == False).sum()))  # noqa: E712
        print(c["date_check"].value_counts().to_dict(), "시가총액", c["mcap_ok"].value_counts(dropna=False).to_dict())


# ── 실적 발표 시각 ────────────────────────────────────────────────────────────

def step_accepted() -> None:
    """CIK마다 실적 발표(SW1 earnings_dates: 분기마다 8-K 2.02, 없으면 10-Q·10-K) → 색인 페이지 Accepted.

    대상: 제출일이 편입 기간(처음 보인 날 − 120일 ~ 마지막 보인 날 + 120일)과 2011-12 ~ 2021-12 안.
    결과: earnings_events.csv (accepted_et, session), accepted.json(캐시).
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import requests

    mem = membership()
    _, ua = su._edgar()
    rows = []
    for r in mem.itertuples():
        f = sw.submissions_rows(sub_jsons(r.cik))
        ev = sw.pick_events(f, "2011-06-01", "2021-12-31")
        e = sw.earnings_dates(ev)
        lo = max(pd.Timestamp(r.first) - pd.Timedelta(days=120), pd.Timestamp("2011-12-01"))
        hi = min(pd.Timestamp(r.last) + pd.Timedelta(days=120), SEAL)
        e = e[(pd.to_datetime(e["filing_date"]) >= lo) & (pd.to_datetime(e["filing_date"]) <= hi)]
        rows.append(e.assign(cik=r.cik))
    e = pd.concat(rows, ignore_index=True)
    cache = json.loads(ACCEPTED_JSON.read_text(encoding="utf-8")) if ACCEPTED_JSON.exists() else {}
    todo = [(c, a) for c, a in zip(e["cik"], e["accn"]) if a not in cache]
    print(f"[accepted] 실적 발표 {len(e)} · 캐시 {len(e) - len(todo)} · 받을 것 {len(todo)}", flush=True)
    lock, last = threading.Lock(), [0.0]

    def one(cik: int, acc: str):
        with lock:  # 시작 간격 0.125초 이상 (SEC 초당 10회 이하)
            w = last[0] + 0.125 - time.monotonic()
            if w > 0:
                time.sleep(w)
            last[0] = time.monotonic()
        u = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}/{acc}-index.htm"
        try:
            resp = requests.get(u, headers={"User-Agent": ua}, timeout=30)
        except Exception as exc:  # noqa: BLE001 — 실패는 모아 보고
            return acc, None, type(exc).__name__
        v = sw.parse_index_accepted(resp.text) if resp.status_code == 200 else None
        return acc, v, "" if v else f"HTTP {resp.status_code}"

    failed = []
    with ThreadPoolExecutor(max_workers=12) as ex:  # SEC 응답이 느려 동시 12개, 시작 간격 0.125초는 그대로
        for i, (acc, v, why) in enumerate(ex.map(lambda x: one(*x), todo)):
            if v:
                cache[acc] = v
            else:
                failed.append((acc, why))
            if (i + 1) % 1000 == 0:
                ACCEPTED_JSON.write_text(json.dumps(cache), encoding="utf-8")
                print(f"[accepted] {i + 1}/{len(todo)} · 실패 {len(failed)}", flush=True)
    ACCEPTED_JSON.write_text(json.dumps(cache), encoding="utf-8")
    e["accepted_et"] = e["accn"].map(cache)
    e["session"] = [sw.session_of(pd.Timestamp(x)) if isinstance(x, str) else None for x in e["accepted_et"]]
    e.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    print(f"[accepted] 완료 · 시각 있음 {int(e['accepted_et'].notna().sum())}/{len(e)} · 실패 {len(failed)}")
    print(e["source"].value_counts().to_dict(), e["session"].value_counts(dropna=False).to_dict())
    for a, why in failed[:20]:
        print(f"  실패 {a}: {why}")


# ── 생존 편향 ────────────────────────────────────────────────────────────────

def exit_reason(f: pd.DataFrame, last_seen: str, still_member: bool) -> str:
    """편입 마지막 보인 날 앞뒤 공시로 사라진 이유 (순수).

    순서: 파산(8-K 1.03, 마지막 보인 날 −365 ~ +365일) → 지수 편출만(마지막 보인 날 +200일 뒤에도 10-K·10-Q 제출)
    → 인수·합병(+−200일 안 8-K 2.01·3.01, Form 25·15, 합병 위임장 DEFM14A) → 불명.
    """
    if still_member:
        return "끝까지 편입(2021-12)"
    d = pd.Timestamp(last_seen)
    fd = pd.to_datetime(f["filing_date"])
    items = f["items"].fillna("").astype(str)
    near = (fd >= d - pd.Timedelta(days=365)) & (fd <= d + pd.Timedelta(days=365))
    if (near & (f["form"] == "8-K") & items.str.contains(r"\b1\.03\b")).any():
        return "파산(8-K 1.03)"
    if (f["form"].isin(["10-K", "10-Q"]) & (fd > d + pd.Timedelta(days=200))).any():
        return "지수 편출만(회사는 계속 공시)"
    near2 = (fd >= d - pd.Timedelta(days=200)) & (fd <= d + pd.Timedelta(days=200))
    deal = (f["form"].isin(["25", "25-NSE", "15-12B", "15-12G", "15-15D", "DEFM14A", "SC TO-T", "SC 14D9"])
            | ((f["form"] == "8-K") & items.str.contains(r"\b(?:2\.01|3\.01)\b")))
    if (near2 & deal).any():
        return "인수·합병(상장폐지)"
    return "불명"


def step_survival() -> None:
    """CIK마다 가격 유무·같은 회사 판정·사라진 이유 → survival.csv와 요약 표."""
    c = pd.read_csv(CHECK_CSV)
    rows = []
    for r in c.to_dict("records"):
        f = filings_table(sub_jsons(int(r["cik"])))
        still = r["last"] >= "2021-11-30"
        rows.append({**r, "reason": exit_reason(f, r["last"], still)})
    s = pd.DataFrame(rows)
    s["usable"] = s["same_company"].fillna(False).astype(bool)
    s.to_csv(SURVIVAL_CSV, index=False, encoding="utf-8-sig")
    tab = s.groupby("reason").agg(n=("cik", "size"), usable=("usable", "sum"))
    tab["gap"] = tab["n"] - tab["usable"]
    tab["gap_pct"] = (tab["gap"] / tab["n"] * 100).round(1)
    print(tab.to_string())
    print("전체", len(s), "쓸 수 있음", int(s["usable"].sum()), "공백", int((~s["usable"]).sum()),
          f"{(~s['usable']).mean() * 100:.1f}%")


# ── 신호 건수 (허용 조회) ─────────────────────────────────────────────────────

def liquid_before(px: pd.DataFrame, d0: pd.Timestamp, window: int = 20) -> tuple[bool, float, float]:
    """반응일 전날 종가 ≥ 5달러, 반응일 전 20거래일 평균 거래대금(종가×거래량) ≥ 500만 달러 (순수, d0 당일 미사용)."""
    h = px.loc[:d0].iloc[:-1]
    if len(h) < window:
        return False, float("nan"), float("nan")
    price = float(h["close"].iloc[-1])
    dv = float((h["close"] * h["volume"]).iloc[-window:].mean())
    return bool(price >= LIQ_PRICE and dv >= LIQ_DOLLAR_VOL), price, dv


def member_on(ivs: list, d: pd.Timestamp) -> tuple[bool, str | None]:
    """그날 편입 중인지와 지수 (순수).

    편입 = [처음 보인 날, 마지막 보인 날] 안 (보고로 확인된 범위만, 앞뒤 불확실 구간은 넣지 않음).
    보고 1번만 보인 구간(n_reports = 1)은 정책(docs/validation_zones.md)대로 뺀다.
    """
    for a, b, idx, n in ivs:
        if n > 1 and a <= d <= b:
            return True, idx
    return False, None


def step_signals() -> None:
    """실적 발표마다 반응일·신호 판정(SW1 signal_on, 반응일까지만) → 연도별 건수, 유동성 필터 통과 비율.

    대상: 반응일에 그 CIK가 편입 중(보고 1번만 보인 구간 제외), 같은 회사 확인 통과, 반응일 앞뒤 한 분기 안
    나스닥100 ∪ S&P 100 소속 아님. 탐색 ②(2012~2016)·검증 ①(2017~2021)로 나눠 센다.
    """
    e = pd.read_csv(EVENTS_CSV)
    s = pd.read_csv(SURVIVAL_CSV)
    usable = set(s.loc[s["usable"], "cik"].astype(int))
    ivs = intervals_by_cik()
    qqq = pd.read_parquet(QQQ)
    check_seal(qqq.loc[:SEAL], "QQQ")
    qqq = qqq.loc[:SEAL]
    days = pd.DatetimeIndex(qqq.index)
    big = pd.read_csv(su.SW1_CIK, parse_dates=["first", "last"]).dropna(subset=["cik"])
    big_ivs = {int(c): (a, b) for c, a, b in zip(big["cik"], big["first"], big["last"])}
    cache: dict[int, pd.DataFrame] = {}
    rows = []
    for r in e.to_dict("records"):
        cik = int(r["cik"])
        row = {"cik": cik, "accn": r["accn"], "source": r["source"], "filing_date": r["filing_date"],
               "accepted_et": r.get("accepted_et"), "session": r.get("session")}
        if not isinstance(r.get("accepted_et"), str):
            rows.append({**row, "status": "발표 시각 없음"})
            continue
        d0 = sw.reaction_day(pd.Timestamp(r["accepted_et"]), days)
        if d0 is None or d0 < pd.Timestamp("2012-01-01") or d0 > SEAL:
            rows.append({**row, "status": "기간 밖"})
            continue
        row["d0"], row["year"] = d0.date(), d0.year
        inm, idx = member_on(ivs.get(cik, []), d0)
        if not inm:
            rows.append({**row, "status": "그날 편입 아님"})
            continue
        row["index"] = idx
        if big_overlap(big_ivs.get(cik), d0):
            rows.append({**row, "status": "대형 겹침(앞뒤 한 분기)"})
            continue
        if cik not in usable:
            rows.append({**row, "status": "가격 공백·다른 회사"})
            continue
        if cik not in cache:
            px = pd.read_csv(PX / f"{cik}.csv", index_col=0, parse_dates=True).sort_index()
            check_seal(px, f"prices/{cik}")
            cache[cik] = px
        px = cache[cik]
        sig = sw.signal_on(px[["close", "volume"]], qqq["close"], d0)
        if sig is None:
            rows.append({**row, "status": "가격 부족(판정 불가)"})
            continue
        liq, price, dv = liquid_before(px, d0)
        rows.append({**row, "status": "판정", **sig, "liquid": liq, "prev_close": price, "dollar_vol20": dv})
    out = pd.DataFrame(rows)
    out.to_csv(SIGNALS_CSV, index=False, encoding="utf-8-sig")
    print(out["status"].value_counts().to_dict())
    j = out[out["status"] == "판정"].copy()
    j["zone"] = j["year"].map(lambda y: "탐색 ②" if y <= 2016 else "검증 ①")
    tab = j.groupby(["zone", "year"]).agg(judged=("signal", "size"), signals=("signal", "sum"),
                                         liquid=("liquid", "sum"),
                                         signals_liquid=("signal", lambda x: int((x & j.loc[x.index, "liquid"]).sum())))
    print(tab.to_string())
    print(j.groupby("zone").agg(judged=("signal", "size"), signals=("signal", "sum"),
                                signals_liquid=("signal", lambda x: int((x & j.loc[x.index, "liquid"]).sum()))).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True,
                    choices=["tickers", "yf", "tiingo", "check", "accepted", "survival", "signals"])
    ap.add_argument("--wait-on-limit", action="store_true", help="Tiingo 한도(429)에서 다음 정시까지 기다렸다 이어 받기")
    a = ap.parse_args()
    if a.step == "tiingo":
        step_tiingo(a.wait_on_limit)
        return
    {"tickers": step_tickers, "yf": step_yf, "check": step_check, "accepted": step_accepted,
     "survival": step_survival, "signals": step_signals}[a.step]()


if __name__ == "__main__":
    main()
