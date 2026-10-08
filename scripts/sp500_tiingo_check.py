"""S&P 500(나스닥 100 제외) 2012~2021 종목의 가격 빈칸을 Tiingo 무료로 메울 수 있는지 점검 (조사만).

docs/research/sp500_universe_check.md "Tiingo 보완" 장의 재료. 수익률·신호 계산 없음.

단계 (--step):
- yf: 670종목 전체를 yfinance로 받아(E1 load_prices, 캐시 data/cache/sp500/prices/) 실패 목록을 만든다
- meta: Tiingo 무료 지원 티커 목록(supported_tickers.zip, API 호출 아님)을 받아 둔다
- tiingo: yfinance 실패 종목을 Tiingo 일별 가격으로 받는다. 무료 한도에 걸리면 멈추고 진행률을 남긴다
  (이미 받은 종목은 캐시를 써서 다시 부르지 않는다 — 다음 실행에서 이어 감)
- report: 위 결과로 표를 만든다 (네트워크 없음)

Tiingo 연결은 feature/moat scripts/fund_fetch_missing_prices.py(H4 F2)의 함수를 그대로 옮겨 왔다.
API 키 값은 출력하지 않는다 (있음/없음만).
"""

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import sp500_universe_check as su  # noqa: E402

CACHE = su.CACHE
YF_RESULT = CACHE / "yf_all.csv"
TIINGO_DIR = CACHE / "tiingo"
TIINGO_LOG = CACHE / "tiingo_log.json"
TIINGO_PROGRESS = CACHE / "tiingo_progress.json"
SUPPORTED_ZIP_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
SUPPORTED_PATH = CACHE / "tiingo_supported_tickers.csv"
FETCH_START, FETCH_END = date(2011, 6, 1), date(2021, 12, 31)  # E1 가격 구간과 같게, 봉인 2022 이전


# ── feature/moat scripts/fund_fetch_missing_prices.py에서 옮김 (동작 변경 없음) ────────────


class TiingoFetchError(RuntimeError):
    """Tiingo API 호출 실패. 메시지에 api_key 값은 절대 넣지 않는다(CLAUDE.md 보안)."""


class TiingoLimitError(TiingoFetchError):
    """무료 한도(시간당·일당 요청 수) 초과 — 이 조사에서 추가. 만나면 멈춘다."""


def fetch_tiingo_daily(ticker: str, start: date, end: date, api_key: str, timeout: float = 30.0) -> list[dict]:
    """Tiingo /tiingo/daily/{ticker}/prices 원본 레코드 목록. 예외: TiingoFetchError(한도면 TiingoLimitError)."""
    import requests

    try:
        resp = requests.get(
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
            params={"startDate": start.isoformat(), "endDate": end.isoformat(), "token": api_key, "format": "json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise TiingoFetchError(f"{ticker}: Tiingo 요청 실패(네트워크): {type(exc).__name__}") from None
    if resp.status_code == 404:
        raise TiingoFetchError(f"{ticker}: Tiingo에 해당 티커 없음(404)")
    if resp.status_code == 429:
        raise TiingoLimitError(f"{ticker}: Tiingo 한도 초과(HTTP 429)")
    if resp.status_code != 200:
        raise TiingoFetchError(f"{ticker}: Tiingo 요청 실패: HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        raise TiingoFetchError(f"{ticker}: Tiingo 응답이 JSON이 아님") from None
    if isinstance(data, dict) and "allocation" in str(data.get("detail", "")).lower():
        raise TiingoLimitError(f"{ticker}: Tiingo 한도 초과 — {str(data.get('detail'))[:120]}")
    if not isinstance(data, list):
        raise TiingoFetchError(f"{ticker}: Tiingo 응답 형식이 예상과 다름: {str(data)[:120]}")
    return data


def records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Tiingo 레코드 → 날짜 인덱스 DataFrame(raw_close, volume, split_factor, div_cash)."""
    if not records:
        return pd.DataFrame(columns=["raw_close", "volume", "split_factor", "div_cash"])
    idx = [pd.Timestamp(r["date"][:10]) for r in records]
    rows = [{"raw_close": r["close"], "volume": r["volume"], "split_factor": r.get("splitFactor", 1.0),
             "div_cash": r.get("divCash", 0.0)} for r in records]
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="date")).sort_index()


def last_real_trading_day(df: pd.DataFrame) -> date | None:
    """volume > 0인 마지막 날 (상폐 직전 거래정지로 같은 가격이 반복되는 날을 빼기 위함)."""
    if df.empty:
        return None
    nonzero = df.index[df["volume"].fillna(0) > 0]
    return (nonzero[-1] if len(nonzero) else df.index[-1]).date()


# ── 이 조사 ─────────────────────────────────────────────────────────────────


def pool() -> pd.DataFrame:
    """2012-01~2021-12 S&P 500 중 나스닥 100이 아닌 종목과 그 기간의 편입 첫 달·마지막 달.

    출력: index=티커(yfinance 형식), 열 first_month, last_month
    """
    fja = su.load_fja()
    months = pd.date_range(su.START, su.END, freq="ME")[1:]
    seen: dict[str, list] = {}
    for m in months:
        n = su.n100_on(m)
        for t in su.fja_on(fja, m):
            y = su.norm(su.base(t))
            if y in n:
                continue
            seen.setdefault(y, [m, m])[1] = m
    df = pd.DataFrame.from_dict(seen, orient="index", columns=["first_month", "last_month"]).sort_index()
    df["in_sp500_now"] = df.index.isin({su.norm(t) for t in fja.iloc[-1]["tickers"]})
    return df


def removal_reasons() -> pd.DataFrame:
    """위키 옛 판 변경 표의 편출 사유. 출력: index=편출 티커, 열 date, reason, removed_name, category."""
    tables = pd.read_html(su.WIKI_PATH)
    ch = tables[1]
    ch.columns = ["date", "added", "added_name", "removed", "removed_name", "reason"]
    ch["date"] = pd.to_datetime(ch["date"], format="%B %d, %Y", errors="coerce")
    ch = ch[ch["removed"].notna()].copy()
    ch["removed"] = ch["removed"].map(su.norm)
    ch["category"] = ch["reason"].map(classify_reason)
    return ch.sort_values("date").groupby("removed").last()[["date", "reason", "removed_name", "category"]]


def classify_reason(text) -> str:
    """편출 사유 문장 → 인수·합병 | 파산 | 분사·분할 | 시가총액 등(회사는 계속 거래) | 기타."""
    r = str(text or "").lower()
    if re.search(r"bankrupt|chapter 11", r):
        return "파산"
    if re.search(r"acqui|merg|bought|buyout|purchas|taken over|taken private|went private|private equity", r):
        return "인수·합병"
    if re.search(r"market cap|market value|capitaliz|reconstitut|reclass|rebalanc|not representative|liquidity|moved|index", r):
        return "시가총액 등(계속 거래)"
    # "X spun off Y", "A replaces B"처럼 편입 쪽만 설명한 문장은 편출 사유를 알 수 없다
    return "기타(사유 불명)"


def step_yf() -> None:
    """670종목 yfinance 수집(E1 load_prices 재사용) → yf_all.csv."""
    import e1_experiments as e1

    e1.PRICE_CACHE = CACHE / "prices"
    p = pool()
    prices, missing = e1.load_prices(list(p.index))
    rows = []
    for t in p.index:
        px = prices.get(t)
        rows.append({"ticker": t, "yf": px is not None,
                     "yf_first": px.index.min().date() if px is not None else None,
                     "yf_last": px.index.max().date() if px is not None else None})
    out = p.join(pd.DataFrame(rows).set_index("ticker"))
    out.to_csv(YF_RESULT, encoding="utf-8-sig")
    print(f"[yf] 전체 {len(out)} · 가격 있음 {int(out['yf'].sum())} · 없음 {int((~out['yf']).sum())}")


def step_meta() -> None:
    """Tiingo 지원 티커 목록(정적 파일, API 키·요청 한도와 무관)을 받는다."""
    import requests

    resp = requests.get(SUPPORTED_ZIP_URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        name = z.namelist()[0]
        SUPPORTED_PATH.write_bytes(z.read(name))
    s = pd.read_csv(SUPPORTED_PATH)
    print(f"[meta] 지원 티커 {len(s)}행, 열 {list(s.columns)}")


def next_hour_wait_seconds(now: float, margin_sec: int = 120) -> int:
    """다음 정시 + margin까지 남은 초 (순수 함수). Tiingo 무료 시간당 한도가 정시에 풀린다고 보고 기다린다."""
    return int(3600 - (now % 3600) + margin_sec)


def _write_progress(order: list[str], log: dict, state: str, **extra) -> None:
    """진행 상황 파일(tiingo_progress.json). PC가 꺼져도 tiingo_log.json으로 이어 가고, 이 파일은 사람이 보는 용도."""
    TIINGO_PROGRESS.write_text(json.dumps({
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"), "state": state,
        "done": len([t for t in order if t in log]), "total": len(order), **extra,
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def step_tiingo(order: list[str], wait_on_limit: bool = False, max_waits: int = 30) -> None:
    """order 순서로 Tiingo 가격을 받는다.

    한도에 걸리면 wait_on_limit=False면 멈추고, True면 다음 정시까지 기다렸다 같은 종목부터 이어 간다
    (최대 max_waits번). 종목마다 결과를 tiingo_log.json에 바로 써서, 중간에 꺼져도 다시 실행하면 이어 간다.
    """
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    api_key = (os.environ.get("TIINGO_API_KEY") or "").strip()
    print(f"[tiingo] TIINGO_API_KEY 있음: {bool(api_key)}", flush=True)
    if not api_key:
        raise SystemExit("TIINGO_API_KEY 없음")
    TIINGO_DIR.mkdir(parents=True, exist_ok=True)
    log = json.loads(TIINGO_LOG.read_text(encoding="utf-8")) if TIINGO_LOG.exists() else {}
    todo = [t for t in order if t not in log]
    print(f"[tiingo] 대상 {len(order)} · 이미 처리 {len(order) - len(todo)} · 이번에 {len(todo)}", flush=True)
    done, waits, i = 0, 0, 0
    while i < len(todo):
        t = todo[i]
        try:
            recs = fetch_tiingo_daily(t, FETCH_START, FETCH_END, api_key)
        except TiingoLimitError as exc:
            if not wait_on_limit or waits >= max_waits:
                print(f"[tiingo] 한도에 걸려 멈춤: {exc}", flush=True)
                _write_progress(order, log, "한도로 멈춤", next_ticker=t)
                break
            sec = next_hour_wait_seconds(time.time())
            waits += 1
            resume = time.strftime("%H:%M:%S", time.localtime(time.time() + sec))
            print(f"[tiingo] 한도 — {sec}초 대기 후 {t}부터 재개 (예정 {resume}, 대기 {waits}회째, 누적 {len([x for x in order if x in log])}/{len(order)})", flush=True)
            _write_progress(order, log, "한도 대기", next_ticker=t, resume_at=resume, waits=waits)
            time.sleep(sec)
            continue
        except TiingoFetchError as exc:
            log[t] = {"status": "실패", "error": str(exc)}
        else:
            df = records_to_frame(recs)
            df.to_csv(TIINGO_DIR / f"{t}.csv")
            log[t] = {"status": "있음" if len(df) else "빈 응답", "rows": len(df)}
        done += 1
        i += 1
        TIINGO_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        _write_progress(order, log, "진행 중", last_ticker=t)
        time.sleep(0.5)
    else:
        _write_progress(order, log, "완료")
    print(f"[tiingo] 이번 실행 처리 {done} · 누적 {len([t for t in order if t in log])}/{len(order)}", flush=True)


def tiingo_order() -> list[str]:
    """요청 순서: 표본 30 중 yfinance 실패 종목 먼저, 그다음 나머지 실패 종목(가나다순)."""
    yf = pd.read_csv(YF_RESULT, index_col=0)
    miss = sorted(yf.index[~yf["yf"]])
    sample = pd.read_csv(CACHE / "sample30.csv")["티커"].tolist()
    first = [t for t in sample if t in miss]
    return first + [t for t in miss if t not in first]


def judge_period(first: pd.Timestamp, last: pd.Timestamp, p_first, p_last, removed_on, tol_days: int = 10) -> str:
    """가격 기간이 S&P 편입 기간과 맞는지 (다른 회사 가격 혼입 판단, 순수 함수).

    입력: first/last(편입 첫 달·마지막 달 말), p_first/p_last(가격 첫날·마지막 실제 거래일), removed_on(위키 편출일 또는 None)
    출력: "맞음" | "앞이 빔" | "끝이 이름" | "다른 회사 의심" | "가격 없음"
    """
    if p_first is None or pd.isna(p_first):
        return "가격 없음"
    p_first, p_last = pd.Timestamp(p_first), pd.Timestamp(p_last)
    month_start = first - pd.offsets.MonthBegin(1)
    if p_first > last or p_last < month_start:
        return "다른 회사 의심"  # 편입 기간과 전혀 안 겹침
    need_end = min(last, pd.Timestamp(su.END))
    if removed_on is not None and not pd.isna(removed_on):
        need_end = min(need_end, pd.Timestamp(removed_on))
    if p_first > max(month_start, pd.Timestamp(FETCH_START)) + pd.Timedelta(days=tol_days):
        return "앞이 빔"
    if p_last < need_end - pd.Timedelta(days=tol_days + 31):
        return "끝이 이름"
    return "맞음"


def step_report() -> None:
    """yfinance 실패 192종목 표 (네트워크 없음) → tiingo_report.csv와 요약 출력."""
    yf = pd.read_csv(YF_RESULT, index_col=0, parse_dates=["first_month", "last_month"])
    miss = yf[~yf["yf"]].copy()
    log = json.loads(TIINGO_LOG.read_text(encoding="utf-8")) if TIINGO_LOG.exists() else {}
    sup = pd.read_csv(SUPPORTED_PATH, parse_dates=["startDate", "endDate"])
    sup["t"] = sup["ticker"].str.upper().str.replace(".", "-", regex=False)
    reasons = removal_reasons()
    rows = []
    for t, r in miss.iterrows():
        rr = reasons.loc[t] if t in reasons.index else None
        removed_on = rr["date"] if rr is not None else None
        cand = sup[(sup["t"] == t) & (sup["assetType"] == "Stock")]
        win_s, win_e = r["first_month"] - pd.offsets.MonthBegin(1), r["last_month"]
        overlap = cand[(cand["startDate"] <= win_e) & (cand["endDate"] >= win_s)]
        row = {
            "ticker": t, "편입 첫 달": r["first_month"].date(), "편입 끝 달": r["last_month"].date(),
            "지금 S&P": bool(r["in_sp500_now"]),
            "위키 편출일": removed_on.date() if removed_on is not None and not pd.isna(removed_on) else "",
            "사유 분류": rr["category"] if rr is not None else "편출 기록 없음(이름 변경 추정)",
            "위키 편출 이름": rr["removed_name"] if rr is not None else "",
            "Tiingo 목록 행 수": len(cand), "목록상 기간 겹침": len(overlap) > 0,
            "Tiingo 요청": log.get(t, {}).get("status", "미요청"),
        }
        f = TIINGO_DIR / f"{t}.csv"
        if row["Tiingo 요청"] == "있음" and f.exists():
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            row["Tiingo 첫날"] = df.index.min().date()
            row["Tiingo 끝날"] = last_real_trading_day(df)
            row["분할 기록"] = int((df["split_factor"] != 1).sum())
            row["배당 기록"] = int((df["div_cash"] > 0).sum())
            row["기간 판정"] = judge_period(r["first_month"], r["last_month"], row["Tiingo 첫날"], row["Tiingo 끝날"], removed_on)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(CACHE / "tiingo_report.csv", index=False, encoding="utf-8-sig")
    req = out[out["Tiingo 요청"] != "미요청"]
    print(f"[report] yfinance 실패 {len(out)} · Tiingo 요청 {len(req)} · 미요청 {len(out) - len(req)}")
    print(req["Tiingo 요청"].value_counts().to_string())
    if "기간 판정" in out:
        print(out["기간 판정"].value_counts().to_string())
    print("사유 분류 (192 전체):")
    print(out["사유 분류"].value_counts().to_string())
    print("Tiingo 목록상 기간 겹침 (192 전체):", int(out["목록상 기간 겹침"].sum()))
    print("미요청 중 목록상 기간 겹침:", int(out[out["Tiingo 요청"] == "미요청"]["목록상 기간 겹침"].sum()), "/", int((out["Tiingo 요청"] == "미요청").sum()))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["yf", "meta", "tiingo", "report"], required=True)
    ap.add_argument("--wait-on-limit", action="store_true", help="Tiingo 한도에 걸리면 다음 정시까지 기다렸다 이어 간다")
    ap.add_argument("--tickers", help="tiingo 단계에서 기본 순서 대신 요청할 티커 목록 파일(한 줄에 하나)")
    a = ap.parse_args()
    if a.step == "yf":
        step_yf()
    elif a.step == "meta":
        step_meta()
    elif a.step == "tiingo":
        order = Path(a.tickers).read_text(encoding="utf-8").split() if a.tickers else tiingo_order()
        step_tiingo(order, wait_on_limit=a.wait_on_limit)
    else:
        step_report()


if __name__ == "__main__":
    main()
