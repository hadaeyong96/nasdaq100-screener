"""RS26 검증 구역 ③ 0단계 — 허용 조회만 (수익 계산 없음, docs/validation_zones.md 2장 4번).

센다: 신호일(매월 마지막 NYSE 거래일, 2012-01 ~ 2021-12)마다
  1) 명단 확보: S&P 500(fja05680, 시점별 행)·S&P 100(위키 옛 판 매달 말, SW1 캐시)·나스닥 100(시점별 역산)
  2) 구역 ③ 종목 수 = S&P 500 − (나스닥 100 ∪ S&P 100) (이름 변경 짝은 같은 회사로 보고 뺀다)
  3) 가격 공백: 구역 ③ 종목×신호월 중 126거래일 전·신호일·체결일(다음 거래일) 종가가 모두 있는 비율
  4) 월별 대상 수: 위 3) 충족 종목, 그중 신호일 종가 5달러 이상 (가격 수준만 보고 수익은 계산하지 않는다)

종목 × 날짜 가드: 가격 파일은 날짜 오름차순으로 한 줄씩 읽다가 2021-12-31을 넘는 줄을 만나면 거기서 읽기를
멈춘다(봉인 ⓐ 2022-01 ~ 를 메모리에 올리지 않음). 2011-06-01 이전 줄은 쓰지 않는다. 가격은 S&P 500 캐시
(data/cache/sp500/prices, 구역 ③ 종목과 탐색 구역 종목이 섞여 있음) 중 구역 ③ 후보 종목만 연다.
2021-12 신호의 체결일(2022-01-03)은 봉인 구간이라 읽지 않고 따로 표시한다.

재료(모두 무료, 이미 받아 둔 캐시 — 새로 받지 않는다):
- data/cache/sp500/fja05680_updated.csv (GitHub fja05680/sp500, MIT)
- data/cache/sw1/sp100_monthly.csv (위키 "S&P 100" 옛 판 매달 말, SW1 0단계)
- data/universe_history (위키 나스닥 100 변경 표 역산 캐시) + data/universe_fallback.csv
- data/cache/sp500/final_gap.csv·renames.csv·prices/ (feat/sp500-universe f49d20e: yfinance + Tiingo 무료)

출력: data/cache/rs26/zone3_monthly.csv, zone3_tickers.csv (git 제외) + 화면 요약
"""

from __future__ import annotations

import csv
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.market_calendar import trading_days_between  # noqa: E402

SP500 = ROOT / "data" / "cache" / "sp500"
SP100_CSV = ROOT / "data" / "cache" / "sw1" / "sp100_monthly.csv"
OUT = ROOT / "data" / "cache" / "rs26"
ZONE_FIRST, ZONE_LAST = pd.Timestamp("2011-06-01"), pd.Timestamp("2021-12-31")  # 가격 읽기 허용 (126일 준비 포함)
SIGNAL_FIRST, SIGNAL_LAST = "2012-01", "2021-12"
LOOKBACK = 126
MIN_PRICE = 5.0
# 수작업 공백(docs/research/sp500_universe_check.md 7-2): BBT 파일은 다른 회사(Beacon Financial) 가격
MANUAL_GAP = {"BBT"}
# 이름 변경 짝 — renames.csv(같은 CIK 확정)에 없지만 나스닥 100·S&P 100 쪽 표기가 달라 빼기에 필요한 짝
EXTRA_ALIASES = [("FB", "META"), ("GOOG", "GOOGL"), ("WAG", "WBA"), ("KFT", "MDLZ"), ("ESRX", "CI"),
                 ("BRCM", "AVGO"), ("TWX", "WBD"), ("CELG", "BMY"), ("MON", "BAYRY")]


class ZoneGuardError(RuntimeError):
    """허용 구역 밖 (종목, 날짜)를 읽으려 할 때."""


def norm(t: str) -> str:
    """BRK.B → BRK-B, 'AAL-199702' 꼬리 제거 (fja05680 옛 티커 표기)."""
    return re.sub(r"-\d{6}$", "", t.strip().upper().replace(".", "-"))


def signal_days() -> list[pd.Timestamp]:
    """2012-01 ~ 2021-12 매월 마지막 NYSE 거래일."""
    days = trading_days_between(date(2011, 6, 1), date(2022, 1, 31))
    s = pd.Series(days, index=days)
    me = s.groupby([days.year, days.month]).max()
    return [d for d in me if SIGNAL_FIRST <= d.strftime("%Y-%m") <= SIGNAL_LAST]


def load_fja() -> pd.DataFrame:
    df = pd.read_csv(SP500 / "fja05680_updated.csv", parse_dates=["date"]).set_index("date").sort_index()
    df["tickers"] = df["tickers"].map(lambda s: {norm(x) for x in s.split(",")})
    return df


def load_sp100() -> tuple[dict[str, set[str]], dict[str, int], dict[str, set[str]]]:
    """매달 말 S&P 100 (위키 옛 판). 결정 4(2026-10-15): 위키에 변경 표가 없어, 그달 말 뒤 **처음 새로 고쳐진 판**의
    명단을 더해 뺀다(옛 판이 늦게 고쳐진 사이 편입된 종목을 놓치지 않게 — 더 많이 빼는 쪽). 2021-12-31 뒤 판은 쓰지 않는다.
    출력: (그달 명단, 판 나이, 그달 명단 ∪ 다음 새 판 명단)"""
    df = pd.read_csv(SP100_CSV)
    df["me"] = pd.to_datetime(df["month_end"])
    df["rd"] = pd.to_datetime(df["rev_date"])
    lists = {str(r["month_end"])[:7]: {norm(x) for x in str(r["tickers"]).split(";")} for _, r in df.iterrows()}
    ages = {str(r["month_end"])[:7]: int(r["rev_age_days"]) for _, r in df.iterrows()}
    filled = {}
    for _, r in df.iterrows():
        key = str(r["month_end"])[:7]
        later = df[(df["rd"] > r["me"]) & (df["rd"] <= ZONE_LAST)].sort_values("rd")
        nxt = lists[str(later.iloc[0]["month_end"])[:7]] if len(later) else set()
        filled[key] = lists[key] | nxt
    return lists, ages, filled


def ndx_on(d: pd.Timestamp, cur: set[str]) -> set[str]:
    from data import universe_history as uh

    return {norm(t) for t in uh.point_in_time_universe(d.date(), cur)}


def alias_map() -> dict[str, set[str]]:
    """티커 → 같은 회사로 보는 티커 묶음(자기 자신 포함)."""
    pairs = [(norm(r["old"]), norm(r["new"])) for _, r in pd.read_csv(SP500 / "renames.csv", encoding="utf-8-sig").iterrows()]
    pairs += EXTRA_ALIASES
    groups: dict[str, set[str]] = {}
    for a, b in pairs:
        g = groups.get(a, {a}) | groups.get(b, {b})
        for t in g:
            groups[t] = g
    return groups


def price_path(ticker: str, gap: pd.DataFrame, renames: dict[str, str]) -> Path | None:
    """final_gap.csv의 가격 출처대로 파일 경로: yfinance → prices/{티커}, yfinance(새 티커) → prices/{새 티커},
    tiingo → tiingo/{티커}. 공백 종목·표에 없는 종목은 None."""
    if ticker not in gap.index or gap.at[ticker, "status"] != "있음" or ticker in MANUAL_GAP:
        return None
    src = str(gap.at[ticker, "price_source"])
    if src == "tiingo":
        return SP500 / "tiingo" / f"{ticker}.csv"
    if src.startswith("yfinance(") and ticker in renames:
        return SP500 / "prices" / f"{renames[ticker]}.csv"
    if src == "yfinance":
        return SP500 / "prices" / f"{ticker}.csv"
    raise RuntimeError(f"{ticker}: 가격 출처 {src}의 파일을 찾지 못했습니다(이름 변경 표 없음)")


def read_closes_guarded(ticker: str, path: Path, allowed: set[str]) -> tuple[pd.Series, bool]:
    """구역 가드를 지키며 종가를 읽는다. 출력: (2011-06-01 ~ 2021-12-31 종가, 2022년 이후 줄이 파일에 있었는지).

    yfinance 저장분은 close(분할 반영), Tiingo 저장분은 raw_close(그날 실제 종가, 분할 미반영)다 —
    0단계는 있는지와 그날 가격 수준(5달러)만 본다. 1단계 수익 계산에는 Tiingo split_factor 반영이 필요하다.
    """
    if ticker not in allowed:
        raise ZoneGuardError(f"{ticker}: 구역 ③ 후보가 아닌 종목의 가격을 열려고 했습니다")
    dates, closes, beyond = [], [], False
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        head = [h.strip().lower() for h in next(rd)]
        i_d = head.index("date") if "date" in head else 0  # yfinance 저장분은 날짜 열 이름이 비어 있다
        i_c = head.index("close") if "close" in head else head.index("raw_close")
        prev = None
        for row in rd:
            d = pd.Timestamp(row[i_d][:10])
            if prev is not None and d < prev:
                raise ZoneGuardError(f"{ticker}: 날짜가 오름차순이 아니라 줄 단위 가드를 쓸 수 없습니다")
            prev = d
            if d > ZONE_LAST:
                beyond = True
                break  # 봉인 구간 줄은 읽지 않는다
            if d < ZONE_FIRST or row[i_c] == "":
                continue
            dates.append(d)
            closes.append(float(row[i_c]))
    s = pd.Series(closes, index=pd.DatetimeIndex(dates), dtype=float)
    if len(s) and (s.index.max() > ZONE_LAST or s.index.min() < ZONE_FIRST):
        raise ZoneGuardError(f"{ticker}: 허용 날짜 밖 가격이 섞였습니다")
    return s[~s.index.duplicated(keep="last")], beyond


def read_prices_guarded(ticker: str, path: Path, allowed: set[str]) -> pd.DataFrame:
    """가드를 지키며(2021-12-31 넘는 줄에서 멈춤) 분할 반영 종가·그날 실제 종가를 읽는다.

    출력: DataFrame(index=날짜, close=분할 반영 종가, raw_close=그날 실제 종가, split=그날 분할 비율(없으면 1)).
    yfinance 저장분은 close가 이미 분할 반영이다(raw_close는 비움). Tiingo 저장분은 raw_close를 그날 이후
    (2021-12-31까지) 분할 비율의 곱으로 나눈다 — 2022년 이후 분할은 구간 안 수익 비율에 영향이 없어 보지 않는다.
    """
    if ticker not in allowed:
        raise ZoneGuardError(f"{ticker}: 구역 ③ 후보가 아닌 종목의 가격을 열려고 했습니다")
    rows, prev = [], None
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        head = [h.strip().lower() for h in next(rd)]
        i_d = head.index("date") if "date" in head else 0
        tiingo = "raw_close" in head
        i_c = head.index("raw_close") if tiingo else head.index("close")
        i_s = head.index("split_factor") if tiingo else None
        for row in rd:
            d = pd.Timestamp(row[i_d][:10])
            if prev is not None and d < prev:
                raise ZoneGuardError(f"{ticker}: 날짜가 오름차순이 아니라 줄 단위 가드를 쓸 수 없습니다")
            prev = d
            if d > ZONE_LAST:
                break
            if d < ZONE_FIRST or row[i_c] == "":
                continue
            sp = float(row[i_s]) if tiingo and row[i_s] not in ("", "0", "0.0") else 1.0
            rows.append((d, float(row[i_c]), sp))
    df = pd.DataFrame(rows, columns=["date", "px", "split"]).drop_duplicates("date", keep="last").set_index("date")
    if tiingo:
        later = df["split"][::-1].cumprod()[::-1].shift(-1).fillna(1.0)  # 다음 날부터 2021-12-31까지 분할 곱
        df["close"], df["raw_close"] = df["px"] / later, df["px"]
    else:
        df["close"], df["raw_close"] = df["px"], float("nan")
    return df[["close", "raw_close", "split"]]


def zone_windows(m: pd.DataFrame) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """종목별 보유 가능 창 (zone3_monthly.csv + 신호월 명단 재계산 없이 zone3_members.csv 사용)."""
    mem = pd.read_csv(OUT / "zone3_members.csv", encoding="utf-8-sig", parse_dates=["signal_date"])
    sig = sorted(m["signal_date"])
    look = dict(zip(m["signal_date"], m["lookback_start"]))
    out = {}
    for t, g in mem.groupby("ticker"):
        first, last = g["signal_date"].min(), g["signal_date"].max()
        nxt = [d for d in sig if d > last]
        hi = (nxt[0] + pd.offsets.MonthEnd(1)) if nxt else ZONE_LAST
        out[t] = (look[first], min(hi, ZONE_LAST))
    return out


SPLIT_RATIOS = [2, 3, 4, 5, 6, 7, 8, 10, 1.5, 2.5, 1.25, 4 / 3]
JUMP_LIMIT = 0.5  # 하루 ±50%
SPLIT_TOL = 0.01  # 흔한 분할 비율과 1% 안이면 분할 모양 (5%는 실제 급등락도 많이 걸렸다)


def jump_rows(ticker: str, src: str, df: pd.DataFrame) -> list[dict]:
    """하루 종가 비율이 1.5 초과·0.5 미만인 날. split_like = 흔한 분할 비율(또는 그 역수)과 5% 안."""
    r = df["close"] / df["close"].shift(1)
    out = []
    for d in r.index[(r > 1 + JUMP_LIMIT) | (r < 1 - JUMP_LIMIT)]:
        x = float(r[d])
        cands = SPLIT_RATIOS + [1 / q for q in SPLIT_RATIOS]
        near = min(cands, key=lambda q: abs(x / q - 1))
        out.append({"ticker": ticker, "source": src, "date": d.date(), "ratio": round(x, 4), "near_split_ratio": round(near, 4),
                    "split_like": abs(x / near - 1) < SPLIT_TOL, "split_recorded_that_day": float(df.at[d, "split"])})
    return out


def step_jumps() -> None:
    """사전 등록 전 점검(결정 3): 구역 ③ 가격 하루 ±50% 움직임 전부 → data/cache/rs26/jumps.csv. 수익 비교 없음."""
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    tk = pd.read_csv(OUT / "zone3_tickers.csv", encoding="utf-8-sig")
    gap = pd.read_csv(SP500 / "final_gap.csv", encoding="utf-8-sig").set_index("ticker")
    renames = {norm(r["old"]): norm(r["final_ticker"]) for _, r in pd.read_csv(SP500 / "renames.csv", encoding="utf-8-sig").iterrows()}
    allowed = set(tk["ticker"])
    m = pd.read_csv(OUT / "zone3_monthly.csv", encoding="utf-8-sig", parse_dates=["signal_date", "lookback_start"])
    # 종목별 보유 가능 창: 첫 소속 신호월의 126거래일 전 ~ 마지막 소속 신호월 다음 달 말(다음 교체일에 판다)
    win = zone_windows(m)
    rows, n_t, n_split = [], {"tiingo": 0, "yfinance": 0}, 0
    for t in tk.loc[tk["price_loaded"], "ticker"]:
        path = price_path(t, gap, renames)
        src = "tiingo" if str(gap.at[t, "price_source"]) == "tiingo" else "yfinance"
        n_t[src] += 1
        df = read_prices_guarded(t, path, allowed)
        n_split += int((df["split"] != 1).sum())
        lo, hi = win[t]
        rows += [dict(r, in_window=lo <= pd.Timestamp(r["date"]) <= hi) for r in jump_rows(t, src, df)]
    j = pd.DataFrame(rows, columns=["ticker", "source", "date", "ratio", "near_split_ratio", "split_like",
                                    "split_recorded_that_day", "in_window"])
    j.to_csv(OUT / "jumps.csv", index=False, encoding="utf-8-sig")
    print(f"[점프] 구역 ③ 가격 종목 tiingo {n_t['tiingo']} · yfinance {n_t['yfinance']}, Tiingo 분할 기록 {n_split}건 반영")
    print(f"[점프] 하루 ±50% 넘는 날 전체 {len(j)}건 · {j['ticker'].nunique()}종목, 보유 가능 창 안 {int(j['in_window'].sum())}건")
    jw = j[j["in_window"]]
    print(jw.to_string(index=False) if len(jw) else "(창 안 없음)")
    bad = sorted(jw.loc[jw["split_like"] & (jw["source"] == "tiingo"), "ticker"].unique())
    print(f"[제외 후보: Tiingo 분할 모양 점프(창 안)] {len(bad)}개 {bad}")
    yb = sorted(jw.loc[jw["split_like"] & (jw["source"] == "yfinance"), "ticker"].unique())
    print(f"[참고: yfinance 분할 모양 점프(창 안)] {len(yb)}개 {yb}")


def main() -> None:
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    sigs = signal_days()
    cal = trading_days_between(date(2011, 6, 1), date(2022, 1, 31))
    fja = load_fja()
    sp100_plain, sp100_age, sp100 = load_sp100()
    cur = {norm(t) for t in pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"]}
    aliases = alias_map()
    gap = pd.read_csv(SP500 / "final_gap.csv", encoding="utf-8-sig").set_index("ticker")

    members = {}
    month_rows = []
    for d in sigs:
        fja_rows = fja.loc[:d]
        sp = fja_rows.iloc[-1]["tickers"]
        fja_age = (d - fja_rows.index[-1]).days
        key = d.strftime("%Y-%m")
        s100 = sp100.get(key)
        sp100_added = len((s100 or set()) - (sp100_plain.get(key) or set()))
        ndx = ndx_on(d, cur)
        excl = ndx | (s100 or set())
        zone = set()
        alias_excluded = 0
        for t in sp:
            grp = aliases.get(t, {t})
            if t in excl:
                continue
            if grp & excl:
                alias_excluded += 1
                continue
            zone.add(t)
        members[d] = zone
        month_rows.append({"signal_date": d.date(), "sp500_n": len(sp), "fja_row_age_days": fja_age,
                           "sp100_available": s100 is not None, "sp100_n": len(s100 or ()),
                           "sp100_rev_age_days": sp100_age.get(key), "sp100_fill_added": sp100_added, "ndx_n": len(ndx),
                           "zone3_n": len(zone), "alias_excluded": alias_excluded})

    all_zone = sorted(set().union(*members.values()))
    allowed = set(all_zone)
    in_gap = set(gap.index)
    renames = {norm(r["old"]): norm(r["final_ticker"]) for _, r in pd.read_csv(SP500 / "renames.csv", encoding="utf-8-sig").iterrows()}
    closes, beyond_files, empty_files = {}, 0, []
    for t in all_zone:
        path = price_path(t, gap, renames)
        if path is None:
            continue
        if not path.exists():
            raise RuntimeError(f"{t}: 가격 파일이 없습니다 ({path})")
        s, beyond = read_closes_guarded(t, path, allowed)
        if s.empty:
            empty_files.append(t)
        closes[t] = s
        beyond_files += int(beyond)
    if empty_files:
        print(f"[경고] '있음'인데 2011-06 ~ 2021-12 종가가 빈 파일 {len(empty_files)}개: {empty_files}")

    for row, d in zip(month_rows, sigs):
        pos = cal.get_loc(d)
        start, fill = cal[pos - LOOKBACK], cal[pos + 1]
        fill_in_zone = fill <= ZONE_LAST
        ok, ok5 = 0, 0
        why = {"no_file": 0, "late_start": 0, "no_signal_close": 0, "no_start_close": 0, "no_fill_close": 0}
        for t in members[d]:
            s = closes.get(t)
            if s is None or s.empty:
                why["no_file"] += 1  # 공백 종목(가격 없음·다른 회사·일부만)
                continue
            if s.index.min() > start:
                why["late_start"] += 1  # 가격 시작이 126거래일 전보다 늦음 — 신규 상장·분사(규칙상 제외) 또는 앞 공백
                continue
            if d not in s.index:
                why["no_signal_close"] += 1
                continue
            if start not in s.index:
                why["no_start_close"] += 1
                continue
            if fill_in_zone and fill not in s.index:
                why["no_fill_close"] += 1
                continue
            ok += 1
            ok5 += int(s[d] >= MIN_PRICE)
        row.update(why)
        row.update({"lookback_start": start.date(), "fill_date": fill.date(), "fill_in_zone": fill_in_zone,
                    "with_prices": ok, "with_prices_ge5": ok5,
                    "price_coverage_pct": round(ok / row["zone3_n"] * 100, 2) if row["zone3_n"] else None})

    m = pd.DataFrame(month_rows)
    m.to_csv(OUT / "zone3_monthly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"signal_date": d.date(), "ticker": t} for d in sigs for t in sorted(members[d])]).to_csv(
        OUT / "zone3_members.csv", index=False, encoding="utf-8-sig")
    tick_rows = []
    for t in all_zone:
        months_in = sum(t in members[d] for d in sigs)
        st = gap.at[t, "status"] if t in in_gap else "표 없음"
        tick_rows.append({"ticker": t, "member_months": months_in, "gap_status": "공백(수작업)" if t in MANUAL_GAP else st,
                          "price_loaded": t in closes})
    tk = pd.DataFrame(tick_rows)
    tk.to_csv(OUT / "zone3_tickers.csv", index=False, encoding="utf-8-sig")

    tm = int(m["zone3_n"].sum())
    print(f"신호월 {len(m)}개 ({m['signal_date'].min()} ~ {m['signal_date'].max()})")
    print(f"[명단] S&P 500 행 있음 {len(m)}/{len(m)}, fja 행 나이 최대 {m['fja_row_age_days'].max()}일·중앙값 {m['fja_row_age_days'].median():.0f}일, "
          f"S&P 500 종목 수 {m['sp500_n'].min()}~{m['sp500_n'].max()}")
    print(f"[명단] S&P 100 있음 {int(m['sp100_available'].sum())}/{len(m)}, 옛 판 나이 중앙값 {m['sp100_rev_age_days'].median():.0f}일·최대 {m['sp100_rev_age_days'].max()}일, "
          f"종목 수 {m['sp100_n'].min()}~{m['sp100_n'].max()}")
    print(f"[명단] S&P 100 보완(다음 새 판)으로 더 뺀 종목: 월 {m['sp100_fill_added'].min()}~{m['sp100_fill_added'].max()}, "
          f"보완이 있었던 달 {int((m['sp100_fill_added'] > 0).sum())}개")
    print(f"[명단] 나스닥 100 종목 수 {m['ndx_n'].min()}~{m['ndx_n'].max()}")
    print(f"[구역 ③] 월별 종목 수 {m['zone3_n'].min()}~{m['zone3_n'].max()} (중앙값 {m['zone3_n'].median():.0f}), "
          f"이름 변경 짝으로 더 뺀 종목 월평균 {m['alias_excluded'].mean():.1f}, 한 번이라도 속한 종목 {len(all_zone)}")
    gs = tk["gap_status"].value_counts().to_dict()
    print(f"[가격·종목] {len(all_zone)}종목 중 쓸 수 있음 {int(tk['price_loaded'].sum())}, 상태 {gs}")
    print(f"[가격·종목×월] 126거래일 전·신호일·체결일 종가 모두 있음 {int(m['with_prices'].sum())}/{tm} "
          f"= {m['with_prices'].sum() / tm * 100:.2f}% (공백 {100 - m['with_prices'].sum() / tm * 100:.2f}%)")
    w = {k: int(m[k].sum()) for k in ("no_file", "late_start", "no_signal_close", "no_start_close", "no_fill_close")}
    print(f"[가격·종목×월 빠진 이유] {w}")
    print(f"[가격·종목×월, 신규 상장(가격 시작 늦음) 뺀 분모] {int(m['with_prices'].sum())}/{tm - w['late_start']} "
          f"= {m['with_prices'].sum() / (tm - w['late_start']) * 100:.2f}%")
    print(f"[가격·월] 확보율 최저 {m['price_coverage_pct'].min():.1f}% ({m.loc[m['price_coverage_pct'].idxmin(), 'signal_date']}), "
          f"90% 미만인 달 {int((m['price_coverage_pct'] < 90).sum())}개")
    print(f"[대상 수] 가격 있음 {m['with_prices'].min()}~{m['with_prices'].max()}, 그중 5달러 이상 {m['with_prices_ge5'].min()}~{m['with_prices_ge5'].max()} (월별 표 zone3_monthly.csv)")
    print(f"[가드] 2022년 이후 줄이 있던 파일 {beyond_files}개 (그 줄부터 읽지 않음). 체결일이 봉인 구간인 신호: "
          f"{[str(r['signal_date']) for r in month_rows if not r['fill_in_zone']]}")
    by_year = m.assign(y=pd.to_datetime(m["signal_date"]).dt.year).groupby("y")[["zone3_n", "with_prices", "with_prices_ge5"]].mean().round(1)
    print("[연도별 월평균]\n" + by_year.to_string())




# ── 계산 전 준비 데이터 (사전 등록 전, 수익 계산 없음) ───────────────────────────
RAW_DIR = OUT / "tiingo_raw"
QQQ_CSV = OUT / "QQQ.csv"


def step_prepare(tickers: list[str]) -> None:
    """QQQ(yfinance, 2011-06-01 ~ 2021-12-31)와, 5달러 판정용 그날 실제 종가(Tiingo raw_close, 같은 기간)를 받는다.

    yfinance 저장분 close는 2021년 뒤 분할까지 반영돼 과거 가격 수준이 작게 보인다(NFLX 2012 $1.32 등) —
    5달러 판정은 그날 실제 가격이어야 하므로, 분할 반영 종가가 5달러 미만인 소속월이 있던 종목만 Tiingo 원시 종가를 받는다.
    받은 줄이 2021-12-31을 넘으면 저장하지 않고 멈춘다. 키 값은 출력하지 않는다.
    """
    import os

    import requests
    import yfinance as yf
    from dotenv import load_dotenv

    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if not QQQ_CSV.exists():
        h = yf.Ticker("QQQ").history(start="2011-06-01", end="2022-01-01", auto_adjust=False, actions=True)
        h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        if h.index.max() > ZONE_LAST:
            raise ZoneGuardError("QQQ: 2021-12-31 뒤 줄을 받았습니다 — 저장하지 않고 멈춥니다")
        h[["Close", "Dividends", "Stock Splits"]].rename(columns={"Close": "close", "Dividends": "div", "Stock Splits": "split"}).to_csv(QQQ_CSV)
        print(f"[준비] QQQ {len(h)}줄 {h.index.min().date()} ~ {h.index.max().date()}")
    load_dotenv(ROOT / ".env")
    key = os.environ.get("TIINGO_API_KEY") or ""
    if not key:
        raise RuntimeError("TIINGO_API_KEY가 없습니다")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for t in tickers:
        f = RAW_DIR / f"{t}.csv"
        if f.exists():
            continue
        r = requests.get(f"https://api.tiingo.com/tiingo/daily/{t.lower()}/prices",
                         params={"startDate": "2011-06-01", "endDate": "2021-12-31"},
                         headers={"Authorization": f"Token {key}", "Content-Type": "application/json"}, timeout=60)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        if r.status_code != 200 or not isinstance(body, list) or not body:
            detail = body.get("detail", "") if isinstance(body, dict) else ""
            failed.append(f"{t}(HTTP {r.status_code} {str(detail)[:80]})")
            continue
        df = pd.DataFrame(body)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        if df["date"].max() > ZONE_LAST:
            raise ZoneGuardError(f"{t}: 2021-12-31 뒤 줄을 받았습니다 — 저장하지 않고 멈춥니다")
        df.rename(columns={"close": "raw_close", "splitFactor": "split_factor", "divCash": "div_cash"})[
            ["date", "raw_close", "volume", "split_factor", "div_cash"]].to_csv(f, index=False)
        print(f"[준비] {t} Tiingo 원시 종가 {len(df)}줄 {df['date'].min().date()} ~ {df['date'].max().date()}")
    if failed:
        print(f"[준비] 받지 못함: {failed}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "jumps":
        step_jumps()
    elif len(sys.argv) > 1 and sys.argv[1] == "prepare":
        step_prepare(sys.argv[2].split(","))
    else:
        main()
