"""폭락 연구 1 관문 0(데이터 감사)·관문 1(하락 사건 표·반토막 진입) — docs/prereg/shock_dip.md.

전략 수익률은 계산하지 않는다(사전 등록 K). 받기만 하고, 사건 표는 가격지수 기준 참고 분석이다.
캐시: data/cache/shock_dip/ (gitignore). 네트워크 실패는 예외로 멈춘다.
실러 파일(.xls)을 읽으려면 xlrd가 필요하다(없으면 멈춤).

실행: python scripts/shock_dip_gate01.py
출력: docs/research/shock_dip_data_audit_stats.md, drawdown_events.md/.csv, halving_entry.md
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import drawdown_events as de  # noqa: E402
from core import l1d  # noqa: E402

CACHE = ROOT / "data" / "cache" / "shock_dip"
OUT = ROOT / "docs" / "research"
END = "2026-10-17"  # 끝 제외 — 2026-10-16 확정 봉까지
JUDGE_END = pd.Timestamp("2021-12-31")

TICKERS = ["^GSPC", "^IXIC", "QQQ", "QLD", "^KS11", "^N225", "^GDAXI", "^FTSE", "^HSI", "^VIX", "^VXO"]
FRED = ["SAHMREALTIME", "BAA10Y", "BAMLH0A0HYM2", "DTB3", "VXOCLS", "VIXCLS"]
EVENT_INDEXES = {"^GSPC": "1950-01-01", "^IXIC": None, "QQQ": None, "^KS11": None}
SHILLER_URL = ("https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/"
               "1449043e-a1ed-4e2e-8e9c-7b9a2a064493/ie_data.xls")
HALVING_LEVELS = [-0.30, -0.40, -0.50]
HORIZONS = {"1년": 252, "3년": 756, "5년": 1260, "10년": 2520}
MIN_BAA_OBS = 252  # [분석 선택] BAA10Y 백분위 최소 표본(약 1년) — 참고 표 전용

# 사후 원인 설명 — 널리 알려진 사건만 적고 나머지는 비운다. 신호·잠정 유형 계산에 쓰지 않는다.
_US = {
    "1956-08": "1957~58 경기침체", "1961-12": "1962 케네디 슬라이드", "1966-02": "1966 신용경색(연준 긴축)",
    "1968-11": "인플레이션·긴축, 1969~70 경기침체", "1973-01": "1차 오일쇼크·1973~75 경기침체",
    "1978-09": "1978-10 금리 급등·달러 약세", "1980-02": "볼커 긴축·신용 통제(1980-03 은값 폭락)",
    "1980-11": "볼커 고금리·1981~82 경기침체", "1981-05": "볼커 고금리·1981~82 경기침체",
    "1987-08": "블랙 먼데이(1987-10-19)", "1989-10": "1989-10 미니 크래시·1990~91 경기침체",
    "1990-07": "이라크의 쿠웨이트 침공·1990~91 경기침체", "1998-07": "러시아 디폴트·LTCM",
    "2000-03": "닷컴 거품 붕괴·2001 경기침체", "2007-10": "서브프라임·글로벌 금융위기",
    "2015-07": "위안화 절하·중국 증시 급락·유가 급락", "2018-08": "연준 금리 인상·미중 무역 분쟁",
    "2018-09": "연준 금리 인상·미중 무역 분쟁", "2020-02": "코로나19 팬데믹",
    "2021-11": "인플레이션·연준 급격한 긴축", "2022-01": "인플레이션·연준 급격한 긴축",
    "2024-12": "2025-04 미국 관세 발표", "2025-02": "2025-04 미국 관세 발표",
}
_KR = {
    "1997-06": "아시아·IMF 외환위기", "1999-07": "대우그룹 사태", "2000-01": "닷컴 거품 붕괴",
    "2007-07": "서브프라임 초기 충격", "2007-10": "글로벌 금융위기", "2011-05": "미국 신용등급 강등·유럽 재정위기",
    "2018-01": "미중 무역 분쟁·반도체 하강 → 2020-03 코로나19", "2021-07": "인플레이션·긴축·반도체 하강",
}
CAUSES = {(tk, ym): why for tk in ["^GSPC", "^IXIC", "QQQ"] for ym, why in _US.items() if why}
CAUSES.update({("^KS11", ym): why for ym, why in _KR.items()})
CAUSES[("QQQ", "2000-01")] = "데이터 의심: 야후 QQQ 2000-01-06 종가가 ^NDX 비율 대비 −3.8% 어긋남(지수는 −11.9%)"


def _get(url: str, timeout: int = 300) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def load_yf(ticker: str) -> pd.DataFrame:
    """yfinance auto_adjust=False의 Close·Adj Close(가장 이른 날 ~ END 전). 빈 결과는 빈 DataFrame."""
    path = CACHE / f"yf_{ticker.replace('^', '')}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf

    df = yf.download(ticker, start="1900-01-01", end=END, auto_adjust=False, progress=False, threads=False)
    out = pd.DataFrame()
    if df is not None and not df.empty:
        for col in ["Close", "Adj Close"]:
            c = df[col]
            out[col] = c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return out


def load_fred(series_id: str) -> pd.Series:
    """키 없는 fredgraph.csv 전체 이력(결측 "."은 NaN 유지)."""
    path = CACHE / f"fred_{series_id}.csv"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"))
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def load_alfred_vintages(series_id: str) -> pd.DatetimeIndex:
    """ALFRED 다운로드 페이지에 나열된 발표일(빈티지) 목록."""
    path = CACHE / f"alfred_vintages_{series_id}.csv"
    if not path.exists():
        html = _get(f"https://alfred.stlouisfed.org/series/downloaddata?seid={series_id}").decode("utf-8", "replace")
        dates = sorted(set(re.findall(r'<option value="(\d{4}-\d{2}-\d{2})"', html)))
        if not dates:
            raise RuntimeError(f"ALFRED {series_id} 발표일 목록 없음")
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.Series(dates, name="vintage").to_csv(path, index=False)
    return pd.DatetimeIndex(pd.read_csv(path)["vintage"])


def load_cape() -> pd.Series:
    """실러 ie_data.xls의 월별 CAPE(인덱스 = 월 시작일). xlrd 필요."""
    path = CACHE / "shiller_cape.csv"
    if not path.exists():
        raw = pd.read_excel(io.BytesIO(_get(SHILLER_URL)), sheet_name="Data", header=None, engine="xlrd")
        d = raw.iloc[8:, [0, 12]].copy()
        d.columns = ["date", "cape"]
        d = d[pd.to_numeric(d["date"], errors="coerce").notna()]
        dt = pd.to_numeric(d["date"])
        year = dt.astype(int)
        month = ((dt - year) * 100).round().astype(int)
        d.index = pd.to_datetime(dict(year=year, month=month, day=1))
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.to_numeric(d["cape"], errors="coerce").rename("cape").to_csv(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)["cape"]


def flat_runs(s: pd.Series, min_len: int = 5) -> int:
    """같은 종가가 min_len일 이상 이어진 구간 수(지수 데이터 멈춤 의심)."""
    same = s.diff().eq(0)
    run_id = (~same).cumsum()
    lengths = same.groupby(run_id).sum() + 1
    return int((lengths >= min_len).sum())


def audit_yf(ticker: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"ticker": ticker, "status": "BLOCKED(빈 결과)"}
    raw = df["Close"]
    c, info = l1d.clean_index(raw)
    gaps = c.index.to_series().diff().dt.days
    r = c.pct_change().abs()
    adj = df["Adj Close"].reindex(c.index)
    return {
        "ticker": ticker,
        "start": c.index[0].date(),
        "end": c.index[-1].date(),
        "rows": len(c),
        "nan_or_nonpos": info["nan"] + info["nonpositive"],
        "dup": info["duplicates"],
        "max_gap_days": int(gaps.max()),
        "max_gap_at": gaps.idxmax().date(),
        "flat5_runs": flat_runs(c),
        "abs_ret_gt20pct": int((r > 0.20).sum()),
        "adj_eq_close": bool((adj - c).abs().max() < 1e-6),
    }


def audit_fred(series_id: str, s: pd.Series) -> dict:
    ok = s.dropna()
    return {
        "series": series_id,
        "start": ok.index[0].date() if len(ok) else None,
        "end": ok.index[-1].date() if len(ok) else None,
        "rows": len(s),
        "missing": int(s.isna().sum()),
    }


def sahm_release_dates(sahm: pd.Series, unrate_vintages: pd.DatetimeIndex) -> pd.Series:
    """월 m 샴 값의 공개일 근사 = 월 m 말일 뒤 첫 UNRATE 발표일(고용보고서). 발표일 목록 이전 월은
    목록 첫날(1960-03-15)로 늦게 잡힌다(보수적). 이후 발표가 없으면 NaT."""
    out = {}
    for m in sahm.index:
        later = unrate_vintages[unrate_vintages > m + pd.offsets.MonthEnd(0)]
        out[m] = later[0] if len(later) else pd.NaT
    return pd.Series(out)


def event_context(day, sahm, sahm_pub, baa, cape, cape_pub, dtb3):
    """사건 −15% 도달일에 알려진 값만으로 샴·BAA10Y 백분위·CAPE 백분위·DTB3 12개월 변화."""
    s = de.asof_value(sahm, sahm_pub, day)
    baa_hist = baa[baa.index <= day].dropna()
    baa_pct = de.expanding_percentile(baa_hist, baa_hist.iloc[-1] if len(baa_hist) else np.nan, MIN_BAA_OBS)
    known_cape = cape[(cape_pub <= day).to_numpy()].dropna()
    cape_pct = de.expanding_percentile(known_cape, known_cape.iloc[-1] if len(known_cape) else np.nan, 120)
    d_now = dtb3[dtb3.index <= day].dropna()
    d_then = dtb3[dtb3.index <= day - pd.Timedelta(days=365)].dropna()
    chg = float(d_now.iloc[-1] - d_then.iloc[-1]) if len(d_now) and len(d_then) else np.nan
    return s, baa_pct, cape_pct, chg


def _cell(v) -> str:
    if isinstance(v, float):
        return "" if np.isnan(v) else (str(int(v)) if v.is_integer() and abs(v) >= 1 else str(v))
    return str(v)


def md_table(df: pd.DataFrame) -> str:
    """DataFrame → 마크다운 표 문자열(tabulate 없이)."""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(_cell(v) for v in row) + " |")
    return "\n".join(out)


def fmt_pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{digits}f}%"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    yf_data = {t: load_yf(t) for t in TICKERS}
    fred = {s: load_fred(s) for s in FRED}
    unrate_v = load_alfred_vintages("UNRATE")
    sahm_v = load_alfred_vintages("SAHMREALTIME")
    cape = load_cape()

    # ── 관문 0 통계 ──
    yrows = [audit_yf(t, df) for t, df in yf_data.items()]
    frows = [audit_fred(s, v) for s, v in fred.items()]
    cape_ok = cape.dropna()
    lines = ["# 폭락 연구 1 관문 0 — 내려받은 데이터 자동 통계", "",
             f"> 생성: scripts/shock_dip_gate01.py, 받은 범위 끝 {END}(제외). 해석·무료 여부·발표 시차는 shock_dip_data_audit.md.", "",
             "## yfinance (auto_adjust=False)", "", md_table(pd.DataFrame(yrows)), "",
             "## FRED fredgraph.csv (키 없음)", "", md_table(pd.DataFrame(frows)), "",
             "## ALFRED 발표일 목록", "",
             f"- UNRATE: {len(unrate_v)}개, {unrate_v[0].date()} ~ {unrate_v[-1].date()}",
             f"- SAHMREALTIME: {len(sahm_v)}개, {sahm_v[0].date()} ~ {sahm_v[-1].date()}", "",
             "## 실러 CAPE (ie_data.xls)", "",
             f"- 월 {len(cape)}행, CAPE 값 있는 첫 달 {cape_ok.index[0]:%Y-%m}, 마지막 {cape_ok.index[-1]:%Y-%m}, 결측 {int(cape.isna().sum())}", ""]
    (OUT / "shock_dip_data_audit_stats.md").write_text("\n".join(lines), encoding="utf-8")

    # ── 관문 1 사건 표 ──
    sahm = fred["SAHMREALTIME"].dropna()
    sahm_pub = sahm_release_dates(sahm, unrate_v)
    cape_pub = pd.Series(cape.index + pd.offsets.MonthBegin(2), index=cape.index)  # [분석 가정] 월 m 값은 m+2월 1일부터 공개
    baa, dtb3 = fred["BAA10Y"], fred["DTB3"]

    ev_rows, half_rows = [], []
    for tk, start in EVENT_INDEXES.items():
        c, _ = l1d.clean_index(yf_data[tk]["Close"])
        if start:
            c = c[c.index >= start]
        events = de.find_drawdown_events(c, -0.15)
        for _, e in events.iterrows():
            s, bp, cp, chg = event_context(e["hit_date"], sahm, sahm_pub, baa, cape, cape_pub, dtb3)
            key = (tk, f"{e['peak_date']:%Y-%m}")
            ev_rows.append({
                "지수": tk,
                "고점일": e["peak_date"].date(), "고점": round(e["peak"], 2),
                "−15% 도달일": e["hit_date"].date(),
                "바닥일": e["trough_date"].date(), "바닥": round(e["trough"], 2),
                "최대 하락": round(e["max_dd"] * 100, 1),
                "고점→바닥 거래일": int(e["peak_to_trough_days"]),
                "고점→회복 거래일": "미회복" if np.isnan(e["peak_to_recovery_days"]) else int(e["peak_to_recovery_days"]),
                "회복일": "미회복" if pd.isna(e["recovery_date"]) else e["recovery_date"].date(),
                "가격/배당": "가격지수(배당 미포함)" if tk != "QQQ" else "가격(분할 반영, 분배금 미포함)",
                "시작 20거래일 −10%": "예" if e["fast20"] else "아니오",
                "도중 20거래일 −10%": "예" if e["any20"] else "아니오",
                "샴(그날)": "UNKNOWN" if np.isnan(s) else round(s, 2),
                "BAA10Y 백분위": "UNKNOWN" if np.isnan(bp) else round(bp * 100, 1),
                "CAPE 백분위": "UNKNOWN" if np.isnan(cp) else round(cp * 100, 1),
                "DTB3 12개월 변화(%p)": "UNKNOWN" if np.isnan(chg) else round(chg, 2),
                "잠정 유형": de.provisional_type(s, bp, cp, chg, bool(e["fast20"])),
                "구간": "참고(판정 구간 밖)" if e["hit_date"] > JUDGE_END else "판정 구간",
                "사후 원인 설명(신호 계산에 사용 금지)": CAUSES.get(key, ""),
            })
            for lv in HALVING_LEVELS:
                tp = de.first_touch_pos(c, e, lv)
                if tp is None:
                    continue
                sp = de.first_above_sma_pos(c, tp, 50)
                for kind, pos in [("처음 닿은 날", tp), ("그 뒤 첫 50일선 위", sp)]:
                    row = {"지수": tk, "고점일": e["peak_date"].date(), "수준": f"{int(lv * 100)}%", "진입": kind}
                    if pos is None:
                        row.update({"진입일": "없음"})
                    else:
                        row.update({"진입일": c.index[pos].date(), "진입가": round(c.iloc[pos], 2)})
                        for name, h in HORIZONS.items():
                            row[name] = de.forward_return(c, pos, h)
                        row["이후 추가 하락"] = de.further_drawdown(c, pos, int(e["end_pos"]))
                    row["구간"] = "참고(판정 구간 밖)" if e["hit_date"] > JUDGE_END else "판정 구간"
                    half_rows.append(row)

    ev = pd.DataFrame(ev_rows)
    ev.to_csv(OUT / "drawdown_events.csv", index=False, encoding="utf-8-sig")
    hv = pd.DataFrame(half_rows)
    write_events_md(ev)
    write_halving_md(hv)
    print(f"사건 {len(ev)}개, 반토막 행 {len(hv)}개 - docs/research/에 저장")


def write_events_md(ev: pd.DataFrame) -> None:
    cols = ["지수", "고점일", "바닥일", "최대 하락", "고점→바닥 거래일", "고점→회복 거래일", "시작 20거래일 −10%",
            "샴(그날)", "BAA10Y 백분위", "CAPE 백분위", "DTB3 12개월 변화(%p)", "잠정 유형", "구간",
            "사후 원인 설명(신호 계산에 사용 금지)"]
    summary = ev.groupby("지수", sort=False).agg(
        사건=("고점일", "size"),
        판정구간=("구간", lambda x: int((x == "판정 구간").sum())),
        빠른하락=("시작 20거래일 −10%", lambda x: int((x == "예").sum())),
        미회복=("회복일", lambda x: int((x == "미회복").sum())),
    )
    types = ev.pivot_table(index="지수", columns="잠정 유형", values="고점일", aggfunc="size", fill_value=0)
    md = [
        "# 하락 사건 표 (폭락 연구 1 관문 1, 참고 분석 — 시험 아님)", "",
        "> 생성: `scripts/shock_dip_gate01.py`, 규칙 `core/drawdown_events.py`. 전체 열은 `drawdown_events.csv`.",
        "> 사건: 종가 기준 직전 최고가 대비 −15% 이하가 된 물속 구간(회복 전 재하락은 같은 사건). 가격지수 기준.",
        "> ^GSPC 1950-01-01~, ^IXIC 1971-02-05~, QQQ 1999-03-10~, ^KS11 1996-12-11~(야후 최초일), 끝 2026-10-16.",
        "> 판정 구간 = −15% 도달일 ≤ 2021-12-31. 그 뒤는 「참고(판정 구간 밖)」.", "",
        "## 해석·분석 선택 (사전 등록 원문에 없음 — 참고 표 전용, 사용자 확인 필요)", "",
        "- 「그날」 = 사건의 −15% 도달일. 그날까지 공개된 값만 쓴다.",
        "- 시작 20거래일 −10%: 고점 다음 20거래일 안에 고점 대비 −10% 이하 종가가 있었는지. "
        "(참고로 「도중 20거래일 −10%」 = 고점~바닥 사이 어느 20거래일 수익률이든 −10% 이하였는지, CSV에만)",
        "- 샴: FRED SAHMREALTIME 현재 판(2019-09 이후 발표본과 과거 값 동일 확인). 월 m 값 공개일 = 월 m 말일 뒤 첫 UNRATE 발표일(ALFRED) — 근사.",
        "- BAA10Y 백분위: 1986-01-02부터 그날까지 일별 값 중 그날 값 이하 비율. 표본 252일 미만이면 UNKNOWN.",
        "- CAPE 백분위: 실러 현재 파일(빈티지 없음 → 사후 수정 반영됨), 월 m 값은 m+2월 1일부터 공개로 가정. 1881-01부터 분포, 표본 120개월 미만 UNKNOWN.",
        "- 잠정 유형: 침체 = 샴 ≥ 0.5 또는 BAA10Y 백분위 ≥ 90 / 거품 = CAPE 백분위 ≥ 90 / 긴축 = DTB3 12개월 상승 ≥ 1.0%p(이 1.0은 분석 가정) "
        "/ 깃발 2개 이상 혼합 / 깃발 없음 + 입력 모두 있음 + 빠른 하락이면 충격 / 그 밖 UNKNOWN.",
        "", "## 읽을 때 주의", "",
        "- ^KS11 사건에도 미국 지표(샴·BAA10Y·실러 CAPE·DTB3)를 그대로 붙였다(해외 판정에 미국 샴을 쓰는 사전 등록 H③과 같은 방식). "
        "한국 경기·밸류에이션이 아니다.",
        "- 실러 CAPE의 1881년부터 누적 백분위는 1999~2021년 달의 72.1%에서 90 이상이다(1950~1998 21.8%, 1972~1998 16.7%). "
        "그래서 1996년 이후 사건 대부분이 「거품」 깃발을 받는다. 사전 등록 B2(CAPE 상위 10% 미만일 때만 진입)는 QQQ 구간에서 진입을 대부분 막을 수 있다.",
        "- BAA10Y 백분위 ≥ 90인 날은 1999~2021 거래일의 27.4%.",
        "- QQQ 2000-01-03 사건(−15.5%, 3거래일)은 야후 QQQ 2000-01-06 종가 이상치 때문일 가능성이 크다(같은 날 ^NDX는 고점 대비 −11.9%). "
        "야후 QQQ 종가를 ^NDX 비율과 비교하면 2% 넘게 어긋난 날이 10일(1999-11-22, 2000-01-06·07, 2000-08-08, 2000-09-21, 2000-11-29, 2000-12-07·08, 2002-06-18, 2008-09-29).",
        "", "## 요약", "", md_table(summary.reset_index()), "", "### 잠정 유형 분포", "", md_table(types.reset_index()), "",
        "## 사건 목록", "", md_table(ev[cols]), "",
    ]
    (OUT / "drawdown_events.md").write_text("\n".join(md), encoding="utf-8")


def write_halving_md(hv: pd.DataFrame) -> None:
    show = hv.copy()
    for col in list(HORIZONS) + ["이후 추가 하락"]:
        show[col] = show[col].map(lambda x: fmt_pct(x)) if col in show else "—"
    judged = hv[hv["구간"] == "판정 구간"]
    summ = []
    for (tk, lv, kind), g in judged.groupby(["지수", "수준", "진입"], sort=False):
        r = {"지수": tk, "수준": lv, "진입": kind, "횟수": len(g)}
        for col in list(HORIZONS) + ["이후 추가 하락"]:
            vals = g[col].dropna() if col in g else pd.Series(dtype=float)
            r[f"{col} 중앙값(n)"] = f"{fmt_pct(vals.median())} ({len(vals)})" if len(vals) else "—"
        r["이후 추가 하락 최악"] = fmt_pct(g["이후 추가 하락"].min()) if "이후 추가 하락" in g and g["이후 추가 하락"].notna().any() else "—"
        summ.append(r)
    md = [
        "# 반토막 진입 분석 (폭락 연구 1 관문 1, 참고 분석 — 시험 아님)", "",
        "> 생성: `scripts/shock_dip_gate01.py`. 대상 사건은 `drawdown_events.md`와 같다. 가격지수(배당 미포함), 비용·세금 없음.",
        "> 진입 ① 사건 안에서 고점 대비 −30%·−40%·−50%에 처음 닿은 날 종가, ② 그 다음 날부터 종가 > 50일 단순이동평균인 첫날 종가.",
        "> 1·3·5·10년 = 252·756·1260·2520거래일 뒤 종가 수익률(데이터가 모자라면 —). 미래 기간에는 2022년 이후 가격도 쓴다(참고 분석).",
        "> 이후 추가 하락 = 진입일부터 그 사건의 전고점 회복 전날까지 최저 종가 ÷ 진입가 − 1(진입이 회복 뒤면 —).", "",
        "## 요약 (판정 구간 사건만, 중앙값과 표본 수)", "", md_table(pd.DataFrame(summ)), "",
        "## 전체 행 (참고 구간 포함)", "", md_table(show), "",
    ]
    (OUT / "halving_entry.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
