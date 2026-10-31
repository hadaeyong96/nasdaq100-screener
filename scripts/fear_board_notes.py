"""「공포 매수판」 참고 통계 (정보용 — 판정 아님): 1999~2021 QQQ로 count별 다음 63·126·252거래일 수익.

- 매일 판은 thermometer.fear_board.get_fear_board 그대로(라이브와 같은 함수), 그날까지 자른 자료로 계산한다.
- 2022-01-01 이후 자료는 계산 금지(봉인 ⓑ): 읽은 모든 자료를 notes.end에서 자르고, 남아 있으면 멈춘다.
  다음 n거래일 수익도 notes.end 안에서 끝나는 날만 쓴다(끝나지 않는 날은 그 기간 통계에서 빠짐).
- 자료: QQQ·^VIX = data/backtest_prices 캐시(분할 반영 Close, 배당 미포함 가격 수익).
  공포·탐욕은 2021년 이전 자료가 없어 전 구간 "자료 없음" → 이 통계는 나머지 4개 조건 중 켜진 수다.
- 결과: docs/fear_board_notes.md

실행: python scripts/fear_board_notes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from data import backtest_prices as bp  # noqa: E402
from scripts.thermometer_backfill import _cut  # noqa: E402 — 같은 봉인 가드
from thermometer.core import load_config  # noqa: E402
from thermometer.fear_board import get_fear_board  # noqa: E402

NOTES_PATH = ROOT / "docs" / "fear_board_notes.md"


def load_prices(cfg: dict) -> dict:
    """캐시의 QQQ·VIX(notes.end에서 자름) → get_fear_board의 prices dict. 공포·탐욕은 빈 값."""
    end = pd.Timestamp(cfg["fear_board"]["notes"]["end"])
    out = {"fear_greed": {}}
    for key, tk in (("qqq", cfg["sma"]["ticker"]), ("vix", cfg["vix"]["ticker"])):
        df = bp._load_cache(tk)
        if df is None or not len(df) or df.index[-1] < end:
            raise SystemExit(f"{tk} 캐시가 {end.date()}까지 없습니다 — 멈춥니다")
        df = _cut(df, end)
        out[key] = df["close"].astype(float) if key == "qqq" else df[["close", "high"]].astype(float)
    return out


def daily_boards(prices: dict, cfg: dict) -> pd.DataFrame:
    """notes.start~end 거래일마다 get_fear_board → DataFrame(date 인덱스, count, n_avail, ok_*)."""
    nt = cfg["fear_board"]["notes"]
    days = prices["qqq"].loc[pd.Timestamp(nt["start"]):pd.Timestamp(nt["end"])].index
    rows = []
    for d in days:
        b = get_fear_board(d, prices=prices, cfg=cfg)
        rows.append({"date": d, "count": b["count"], "n_avail": b["n_avail"],
                     **{f"ok_{it['key']}": it["ok"] for it in b["items"]}})
    return pd.DataFrame(rows).set_index("date")


def forward_returns(q: pd.Series, horizons) -> pd.DataFrame:
    """종가 → 날짜별 다음 h거래일 수익 %(h거래일 뒤 종가 / 그날 종가 − 1). 자료 끝을 넘으면 NaN.
    (통계용 — 신호 계산에는 쓰지 않는다.)"""
    v = q.to_numpy(dtype=float)
    out = {}
    for h in horizons:
        r = np.full(len(v), np.nan)
        r[:len(v) - h] = (v[h:] / v[:len(v) - h] - 1.0) * 100.0
        out[h] = r
    return pd.DataFrame(out, index=q.index)


def summarize(fwd: pd.DataFrame, mask: pd.Series, horizons) -> dict:
    """mask가 True인 날의 h별 {n, mean, median, p10, pos}."""
    out = {}
    for h in horizons:
        x = fwd.loc[mask[mask].index, h].dropna()
        out[h] = {"n": len(x), "mean": x.mean(), "median": x.median(), "p10": x.quantile(0.1), "pos": (x > 0).mean() * 100}
    return out


def episodes(days: pd.DatetimeIndex, index: pd.DatetimeIndex, gap: int) -> list[dict]:
    """켜진 날 목록 → 묶음 [{first, last, days}]. 앞 켜진 날과 거래일 위치 차이가 gap 이하면 같은 묶음(gap=1 = 연속)."""
    pos = index.get_indexer(days)
    out: list[dict] = []
    for p, d in zip(pos, days):
        if out and p - out[-1]["_p"] <= gap:
            out[-1].update(last=d, days=out[-1]["days"] + 1, _p=p)
        else:
            out.append({"first": d, "last": d, "days": 1, "_p": p})
    return out


def _row(label: str, st: dict, horizons) -> str:
    cells = []
    for h in horizons:
        s = st[h]
        cells.append("—" if s["n"] == 0 else f"{s['mean']:+.1f}% / {s['median']:+.1f}% / {s['p10']:+.1f}% / {s['pos']:.0f}% (n={s['n']:,})")
    return f"| {label} | " + " | ".join(cells) + " |"


def write_notes(df: pd.DataFrame, fwd: pd.DataFrame, cfg: dict) -> str:
    fb, nt = cfg["fear_board"], cfg["fear_board"]["notes"]
    hz = list(nt["horizons"])
    gap = int(nt["episode_gap_days"])
    avail4 = df["n_avail"] == 4
    fear, dip, every = avail4 & (df["count"] >= 3), avail4 & (df["count"] == 2), avail4
    fear_days = df.index[fear]
    streaks = episodes(fear_days, df.index, 1)
    groups = episodes(fear_days, df.index, gap)
    first_streak = pd.Series(df.index.isin([e["first"] for e in streaks]), index=df.index)
    first_group = pd.Series(df.index.isin([e["first"] for e in groups]), index=df.index)
    head = " | ".join(f"다음 {h}거래일" for h in hz)
    table = [f"| 구분 | {head} |", "| --- |" + " --- |" * len(hz),
             _row(f"공포: 4개 중 3개 이상 ({int(fear.sum()):,}일)", summarize(fwd, fear, hz), hz),
             _row(f"눌림: 4개 중 2개 ({int(dip.sum()):,}일)", summarize(fwd, dip, hz), hz),
             _row(f"모든 날 ({int(every.sum()):,}일)", summarize(fwd, every, hz), hz),
             _row(f"공포 연속 묶음 첫날 ({len(streaks)}번)", summarize(fwd, first_streak, hz), hz),
             _row(f"공포 하락 묶음 첫날 ({len(groups)}번, 간격 {gap}거래일 이내 합침)", summarize(fwd, first_group, hz), hz)]
    by_cnt = df.loc[avail4, "count"].value_counts().sort_index()
    cnt_rows = "\n".join(f"| {c} | {n:,} | {n / avail4.sum() * 100:.1f}% |" for c, n in by_cnt.items())
    ok_share = " · ".join(f"{fb['names'][k]} {df.loc[avail4, f'ok_{k}'].astype(bool).mean() * 100:.1f}%"
                          for k in ("rsi", "bollinger", "trend", "vix"))
    ep_rows = "\n".join(
        f"| {e['first'].date()} | {e['last'].date()} | {e['days']} | {int(df.loc[e['first']:e['last'], 'count'].max())} | "
        + " | ".join("—" if pd.isna(fwd.loc[e['first'], h]) else f"{fwd.loc[e['first'], h]:+.1f}%" for h in hz) + " |"
        for e in groups)
    q_first = df.index[avail4][0].date() if avail4.any() else None
    text = f"""# 공포 매수판 참고 통계 (정보용 — 판정 아님)

> `scripts/fear_board_notes.py` (2026-10-31). 매일 판은 `thermometer/fear_board.py` `get_fear_board`(라이브와 같은 함수)로 그날까지의 자료만 써서 구했다.
> 기간 {nt['start']} ~ {nt['end']} (2022-01-01 이후 자료는 읽지도 계산하지도 않음, 봉인 ⓑ). 다음 n거래일 수익도 {nt['end']} 안에서 끝나는 날만 넣었다.
> 수익은 QQQ 종가(분할 반영, 배당 미포함) 기준 가격 수익이다. 이 숫자로 규칙을 만들려면 사전 등록 후 검증 구역에서 따로 시험해야 한다.

## 자료 상황 — **공포·탐욕 자료 없음 → 전 구간 "4개 조건 중"으로 따로 표시**

- 공포·탐욕(①)은 2021년 이전 자료가 없다(캐시는 2025-08부터). 그래서 이 문서의 모든 통계는 ②~⑤ **4개 조건 중** 켜진 수다.
  라이브 판(5개 중 3개 이상 = 공포)과 기준이 같지 않다 — 여기서 "공포"는 **4개 중 3개 이상**, "눌림"은 **4개 중 2개**.
- 4개 조건이 모두 계산되는 첫날: {q_first} (QQQ 캐시 첫날 {df.index[0].date()} 뒤 60일선·21거래일 전 20일선이 찬 뒤). 그 전 날은 빼고 셌다.
- 조건별 켜진 날 비율: {ok_share}

| 켜진 수(4개 중) | 일수 | 비율 |
| --- | --- | --- |
{cnt_rows}

## 다음 63·126·252거래일 QQQ 수익

칸마다 **평균 / 중앙값 / 하위 10% / 양수 비율** (n = 그 기간이 {nt['end']} 안에서 끝나는 날 수). 날마다 기간이 겹치므로 n은 독립 표본 수가 아니다.

{chr(10).join(table)}

## 공포 구간 묶음

- 연속된 공포일(거래일 하루도 안 끊긴 묶음): **{len(streaks)}번**
- 같은 하락으로 합친 묶음(앞 공포일과 {gap}거래일 이내면 같은 묶음, `config/thermometer.yaml` fear_board.notes.episode_gap_days): **{len(groups)}번**
- 아래는 합친 묶음마다 첫날 기준.

| 첫날 | 마지막 공포일 | 공포일 수 | 최대 켜진 수 | {head} |
| --- | --- | --- | --- |{" --- |" * len(hz)}
{ep_rows}
"""
    NOTES_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    cfg = load_config()
    prices = load_prices(cfg)
    df = daily_boards(prices, cfg)
    if df.index.max() > pd.Timestamp(cfg["fear_board"]["notes"]["end"]):
        raise SystemExit("봉인 구간 날짜가 섞였습니다 — 멈춥니다")
    fwd = forward_returns(prices["qqq"].loc[df.index[0]:], cfg["fear_board"]["notes"]["horizons"]).reindex(df.index)
    print(write_notes(df, fwd, cfg))


if __name__ == "__main__":
    main()
