"""시장 온도계 과거 시계열(연구용 백필, 정보용 — 전략 판단 아님): 1999-01-01 ~ 2021-12-31 매일 온도.

- 계산은 thermometer.core.get_temperature 그대로(라이브와 같은 함수). 출력 data/thermometer/history_backfill.csv
- 2022-01-01 이후 자료는 계산 금지(봉인): 읽은 모든 자료를 2021-12-31에서 자르고, 남아 있으면 멈춘다.
- 자료: QQQ·^VIX(data/backtest_prices 캐시, 없으면 2021-12-31까지 받음), 시장 폭 = 시점별 나스닥100
  (data/universe_history, 현재 명단은 data/universe_fallback.csv) × 백테스트 가격 캐시(2006년부터만 있음 →
  그 전 구간은 시장 폭 자료 없음). 공포·탐욕은 2021년 이전 자료가 없어 전 구간 자료 없음(나머지로 환산).
- 참고 통계(a~d)를 docs/thermometer_notes.md에 쓴다.

실행: python scripts/thermometer_backfill.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from data import backtest_prices as bp  # noqa: E402
from data import universe_history as uh  # noqa: E402
from thermometer import history  # noqa: E402
from thermometer.core import get_temperature, load_config  # noqa: E402

NOTES_PATH = ROOT / "docs" / "thermometer_notes.md"


def _cut(obj, end: pd.Timestamp):
    """봉인 가드: end 뒤 줄을 잘라 내고, 잘린 결과에 end 뒤 날짜가 남아 있으면 멈춘다."""
    out = obj.loc[:end]
    if len(out) and out.index.max() > end:
        raise SystemExit(f"봉인 구간 자료가 섞였습니다(마지막 {out.index.max().date()}) — 멈춥니다")
    return out


def _series(ticker: str, start: date, end: date) -> pd.DataFrame:
    cached = bp._load_cache(ticker)
    if cached is not None and len(cached) and cached.index[-1].date() >= end:
        return cached
    df, _ = bp.fetch_history(ticker, start, end)
    return df


def load_prices(cfg: dict) -> tuple[dict, dict]:
    """백필 자료 → (prices dict, 자료 상황 dict)."""
    start = date.fromisoformat(cfg["history"]["backfill_start"])
    end = date.fromisoformat(cfg["history"]["backfill_end"])
    end_ts = pd.Timestamp(end)
    qqq = _cut(_series(cfg["sma"]["ticker"], start, end), end_ts)["close"].astype(float)
    vix = _cut(_series(cfg["vix"]["ticker"], start, end), end_ts)[["close", "high"]].astype(float)

    current = set(pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"])
    checkpoints = uh.membership_checkpoints(current, uh.get_changes(), start)
    ever = set().union(*(m for d, m in checkpoints if d <= end))
    closes, no_price = {}, []
    for t in sorted(ever):
        df = bp._load_cache(t)
        if df is None or not len(df):
            no_price.append(t)
            continue
        closes[t] = _cut(df, end_ts)["close"].astype(float)
    stocks = pd.DataFrame(closes).sort_index()
    stocks = stocks.loc[pd.Timestamp(start):]
    prices = {"qqq": qqq, "vix": vix, "stocks": stocks, "fear_greed": {},
              "members": lambda ts: uh.universe_on(checkpoints, ts.date())}
    info = {"start": start, "end": end, "qqq_first": qqq.index[0].date(), "vix_first": vix.index[0].date(),
            "stocks_first": stocks.index[0].date() if len(stocks) else None,
            "n_ever": len(ever), "n_price": len(closes), "no_price": no_price}
    return prices, info


def up_crosses(gap: pd.Series) -> list[int]:
    """이격률 Series(NaN 제외, 위치 기준) → 아래(≤0)에서 위(>0)로 넘어선 날의 위치 목록."""
    g = gap.to_numpy()
    return [i for i in range(1, len(g)) if g[i - 1] <= 0 < g[i]]


def stats(df: pd.DataFrame) -> dict:
    """백필 표(date 인덱스, raw_*/deg_*/temp/zone) → 참고 통계 a~d."""
    both = df.dropna(subset=["raw_sma_fast", "raw_sma_slow"])
    gf, gs = both["raw_sma_fast"].astype(float), both["raw_sma_slow"].astype(float)
    cf, cs = up_crosses(gf), up_crosses(gs)
    leads = []
    for t in cs:  # a) 200일선 상향 돌파일마다, 그날 또는 그 전 가장 최근의 150일선 상향 돌파일까지의 거래일 수
        prev = [s for s in cf if s <= t]
        if prev:
            leads.append(t - prev[-1])

    def fail_rate(g: pd.Series, crosses: list[int]) -> tuple[int, int]:
        v = g.to_numpy()
        done = [i for i in crosses if i + 20 < len(v)]
        fails = sum(1 for i in done if (v[i + 1:i + 21] <= 0).any())
        return fails, len(done)

    ff, fn = fail_rate(gf, cf)
    sf, sn = fail_rate(gs, cs)
    partial = (both["deg_sma_fast"].astype(float) + both["deg_sma_slow"].astype(float))
    t = df["temp"].dropna()
    zone_share = df.loc[t.index, "zone"].value_counts(normalize=True)
    lead_s = pd.Series(leads, dtype=float)
    return {
        "a": {"n": len(leads), "mean": lead_s.mean(), "median": lead_s.median(), "same_day": int((lead_s == 0).sum())},
        "b": {"fast": (ff, fn), "slow": (sf, sn)},
        "c": {"p10": partial.quantile(0.1), "p50": partial.quantile(0.5), "p90": partial.quantile(0.9), "n": len(partial)},
        "d": zone_share, "n_temp": len(t),
        "missing": df["missing"].fillna("").value_counts(),
    }


def write_notes(st: dict, info: dict, cfg: dict) -> str:
    zones = [z["name"] for z in cfg["zones"]]
    names = cfg["names"]
    miss = "\n".join(f"| {(' · '.join(names[k] for k in m.split('|')) if m else '없음')} | {n:,} |"
                     for m, n in st["missing"].items())
    a, b, c = st["a"], st["b"], st["c"]
    pct = lambda x, n: f"{x / n * 100:.1f}% ({x}/{n})" if n else "—"  # noqa: E731
    text = f"""# 시장 온도계 참고 통계 (정보용 — 전략 판단 아님)

> `scripts/thermometer_backfill.py` (2026-10-28). 계산 함수 `thermometer/core.py` `get_temperature` (라이브와 같음), 설정 `config/thermometer.yaml`.
> 기간 {info['start']} ~ {info['end']} (2022-01-01 이후 과거 시계열은 계산하지 않음). 결과 `data/thermometer/history_backfill.csv`.
> 이 숫자로 매매 규칙을 만들려면 사전 등록 후 검증 구역에서 따로 시험해야 한다.

## 자료 상황

- QQQ {info['qqq_first']}부터 (150일선 {cfg['sma']['fast_days']}거래일·200일선 {cfg['sma']['slow_days']}거래일이 찬 뒤부터 ①② 계산) · VIX {info['vix_first']}부터
- 시장 폭: 시점별 나스닥100(위키 변경 이력 역산, 2007-02 전은 그때 명단 그대로) × 백테스트 가격 캐시 — 가격이 {info['stocks_first']}부터만 있어 그 전은 자료 없음.
  2006~2021 명단에 든 {info['n_ever']}종목 중 가격 있는 종목 {info['n_price']}개(편입 제외·상장 폐지 종목 가격이 없음). 확보율 {cfg['breadth']['min_coverage']:.0%} 미만인 날은 자료 없음
  → 시장 폭이 계산된 날은 **{info['breadth_first']} ~ {info['breadth_last']} {info['breadth_days']:,}일뿐**이다. 그 전 온도는 ①②④ 세 항목(60도 만점)을 100도로 환산한 값이라 뒤 구간과 바로 비교하기 어렵다.
- 공포·탐욕: 2021년 이전 자료 없음 → 전 구간 자료 없음(나머지 항목으로 100도 환산).

| 자료 없음 항목 | 일수 |
| --- | --- |
{miss}

## a) 150일선이 200일선보다 며칠 먼저 넘어섰나

QQQ가 200일선을 아래→위로 넘어선 날 {a['n']}번마다, 그날 또는 그 전 가장 최근에 150일선을 아래→위로 넘어선 날까지의 거래일 수.
평균 **{a['mean']:.1f}거래일**, 중앙값 {a['median']:.0f}거래일 (같은 날 {a['same_day']}번).

## b) 넘어선 뒤 20거래일 안에 다시 아래로 내려간 비율

- 150일선: **{pct(*b['fast'])}**
- 200일선: **{pct(*b['slow'])}**

## c) ①②만으로 만든 40도 만점 부분 온도 분포 ({c['n']:,}일)

하위 10% **{c['p10']:.1f}도** · 중앙값 **{c['p50']:.1f}도** · 상위 10% **{c['p90']:.1f}도**

## d) 전체 온도 구간별 일수 비율 ({st['n_temp']:,}일, 자료 없는 항목은 환산)

| 구간 | 비율 |
| --- | --- |
""" + "\n".join(f"| {z} | {st['d'].get(z, 0) * 100:.1f}% |" for z in zones) + "\n"
    NOTES_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    cfg = load_config()
    prices, info = load_prices(cfg)
    days = prices["qqq"].loc[pd.Timestamp(info["start"]):pd.Timestamp(info["end"])].index
    rows = [history.row_of(get_temperature(d, prices=prices, cfg=cfg)) for d in days]
    history.write_rows(rows, cfg["history"]["backfill_path"])
    df = pd.DataFrame(rows).set_index("date").replace("", None).infer_objects()
    br = df["raw_breadth"].dropna()
    info.update(breadth_first=br.index.min(), breadth_last=br.index.max(), breadth_days=len(br))
    print(write_notes(stats(df), info, cfg))
    if info["no_price"]:
        print(f"가격 캐시 없는 명단 종목 {len(info['no_price'])}개: {', '.join(info['no_price'])}")


if __name__ == "__main__":
    main()
