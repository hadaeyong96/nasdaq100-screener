"""시장 온도계 계산 — 온도는 이 모듈의 get_temperature 한 함수에서만 구한다 (2026-10-28).

항목 5개 × 각 0~20도, 합계 0~100도(정수 반올림). 기준값은 config/thermometer.yaml.
 ① 150일선: QQQ 이격률 x% → 20 × clip((x + 10) / 20, 0, 1)
 ② 200일선: 같은 식
 ③ 시장 폭: 그날 구성 종목 중 종가 > 자기 SMA50 비율 p% → p ÷ 5 (직전 10거래일 안에 20% 미만 → 오늘 50% 이상 = 강한 회복)
 ④ VIX v: 20 × clip((35 − v) / (35 − 12), 0, 1) (직전 21거래일 최고 25 이상이면 "지난달 …에서 하락")
 ⑤ 공포·탐욕 f: f ÷ 5
자료 없는 항목은 빼고 나머지를 100도 만점으로 환산한다(4개면 합계 × 100/80).

미래 자료 금지: 모든 입력을 그날(date)까지 잘라서 쓴다 — 음수 shift 없음. prices를 주면 네트워크·파일을 건드리지
않는 순수 계산이고(같은 입력 → 같은 출력), 안 주면 thermometer.data.load_live로 오늘 자료를 받는다.

prices 형식(dict):
  qqq: Series(종가, DatetimeIndex 오름차순)
  stocks: DataFrame(종가, 날짜 × 티커) — 없으면 시장 폭 자료 없음
  members: None(= stocks의 모든 열) | 티커 집합 | 날짜(Timestamp) → 티커 집합 함수 (그날 구성 종목)
  vix: DataFrame(close, high) 또는 Series(종가)
  fear_greed: {"YYYY-MM-DD": 값} 또는 Series
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "thermometer.yaml"
KEYS = ("sma_fast", "sma_slow", "breadth", "vix", "fear_greed")


def load_config(path: Path | None = None) -> dict:
    """config/thermometer.yaml → dict."""
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clip01(x: float) -> float:
    return min(1.0, max(0.0, x))


def round_half_up(x: float) -> int:
    """정수 반올림(0.5는 올림) — 파이썬 round()의 짝수 반올림을 피한다."""
    return int(math.floor(x + 0.5 + 1e-9))


def score_gap(x_pct: float, cfg: dict) -> float:
    """이격률 x%(종가/SMA − 1) → 0~item_max도."""
    s = cfg["sma"]
    lo, hi = float(s["gap_low_pct"]), float(s["gap_high_pct"])
    return cfg["item_max"] * _clip01((x_pct - lo) / (hi - lo))


def score_breadth(p_pct: float, cfg: dict) -> float:
    """50일선 위 종목 비율 p%(0~100) → p ÷ 5 (item_max 20 기준 = p × item_max / 100)."""
    return cfg["item_max"] * _clip01(p_pct / 100.0)


def score_vix(v: float, cfg: dict) -> float:
    """VIX v → 0~item_max도 (high 이상 0도, low 이하 item_max도)."""
    hi, lo = float(cfg["vix"]["high"]), float(cfg["vix"]["low"])
    return cfg["item_max"] * _clip01((hi - v) / (hi - lo))


def score_fear_greed(f: float, cfg: dict) -> float:
    """공포·탐욕 f(0~100) → f ÷ 5."""
    return cfg["item_max"] * _clip01(f / 100.0)


def zone_of(temp: int, cfg: dict) -> dict:
    """정수 온도 → {"name", "desc"} (config zones에서 max 이하인 첫 구간)."""
    for z in cfg["zones"]:
        if temp <= z["max"]:
            return {"name": z["name"], "desc": z["desc"]}
    last = cfg["zones"][-1]
    return {"name": last["name"], "desc": last["desc"]}


def fear_greed_label(f: float, cfg: dict) -> str:
    """공포·탐욕 값 → 공포|중립|탐욕."""
    for b in cfg["fear_greed"]["labels"]:
        if f <= b["max"]:
            return b["label"]
    return cfg["fear_greed"]["labels"][-1]["label"]


def _sma_gap(q: pd.Series, days: int) -> float | None:
    """그날까지 자른 종가 q → 이격률 %(자료가 days일 미만이면 None)."""
    if len(q) < days:
        return None
    sma = float(q.iloc[-days:].mean())
    return (float(q.iloc[-1]) / sma - 1.0) * 100.0


def _members_on(prices: dict, ts: pd.Timestamp, columns) -> list[str]:
    m = prices.get("members")
    if m is None:
        names = set(columns)
    elif callable(m):
        names = set(m(ts))
    else:
        names = set(m)
    return sorted(names)


def _breadth(prices: dict, ts: pd.Timestamp, cfg: dict) -> dict | None:
    """그날 시장 폭 → {"pct", "strong_recovery", "coverage", "n"} 또는 None(자료 없음)."""
    stocks = prices.get("stocks")
    if stocks is None or len(stocks) == 0:
        return None
    b = cfg["breadth"]
    days, look = int(b["sma_days"]), int(b["recovery_lookback"])
    s = stocks.loc[:ts]
    if len(s) == 0 or s.index[-1] != ts:  # 그날 종목 종가가 없으면(지연) 오래된 값으로 대신하지 않는다
        return None
    members = _members_on(prices, ts, s.columns)
    if not members:
        return None
    tail = s.reindex(columns=members).iloc[-(days + look):].astype(float)
    sma = tail.rolling(days, min_periods=days).mean()
    valid = sma.notna() & tail.notna()
    above = (tail > sma) & valid
    n_valid = valid.sum(axis=1)
    coverage = float(n_valid.iloc[-1]) / len(members)
    if coverage < float(b["min_coverage"]):
        return None
    pct = above.sum(axis=1) / n_valid.where(n_valid > 0) * 100.0
    today = float(pct.iloc[-1])
    before = pct.iloc[-(look + 1):-1].dropna()
    strong = bool(len(before) and (before < float(b["recovery_low_pct"])).any() and today >= float(b["recovery_high_pct"]))
    return {"pct": today, "strong_recovery": strong, "coverage": coverage, "n": int(n_valid.iloc[-1])}


def _vix(prices: dict, ts: pd.Timestamp, cfg: dict) -> dict | None:
    """그날 VIX → {"v", "peak"(덧붙일 지난달 최고, 없으면 None)} 또는 None."""
    raw = prices.get("vix")
    if raw is None or len(raw) == 0:
        return None
    df = raw.to_frame("close") if isinstance(raw, pd.Series) else raw
    df = df.loc[:ts]
    if len(df) == 0 or df.index[-1] != ts or pd.isna(df["close"].iloc[-1]):
        return None
    v = float(df["close"].iloc[-1])
    col = "high" if "high" in df.columns else "close"
    prev = df[col].iloc[-(int(cfg["vix"]["peak_lookback"]) + 1):-1].dropna()
    peak = float(prev.max()) if len(prev) else None
    show = peak is not None and peak >= float(cfg["vix"]["peak_mark"]) and v < peak
    return {"v": v, "peak": peak if show else None}


def _fear_greed(prices: dict, ts: pd.Timestamp, cfg: dict) -> float | None:
    """그날(또는 stale_days 안의 가장 최근) 공포·탐욕 값. 없으면 None."""
    fg = prices.get("fear_greed")
    if fg is None or len(fg) == 0:
        return None
    items = fg.items() if isinstance(fg, dict) else ((str(pd.Timestamp(k).date()), v) for k, v in fg.items())
    day = ts.date().isoformat()
    cands = [(str(k)[:10], v) for k, v in items if str(k)[:10] <= day and v is not None and not pd.isna(v)]
    if not cands:
        return None
    d, v = max(cands)
    if (ts.date() - pd.Timestamp(d).date()).days > int(cfg["fear_greed"]["stale_days"]):
        return None
    return float(v)


def _compute(prices: dict, ts: pd.Timestamp, cfg: dict) -> dict:
    """ts(그날, 자료가 있는 거래일)까지 자른 자료로 온도 한 번 계산 (어제 대비는 빼고)."""
    q = prices["qqq"].loc[:ts].dropna().astype(float)
    parts: dict[str, dict] = {k: {"deg": None, "raw": None} for k in KEYS}
    gf = _sma_gap(q, int(cfg["sma"]["fast_days"]))
    gs = _sma_gap(q, int(cfg["sma"]["slow_days"]))
    if gf is not None:
        parts["sma_fast"] = {"deg": score_gap(gf, cfg), "raw": gf}
    if gs is not None:
        parts["sma_slow"] = {"deg": score_gap(gs, cfg), "raw": gs}
    br = _breadth(prices, ts, cfg)
    if br is not None:
        parts["breadth"] = {"deg": score_breadth(br["pct"], cfg), "raw": br["pct"],
                            "strong_recovery": br["strong_recovery"], "coverage": br["coverage"]}
    vx = _vix(prices, ts, cfg)
    if vx is not None:
        parts["vix"] = {"deg": score_vix(vx["v"], cfg), "raw": vx["v"], "peak": vx["peak"]}
    fg = _fear_greed(prices, ts, cfg)
    if fg is not None:
        parts["fear_greed"] = {"deg": score_fear_greed(fg, cfg), "raw": fg, "label": fear_greed_label(fg, cfg)}

    have = [k for k in KEYS if parts[k]["deg"] is not None]
    missing = [k for k in KEYS if parts[k]["deg"] is None]
    temp = zone = None
    if have:
        total = sum(parts[k]["deg"] for k in have)
        temp = round_half_up(total * 100.0 / (cfg["item_max"] * len(have)))
        zone = zone_of(temp, cfg)
    note = None
    if gf is not None and gs is not None:
        it = cfg["interpretation"]
        key = {(True, True): "both_above", (True, False): "fast_above_slow_below",
               (False, False): "both_below", (False, True): "fast_below_slow_above"}[(gf > 0, gs > 0)]
        note = it[key]
    return {"temp": temp, "zone": zone["name"] if zone else None, "zone_desc": zone["desc"] if zone else None,
            "parts": parts, "note": note, "missing": missing}


def get_temperature(date, prices: dict | None = None, cfg: dict | None = None) -> dict:
    """그날의 시장 온도.

    입력: date(YYYY-MM-DD 문자열·date·Timestamp — 과거 날짜도 됨), prices(위 모듈 설명 형식, 없으면 오늘 자료를 받음),
         cfg(config/thermometer.yaml, 없으면 읽음)
    출력: {"date", "data_date"(그날까지 QQQ 마지막 거래일), "temp"(정수 0~100|None), "zone", "zone_desc",
          "parts"{항목: {"deg"(0~20|None), "raw"(원값|None), …}}, "note"(해석 한 줄|None), "missing"[항목],
          "prev_date", "prev_temp", "delta"(어제 대비 도수|None)}
    """
    cfg = cfg or load_config()
    ts = pd.Timestamp(date).normalize()
    if prices is None:
        from thermometer.data import load_live

        prices = load_live(ts.date(), cfg)
    q = prices["qqq"].loc[:ts].dropna()
    out: dict = {"date": ts.date().isoformat(), "data_date": None, "prev_date": None, "prev_temp": None, "delta": None}
    if len(q) == 0:
        out.update({"temp": None, "zone": None, "zone_desc": None, "note": None,
                    "parts": {k: {"deg": None, "raw": None} for k in KEYS}, "missing": list(KEYS)})
        return out
    day = q.index[-1]
    out["data_date"] = day.date().isoformat()
    out.update(_compute(prices, day, cfg))
    if len(q) >= 2:
        prev_day = q.index[-2]
        prev = _compute(prices, prev_day, cfg)
        out["prev_date"] = prev_day.date().isoformat()
        out["prev_temp"] = prev["temp"]
        if prev["temp"] is not None and out["temp"] is not None:
            out["delta"] = out["temp"] - prev["temp"]
    return out
