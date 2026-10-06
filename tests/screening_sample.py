"""합성 데이터로 "오늘의 스크리닝" 견본 보고서를 만든다 (네트워크 없음).

네 레인(1차·2차·3차·재진입)이 모두 차고, 추천·제외·대기·안 산 1차가 섞이도록 종목별
지표 줄을 손으로 정한 뒤, 실제 엔진 경로(engine.daily.simulate_since → build_report_summary
→ notify.report_html)를 그대로 지나가게 한다. 종목 이름은 화면 모양을 보여 주기 위한 것이고
숫자는 모두 지어낸 값이다.

사용법: python -m tests.screening_sample   → outputs/screening_sample.html
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core import state as st
from data.fills import FillsResult
from data.fx import FxRateResult
from data.market_calendar import trading_days_between

ROOT = Path(__file__).resolve().parents[1]
AS_OF = pd.Timestamp("2026-10-05")
N = 60
INDEX = pd.bdate_range(end=AS_OF, periods=N, name="date")
FX = 1400.0
# 견본은 네 레인을 모두 채우려고 보유 종목이 많아, 동시 보유 한도만 12로 올려 둔다(실제 config는 8).
SAMPLE_MAX_CONCURRENT = 12

NAMES = {
    "ROP": "로퍼 테크놀로지스", "ODFL": "올드 도미니언", "PAYX": "페이첵스", "WDAY": "워크데이", "IDXX": "아이덱스",
    "CPRT": "코파트", "CSX": "CSX", "MCHP": "마이크로칩", "XEL": "엑셀 에너지", "CEG": "컨스텔레이션",
    "ALAB": "아스테라 랩스", "LITE": "루멘텀", "AXON": "액슨", "AVGO": "브로드컴", "ORLY": "오라일리",
    "ADP": "ADP", "COST": "코스트코", "PEP": "펩시코", "MSFT": "마이크로소프트", "AAPL": "애플",
    "GOOGL": "알파벳 A", "NVDA": "엔비디아", "AMZN": "아마존", "META": "메타", "TSLA": "테슬라",
    "ZS": "지스케일러", "DDOG": "데이터독", "TEAM": "아틀라시안",
}


def _frame(c0: float, c1: float, *, rsi_prev: float, rsi: float, macd_norm_prev: float, macd_norm: float,
           sig_norm_prev: float, sig_norm: float, cloud=(None, None), spans=(None, None), vol=1.0, bb=0.5,
           gap_pct: float = 0.3, whipsaw: bool = False, low_dip: float | None = None,
           bottom: float | None = None) -> pd.DataFrame:
    """종가가 c0 → c1로 움직이는 60거래일 지표 표. 마지막 두 날의 RSI·MACD·시그널은 직접 정한다.

    gc·dc·macd_norm·chikou_ok·future_yang·swing_low는 값에서 계산해 표의 숫자와 신호가 서로 맞게 둔다.
    """
    if bottom is None:
        close = np.linspace(c0, c1, N)
    else:  # V자: 12거래일 전 바닥(bottom)까지 내려갔다가 c1로 반등
        close = np.concatenate([np.linspace(c0, bottom, N - 12), np.linspace(bottom, c1, 13)[1:]])
    df = pd.DataFrame(index=INDEX)
    df["close"] = close
    df["open"] = close * 0.998
    df.iloc[-1, df.columns.get_loc("open")] = close[-2] * (1 + gap_pct / 100)
    df["high"] = np.maximum(df["open"], df["close"]) * 1.006
    df["low"] = np.minimum(df["open"], df["close"]) * 0.992
    if low_dip is not None:
        df.iloc[-4, df.columns.get_loc("low")] = low_dip
    df["volume"] = 1_000_000.0
    df["rsi"] = 45.0
    df.iloc[-2, df.columns.get_loc("rsi")] = rsi_prev
    df.iloc[-1, df.columns.get_loc("rsi")] = rsi
    mn = np.full(N, macd_norm_prev - 0.3)
    sn = np.full(N, sig_norm_prev - 0.2)
    if whipsaw:  # 최근 20거래일에 교차를 여러 번 만든다
        for k in range(-16, -2, 3):
            mn[k], sn[k] = sn[k] + 0.2, sn[k]
    mn[-2], mn[-1], sn[-2], sn[-1] = macd_norm_prev, macd_norm, sig_norm_prev, sig_norm
    df["macd"] = mn * close / 100
    df["signal"] = sn * close / 100
    df["hist"] = df["macd"] - df["signal"]
    df["gc"] = (df["macd"].shift(1) <= df["signal"].shift(1)) & (df["macd"] > df["signal"])
    df["dc"] = (df["macd"].shift(1) >= df["signal"].shift(1)) & (df["macd"] < df["signal"])
    df["macd_norm"] = df["macd"] / df["close"] * 100
    df["tenkan"] = df["kijun"] = close
    top, bot = cloud
    df["cloud_top"] = top if top is not None else close * 1.08
    df["cloud_bot"] = bot if bot is not None else close * 1.04
    sa, sb = spans
    df["span_a"] = sa if sa is not None else close * 0.97
    df["span_b"] = sb if sb is not None else close * 1.01
    df["future_yang"] = df["span_a"] > df["span_b"]
    D = 26
    df["chikou_ok"] = df["close"] > df["high"].shift(D)
    df["chikou_broken"] = False
    df["vol_ratio"] = 1.0
    df.iloc[-1, df.columns.get_loc("vol_ratio")] = vol
    df["swing_low"] = df["low"].rolling(10).min()
    df["bb_width_pct"] = bb
    return df


def _down(c0, c1, rsi_prev, rsi, vol, thick_pct, **kw):
    """하락 뒤 RSI 30 돌파 종목(1차 레인용). 구름은 가격 위, 두께 thick_pct%."""
    top = c1 * 1.10
    return _frame(c0, c1, rsi_prev=rsi_prev, rsi=rsi, macd_norm_prev=-2.6, macd_norm=-2.5, sig_norm_prev=-2.2, sig_norm=-2.2,
                  cloud=(top, top - c1 * thick_pct / 100), vol=vol, **kw)


def _scout(c0, c1, *, gc: bool, rsi: float, norm: float, vol=1.0, thick=4.0, **kw):
    """1차 보유(정찰) 종목(12거래일 전 바닥 뒤 반등). gc=True면 오늘 골든크로스."""
    kw.setdefault("bottom", min(c0, c1) * 0.93)
    top = c1 * 1.06
    if gc:
        return _frame(c0, c1, rsi_prev=rsi - 3, rsi=rsi, macd_norm_prev=norm - 0.15, macd_norm=norm, sig_norm_prev=norm - 0.05,
                      sig_norm=norm - 0.08, cloud=(top, top - c1 * thick / 100), vol=vol, **kw)
    return _frame(c0, c1, rsi_prev=rsi - 1, rsi=rsi, macd_norm_prev=norm - 0.3, macd_norm=norm - 0.25, sig_norm_prev=norm, sig_norm=norm,
                  cloud=(top, top - c1 * thick / 100), vol=vol, **kw)


def _trend(c0, c1, *, gc: bool, rsi: float, norm: float, vol=1.0, thick=2.5, **kw):
    """구름 위 상승 추세 종목(재진입·3차 레인용)."""
    top = c1 * 0.95
    spans = (c1 * 0.97, c1 * 0.93)  # 앞구름 양운
    if gc:
        return _frame(c0, c1, rsi_prev=rsi - 2, rsi=rsi, macd_norm_prev=norm - 0.1, macd_norm=norm, sig_norm_prev=norm - 0.05,
                      sig_norm=norm - 0.06, cloud=(top, top - c1 * thick / 100), spans=spans, vol=vol, **kw)
    return _frame(c0, c1, rsi_prev=rsi, rsi=rsi, macd_norm_prev=norm + 0.2, macd_norm=norm + 0.25, sig_norm_prev=norm, sig_norm=norm,
                  cloud=(top, top - c1 * thick / 100), spans=spans, vol=vol, **kw)


def _bd(n: int) -> pd.Timestamp:
    """오늘에서 n거래일 전."""
    return INDEX[-1 - n]


def sample_inputs() -> dict:
    """견본용 (cfg, indicator_map, states, earnings_map, recent_events, plan) 묶음."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["risk"]["max_concurrent_positions"] = SAMPLE_MAX_CONCURRENT
    cfg["plan"]["max_slots"] = SAMPLE_MAX_CONCURRENT

    im = {
        # 1차 레인
        "ROP": _down(560, 470, 27.1, 33.0, 1.7, 2.6, low_dip=455.0),
        "ODFL": _down(210, 150, 27.8, 31.2, 1.6, 4.2),
        "PAYX": _down(150, 128, 29.0, 30.4, 1.3, 5.1),
        "WDAY": _down(270, 220, 28.0, 30.5, 0.9, 7.4),
        "IDXX": _down(640, 560, 26.5, 32.2, 1.4, 3.9),
        "CPRT": _down(58, 47, 28.4, 30.9, 1.1, 5.5),
        "CSX": _down(37, 31, 29.3, 31.0, 1.0, 6.5, bottom=28.5),  # 2차 보유 중(3차 레인에도 나옴)
        # 2차 레인 (1차 보유)
        "MCHP": _scout(70, 62, gc=True, rsi=48.2, norm=-0.3, vol=1.6, thick=2.5),
        "XEL": _scout(68, 63, gc=True, rsi=44.0, norm=-1.2, vol=1.3, thick=4.5),
        "CEG": _scout(300, 280, gc=True, rsi=71.3, norm=-0.6, vol=1.2, thick=5.0),
        "ALAB": _scout(150, 120, gc=False, rsi=41.5, norm=-1.8),
        "LITE": _scout(90, 76, gc=False, rsi=38.0, norm=-2.2),
        # 3차 레인 (2차 보유)
        "AXON": _trend(560, 640, gc=False, rsi=58.0, norm=0.6, vol=1.3, gap_pct=1.2),
        "AVGO": _trend(300, 340, gc=False, rsi=61.0, norm=0.5, gap_pct=5.2),
        # 재진입 레인 (대기 + 구름 위)
        "ORLY": _trend(90, 104, gc=True, rsi=57.4, norm=0.12, vol=1.6, thick=4.0),
        "ADP": _trend(280, 305, gc=True, rsi=49.1, norm=-0.31, vol=1.2),
        "COST": _trend(880, 960, gc=True, rsi=55.0, norm=-0.8),
        "PEP": _trend(150, 160, gc=True, rsi=55.0, norm=-0.2, whipsaw=True),
        "MSFT": _trend(480, 520, gc=False, rsi=60.0, norm=0.4),
        "AAPL": _trend(220, 240, gc=False, rsi=62.0, norm=0.5),
        "GOOGL": _trend(240, 255, gc=False, rsi=58.0, norm=0.3),
        # 신호 없음
        "NVDA": _scout(190, 185, gc=False, rsi=52.0, norm=0.1),
        "AMZN": _scout(230, 220, gc=False, rsi=47.0, norm=-0.4),
        "META": _scout(760, 720, gc=False, rsi=46.0, norm=-0.6),
        "TSLA": _scout(450, 430, gc=False, rsi=50.0, norm=-0.2),
    }

    states = {t: st.init_state(t, NAMES[t]) for t in im}

    def scout(t, a1_n, qty, price):
        df = im[t]
        a1 = _bd(a1_n)
        states[t].update(state="정찰", units={"1": qty}, entries={"1": price}, a1_date=a1, stop=float(df.loc[a1, "swing_low"]))

    def confirmed(t, a1_n, q1, q2, p, grade="S"):
        df = im[t]
        a1 = _bd(a1_n)
        states[t].update(state="확인", units={"1": q1, "2": q2}, entries={"1": p, "2": p * 1.03}, a1_date=a1,
                         stop=float(df.loc[a1, "swing_low"]), grade=grade)

    scout("MCHP", 5, 7, 63.1)
    scout("XEL", 6, 8, 62.5)
    scout("CEG", 8, 2, 276.0)
    scout("ALAB", 2, 3, 121.0)
    scout("LITE", 4, 6, 78.0)
    confirmed("AXON", 14, 1, 2, 590.0)
    confirmed("AVGO", 12, 2, 4, 320.0)
    confirmed("CSX", 7, 20, 40, 32.0)
    states["CPRT"]["cooldown_until"] = AS_OF + pd.tseries.offsets.BDay(3)

    earnings = {t: None for t in im}
    earnings["IDXX"] = AS_OF + pd.tseries.offsets.BDay(2)
    for t in ("ROP", "ODFL", "MCHP", "XEL", "AXON", "ORLY"):
        earnings[t] = AS_OF + pd.Timedelta(days=30)

    recent_events = [
        {"date": _bd(5), "ticker": "MCHP", "kind": "A1"},
        {"date": _bd(7), "ticker": "ZS", "kind": "A1"},
        {"date": _bd(6), "ticker": "ZS", "kind": "UNFILLED", "stage": "A1"},
        {"date": _bd(4), "ticker": "DDOG", "kind": "A1"},
        {"date": _bd(1), "ticker": "DDOG", "kind": "STOP"},
        {"date": _bd(3), "ticker": "TEAM", "kind": "A1"},
    ]
    plan = {"ROP": 10_000_000.0, "MCHP": 4_000_000.0, "AXON": 8_000_000.0}
    return {"cfg": cfg, "im": im, "states": states, "earnings": earnings, "recent_events": recent_events, "plan": plan}


def sample_summary() -> dict:
    """견본 입력을 실제 엔진 경로로 하루 돌려 보고서 summary를 만든다(live 모드, 실제 체결 없음)."""
    from engine.daily import build_report_summary, simulate_since

    x = sample_inputs()
    cfg, im = x["cfg"], x["im"]
    sim = simulate_since(
        im, {t: [AS_OF] for t in im}, x["states"], cfg, x["earnings"], {t: set() for t in im}, pd.DataFrame(),
        virtual_fill=False, max_concurrent=cfg["risk"]["max_concurrent_positions"],
    )
    future = list(trading_days_between((AS_OF + pd.Timedelta(days=1)).date(), (AS_OF + pd.Timedelta(days=45)).date()))
    summary = build_report_summary(
        "live", cfg, im, NAMES, x["earnings"], sim["states"], sim["today_events"], sim["as_of_by_ticker"], [],
        FillsResult(), [], cfg["risk"]["max_concurrent_positions"], FxRateResult(FX, AS_OF.date().isoformat(), False, None),
        plan_by_ticker=x["plan"], default_budget_krw=5_000_000.0,
        recent_events=x["recent_events"] + sim["today_events"], future_trading_days=future,
    )
    summary["_sim"] = sim
    return summary


def main() -> Path:
    from notify import report_html

    summary = sample_summary()
    cfg = sample_inputs()["cfg"]
    html = report_html._env.get_template("report.html.j2").render(**report_html.build_context(summary, cfg))
    banner = (
        '<div class="stale-banner" style="background:var(--warn)">견본입니다. 종목 이름은 화면 모양용이고 숫자는 모두 합성 데이터입니다 '
        f'(동시 보유 한도만 {SAMPLE_MAX_CONCURRENT}로 올림).</div>'
    )
    html = html.replace('<div class="wrap">', '<div class="wrap">\n' + banner, 1)
    out = ROOT / "outputs" / "screening_sample.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main())
