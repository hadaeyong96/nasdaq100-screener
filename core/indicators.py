"""정리본 9장(지표 계산식과 미래 데이터 방지)의 지표를 계산하는 순수 함수 모음.

- 네트워크·파일·DB·현재 시각에 접근하지 않는다. DataFrame과 설정값만 받는다.
- 같은 입력이면 항상 같은 출력을 낸다 (라이브 실행·백테스트 공용).
- 미래 데이터 금지: `shift(-n)` 같은 음수 이동을 쓰지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """정리본 9장의 모든 지표 열을 추가한다.

    입력:
        df — DataFrame(open, high, low, close, volume), 날짜 오름차순 인덱스
        cfg — config.yaml 로드값 (cfg["indicators"] 아래에서 설정을 읽는다)
    출력:
        df에 아래 열을 추가한 새 DataFrame:
        macd, signal, hist, gc, dc, macd_norm,
        rsi,
        tenkan, kijun, span_a, span_b, cloud_top, cloud_bot,
        future_yang, chikou_ok, chikou_broken,
        vol_ratio, swing_low, bb_width_pct,
        bars, insufficient_history
    """
    ind_cfg = cfg["indicators"]
    out = df.copy()

    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    # ── MACD (12, 26, 9) ────────────────────────────────────────────
    macd_cfg = ind_cfg["macd"]
    ema_fast = close.ewm(span=macd_cfg["fast"], adjust=False).mean()
    ema_slow = close.ewm(span=macd_cfg["slow"], adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_cfg["signal"], adjust=False).mean()

    out["macd"] = macd
    out["signal"] = signal
    out["hist"] = macd - signal
    out["gc"] = (macd.shift(1) <= signal.shift(1)) & (macd > signal)
    out["dc"] = (macd.shift(1) >= signal.shift(1)) & (macd < signal)
    out["macd_norm"] = macd / close * 100

    # ── RSI (Wilder, 14) ────────────────────────────────────────────
    rsi_period = ind_cfg["rsi"]["period"]
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.where(loss != 0, 100.0)  # loss가 0이면 100으로 처리
    out["rsi"] = rsi

    # ── 일목균형표 (전환선 9, 기준선 26, 선행스팬 B 52) ────────────────
    ichi_cfg = ind_cfg["ichimoku"]
    D = ind_cfg["ichimoku_shift"]

    tenkan = (high.rolling(ichi_cfg["tenkan"]).max() + low.rolling(ichi_cfg["tenkan"]).min()) / 2
    kijun = (high.rolling(ichi_cfg["kijun"]).max() + low.rolling(ichi_cfg["kijun"]).min()) / 2
    span_a = (tenkan + kijun) / 2  # 오늘 계산 = 앞구름
    span_b = (high.rolling(ichi_cfg["senkou_b"]).max() + low.rolling(ichi_cfg["senkou_b"]).min()) / 2

    out["tenkan"] = tenkan
    out["kijun"] = kijun
    out["span_a"] = span_a
    out["span_b"] = span_b
    out["cloud_top"] = np.maximum(span_a, span_b).shift(D)  # 오늘 가격 아래의 구름
    out["cloud_bot"] = np.minimum(span_a, span_b).shift(D)
    out["future_yang"] = span_a > span_b  # 앞구름 양운
    out["chikou_ok"] = close > high.shift(D)  # 후행스팬 돌파
    out["chikou_broken"] = close < span_b.shift(2 * D)  # 후행스팬 < 선행 B (E3)

    # ── 거래량 · 스윙 저점 · 볼린저 밴드폭 ──────────────────────────────
    vol_ma_period = ind_cfg["volume_ma"]["period"]
    out["vol_ratio"] = volume / volume.rolling(vol_ma_period).mean()

    swing_period = ind_cfg["swing_low"]["period"]
    out["swing_low"] = low.rolling(swing_period).min()

    bb_cfg = ind_cfg["bollinger"]
    bb_mid = close.rolling(bb_cfg["period"]).mean()
    bb_sd = close.rolling(bb_cfg["period"]).std()
    bb_width = bb_cfg["std_mult"] * 2 * bb_sd / bb_mid
    out["bb_width_pct"] = bb_width.rolling(bb_cfg["percentile_period"]).rank(pct=True)

    # ── 보유 거래일 수 · 데이터 부족 플래그 (P1.1) ──────────────────────
    # bars: 이 종목이 시작일부터 해당 날짜까지 보유한 거래일 수(누적 봉 개수).
    # 상장 기간이 짧은 종목(예: 최근 IPO)은 마지막 행의 bars 값이 곧
    # "보유 거래일 수"가 되어, 지표별로 계산 가능한지 바로 확인할 수 있다.
    out["bars"] = np.arange(1, len(out) + 1)

    # insufficient_history: 가장 긴 lookback이 필요한 지표(일목 구름 — 52+D일,
    # 볼린저 백분위 — bollinger.period + percentile_period일)가 아직 NaN이면 True.
    # 상장 기간이 짧아 이 지표들을 아직 계산할 수 없다는 뜻이다.
    long_lookback_cols = ["cloud_top", "cloud_bot", "bb_width_pct"]
    out["insufficient_history"] = out[long_lookback_cols].isna().any(axis=1)

    return out


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR(Average True Range) — 순수 함수 (P5-3 E1 실험용, compute_indicators엔 없음).

    입력: df(open/high/low/close), period(기본 14)
    출력: True Range를 alpha=1/period로 지수평활한 Series (df와 같은 인덱스)
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """OBV(On-Balance Volume) — 순수 함수 (P5-3 C3 실험용).

    입력: df(close, volume)
    출력: 종가가 전일보다 오르면 +거래량, 내리면 -거래량을 누적한 Series
    """
    close, volume = df["close"], df["volume"]
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def last_cross_date(series_bool: pd.Series) -> pd.Timestamp | None:
    """불리언 시리즈(gc 또는 dc)에서 가장 최근 True의 날짜(인덱스)를 구한다.

    입력: DatetimeIndex를 가진 bool Series
    출력: 가장 최근 True의 인덱스 값, True가 하나도 없으면 None
    """
    true_index = series_bool.index[series_bool.fillna(False)]
    if len(true_index) == 0:
        return None
    return true_index[-1]
