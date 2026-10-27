"""테스트 공용 픽스처. 네트워크 없이 합성 데이터로 돈다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RealTelegramSendBlocked(RuntimeError):
    """테스트 중 api.telegram.org로 실제 요청을 보내려 할 때 (2026-10-17 — 테스트가 진짜 알림을 보낸 사고 뒤)."""


@pytest.fixture(autouse=True)
def _no_real_telegram(monkeypatch, tmp_path):
    """모든 테스트: GitHub Actions가 아닌 것으로 보고(GITHUB_ACTIONS 지움 → 공장 알림 dry-run),
    requests.post로 api.telegram.org에 가는 요청은 막는다(.env가 import 때 읽혀 진짜 토큰이 있어도 안전).
    테스트가 requests.post를 따로 바꿔 끼우면 그쪽이 이긴다."""
    import requests

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    real_post = requests.post

    def guarded(url, *a, **k):
        if "api.telegram.org" in str(url):
            raise RealTelegramSendBlocked("테스트 중 텔레그램 실제 발송 금지")
        return real_post(url, *a, **k)

    monkeypatch.setattr(requests, "post", guarded)
    # 공장 알림: 재시도 대기 0, 실패 보관은 임시 폴더(진짜 outputs/에 남지 않게)
    from factory import notify as fnotify

    monkeypatch.setattr(fnotify, "_sleep", lambda s: None)
    monkeypatch.setattr(fnotify, "PENDING_DIR", tmp_path / "notify_pending")
    # 10번 동전주 장부(2026-10-21): 목록·가격 받기는 네트워크라 막는다 — 주입하지 않은 테스트에선 "목록 못 받음"(선정 미룸)으로 돈다
    from data import penny_universe as pu

    def _no_net(*a, **k):
        raise pu.PennyListError("테스트 중 네트워크 없음")

    monkeypatch.setattr(pu, "fetch_lists", _no_net)
    monkeypatch.setattr(pu, "download_batch", _no_net)
    monkeypatch.setattr(pu, "history", _no_net)
    # 웅덩이 v3.1 가상 장부(2026-10-23): 구성 종목·가격·EDGAR는 네트워크라 막는다 — 주입하지 않은 테스트에선 장부만 오류로 건너뛴다
    from engine import puddle_v31 as pv31

    def _no_net_pv(*a, **k):
        raise RuntimeError("테스트 중 네트워크 없음(웅덩이 v3.1)")

    monkeypatch.setattr(pv31, "_default_universe", _no_net_pv)
    monkeypatch.setattr(pv31, "_default_history", _no_net_pv)
    monkeypatch.setattr(pv31, "_default_shares", _no_net_pv)  # 웅덩이 v3.2·반등 확인 v2(2026-10-27) 시가총액용 주식 수
    monkeypatch.setattr(pv31.FilingChecker, "_edgar", _no_net_pv)  # 웅덩이 반등 확인(2026-10-24)도 같은 World·FilingChecker를 쓴다
    # 아침 편지 v3(2026-10-24): 뉴스 RSS·S&P500·SOXX 종가는 네트워크라 막는다 — 주입하지 않은 테스트에선 뉴스 실패·지수 줄 없음
    from data import news as dnews
    from engine import daily as _daily

    def _no_net_news(*a, **k):
        raise ConnectionError("테스트 중 네트워크 없음(뉴스)")

    monkeypatch.setattr(dnews, "_get", _no_net_news)
    monkeypatch.setattr(dnews, "PAUSE", 0)
    monkeypatch.setattr(_daily, "_fetch_index_closes", lambda cfg: {})


@pytest.fixture(scope="session")
def cfg() -> dict:
    """config.yaml을 그대로 읽어 반환한다."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_synthetic_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """네트워크 없이 쓰는 합성 OHLCV 데이터를 만든다.

    입력: 생성할 행 수, 난수 시드
    출력: DataFrame(open, high, low, close, volume), 영업일 오름차순 인덱스
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.05, scale=1.2, size=n)
    close = 100 + np.cumsum(steps)
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0.1, 1.5, size=n)
    low = close - rng.uniform(0.1, 1.5, size=n)
    low = np.minimum(low, close - 0.01)
    open_ = low + rng.uniform(0, 1, size=n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    index = pd.bdate_range("2020-01-02", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
