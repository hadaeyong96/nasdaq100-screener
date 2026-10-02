"""DB에서 다시 읽은 a1_date(문자열)로 A2 매수 행을 만들 때 깨지지 않는지 확인한다 (회귀 방지).

발견된 버그: store/db.py의 _deserialize는 날짜 칸(a1_date 등)을 저장된 문자열
그대로 돌려준다. 그런데 engine/daily.py의 _build_buy_row A2 분기가
states_after["a1_date"].date()를 불러, A1을 지난 실행에서 받고 오늘 A2가 난
종목에서 AttributeError: 'str' object has no attribute 'date'로 실행 전체가
실패했다 (2026-10-02 paper 수동 실행).

수정: pd.Timestamp(...)로 감싸 문자열·Timestamp 둘 다 처리한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine import daily as engine_daily
from store import db


def _indicator_df(end_date: str = "2026-10-02", periods: int = 40) -> pd.DataFrame:
    """_build_buy_row가 읽는 지표 열만 갖춘 합성 DataFrame."""
    index = pd.bdate_range(end=end_date, periods=periods, name="date")
    close = 100 + np.linspace(0, 5, periods)
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "close": close,
            "rsi": np.full(periods, 45.0),
            "macd_norm": np.full(periods, 0.3),
            "vol_ratio": np.full(periods, 1.2),
            "cloud_top": close - 1,
            "cloud_bot": close - 3,
            "bb_width_pct": np.full(periods, 0.05),
        },
        index=index,
    )


def _scout_state(a1_date) -> dict:
    return {
        "ticker": "TEST", "name_kr": "테스트", "state": "정찰",
        "units": {"1": 10}, "entries": {"1": 100.0}, "stop": 95.0,
        "a1_date": a1_date, "cooldown_until": None, "sent_alerts": [],
        "updated_at": pd.Timestamp("2026-09-25"), "b_entry_date": None, "b_total_qty": None,
        "pending": None,
    }


def _a2_row(states_after: dict, df: pd.DataFrame, cfg: dict) -> dict:
    as_of = df.index[-1]
    event = {"date": as_of, "ticker": "TEST", "kind": "A2", "unit": "2", "price": float(df.loc[as_of, "close"]), "score": 0}
    return engine_daily._build_buy_row(event, df, states_after, cfg, {"TEST": "테스트"}, {"TEST": None})


def test_a2_row_after_db_roundtrip_formats_a1_date(tmp_path, cfg):
    """A1(Timestamp)으로 저장 → load_all_positions(문자열로 돌아옴) → A2 행이 오류 없이 날짜를 낸다."""
    conn = db.connect(tmp_path / "paper_state.db")
    db.save_position(conn, _scout_state(pd.Timestamp("2026-09-25")))
    loaded = db.load_all_positions(conn)["TEST"]
    conn.close()

    assert isinstance(loaded["a1_date"], str)  # 버그 전제: DB는 문자열로 돌려준다
    row = _a2_row(loaded, _indicator_df(), cfg)
    assert row["a1_date"] == "2026-09-25"


@pytest.mark.parametrize("a1_date", [pd.Timestamp("2026-09-25"), "2026-09-25", "2026-09-25 00:00:00"])
def test_a2_row_accepts_timestamp_and_string_a1_date(a1_date, cfg):
    """같은 실행 안(Timestamp)과 DB 경유(문자열) 모두 같은 "YYYY-MM-DD"를 낸다."""
    row = _a2_row(_scout_state(a1_date), _indicator_df(), cfg)
    assert row["a1_date"] == "2026-09-25"


def test_a2_row_without_a1_date_is_blank(cfg):
    row = _a2_row(_scout_state(None), _indicator_df(), cfg)
    assert row["a1_date"] == ""
