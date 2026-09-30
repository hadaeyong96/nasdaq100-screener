"""scripts/moat_h4_backtest.py의 순수·저수준 함수 테스트 — 네트워크 없이 돈다.

SEC EDGAR·yfinance 요청이 들어가는 main()은 여기서 테스트하지 않는다(통합 테스트가
아니라, 가격 계산·날짜 계산·인수·상장폐지 대체 로직만 합성 데이터로 검증한다).
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import moat_h4_backtest as h4

COSTS_CFG = {"slippage_pct": 0.05, "commission_buy_pct": 0.07, "commission_sell_pct": 0.07}
DIV_WH_RATE = 0.15


def _df(dates: list[str], **cols) -> pd.DataFrame:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame(cols, index=idx)


def test_first_trading_day_of_month_picks_earliest():
    days = [pd.Timestamp("2019-03-29"), pd.Timestamp("2019-04-01"), pd.Timestamp("2019-04-02")]
    assert h4.first_trading_day_of_month(days, 2019, 4) == pd.Timestamp("2019-04-01")


def test_first_trading_day_of_month_raises_when_absent():
    days = [pd.Timestamp("2019-03-29")]
    with pytest.raises(ValueError):
        h4.first_trading_day_of_month(days, 2019, 4)


def test_next_trading_day_after_finds_first_strictly_later():
    days = [pd.Timestamp("2019-04-01"), pd.Timestamp("2019-04-02"), pd.Timestamp("2019-04-03")]
    assert h4.next_trading_day_after(days, pd.Timestamp("2019-04-01")) == pd.Timestamp("2019-04-02")


def test_next_trading_day_after_raises_when_no_later_day():
    days = [pd.Timestamp("2019-04-01")]
    with pytest.raises(ValueError):
        h4.next_trading_day_after(days, pd.Timestamp("2019-04-01"))


def test_stock_leg_return_normal_case_no_note():
    entry, exit_ = pd.Timestamp("2019-04-02"), pd.Timestamp("2020-04-01")
    df = _df(["2019-04-02", "2020-04-01"], open=[100.0, 150.0], close=[99.0, 148.0])
    price_map = {"AAA": df}
    div_map = {"AAA": pd.Series(dtype=float)}
    result, note = h4.stock_leg_return("AAA", entry, exit_, False, price_map, div_map, df, pd.Series(dtype=float), COSTS_CFG, DIV_WH_RATE)
    assert note is None
    pf, dy = result
    buy_price = 100.0 * 1.0005 * 1.0007
    sell_price = 150.0 * 0.9995 * 0.9993
    assert pf == pytest.approx(sell_price / buy_price)
    assert dy == pytest.approx(0.0)


def test_stock_leg_return_missing_entry_price_returns_none():
    df = _df(["2020-04-01"], open=[150.0], close=[148.0])
    result, note = h4.stock_leg_return("AAA", pd.Timestamp("2019-04-02"), pd.Timestamp("2020-04-01"), False, {"AAA": df}, {}, df, pd.Series(dtype=float), COSTS_CFG, DIV_WH_RATE)
    assert result is None
    assert note is not None


def test_stock_leg_return_final_mark_uses_close_without_sell_costs():
    entry, exit_ = pd.Timestamp("2021-04-02"), pd.Timestamp("2021-12-31")
    df = _df(["2021-04-02", "2021-12-31"], open=[100.0, 200.0], close=[99.0, 210.0])
    result, note = h4.stock_leg_return("AAA", entry, exit_, True, {"AAA": df}, {}, df, pd.Series(dtype=float), COSTS_CFG, DIV_WH_RATE)
    assert note is None
    pf, _ = result
    buy_price = 100.0 * 1.0005 * 1.0007
    assert pf == pytest.approx(210.0 / buy_price)  # 매도비용 없음(평가일 뿐)


def test_stock_leg_return_delisted_mid_period_falls_back_to_qqqm():
    entry, exit_ = pd.Timestamp("2019-04-02"), pd.Timestamp("2020-04-01")
    # AAA는 2019-09-03에 데이터가 끝난다(인수·상장폐지로 본다)
    stock_df = _df(["2019-04-02", "2019-09-03"], open=[100.0, 60.0], close=[99.0, 58.0])
    cash_df = _df(["2019-04-02", "2019-09-03", "2019-09-04", "2020-04-01"], open=[300.0, 310.0, 305.0, 330.0], close=[299.0, 309.0, 304.0, 329.0])
    result, note = h4.stock_leg_return("AAA", entry, exit_, False, {"AAA": stock_df}, {}, cash_df, pd.Series(dtype=float), COSTS_CFG, DIV_WH_RATE)
    assert result is not None
    assert "상장폐지" in note
    pf, dy = result
    buy_price = 100.0 * 1.0005 * 1.0007
    sell_at_delist = 58.0 * 0.9995 * 0.9993
    pf1 = sell_at_delist / buy_price
    qqqm_buy = 305.0 * 1.0005 * 1.0007  # 2019-09-04 시가로 QQQM 편입
    qqqm_sell = 330.0 * 0.9995 * 0.9993  # 2020-04-01 시가로 정상 매도
    pf2 = qqqm_sell / qqqm_buy
    assert pf == pytest.approx(pf1 * pf2)
    assert dy == pytest.approx(0.0)


def test_stock_leg_return_delisted_with_no_qqqm_data_after_uses_only_delist_leg():
    entry, exit_ = pd.Timestamp("2019-04-02"), pd.Timestamp("2020-04-01")
    stock_df = _df(["2019-04-02", "2019-09-03"], open=[100.0, 60.0], close=[99.0, 58.0])
    cash_df = _df(["2019-04-02", "2019-09-03"], open=[300.0, 310.0], close=[299.0, 309.0])  # 상폐 이후 QQQM 데이터도 없음
    result, note = h4.stock_leg_return("AAA", entry, exit_, False, {"AAA": stock_df}, {}, cash_df, pd.Series(dtype=float), COSTS_CFG, DIV_WH_RATE)
    assert result is not None
    assert "QQQM 편입 실패" in note
    pf, _ = result
    buy_price = 100.0 * 1.0005 * 1.0007
    sell_at_delist = 58.0 * 0.9995 * 0.9993
    assert pf == pytest.approx(sell_at_delist / buy_price)


def test_build_group_periods_excludes_unpriceable_tickers_and_reweights():
    stock_returns_by_year = {y: {"A": (1.1, 0.0), "B": (1.2, 0.0)} for y in h4.YEARS}
    sectors_by_year = {y: {"A": "Tech", "B": "Tech", "C": "Tech"} for y in h4.YEARS}
    tickers_by_year = {y: ["A", "B", "C"] for y in h4.YEARS}  # C는 가격이 없어 returns에 없음
    execution_dates = {y: pd.Timestamp(f"{y}-04-02") for y in h4.YEARS}
    final_mark_date = pd.Timestamp("2021-12-31")
    fx_by_date = {execution_dates[y].date().isoformat(): 1200.0 for y in h4.YEARS}
    fx_by_date[final_mark_date.date().isoformat()] = 1200.0

    periods, triggered = h4.build_group_periods(tickers_by_year, stock_returns_by_year, sectors_by_year, execution_dates, final_mark_date, fx_by_date, 30)
    assert len(periods) == len(h4.YEARS)
    # C가 빠졌으니 동일비중(0.5,0.5)이고 업종캡(같은 업종 100%>>30%)이 걸려도 재배분할 곳이
    # 없어(전부 Tech) 그대로 0.5/0.5 -> 캡이 "걸리지 않음"으로 봐야 함(전부 한 업종이라 불가능)
    assert triggered[h4.YEARS[0]] is False
    assert periods[0].price_return_factor == pytest.approx(0.5 * 1.1 + 0.5 * 1.2)


def test_build_group_periods_sector_cap_triggers_when_redistribution_possible():
    # A·B·C가 다 Tech(동일비중이면 75%, 캡 30% 초과), D만 Health(25%, 캡 아래) — Health로
    # 초과분을 받을 여유가 있어 실제로 재배분이 일어나야 하는 경우.
    stock_returns_by_year = {y: {"A": (1.1, 0.0), "B": (1.2, 0.0), "C": (1.3, 0.0), "D": (1.0, 0.0)} for y in h4.YEARS}
    sectors_by_year = {y: {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Health"} for y in h4.YEARS}
    tickers_by_year = {y: ["A", "B", "C", "D"] for y in h4.YEARS}
    execution_dates = {y: pd.Timestamp(f"{y}-04-02") for y in h4.YEARS}
    final_mark_date = pd.Timestamp("2021-12-31")
    fx_by_date = {execution_dates[y].date().isoformat(): 1200.0 for y in h4.YEARS}
    fx_by_date[final_mark_date.date().isoformat()] = 1200.0

    periods, triggered = h4.build_group_periods(tickers_by_year, stock_returns_by_year, sectors_by_year, execution_dates, final_mark_date, fx_by_date, 30)
    assert triggered[h4.YEARS[0]] is True


def test_qqqm_daily_shape_rows_relative_to_window_start():
    df = _df(["2019-04-02", "2019-04-03", "2019-04-04"], close=[100.0, 90.0, 110.0])
    fx_by_date = {d: 1200.0 for d in ("2019-04-02", "2019-04-03", "2019-04-04")}
    rows = h4.qqqm_daily_shape_rows(df, pd.Timestamp("2019-04-02"), pd.Timestamp("2019-04-04"), fx_by_date)
    assert [r["total_krw"] for r in rows] == pytest.approx([1200.0, 1080.0, 1320.0])
