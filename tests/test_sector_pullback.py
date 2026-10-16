"""core/sector_pullback.py · scripts/sector_pullback_data.read_guarded 테스트 (네트워크 없음, 합성 데이터)."""

import numpy as np
import pandas as pd
import pytest

from core import sector_pullback as sp


def _market(D=420, C=8, seed=1):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2011-01-03", periods=D)
    drift = np.linspace(-0.002, 0.003, C)
    close = 50 * np.exp(np.cumsum(rng.normal(drift, 0.02, size=(D, C)), axis=0))
    # 주기적인 오르내림 → MACD 돌파·눌림이 생기게
    close *= (1 + 0.15 * np.sin(np.arange(D)[:, None] / 15 + np.arange(C)))
    close[:30, 0] = np.nan  # 늦게 상장
    close[300:, 1] = np.nan  # 상장폐지
    div = np.zeros((D, C))
    div[::63, 2] = 0.3
    member = np.ones((D, C), dtype=bool)
    member[:, 7] = False
    q = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, D)))
    mk = sp.Market(days=days, companies=[f"T{j}" for j in range(C)], close=close, div=div, member=member,
                   sector=np.array([0, 0, 0, 1, 1, 1, 2, 2]), sectors=["가", "나", "다"], qclose=q, qdiv=np.zeros(D))
    sp.build_indicators(mk)
    return mk


def _truncate(mk, t):
    m = sp.Market(days=mk.days[: t + 1], companies=mk.companies, close=mk.close[: t + 1].copy(), div=mk.div[: t + 1].copy(),
                  member=mk.member[: t + 1].copy(), sector=mk.sector, sectors=mk.sectors, qclose=mk.qclose[: t + 1].copy(),
                  qdiv=mk.qdiv[: t + 1].copy())
    sp.build_indicators(m)
    return m


@pytest.mark.parametrize("t", [200, 333, 419])
def test_no_lookahead_indicators_ranks_signals_exits(t):
    full = _market()
    cut = _truncate(full, t)
    for k in ("ret126", "ret63", "sma20", "sma50", "macd"):
        np.testing.assert_allclose(cut.ind[k], full.ind[k][: t + 1], equal_nan=True)
    for v in sp.VARIANTS:
        vf, vc = sp.variant_data(full, v), sp.variant_data(cut, v)
        if vf.rank is not None:
            # 판정일 표시는 마지막 날만 다를 수 있다(잘린 쪽은 그 주가 끝났는지 모름) → 그 전날까지 같아야 함
            np.testing.assert_array_equal(vc.rank[:t], vf.rank[:t])
        ef, ec = sp.signal_events(full, vf, 0, len(full.days)), sp.signal_events(cut, vc, 0, len(cut.days))
        key = lambda es: [(e["sig"], e["c"]) for e in es if e["sig"] < t]
        assert key(ec) == key(ef)
        np.testing.assert_array_equal(sp.exit_flags(cut, vc)[:t], sp.exit_flags(full, vf)[:t])


def test_signals_exist_and_respect_rules():
    mk = _market()
    vd = sp.variant_data(mk, "B0")
    ev = sp.signal_events(mk, vd, 130, len(mk.days) - 1)
    for e in ev:
        i, j = e["sig"], e["c"]
        assert mk.member[i, j] and e["score"] > 0
        assert 1 <= vd.rank[i, mk.sector[j]] <= sp.TOP_BUY
        assert mk.ind["macd"][i, j] > 0 and mk.close[i, j] > mk.ind["sma50"][i, j]
    b2 = sp.signal_events(mk, sp.variant_data(mk, "B2"), 130, len(mk.days) - 1)
    assert len(b2) >= len(ev)
    assert all(e["c"] != 7 for e in b2)  # 대상 아님


def test_combine_shares_listing_date_and_splice():
    idx = pd.bdate_range("2014-03-31", periods=6)
    a = pd.DataFrame({"close": [10.0, 11, np.nan, 12, 12, 13], "div": 0.0}, index=idx)  # 가장 김(5행)
    b = pd.DataFrame({"close": [20.0, 22, 23, 24, 24, 26], "div": [0, 0, 1.0, 0, 0, 0]}, index=idx)
    rep, out = sp.combine_shares({"GOOG": a, "GOOGL": b}, {"GOOGL": pd.Timestamp("2014-04-03")})
    assert rep == "GOOG"
    # GOOGL은 상장일(4/3) 전을 버림 → 4/2는 어느 주식에도 값이 없어 빠진다
    assert idx[2] not in out.index and out.loc[idx[3], "close"] / out.loc[idx[1], "close"] == pytest.approx(12 / 11)
    b2 = pd.DataFrame({"close": [np.nan, 22, 23, 24, np.nan, np.nan], "div": [0, 0, 1.0, 0, 0, 0]}, index=idx)
    rep2, out2 = sp.combine_shares({"GOOG": a, "GOOGL": b2})
    assert rep2 == "GOOG"
    # 4/2(GOOG 빈 날)는 GOOGL 수익률(22→23)로 잇고, 4/3은 GOOG 자기 수익률(4/1 11 → 4/3 12)로 다시 맞춘다
    assert out2.loc[idx[2], "close"] / out2.loc[idx[1], "close"] == pytest.approx(23 / 22)
    assert out2.loc[idx[3], "close"] / out2.loc[idx[1], "close"] == pytest.approx(12 / 11)
    assert out2.loc[idx[2], "div"] == pytest.approx(1.0 / 23 * out2.loc[idx[2], "close"])  # 배당 수익률 유지
    assert out2.loc[idx[5], "close"] == pytest.approx(13.0)  # 대표 주식 마지막 가격 수준


def test_run_book_cash_conservation_without_costs():
    mk = _market()
    z = sp.Costs(one_way=0.0, tax=0.0, withhold=0.0)
    vd = sp.variant_data(mk, "B2")
    ev = sp.signal_events(mk, vd, 130, len(mk.days) - 1)
    r = sp.run_book(mk, 130, len(mk.days) - 1, z, events=ev, exits=sp.exit_flags(mk, vd))
    assert len(r["trades"]) > 0
    # 비용·세금 없으면 끝 값 = 처음 + 매매 손익 + QQQ 손익
    assert r["final"] == pytest.approx(100_000 + sum(t["pnl"] for t in r["trades"]) + r["qqq_pnl"])
    assert r["curve"][-1] == pytest.approx(r["final"], rel=1e-9)
    # 신호 없으면 QQQ 그냥 보유와 같다
    q = sp.run_qqq(mk, 130, len(mk.days) - 1, z)
    e = sp.run_book(mk, 130, len(mk.days) - 1, z, events=[], exits=None)
    assert e["final"] == pytest.approx(q["final"])


def test_run_book_delisted_position_sold_at_last_close():
    mk = _market()
    sched = [{"c": 1, "buy": 250, "sell": 380}]  # 회사 1은 300일부터 가격 없음
    r = sp.run_book(mk, 130, 419, sp.Costs(0.0, 0.0, 0.0), schedule=sched)
    t = r["trades"][0]
    assert t["sell"] == 300 and t["why"].startswith("가격 끝")
    assert t["proceeds"] == pytest.approx(t["basis"] * mk.close[299, 1] / mk.close[250, 1])


def test_tax_and_costs_reduce_final():
    mk = _market()
    vd = sp.variant_data(mk, "B2")
    ev = sp.signal_events(mk, vd, 130, 419)
    ex = sp.exit_flags(mk, vd)
    a = sp.run_book(mk, 130, 419, sp.Costs(0.0, 0.0, 0.0), events=ev, exits=ex)
    b = sp.run_book(mk, 130, 419, sp.Costs(0.003, 0.0, 0.0), events=ev, exits=ex)
    assert b["final"] < a["final"]


def test_random_schedule_keeps_dates_and_excludes_signal_company():
    mk = _market()
    trades = [{"c": 3, "buy": 200, "sell": 250}, {"c": 4, "buy": 210, "sell": 260}]
    s = sp.random_schedule(mk, trades, np.random.default_rng(0))
    assert [(x["buy"], x["sell"]) for x in s] == [(200, 250), (210, 260)]
    assert s[0]["c"] != 3 and s[1]["c"] not in (4, s[0]["c"]) and all(x["c"] != 7 for x in s)


def test_regime_and_week_end():
    q = np.concatenate([np.linspace(100, 200, 300), np.linspace(200, 150, 30)])
    r = sp.regime(q)
    assert r[100] == -1 and r[299] == 0 and r[-1] == 2
    days = pd.bdate_range("2021-12-27", periods=6)  # 월~금, 다음 월
    assert list(sp.week_end_mask(days)) == [False, False, False, False, True, True]


def test_read_guarded_stops_at_2022(tmp_path):
    from scripts.sector_pullback_data import read_guarded

    p = tmp_path / "x.csv"
    p.write_text("date,close,div\n2021-12-30,10,0\n2021-12-31,11,0.5\n2022-01-03,999,0\n2022-01-04,1000,0\n", encoding="utf-8")
    df = read_guarded(p)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2021-12-30", "2021-12-31"]
    t = tmp_path / "t.csv"
    t.write_text("date,raw_close,volume,split_factor,div_cash\n2021-12-29,100,1,1,0\n2021-12-30,50,1,2,1\n"
                 "2021-12-31,51,1,1,0\n2022-01-03,10,1,5,0\n", encoding="utf-8")
    d2 = read_guarded(t)
    assert d2["close"].tolist() == [50.0, 50.0, 51.0]  # 2022년 분할(5)은 보지 않는다
    assert d2["div"].tolist() == [0.0, 1.0, 0.0]


def test_schedule_buy_without_money_is_recorded_not_crash():
    mk = _market()
    # 같은 날 11건을 사려 하면 10건 뒤에는 총자산의 1/10씩 계속 사서 돈이 바닥난다 → 체결 불가로 기록
    sched = [{"c": j % 7, "buy": 200 + j // 7, "sell": 300} for j in range(14)]
    r = sp.run_book(mk, 130, 419, sp.Costs(0.0, 0.0, 0.0), schedule=sched)
    assert len(r["trades"]) + len(r["no_fill"]) == 14
