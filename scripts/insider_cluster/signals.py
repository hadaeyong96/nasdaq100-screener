"""내부자 집단 매수 — 공시 → 신호 (docs/prereg/insider_cluster.md 3절). 순수 함수, 가격을 쓰지 않는다.

입력 표:
  filings: accession, filing_date(Timestamp), doc_type("4"/"4/A"), issuer_cik(int), period(str)
  owners: accession, owner_cik(int), relationship(str)
  trades: accession, trans_date(Timestamp), code, ad, shares(float), price(float)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday)

from scripts.insider_cluster.gate import CUTOFF, assert_dates

WINDOW_TRADING_DAYS = 30
MIN_PEOPLE = 3
SIGNAL_FIRST = pd.Timestamp("2017-01-01")
EXTRA_CLOSED = [pd.Timestamp("2018-12-05")]  # 조지 H. W. 부시 추모일


class _NyseCalendar(AbstractHolidayCalendar):
    rules = [Holiday("NewYear", month=1, day=1, observance=sunday_to_monday), USMartinLutherKingJr, USPresidentsDay, GoodFriday,
             USMemorialDay, Holiday("July4", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
             Holiday("Christmas", month=12, day=25, observance=nearest_workday)]


def nyse_days(start: str = "2015-01-01", end: str = "2021-12-31") -> pd.DatetimeIndex:
    """NYSE 거래일(2015~2021 규칙). 출력: 오름차순 DatetimeIndex. 끝은 2021-12-31 이하로 자른다."""
    stop = min(pd.Timestamp(end), CUTOFF)
    hol = _NyseCalendar().holidays(start=start, end=stop)
    days = pd.bdate_range(start, stop)
    return days[~days.isin(hol) & ~days.isin(EXTRA_CLOSED)]


def is_insider(relationship: str) -> bool:
    """임원 또는 이사면 True. 10% 대주주·기타만이면 False."""
    r = str(relationship)
    return "Director" in r or "Officer" in r


def eligible_purchases(filings: pd.DataFrame, owners: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """코드 P·취득·Form 4(/A)·거래 날짜 ≤ 접수일 거래만 남기고 해당 내부자 묶음을 붙인다.

    출력: (표 — accession, filing_date, doc_type, issuer_cik, period, insiders(CIK 튜플), trans_date, amount; 4/A는 거래가
    없어도 trans_date NaT 행 1개로 남는다, 건수 dict).
    """
    assert_dates(filings["filing_date"])
    f = filings[filings["doc_type"].isin(["4", "4/A"])]
    own = owners[owners["relationship"].map(is_insider)]
    ins = own.groupby("accession")["owner_cik"].apply(lambda s: tuple(sorted(set(int(x) for x in s))))
    f = f[f["accession"].isin(ins.index)].copy()
    f["insiders"] = f["accession"].map(ins)
    p = trades[(trades["code"] == "P") & (trades["ad"] == "A")].merge(f, on="accession", how="inner")
    bad = p["trans_date"].isna() | (p["trans_date"] > p["filing_date"])
    stats = {"p_rows": int(len(p)), "bad_trans_date": int(bad.sum())}
    p = p[~bad].copy()
    p["amount"] = p["shares"].fillna(0.0) * p["price"].fillna(0.0)
    has_rows = set(p["accession"])
    amend = f[(f["doc_type"] == "4/A") & ~f["accession"].isin(has_rows)].copy()  # 매수가 빠진 정정: 앞선 매수를 지운다
    amend["trans_date"], amend["amount"] = pd.NaT, 0.0
    cols = ["accession", "filing_date", "doc_type", "issuer_cik", "period", "insiders", "trans_date", "amount"]
    return pd.concat([p[cols], amend[cols]], ignore_index=True), stats


class _People:
    """회사 안 내부자 묶음(공동 신고는 1명)."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, ids: tuple) -> None:
        for y in ids[1:]:
            self.parent[self.find(y)] = self.find(ids[0])


def _find_cluster(rows: list[dict], people: _People, min_people: int, window: int) -> dict | None:
    """알려진 매수 중 거래일 창 안에 min_people명 이상이 있으면 그 창 요약. 없으면 None."""
    rows = sorted(rows, key=lambda r: r["pos"])
    for i in range(len(rows)):
        names, amount, last = set(), 0.0, i
        for j in range(i, len(rows)):
            if rows[j]["pos"] - rows[i]["pos"] > window - 1:
                break
            names.add(people.find(rows[j]["insiders"][0]))
            amount += rows[j]["amount"]
            last = j
        if len(names) >= min_people:
            return {"people": len(names), "amount_usd": amount, "first_trade": rows[i]["trans_date"], "last_trade": rows[last]["trans_date"]}
    return None


def cluster_signals(purchases: pd.DataFrame, days: pd.DatetimeIndex, min_people: int = MIN_PEOPLE,
                    window: int = WINDOW_TRADING_DAYS, first: pd.Timestamp = SIGNAL_FIRST) -> pd.DataFrame:
    """회사별 접수일 순서로 처리해 집단 매수 신호를 만든다.

    입력: eligible_purchases 표, 거래일 달력, 기준 인원, 창(거래일), 신호 시작(앞선 신호는 소진만).
    출력: 표 — cik, confirm(신호 확인일 = 접수일), people, amount_usd, first_trade, last_trade. confirm, cik 오름차순.
    """
    assert_dates(purchases["filing_date"])
    out = []
    pos = np.searchsorted(days.values, purchases["trans_date"].values)
    pur = purchases.assign(pos=pos)
    for cik, g in pur.groupby("issuer_cik", sort=True):
        active: dict[tuple, list[dict]] = {}
        consumed: set[tuple] = set()
        people = _People()
        for date, day in g.groupby("filing_date", sort=True):
            for acc, one in day.groupby("accession", sort=True):
                r0 = one.iloc[0]
                key = (r0["insiders"], r0["period"])
                people.join(r0["insiders"])
                if r0["doc_type"] == "4/A":
                    active.pop(key, None)
                rows = [{"insiders": r.insiders, "trans_date": r.trans_date, "pos": int(r.pos), "amount": float(r.amount)}
                        for r in one.itertuples() if pd.notna(r.trans_date) and (r.insiders, r.trans_date) not in consumed]
                if rows:
                    active.setdefault(key, []).extend(rows)
            known = [r for rows in active.values() for r in rows]
            hit = _find_cluster(known, people, min_people, window) if len(known) >= min_people else None
            if hit:
                if date >= first:
                    out.append({"cik": int(cik), "confirm": date, **hit})
                consumed.update((r["insiders"], r["trans_date"]) for r in known)
                active.clear()
    cols = ["cik", "confirm", "people", "amount_usd", "first_trade", "last_trade"]
    return pd.DataFrame(out, columns=cols).sort_values(["confirm", "cik"]).reset_index(drop=True)


def use_and_exec(confirm: pd.Timestamp, days: pd.DatetimeIndex) -> tuple[int | None, int | None]:
    """접수일 → (신호 사용 거래일 위치 t+1, 체결 거래일 위치 t+2). 달력을 넘으면 None."""
    use = int(np.searchsorted(days.values, np.datetime64(confirm), side="right"))
    ex = use + 1
    return (use if use < len(days) else None), (ex if ex < len(days) else None)
