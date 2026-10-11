"""S&P 400·600 명단 복원 조사 스크립트의 순수 함수 (네트워크 없음)."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import smallmid_universe_check as sm  # noqa: E402

NPORT = """<?xml version="1.0"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
 <formData>
  <genInfo><seriesId>S000004307</seriesId><repPdDate>2021-12-31</repPdDate></genInfo>
  <invstOrSecs>
   <invstOrSec><name>United Bankshares Inc/WV</name><lei>X</lei><title>United Bankshares Inc/WV</title>
    <cusip>909907107</cusip><identifiers><isin value="US9099071071"/></identifiers><balance>10</balance>
    <units>NS</units><valUSD>100</valUSD><payoffProfile>Long</payoffProfile><assetCat>EC</assetCat>
    <issuerCat>CORP</issuerCat><invCountry>US</invCountry></invstOrSec>
   <invstOrSec><name>BlackRock Cash Funds: Institutional</name><cusip>066922477</cusip><balance>5</balance>
    <units>NS</units><valUSD>5</valUSD><payoffProfile>Long</payoffProfile><assetCat>STIV</assetCat></invstOrSec>
   <invstOrSec><name>S&amp;P Mid 400 Emini</name><balance>3</balance><units>NC</units><valUSD>1</valUSD>
    <payoffProfile>Long</payoffProfile><assetCat>DFE</assetCat></invstOrSec>
  </invstOrSecs>
 </formData>
</edgarSubmission>"""


def test_parse_nport_and_classify():
    head, df = sm.parse_nport(NPORT)
    assert head == {"series": "S000004307", "rep_date": "2021-12-31"}
    assert list(df["name"]) == ["United Bankshares Inc/WV", "BlackRock Cash Funds: Institutional", "S&P Mid 400 Emini"]
    assert df.loc[0, "isin"] == "US9099071071"
    assert [sm.classify_holding(r) for r in df.to_dict("records")] == ["주식", "현금성", "파생(선물 등)"]
    assert sm.classify_holding({"name": "Contra Abiomed CVR", "asset_cat": "EC"}) == "권리·CVR"
    assert sm.classify_holding({"name": "Foo Pref", "asset_cat": "EP"}) == "기타(EP)"


def test_parse_atom_and_index():
    atom = ("<feed><entry><content><accession-number>0001-21-1</accession-number><filing-date>2022-02-25</filing-date>"
            "<filing-href>https://x/0001-21-1-index.htm</filing-href><filing-type>NPORT-P</filing-type></content></entry></feed>")
    assert sm.parse_atom_filings(atom) == [{"form": "NPORT-P", "filed": "2022-02-25",
                                            "href": "https://x/0001-21-1-index.htm", "accession": "0001-21-1"}]
    idx = ('<div class="infoHead">Period of Report</div>\n<div class="info">2021-12-31</div>'
           '<tr><td>1</td><td></td><td><a href="/Archives/a/b.htm">b.htm</a></td><td>N-Q</td><td>123</td></tr>')
    info = sm.parse_filing_index(idx)
    assert info["period"] == "2021-12-31"
    assert info["docs"] == [("/Archives/a/b.htm", "N-Q", "123")]


def test_clean_fund_name_and_key():
    assert sm.clean_fund_name("United Bankshares Inc/WV") == "United Bankshares Inc"
    assert sm.clean_fund_name("Kroger Co/The") == "Kroger Co"
    assert sm.clean_fund_name("HFF Inc. Class A") == "HFF Inc."
    assert sm.match_key("Cato Corp. (The) Class A") == sm.match_key("CATO CORP")


def test_flatten_and_parse_nq_holdings():
    html = ("<p>Schedule of Investments (Unaudited)</p><p>iSHARES&reg; CORE S&amp;P MID-CAP ETF</p>"
            "<p>December 31, 2016</p><table><tr><td>Security</td><td>Shares</td><td>Value</td></tr>"
            "<tr><td>COMMON STOCKS &mdash; 99.77%</td></tr><tr><td>AEROSPACE &amp; DEFENSE &mdash; 2.14%</td></tr>"
            "<tr><td>B/E Aerospace Inc.</td><td>2,169,040</td><td>$</td><td>130,554,517</td></tr>"
            "<tr><td>Esterline Technologies<br>Corp.<sup>a</sup></td><td>636,698</td><td>56,793,462</td></tr>"
            "<tr><td>* eBay Inc.</td><td>10</td><td>20</td></tr>"
            "<tr><td>MarketAxess Holdings Inc.</td><td>3</td><td>4</td></tr>"
            "<tr><td>Food Products (continued)</td></tr><tr><td>Tootsie Roll Industries Inc.</td><td>5</td><td>6</td></tr>"
            "<tr><td>TOTAL COMMON STOCKS</td><td>1</td><td>2</td></tr>"
            "<tr><td>Ignored Co.</td><td>1</td><td>2</td></tr></table>"
            "<p>Schedule of Investments</p><p>iSHARES CORE S&amp;P SMALL-CAP ETF</p>"
            "<table><tr><td>COMMON STOCKS</td></tr><tr><td>Other Fund Inc.</td><td>1</td><td>2</td></tr></table>")
    lines = sm.flatten_html(html)
    pages = sm.fund_pages(lines, sm.NQ_HEAD["sp400"])
    df = sm.parse_nq_holdings(pages)
    assert list(df["name"]) == ["B/E Aerospace Inc.", "Esterline Technologies Corp.", "eBay Inc.",
                                "MarketAxess Holdings Inc.", "Tootsie Roll Industries Inc."]
    assert df.loc[0, "shares"] == "2,169,040" and df.loc[0, "value"] == "130,554,517"
    # 다른 펀드 쪽은 들어오지 않는다
    small = sm.parse_nq_holdings(sm.fund_pages(lines, sm.NQ_HEAD["sp600"]))
    assert list(small["name"]) == ["Other Fund Inc."]


def test_growth_value_fund_not_matched():
    lines = ["Vanguard S&P Mid-Cap 400 Growth Index Fund", "Schedule of Investments", "COMMON STOCKS", "A Inc.", "1", "2"]
    assert sm.fund_pages(lines, sm.NQ_HEAD["sp400"]) == []


def test_active_near_and_resolve():
    rows = pd.DataFrame({"form": ["10-K", "8-K"], "filing_date": ["2016-03-01", "2021-01-01"]})
    assert sm.active_near(rows, "2016-12-31")
    assert not sm.active_near(rows, "2021-12-31")  # 8-K만 있음
    act = {1: True, 2: False, 3: True}
    f = lambda c, d: act[c]  # noqa: E731
    assert sm.resolve(set(), "2021-12-31", f) == (None, "후보 없음")
    assert sm.resolve({1, 2}, "2021-12-31", f) == (1, "확정")
    assert sm.resolve({1, 3}, "2021-12-31", f) == (None, "모호")
    assert sm.resolve({2}, "2021-12-31", f) == (None, "활동 없음")


def test_diff_reason():
    assert sm.diff_reason("cik:1", {"cik:9"}, {"cik:1"}) == "기준일 차이(뒤 보고에 있음)"
    assert sm.diff_reason("cik:1", {"cik:1"}, None) == "기준일 차이(앞 보고에 있음)"
    assert sm.diff_reason("key:foo", None, None) == "CIK 미확정(이름 표기 차이 가능)"
    assert sm.diff_reason("cik:1", None, None) == "설명 안 됨"


def test_clean_fund_name_apostrophe_and_cos():
    assert sm.match_key("America's Car-Mart Inc/TX") == sm.match_key("AMERICAS CAR-MART INC")
    assert sm.match_key("BJ’s Restaurants Inc.") == sm.match_key("BJS RESTAURANTS INC")
    assert sm.match_key("Carlisle Cos. Inc.") == sm.match_key("CARLISLE COMPANIES INC")
    assert "companies" not in sm.match_key("Cosan Ltd")  # 'Cos'는 단어 전체만


def test_loose_key_word_order_and_initials():
    assert sm.match_key("A.O. Smith Corp.") != sm.match_key("SMITH A O CORP")
    assert sm.loose_key("A.O. Smith Corp.") == sm.loose_key("SMITH A O CORP")
    assert sm.loose_key("EW Scripps Co.") == sm.loose_key("SCRIPPS E W CO")
    assert sm.loose_key("HB Fuller Co") == sm.loose_key("FULLER H B CO")
    assert sm.match_key("F.N.B. Corp.") == sm.match_key("FNB CORP/PA/")
    assert sm.match_key("Eaton Vance Corp. NVS") == sm.match_key("EATON VANCE CORP")
    assert sm.match_key("J. Crew Group Inc.") == sm.match_key("J Crew Group Inc")  # 이니셜 하나는 그대로


def test_name_at_uses_former_name_period():
    sub = {"name": "CNX RESOURCES CORP",
           "formerNames": [{"name": "CONSOL ENERGY INC", "from": "1999-01-01T00:00:00.000Z", "to": "2017-11-28T00:00:00.000Z"}]}
    assert sm.name_at(sub, "2016-12-31") == "CONSOL ENERGY INC"
    assert sm.name_at(sub, "2019-12-31") == "CNX RESOURCES CORP"


def test_resolve_picks_by_name_then():
    active = lambda c, d: True  # noqa: E731
    same = lambda c, d: c == 2  # noqa: E731
    assert sm.resolve({1, 2}, "2016-12-31", active, same) == (2, "확정(이름 기간)")
    assert sm.resolve({1, 2}, "2016-12-31", active, lambda c, d: True) == (None, "모호")


def test_cusip_consistency_verdicts():
    rows = pd.DataFrame([
        {"cusip": "A0000000A", "period": "2020-03-31", "name": "Foo Inc", "cik": 1},
        {"cusip": "A0000000A", "period": "2020-06-30", "name": "Foo Inc", "cik": 1},
        {"cusip": "B0000000B", "period": "2020-03-31", "name": "Bar Inc", "cik": 2},
        {"cusip": "B0000000B", "period": "2020-06-30", "name": "Bar Inc", "cik": None},
        {"cusip": "C0000000C", "period": "2020-03-31", "name": "Baz Inc", "cik": 3},
        {"cusip": "C0000000C", "period": "2020-06-30", "name": "Baz Inc", "cik": 4},
        {"cusip": "D0000000D", "period": "2020-03-31", "name": "Old Name Inc", "cik": 5},
        {"cusip": "D0000000D", "period": "2020-06-30", "name": "New Name Inc", "cik": 6},
        {"cusip": "E0000000E", "period": "2020-03-31", "name": "Qux Inc", "cik": None},
    ])
    v = dict(zip(*sm.cusip_consistency(rows)[["cusip", "verdict"]].T.values))
    assert v == {"A0000000A": "일치", "B0000000B": "일부 기간 미확정", "C0000000C": "같은 이름 다른 CIK(오매칭 의심)",
                 "D0000000D": "이름·CIK 함께 바뀜(재편·합병 또는 오매칭)", "E0000000E": "전 기간 미확정"}
    # 자리표시 CUSIP(외국 법인 등)은 검사에서 뺀다
    ph = pd.DataFrame([{"cusip": "000000000", "period": "2020-03-31", "name": "X NV", "cik": 7},
                       {"cusip": "000000000", "period": "2020-03-31", "name": "Y PLC", "cik": 8}])
    assert sm.cusip_consistency(ph).empty


def test_presence_runs_and_intervals():
    assert sm.presence_runs([True, False, True, False, False, True]) == [(0, 2, 1), (5, 5, 0)]
    dates = ["2020-02-29", "2020-03-31", "2020-05-31", "2020-06-30"]
    iv = sm.intervals_for(dates, [False, True, True, False])
    assert iv == [{"entry_earliest": "2020-03-01", "entry_latest": "2020-03-31", "exit_earliest": "2020-05-31",
                   "exit_latest": "2020-06-29", "n_reports": 2, "gaps_filled": 0}]
    iv = sm.intervals_for(dates, [True, True, True, True])
    assert iv[0]["entry_earliest"] is None and iv[0]["exit_latest"] is None


def test_gap_reason():
    assert sm.gap_reason("Signature Bank/New York NY", "활동 없음", False).startswith("은행")
    assert sm.gap_reason("Core Laboratories NV", "활동 없음", True).startswith("외국")
    assert sm.gap_reason("J. Crew Group Inc. Escrow", "후보 없음", False).startswith("주식 아님")
    assert sm.gap_reason("Foo Inc", "모호", False).startswith("같은 이름")
