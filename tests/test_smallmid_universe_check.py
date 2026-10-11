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


def test_alive_and_ending():
    rows = pd.DataFrame({"form": ["10-Q", "10-K", "8-K"], "filing_date": ["2020-11-06", "2020-03-01", "2021-05-01"]})
    assert sm.alive_at(rows, "2020-09-30")          # 2020-11-06 10-Q가 뒤 200일 안
    assert not sm.alive_at(rows, "2021-02-28")      # 뒤에는 8-K뿐 (옛 Dime 같은 경우)
    assert sm.ending_at(rows, "2021-02-28")         # 마지막 10-Q가 114일 전
    assert not sm.ending_at(rows, "2021-03-31")     # 145일 전 → 인수 직전 아님
    assert not sm.ending_at(rows, "2020-09-30")     # 뒤에 10-Q가 있으면 인수 직전 아님


def test_resolve_prefers_alive_then_ending():
    alive = lambda c, d: c == 2  # noqa: E731
    ending = lambda c, d: c == 1  # noqa: E731
    assert sm.resolve({1, 2}, "2021-02-28", alive, None, ending) == (2, "확정")
    assert sm.resolve({1}, "2021-02-28", lambda c, d: False, None, ending) == (1, "확정(인수 직전)")
    assert sm.resolve({1}, "2021-02-28", lambda c, d: False) == (None, "활동 없음")


def test_override_for():
    ov = pd.DataFrame({"name_regex": [r"First BanCorp(?!.*Southern).*", r"First Bancorp/Southern Pines.*",
                                      r"Bank of the Ozarks.*"],
                       "valid_from": [None, None, "2011-01-01"], "valid_to": [None, None, "2017-06-22"],
                       "cik": [1057706, 811589, 1038205], "evidence": ["", "", ""]})
    assert sm.override_for(ov, "First BanCorp./Puerto Rico", "2016-12-31") == 1057706
    assert sm.override_for(ov, "First Bancorp/Southern Pines NC", "2021-12-31") == 811589
    assert sm.override_for(ov, "Bank of the Ozarks Inc.", "2016-12-31") == 1038205
    assert sm.override_for(ov, "Bank of the Ozarks Inc.", "2018-03-31") is None
    assert sm.override_for(ov, "Other Inc.", "2018-03-31") is None


def test_resolve_drops_operating_partnership():
    alive = lambda c, d: True  # noqa: E731
    lp = lambda c, d: c == 2  # noqa: E731  2 = 'Corporate Office Properties, L.P.'
    assert sm.resolve({1, 2}, "2016-12-31", alive, lambda c, d: True, None, lp) == (1, "확정(운영 조합 제외)")
    assert sm.PARTNERSHIP_RE.search("Corporate Office Properties, L.P.")
    assert sm.PARTNERSHIP_RE.search("REALOGY GROUP LLC")
    assert not sm.PARTNERSHIP_RE.search("Realogy Holdings Corp.")
    assert not sm.PARTNERSHIP_RE.search("Alpine Group Inc")


def test_fill_by_cusip():
    m = pd.DataFrame([
        {"name": "Gannett Co Inc", "period": "2020-06-30", "cik": 1579684, "status": "확정(이름 기간)", "how": "엄격 키"},
        {"name": "Gannett Media Corp", "period": "2021-02-28", "cik": None, "status": "활동 없음", "how": "엄격 키"},
        {"name": "Twin A", "period": "2021-02-28", "cik": 1, "status": "확정", "how": "엄격 키"},
        {"name": "Twin B", "period": "2021-03-31", "cik": 2, "status": "확정", "how": "엄격 키"},
        {"name": "Twin C", "period": "2021-06-30", "cik": None, "status": "모호", "how": "엄격 키"},
    ])
    rows = pd.DataFrame([
        {"name": "Gannett Co Inc", "cusip": "36472T109", "period": "2020-06-30"},
        {"name": "Gannett Media Corp", "cusip": "36472T109", "period": "2021-02-28"},
        {"name": "Twin A", "cusip": "999999999", "period": "2021-02-28"},
        {"name": "Twin B", "cusip": "999999999", "period": "2021-03-31"},
        {"name": "Twin C", "cusip": "999999999", "period": "2021-06-30"},
    ])
    out = sm.fill_by_cusip(m, rows).set_index("name")
    assert out.loc["Gannett Media Corp", "cik"] == 1579684 and out.loc["Gannett Media Corp", "status"] == "확정(CUSIP)"
    assert pd.isna(out.loc["Twin C", "cik"])  # 같은 CUSIP에 확정 CIK가 둘이면 채우지 않음


def test_resolve_merger_eve_prefers_old_company_by_name_then():
    # 2020-12-31 Dime: 옛 Dime(1)은 공시 끝(인수 직전), 이름 이어받을 Bridge(2)만 살아 있음, 그 시점 이름은 Bridge Bancorp
    alive = lambda c, d: c == 2  # noqa: E731
    ending = lambda c, d: c == 1  # noqa: E731
    named = lambda c, d: c == 1  # noqa: E731
    assert sm.resolve({1, 2}, "2020-12-31", alive, named, ending, None, ending_if_none_alive=False) == (1, "확정(인수 직전)")
    # 그 시점 이름이 살아 있는 후보와 맞으면 그대로
    assert sm.resolve({1, 2}, "2021-03-31", alive, lambda c, d: True, ending, gone=lambda c, d: c == 1) == (2, "확정")
    # 첫 판정(ending_if_none_alive=False): 옛 회사가 이미 상장폐지 서류를 냈으면(Dime 25-NSE 2021-01-29)
    # 인수 직전으로 가지 않고 '활동 없음' → 이름 검색으로 새 회사를 찾게 둔다
    assert sm.resolve({1}, "2021-02-28", lambda c, d: False, named, ending, None, ending_if_none_alive=False,
                      gone=lambda c, d: True) == (None, "활동 없음")


def test_gone_before_and_merger_close():
    dime = pd.DataFrame({"form": ["10-Q", "25-NSE", "15-12B"], "filing_date": ["2020-11-06", "2021-01-29", "2021-02-11"],
                         "items": ["", "", ""]})
    assert not sm.gone_before(dime, "2020-12-31")
    assert sm.gone_before(dime, "2021-02-28")
    xperi = pd.DataFrame({"form": ["10-Q", "8-K"], "filing_date": ["2020-05-06", "2020-06-01"], "items": ["", "2.01,3.01"]})
    assert not sm.gone_before(xperi, "2020-05-31")
    assert sm.gone_before(xperi, "2020-06-30")
    # 합병 완료 전: 옛 회사(1, 인수 직전, 아직 상장) / 완료 후: 새 회사(2)
    alive = lambda c, d: c == 2  # noqa: E731
    ending = lambda c, d: c == 1  # noqa: E731
    named = lambda c, d: True  # noqa: E731  두 회사 모두 그 시점 이름이 같음 (Xperi Corp / Xperi Holding Corp)
    assert sm.resolve({1, 2}, "2020-05-31", alive, named, ending, gone=lambda c, d: False)[0] == 1
    assert sm.resolve({1, 2}, "2021-02-28", alive, named, ending, gone=lambda c, d: c == 1)[0] == 2


def test_clean_fund_name_state_suffix_guards():
    assert sm.match_key("M/I Homes Inc.") == sm.match_key("M/I HOMES, INC.")
    assert sm.match_key("M/I Homes Inc.") != ""
    assert sm.match_key("Signature Bank/New York NY") == sm.match_key("Signature Bank")
    assert sm.match_key("Universal Display Corp") == sm.match_key("UNIVERSAL DISPLAY CORP \PA\\")
    assert sm.match_key("Travel + Leisure Co") == sm.match_key("Travel & Leisure Co.")
    assert sm.match_key("United Bankshares Inc/WV") == sm.match_key("UNITED BANKSHARES INC/WV/")
    assert "ferry" in sm.match_key("Korn/Ferry International")
    assert "frost" in sm.match_key("Cullen/Frost Bankers Inc")
    assert sm.match_key("Diodes Inc.") == sm.match_key("DIODES INC /DEL/")
    assert sm.match_key("Service Corp. International/U.S.") == sm.match_key("SERVICE CORPORATION INTERNATIONAL")
    assert sm.match_key("First BanCorp/Puerto Rico") == sm.match_key("FIRST BANCORP /PR/")
    assert sm.match_key("Independent Bank Corp./Rockland MA") == sm.match_key("INDEPENDENT BANK CORP")


def test_resolve_stale_filer_but_not_merged_away():
    never = lambda c, d: False  # noqa: E731
    # 재작성으로 10-K·10-Q가 늦은 같은 회사: 살아 있음·인수 직전 아님, ±400일 안 공시 있음, 상장폐지 서류 없음
    assert sm.resolve({1}, "2017-12-31", never, None, never, gone=never, stale=lambda c, d: True) == (1, "확정(공시 지연)")
    # 합병으로 사라진 회사: 상장폐지 서류가 있으면 붙이지 않음
    assert sm.resolve({1}, "2021-03-31", never, None, never, gone=lambda c, d: True, stale=lambda c, d: True) == (None, "활동 없음")


def test_full_key_and_holding_vs_subsidiary():
    assert sm.match_key("Graham Holdings Co. Class B") == sm.match_key("GRAHAM CORP")  # 이름 키로는 못 가름
    assert sm.full_key("Graham Holdings Co. Class B") == sm.full_key("GRAHAM HOLDINGS CO")
    assert sm.full_key("Graham Holdings Co. Class B") != sm.full_key("GRAHAM CORP")
    assert sm.match_key("Vertex Pharmaceuticals Inc.") == sm.match_key("VERTEX PHARMACEUTICALS INC / MA")
    alive = lambda c, d: True  # noqa: E731
    full = lambda c, d: c == 1  # noqa: E731
    assert sm.resolve({1, 2}, "2016-12-31", alive, lambda c, d: True, same_full_then=full) == (1, "확정(전체 이름)")


def test_ending_subsidiary_does_not_beat_alive_parent_with_full_name():
    # 2019-11-30: 본체 O-I(1)는 살아 있고 전체 이름 일치, 자회사 O-I Group(2)은 마지막 10-Q 직후(인수 직전처럼 보임)
    alive = lambda c, d: c == 1  # noqa: E731
    ending = lambda c, d: c == 2  # noqa: E731
    named = lambda c, d: True  # noqa: E731
    full = lambda c, d: c == 1  # noqa: E731
    assert sm.resolve({1, 2}, "2019-11-30", alive, named, ending, gone=lambda c, d: False, same_full_then=full)[0] == 1
