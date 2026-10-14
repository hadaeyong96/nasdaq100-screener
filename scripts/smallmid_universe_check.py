"""S&P 400·600 시점별 명단 복원 가능성 조사 — 일회성 점검 스크립트 (조사만, 수익 계산 금지).

출처: SEC에 공개된 지수 펀드 보유 공시. 펀드가 지수를 완전 복제하면 보고일의 보유 종목 = 지수 종목.
  - IJH (S&P 400) iShares Trust CIK 1100663 시리즈 S000004307 — 분기 말 3·6·9·12월
  - IJR (S&P 600) iShares Trust CIK 1100663 시리즈 S000004313
  - IVOO (S&P 400) Vanguard Admiral Funds CIK 891190 시리즈 S000030014 — 분기 말 2·5·8·11월
  - VIOO (S&P 600) Vanguard Admiral Funds CIK 891190 시리즈 S000030017
2019 이후 NPORT-P(XML)는 구조화돼 있고, 그 전 N-Q·N-CSR는 HTML이라 펀드 구역을 찾아 표를 읽는다.
방법은 공개 문서(HistoricIndexMembership METHOD.md)를 참고만 했고, 코드는 쓰지 않았다.

단계:
  --step filings  네 시리즈의 NPORT-P·N-Q 공시 목록 → data/cache/smallmid/filings.csv
  --step nport    2019~2021 NPORT-P 원문 받기·보유 종목 표 → data/cache/smallmid/holdings/*.csv
  --step nq       2019년 이전 표본(2016-12·2011-12 근처) N-Q 원문 받기·보유 종목 표
  --step match    이름 → CIK 매칭 (EDGAR cik-lookup-data + 현재 티커 표) → match.csv
  --step report   종목 수·매칭률·펀드 간 차이 원인 표 → 화면 출력 (문서에 옮김)

가격·수익은 읽지 않는다.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "cache" / "smallmid"
RAW = CACHE / "raw"
HOLD = CACHE / "holdings"
FILINGS_CSV = CACHE / "filings.csv"

FUNDS = [
    # (펀드, 지수, 등록인 CIK, 시리즈 ID, N-Q 문서 안 펀드 구역 제목 정규식)
    ("IJH", "sp400", 1100663, "S000004307", r"S\s*&\s*P\s+MID-?\s*CAP\s+(?:400\s+INDEX\s+FUND|ETF)"),
    ("IJR", "sp600", 1100663, "S000004313", r"S\s*&\s*P\s+SMALL-?\s*CAP\s+(?:600\s+INDEX\s+FUND|ETF)"),
    ("IVOO", "sp400", 891190, "S000030014", r"S\s*&\s*P\s+MID-?\s*CAP\s+400\s+INDEX\s+FUND"),
    ("VIOO", "sp600", 891190, "S000030017", r"S\s*&\s*P\s+SMALL-?\s*CAP\s+600\s+INDEX\s+FUND"),
]
NPORT_NS = {"n": "http://www.sec.gov/edgar/nport"}


def _edgar():
    """우리 EDGAR 요청 함수와 User-Agent (네트워크)."""
    from data.edgar import _get, get_user_agent

    return _get, get_user_agent()


def cached_get(url: str, path: Path) -> str:
    """url 원문을 path에 캐시하고 문자열로 돌려준다 (네트워크·파일)."""
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    get, ua = _edgar()
    text = get(url, ua).text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


# ── 공시 목록 ──────────────────────────────────────────────────────────────

def parse_atom_filings(atom: str) -> list[dict]:
    """EDGAR browse-edgar atom 응답 → [{form, filed, href, accession}] (순수)."""
    out = []
    for block in re.findall(r"<entry>(.*?)</entry>", atom, re.S):
        form = re.search(r"<filing-type>([^<]+)</filing-type>", block)
        filed = re.search(r"<filing-date>([^<]+)</filing-date>", block)
        href = re.search(r"<filing-href>([^<]+)</filing-href>", block)
        acc = re.search(r"<accession-number>([^<]+)</accession-number>", block)
        if form and filed and href:
            out.append({"form": form.group(1), "filed": filed.group(1), "href": href.group(1),
                        "accession": acc.group(1) if acc else href.group(1).rsplit("/", 1)[-1].replace("-index.htm", "")})
    return out


def parse_filing_index(html: str) -> dict:
    """공시 색인 페이지 → {period, docs: [(href, type, size)]} (순수)."""
    per = re.search(r"Period of Report</div>\s*<div class=\"info\">([^<]+)", html)
    docs = re.findall(r'href="(/Archives/[^"]+)"[^<]*</a></td>\s*<td[^>]*>([^<]*)</td>\s*<td[^>]*>(\d*)', html)
    return {"period": per.group(1).strip() if per else None, "docs": docs}


def step_filings() -> None:
    """네 시리즈의 NPORT-P·N-Q 목록과 보고 기준일 → filings.csv."""
    rows = []
    for fund, index, cik, sid, _ in FUNDS:
        for form in ("NPORT-P", "N-Q", "N-CSR"):  # N-CSR 검색은 N-CSRS(반기)도 함께 돌려준다
            url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={sid}&type={form}"
                   f"&dateb=&owner=include&count=100&output=atom")
            atom = cached_get(url, RAW / "atom" / f"{sid}_{form}.xml")
            for f in parse_atom_filings(atom):
                if form == "N-CSR" and f["filed"] > "2019-12-31":  # 2019-08 Vanguard 연 보고까지, 그 뒤는 NPORT
                    continue
                idx = cached_get(f["href"], RAW / "index" / f"{f['accession']}.htm")
                info = parse_filing_index(idx)
                rows.append({"fund": fund, "index": index, "cik": cik, "series": sid, **f, "period": info["period"]})
    df = pd.DataFrame(rows).sort_values(["fund", "form", "period"])
    CACHE.mkdir(parents=True, exist_ok=True)
    df.to_csv(FILINGS_CSV, index=False, encoding="utf-8-sig")
    print(df.groupby(["fund", "form"]).agg(n=("period", "size"), first=("period", "min"), last=("period", "max")))


# ── NPORT-P (2019~) ────────────────────────────────────────────────────────

LAST_PERIOD = "2021-12-31"  # 봉인 구간(2022~) 공시는 받지 않는다


def parse_nport(xml: str) -> tuple[dict, pd.DataFrame]:
    """NPORT-P primary_doc.xml → (머리 정보, 보유 종목 표) (순수).

    머리: {series, rep_date}. 표 열: name, title, cusip, isin, lei, balance, units, val_usd, asset_cat,
    issuer_cat, country, payoff. 행 = invstOrSec 하나.
    """
    root = ET.fromstring(xml.encode("utf-8"))
    def t(node, tag):
        x = node.find(f".//n:{tag}", NPORT_NS)
        return x.text.strip() if x is not None and x.text else None
    head = {"series": t(root, "seriesId"), "rep_date": t(root, "repPdDate")}
    rows = []
    for s in root.iterfind(".//n:invstOrSec", NPORT_NS):
        isin = s.find("n:identifiers/n:isin", NPORT_NS)
        rows.append({"name": t(s, "name"), "title": t(s, "title"), "cusip": t(s, "cusip"),
                     "isin": isin.get("value") if isin is not None else None, "lei": t(s, "lei"),
                     "balance": t(s, "balance"), "units": t(s, "units"), "val_usd": t(s, "valUSD"),
                     "asset_cat": t(s, "assetCat"), "issuer_cat": t(s, "issuerCat"),
                     "country": t(s, "invCountry"), "payoff": t(s, "payoffProfile")})
    return head, pd.DataFrame(rows)


def classify_holding(row: dict) -> str:
    """보유 한 줄 → '주식' 또는 지수 종목이 아닌 이유 (순수).

    NPORT assetCat: EC 보통주, STIV 단기 투자(머니마켓·담보), DFE 선물 등 파생, 그 밖은 기타.
    """
    cat = (row.get("asset_cat") or "").upper()
    name = (row.get("name") or "").upper()
    if cat == "STIV" or "MONEY MARKET" in name or "CASH COLLATERAL" in name:
        return "현금성"
    if cat.startswith("D"):
        return "파생(선물 등)"
    if re.search(r"\b(RIGHTS?|CVR|CONTINGENT VALUE|WARRANTS?|ESCROW)\b", name):
        return "권리·CVR"
    if cat != "EC":
        return f"기타({cat or '없음'})"
    if (row.get("payoff") or "Long") != "Long":
        return "숏"
    return "주식"


def step_nport() -> None:
    """2021-12-31까지 NPORT-P 원문을 받아 펀드·기준일별 보유 표 저장."""
    f = pd.read_csv(FILINGS_CSV)
    f = f[(f["form"] == "NPORT-P") & (f["period"] <= LAST_PERIOD)]
    for _, r in f.iterrows():
        out = HOLD / f"{r['fund']}_{r['period']}.csv"
        if out.exists():
            continue
        base = r["href"].rsplit("/", 1)[0]
        xml = cached_get(f"{base}/primary_doc.xml", RAW / "nport" / f"{r['accession']}.xml")
        head, df = parse_nport(xml)
        if head["series"] != r["series"]:
            raise SystemExit(f"시리즈 불일치: {r['accession']} {head['series']} != {r['series']}")
        df["class"] = [classify_holding(x) for x in df.to_dict("records")]
        HOLD.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(r["fund"], r["period"], head["rep_date"], len(df), df["class"].value_counts().to_dict())


# ── N-Q (2019 이전, 표본 2시점) ────────────────────────────────────────────

NQ_SAMPLES = {"IJH": ["2016-12-31", "2011-12-31"], "IJR": ["2016-12-31", "2011-12-31"],
              "IVOO": ["2016-11-30", "2011-11-30"], "VIOO": ["2016-11-30", "2011-11-30"]}


def nq_doc_path(r: dict) -> Path:
    """N-Q 본문 캐시 경로 (한 공시에 여러 펀드가 같이 있어 공시 번호로 저장)."""
    return RAW / "nq" / f"{r['accession']}.htm"


NQ_HEAD = {"sp400": r"S&P\s*MID-?\s*CAP(?:\s*400)?\s+(?:INDEX\s+FUND|ETF)",
           "sp600": r"S&P\s*SMALL-?\s*CAP(?:\s*600)?\s+(?:INDEX\s+FUND|ETF)"}
NQ_END = re.compile(r"(?i)^(total\s+common\s+stocks|short-term\s+investments|money\s+market\s+funds?|"
                    r"temporary\s+cash\s+investments?|total\s+investments)")
_NUM = re.compile(r"^\(?\$?\s*[\d,]+(?:\.\d+)?\)?$")


def flatten_html(html_text: str) -> list[str]:
    """HTML → 텍스트 줄 목록. 칸 경계는 줄을 나누고, 빈 줄·'|'·'$'만 있는 줄은 버린다 (순수)."""
    import html as _html

    t = re.sub(r"(?is)<sup[^>]*>.*?</sup>", "", html_text)  # 각주 위첨자 (Inc.<sup>a,b</sup>)
    t = re.sub(r"(?i)<br\s*/?>|</(?:p|tr|div|h\d|t[dh])>", "\n", t)
    t = _html.unescape(re.sub(r"<[^>]+>", "", t))
    t = re.sub(r"[ \t\xa0]+", " ", t)
    return [x.strip() for x in t.split("\n") if x.strip() and x.strip() not in {"|", "$", "—", "-"}]


def fund_pages(lines: list[str], head_pat: str) -> list[str]:
    """'Schedule of Investments' 줄을 쪽 경계로 보고, 그 앞뒤 4줄에 펀드 제목이 있는 쪽만 이어 붙인다 (순수).

    E-mini(선물) 줄은 제목으로 보지 않는다. Growth·Value 펀드는 제목 정규식에 안 맞는다.
    """
    # 쪽 경계: 'Schedule of Investments'(2018 iShares는 'Schedule of' / 'Investments'로 나뉨),
    # Vanguard 연·반기 보고는 'Statement of Net Assets'
    soi = [i for i, x in enumerate(lines)
           if re.match(r"(?i)^(schedule of(\s+investments)?|statement of net assets)\s*(\(unaudited\))?\s*$", x)
           or re.match(r"(?i)^schedule of investments", x)] + [len(lines)]
    out: list[str] = []
    for a, b in zip(soi, soi[1:]):
        near = " ".join(x for x in lines[max(0, a - 3):a + 5] if "mini" not in x.lower())
        if re.search(head_pat, near, re.I):
            out += lines[a:b]
    return out


def _clean_holding_name(x: str) -> str:
    """'* eBay Inc.', 'Kroger Co. (The)a,b' 같은 표시를 정리 (순수)."""
    x = re.sub(r"^[*†‡§^,\s]+", "", x)
    x = re.sub(r"(?<=[.)])[a-h](?:,[a-h])*$", "", x)  # 'Inc.a,b' 각주 문자
    x = re.sub(r"\s*\([a-h]\)(?:\s*\([a-h]\))*$", "", x)
    return x.strip()


def parse_nq_holdings(lines: list[str]) -> pd.DataFrame:
    """펀드 쪽 줄 → 보통주 보유 표 (name, shares, value). 이름 줄 뒤에 숫자 두 개가 오면 한 종목 (순수).

    'COMMON STOCKS' 이후부터 끝 표시(TOTAL COMMON STOCKS·SHORT-TERM·MONEY MARKET 등) 전까지만 읽는다.
    업종 제목(퍼센트 포함)·표 머리는 이름으로 보지 않는다.
    """
    rows, started, buf = [], False, []
    toks = [x.rstrip(" |").strip() for x in lines]
    toks = [x for x in toks if x and x not in {"|", "$"}]
    # 표 머리 단어는 줄 전체가 그 단어일 때만 (MarketAxess 같은 이름을 버리지 않게)
    header = re.compile(r"(?i)^(?:security|shares|value|market|market value|\(\$000\))$|"
                        r"^(?:schedule of investments|ishares|vanguard|\(unaudited\)|as of )|^[a-z]+ \d{1,2}, \d{4}$")
    i = 0
    while i < len(toks):
        x = toks[i]
        if re.match(r"(?i)^common\s+stocks", x) or (not started and re.match(r"(?i)^security$", x)):
            started, buf = True, []  # 2018 iShares N-Q는 'COMMON STOCKS' 제목 없이 표 머리 'Security' 뒤 바로 업종
        elif started and NQ_END.match(x):
            break
        elif not started:
            pass
        elif ("%" in x or re.search(r"\(\s*continued\s*\)", x, re.I) or header.match(x)
              or re.search(NQ_HEAD["sp400"] + "|" + NQ_HEAD["sp600"], x, re.I)):
            buf = []  # 업종 제목('— 1.0%', 'Food Products (continued)')·쪽 머리: 이름이 아님
        elif re.search(r"[A-Za-z]", x) and not _NUM.match(x):
            buf = (buf + [x])[-3:]  # 이름이 두 줄 이상으로 접힌 경우
        elif buf and _NUM.match(x) and i + 1 < len(toks) and _NUM.match(toks[i + 1]):
            rows.append({"name": _clean_holding_name(" ".join(buf)), "shares": x, "value": toks[i + 1]})
            buf = []
            i += 2
            continue
        i += 1
    return pd.DataFrame(rows, columns=["name", "shares", "value"])


def step_nq() -> None:
    """2019년 이전 표본 2시점: N-Q 본문에서 펀드 구역을 찾아 보통주 표 저장."""
    f = pd.read_csv(FILINGS_CSV)
    for fund, periods in NQ_SAMPLES.items():
        index = [x for x in FUNDS if x[0] == fund][0][1]
        for p in periods:
            r = f[(f["fund"] == fund) & (f["form"] == "N-Q") & (f["period"] == p)]
            if r.empty:
                print(fund, p, "공시 없음")
                continue
            lines = flatten_html(fetch_nq_doc(r.iloc[0].to_dict()))
            df = parse_nq_holdings(fund_pages(lines, NQ_HEAD[index]))
            df["class"] = "주식"
            HOLD.mkdir(parents=True, exist_ok=True)
            df.to_csv(HOLD / f"{fund}_{p}.csv", index=False, encoding="utf-8-sig")
            print(fund, p, "보통주", len(df), "중복 이름", int(df["name"].duplicated().sum()))


def fetch_nq_doc(r: dict) -> str:
    """N-Q 공시 색인에서 본문(N-Q 종류 문서 중 가장 큰 것)을 받는다 (네트워크·파일)."""
    idx = cached_get(r["href"], RAW / "index" / f"{r['accession']}.htm")
    docs = [d for d in parse_filing_index(idx)["docs"] if d[1].startswith("N-Q") and d[0].endswith((".htm", ".txt"))]
    if not docs:
        raise SystemExit(f"N-Q 본문 없음: {r['accession']}")
    href = max(docs, key=lambda d: int(d[2] or 0))[0]
    return cached_get("https://www.sec.gov" + href, nq_doc_path(r))


SIZE_OK = {"sp400": (370, 430), "sp600": (560, 640)}  # 기대 크기 (보통주 줄 수)
REPORTS_CSV = CACHE / "reports.csv"
FIRST_PERIOD, LAST_HTML_PERIOD = "2011-05-31", "2019-08-31"  # 2011-12 앞 편입 기간 확인용으로 반년 앞부터


def report_docs(r: dict) -> list[tuple[str, str, str]]:
    """공시 색인 → 본문 후보 (N-Q·N-CSR·N-CSRS 종류의 .htm·.txt, 큰 것부터) (네트워크·파일)."""
    idx = cached_get(r["href"], RAW / "index" / f"{r['accession']}.htm")
    docs = [d for d in parse_filing_index(idx)["docs"]
            if d[1].startswith(("N-Q", "N-CSR")) and d[0].endswith((".htm", ".txt"))]
    return sorted(docs, key=lambda d: -int(d[2] or 0))


def read_report(r: dict, index: str) -> tuple[pd.DataFrame, str]:
    """한 공시의 본문 문서들에서 그 펀드 보통주 표를 찾는다. 기대 크기에 맞는 첫 문서를 쓴다.

    출력: (표, 문서 경로). 맞는 문서가 없으면 가장 많이 읽힌 표.
    """
    best, best_doc = pd.DataFrame(columns=["name", "shares", "value"]), ""
    lo, hi = SIZE_OK[index]
    for k, (href, _, _) in enumerate(report_docs(r)):
        path = RAW / "nq" / (f"{r['accession']}.htm" if k == 0 else f"{r['accession']}_{k}.htm")
        df = parse_nq_holdings(fund_pages(flatten_html(cached_get("https://www.sec.gov" + href, path)), NQ_HEAD[index]))
        if lo <= len(df) <= hi:
            return df, href
        if len(df) > len(best):
            best, best_doc = df, href
    return best, best_doc


def step_reports() -> None:
    """2011-05 ~ 2019-08 N-Q·N-CSR·N-CSRS 전부 읽기. 기대 크기 검사 결과는 reports.csv, 통과한 것만 보유 표로 저장.

    정정 공시(/A)는 원 공시가 검사를 못 통과할 때만 쓴다.
    """
    f = pd.read_csv(FILINGS_CSV)
    f = f[f["form"].str.startswith(("N-Q", "N-CSR")) & (f["period"] >= FIRST_PERIOD) & (f["period"] <= LAST_HTML_PERIOD)]
    log = []
    for (fund, period), g in f.groupby(["fund", "period"]):
        index = [x for x in FUNDS if x[0] == fund][0][1]
        lo, hi = SIZE_OK[index]
        g = g.assign(amend=g["form"].str.endswith("/A")).sort_values("amend")
        res = None
        for _, r in g.iterrows():
            df, doc = read_report(r.to_dict(), index)
            ok = lo <= len(df) <= hi
            log.append({"fund": fund, "index": index, "period": period, "form": r["form"], "accession": r["accession"],
                        "doc": doc, "n": len(df), "ok": ok})
            if ok:
                res = df
                break
        if res is not None:
            res = res.assign(**{"class": "주식"})
            res.to_csv(HOLD / f"{fund}_{period}.csv", index=False, encoding="utf-8-sig")
        print(fund, period, "→", log[-1]["form"], log[-1]["n"], "통과" if res is not None else "실패")
    pd.DataFrame(log).to_csv(REPORTS_CSV, index=False, encoding="utf-8-sig")


# ── 이름 → CIK ─────────────────────────────────────────────────────────────

CIK_LOOKUP_URL = "https://www.sec.gov/Archives/edgar/cik-lookup-data.txt"
MATCH_CSV = CACHE / "match.csv"
ACTIVE_FORMS = {"10-K", "10-Q", "10-K405", "10-KT", "10-QT"}


_SW = None


def _sw():
    """SW1 조사 스크립트의 이름 정규화·submissions 함수 (같은 규칙을 다시 쓴다)."""
    global _SW
    if _SW is None:
        if str(ROOT / "scripts") not in sys.path:
            sys.path.insert(0, str(ROOT / "scripts"))
        import sw1_data_check as sw

        _SW = sw
    return _SW


def clean_fund_name(name: str) -> str:
    """펀드 보고서·EDGAR 이름에서 매칭에 방해되는 표기를 정리 (순수).

    - '/WV', '/The', 'Class A', 'Cl A', '- A' 같은 주 등록지·주식 종류 표기를 뗀다
    - 아포스트로피(' ’)는 지운다: America's → Americas (EDGAR는 AMERICAS로 씀)
    - 약어 Cos → Companies (Carlisle Cos. ↔ CARLISLE COMPANIES INC)
    """
    x = str(name)
    # 주 등록지·The 꼬리: Inc/WV, Co/The, Bank/New York NY, Corp/DE/ (M/I Homes의 '/I Homes'는 꼬리가 아님)
    # 꼬리로 보는 것: 2~4자 약어(WV, DEL, U.S.), The, '지명 + 주 코드'(New York NY), Puerto Rico·United States.
    # Korn/Ferry International, Cullen/Frost Bankers Inc 같은 이름 안의 '/'는 건드리지 않는다.
    x = re.sub(r"\s*/\s?(?:The|[A-Z](?:\.?[A-Z]){1,3}\.?|(?:[A-Z][a-z]+ )+[A-Z]{2}|Puerto Rico|United States)/?\s*$", " ", x)
    x = re.sub(r"\s*\\[A-Z]{2}\\\s*$", " ", x)  # EDGAR 'UNIVERSAL DISPLAY CORP \PA\'
    x = x.replace("+", "&")  # Travel + Leisure ↔ Travel & Leisure
    x = re.sub(r"[,\s-]+(?:class|cl)\.?\s+[a-z]\b.*$", " ", x, flags=re.I)
    x = re.sub(r"\s+-\s+[a-z]$", " ", x, flags=re.I)
    x = re.sub(r"['’`]", "", x)
    x = re.sub(r"\bCos\b\.?", "Companies", x, flags=re.I)
    x = re.sub(r"[,\s]+NVS\.?$", "", x)  # 무의결권 표시 (Eaton Vance Corp. NVS)
    # 이니셜 붙이기: A.O. Smith → AO Smith, EDGAR 'SCRIPPS E W CO' → 'SCRIPPS EW CO'
    x = re.sub(r"\b(?:[A-Za-z](?:\.\s?|\s)){1,3}[A-Za-z]\b\.?",
               lambda mm: re.sub(r"[.\s]", "", mm.group(0)) + " ", x)
    return re.sub(r"\s+", " ", x).strip()


def match_key(name: str) -> str:
    """엄격 키 = 정리 뒤 SW1·S&P 500 점검과 같은 norm_name (순수)."""
    return _sw().norm_name(clean_fund_name(name))


_LEGAL = {"inc", "corp", "corporation", "co", "company", "the", "plc", "ltd", "limited", "class", "a", "b", "c",
          "de", "nv", "sa", "lp", "llc", "incorporated"}


def full_key(name: str) -> str:
    """전체 이름 키: 법인 형태만 빼고 Holdings·Group·Trust는 남긴다 (지주회사·자회사 가르기용) (순수)."""
    x = re.sub(r"\(.*?\)", " ", clean_fund_name(name).lower()).replace("&", " and ")
    return " ".join(w for w in re.findall(r"[a-z0-9]+", x) if w not in _LEGAL)


def loose_key(name: str) -> str:
    """느슨한 키 = 엄격 키의 단어를 정렬 (어순·이니셜 위치 무시: A.O. Smith ↔ SMITH A O) (순수)."""
    return " ".join(sorted(match_key(name).split()))


def load_name_index() -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """EDGAR cik-lookup-data(옛 이름 포함 모든 등록인) → ({엄격 키: {cik}}, {느슨한 키: {cik}}) (네트워크 1회·캐시)."""
    text = cached_get(CIK_LOOKUP_URL, RAW / "cik-lookup-data.txt")
    strict: dict[str, set[int]] = {}
    loose: dict[str, set[int]] = {}
    for line in text.splitlines():
        m = re.match(r"^(.*):(\d{10}):\s*$", line)
        if not m:
            continue
        k = match_key(m.group(1))
        if k:
            cik = int(m.group(2))
            strict.setdefault(k, set()).add(cik)
            loose.setdefault(" ".join(sorted(k.split())), set()).add(cik)
    return strict, loose


def name_at(sub: dict, date: str) -> str:
    """submissions JSON → 그 날짜에 쓰던 등록 이름 (formerNames의 from~to 기간, 없으면 현재 이름) (순수)."""
    d = pd.Timestamp(date)
    for f in sub.get("formerNames", []) or []:
        a, b = f.get("from"), f.get("to")
        if a and b and pd.Timestamp(a[:10]) <= d <= pd.Timestamp(b[:10]):
            return f.get("name", "")
    return sub.get("name", "")


def active_near(rows: pd.DataFrame, rep_date: str, days: int = 400) -> bool:
    """submissions 행 표에 rep_date ± days 안에 10-K·10-Q 제출이 있는지 (순수)."""
    if rows.empty:
        return False
    d = pd.to_datetime(rows.loc[rows["form"].isin(ACTIVE_FORMS), "filing_date"])
    r = pd.Timestamp(rep_date)
    return bool(((d >= r - pd.Timedelta(days=days)) & (d <= r + pd.Timedelta(days=days))).any())


ALIVE_AFTER_DAYS = 200   # 살아 있는 회사: 기준일 뒤 이 기간 안에 10-K·10-Q
ENDING_BEFORE_DAYS = 120  # 인수 직전: 기준일 뒤 공시는 없지만 마지막 10-K·10-Q가 기준일 전 이 기간 안


def alive_at(rows: pd.DataFrame, rep_date: str, after: int = ALIVE_AFTER_DAYS) -> bool:
    """그 기준일에 살아 있는 등록인인지: 기준일 다음 날 ~ after일 안에 10-K·10-Q 제출 (순수).

    ±400일 창은 합병으로 사라진 회사(옛 Dime·옛 Gannett)를 살려 둬서 바꿨다 (문서 9장).
    """
    if rows.empty:
        return False
    d = pd.to_datetime(rows.loc[rows["form"].isin(ACTIVE_FORMS), "filing_date"])
    r = pd.Timestamp(rep_date)
    return bool(((d > r) & (d <= r + pd.Timedelta(days=after))).any())


def ending_at(rows: pd.DataFrame, rep_date: str, before: int = ENDING_BEFORE_DAYS) -> bool:
    """인수·상장폐지 직전: 기준일 뒤 10-K·10-Q는 없고, 마지막 것이 기준일 전 before일 안 (순수)."""
    if rows.empty:
        return False
    d = pd.to_datetime(rows.loc[rows["form"].isin(ACTIVE_FORMS), "filing_date"])
    r = pd.Timestamp(rep_date)
    if (d > r).any():
        return False
    return bool(((d <= r) & (d >= r - pd.Timedelta(days=before))).any())


PARTNERSHIP_RE = re.compile(r"\b(L\.?\s?P\.?|LLC|OPERATING PARTNERSHIP)\s*$|\bL\.?P\.?\b", re.I)


DEREG_FORMS = {"15-12B", "15-12G", "15-15D", "25", "25-NSE"}


def gone_before(rows: pd.DataFrame, rep_date: str) -> bool:
    """기준일까지 상장폐지·등록 해지 서류를 냈는지: 마지막 10-K·10-Q 뒤 ~ 기준일에 Form 15·25 또는 8-K 2.01·3.01 (순수).

    합병 완료 전(옛 회사가 정답, Xperi 2020-05-31)과 완료 후(새 회사가 정답, Dime 2021-02-28)를 가른다.
    """
    if rows.empty:
        return False
    fd = pd.to_datetime(rows["filing_date"])
    r = pd.Timestamp(rep_date)
    q = fd[rows["form"].isin(ACTIVE_FORMS) & (fd <= r)]
    since = q.max() if len(q) else pd.Timestamp("1900-01-01")
    win = (fd > since) & (fd <= r)
    items = rows.get("items", pd.Series("", index=rows.index)).fillna("").astype(str)
    dereg = rows["form"].isin(DEREG_FORMS) | ((rows["form"] == "8-K") & items.str.contains(r"\b(?:2\.01|3\.01)\b"))
    return bool((win & dereg).any())


def resolve(cands: set[int], rep_date: str, alive, same_name_then=None, ending=None,
            is_partnership=None, ending_if_none_alive: bool = True, gone=None,
            stale=None, same_full_then=None) -> tuple[int | None, str]:
    """후보 CIK들 → (CIK, 상태) (순수).

    alive(cik, d): 그 시점 살아 있음(기준일 뒤 200일 안 10-K·10-Q). same_name_then(cik, d): 그 시점 등록 이름이
    펀드 이름과 같은 키(옛 이름 사용 기간). is_partnership(cik, d): 그 시점 이름이 운영 조합(L.P.·LLC).
    ending(cik, d): 인수 직전(마지막 10-K·10-Q가 기준일 전 120일 안). 살아 있는 유일 후보의 그 시점 이름이 안 맞을 때,
    또는 (ending_if_none_alive면) 살아 있는 후보가 없을 때 본다. 첫 판정은 ending_if_none_alive=False로 불러
    이름 검색이 먼저 오게 한다 (합병이 끝난 뒤 기준일은 새 회사가 정답).
    stale(cik, d): ±400일 안 10-K·10-Q (공시 지연 회사 구제용).
    상태: 확정 | 확정(이름 기간) | 확정(운영 조합 제외) | 확정(인수 직전) | 확정(공시 지연) | 모호 | 활동 없음 | 후보 없음.
    """
    if not cands:
        return None, "후보 없음"
    act = sorted(c for c in cands if alive(c, rep_date))
    # 합병 직전 기준일: 옛 회사는 공시가 끝나 '살아 있음'에서 빠지고, 그 이름을 이어받은 회사만 남는다
    # (Caesars 2020-06, Dime 2020-12, Xperi 2020-05, Aaron's 2020-09, SEACOR 2021-02). 그 시점 이름이 맞는
    # '인수 직전' 후보가 아직 상장폐지 서류(Form 15·25, 8-K 2.01·3.01)를 안 냈으면 그 옛 회사가 정답이다.
    if same_name_then is not None and ending is not None:
        end_named = sorted(c for c in cands if c not in act and ending(c, rep_date) and same_name_then(c, rep_date)
                           and not (gone is not None and gone(c, rep_date))
                           and not (is_partnership is not None and is_partnership(c, rep_date)))
        # 살아 있는 후보가 전체 이름까지 맞으면(Owens-Illinois Inc ↔ 자회사 Owens-Illinois Group) 그쪽이 정답
        if same_full_then is not None and any(same_full_then(a, rep_date) for a in act):
            end_named = [c for c in end_named if same_full_then(c, rep_date)]
        if len(end_named) == 1:
            return end_named[0], "확정(인수 직전)"
    if len(act) == 1:
        return act[0], "확정"
    if len(act) > 1:
        if same_name_then is not None:
            same = [c for c in act if same_name_then(c, rep_date)]
            if len(same) == 1:
                return same[0], "확정(이름 기간)"
        if same_full_then is not None:  # 지주회사와 같은 이름 자회사(Graham Holdings·Graham Corp 등): 전체 이름 비교
            full = [c for c in act if same_full_then(c, rep_date)]
            if len(full) == 1:
                return full[0], "확정(전체 이름)"
        if is_partnership is not None:  # 리츠·지주 본체와 같은 이름의 운영 조합(L.P.·LLC)은 상장 주식이 아님
            corp = [c for c in act if not is_partnership(c, rep_date)]
            if len(corp) == 1:
                return corp[0], "확정(운영 조합 제외)"
        return None, "모호"
    if ending is not None and ending_if_none_alive:
        end = sorted(c for c in cands if ending(c, rep_date)
                     and not (is_partnership is not None and is_partnership(c, rep_date)))
        if len(end) == 1:
            return end[0], "확정(인수 직전)"
        if len(end) > 1:
            return None, "모호"
        # 재작성 등으로 10-K·10-Q가 늦은 같은 회사(Super Micro 2017~18 등): ±400일 안 공시가 있고 상장폐지 서류가
        # 없는 후보가 하나면 인정. 합병으로 사라진 회사는 gone이라 다시 붙지 않는다(Dime·Gannett·A. Schulman).
        if stale is not None:
            late = sorted(c for c in cands if stale(c, rep_date) and not (gone is not None and gone(c, rep_date))
                          and not (is_partnership is not None and is_partnership(c, rep_date)))
            if len(late) == 1:
                return late[0], "확정(공시 지연)"
    return None, "활동 없음"


def prefetch_submissions(ciks: set[int], ua: str, workers: int = 6, per_sec: float = 8.0) -> None:
    """후보 CIK submissions를 미리 받아 캐시 (네트워크). SEC 한도(초당 10건) 아래로 묶어서 동시에 요청.

    실패는 조용히 넘기지 않고 끝에 개수를 출력한다 (본 매칭에서 다시 시도·보고됨).
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    sw = _sw()
    todo = sorted(c for c in ciks if not (sw.SUB_CACHE / f"CIK{c:010d}.json").exists())
    lock, last, failed = threading.Lock(), [0.0], []

    def one(c: int) -> None:
        with lock:  # 요청 시작 간격을 1/per_sec 이상으로
            wait = last[0] + 1.0 / per_sec - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            last[0] = time.monotonic()
        try:
            sw.fetch_submissions(c, ua)
        except Exception:  # noqa: BLE001
            failed.append(c)

    print(f"submissions 미리 받기: {len(todo)}개 (캐시 {len(ciks) - len(todo)}개)")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))
    print(f"미리 받기 끝, 실패 {len(failed)}개")


OVERRIDES_CSV = ROOT / "data" / "smallmid_cik_overrides.csv"


def load_overrides(path: Path = OVERRIDES_CSV) -> pd.DataFrame:
    """매칭 실패 대조표 (펀드 이름 정규식 + 유효 기간 → CIK, 근거). EDGAR에서 손으로 확인한 값 (파일).

    이름 키가 아니라 원래 이름의 정규식(대소문자 무시, 전체 일치)으로 맞춘다 — 'First BanCorp'(PR)와
    'First Bancorp/Southern Pines NC'처럼 정리하면 같은 키가 되는 다른 회사가 있기 때문.
    """
    if not path.exists():
        return pd.DataFrame(columns=["name_regex", "valid_from", "valid_to", "cik", "evidence"])
    return pd.read_csv(path, comment="#", dtype={"valid_from": str, "valid_to": str})


def override_for(ov: pd.DataFrame, name: str, rep: str) -> int | None:
    """대조표에서 (이름, 기준일)에 맞는 CIK. 유효 기간은 양 끝 포함, 비면 제한 없음. 둘 이상 맞으면 None (순수)."""
    if ov.empty:
        return None
    hit = [int(r["cik"]) for r in ov.to_dict("records")
           if re.fullmatch(r["name_regex"], str(name).strip(), re.I)
           and (pd.isna(r["valid_from"]) or r["valid_from"] <= rep)
           and (pd.isna(r["valid_to"]) or r["valid_to"] >= rep)]
    return hit[0] if len(set(hit)) == 1 else None


def nport_cusip_rows() -> pd.DataFrame:
    """NPORT 보유 표의 주식 줄 (name, cusip, period) (파일)."""
    rows = []
    for p in sorted(HOLD.glob("*.csv")):
        fund, period = p.stem.split("_", 1)
        d = pd.read_csv(p)
        if "cusip" in d.columns:
            rows.append(d.loc[d["class"] == "주식", ["name", "cusip"]].assign(period=period))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["name", "cusip", "period"])


def fill_by_cusip(m: pd.DataFrame, rows: pd.DataFrame | None = None) -> pd.DataFrame:
    """NPORT 기간(2019-09~): 이름으로 못 찾은 (이름, 기준일)을 같은 CUSIP의 확정 CIK로 채움 (rows를 주면 순수).

    같은 CUSIP에 이름 매칭으로 확정된 CIK가 정확히 하나일 때만. 상태 '확정(CUSIP)'. 독립 검증(9장)은 이 행을 빼고 센다.
    """
    r = nport_cusip_rows() if rows is None else rows
    if r.empty:
        return m
    r = r[r["cusip"].astype(str).str.fullmatch(r"[0-9A-Z]{9}") & (r["cusip"].astype(str) != "000000000")]
    r = r.merge(m[["name", "period", "cik", "status"]], on=["name", "period"], how="left")
    ok = r[r["status"].fillna("").str.startswith("확정")].dropna(subset=["cik"])
    one = {}
    for cu, s in ok.groupby("cusip")["cik"]:
        v = set(s.astype(int))
        if len(v) == 1:
            one[cu] = v.pop()
    need = r[~r["status"].fillna("").str.startswith("확정") & r["cusip"].isin(one)]
    fill = {(n, p): one[c] for n, p, c in zip(need["name"], need["period"], need["cusip"])}
    m = m.copy()
    for i, (n, p) in enumerate(zip(m["name"], m["period"])):
        if (n, p) in fill:
            m.loc[m.index[i], ["cik", "status", "how"]] = [fill[(n, p)], "확정(CUSIP)", "CUSIP"]
    return m


def step_match() -> None:
    """보유 표의 (이름, 기준일) → CIK. 같은 이름이 시기마다 다른 회사일 수 있어 기준일마다 따로 판정한다.

    순서: 엄격 키 후보 → 없으면 느슨한 키 후보 → 그 시점 10-K·10-Q 제출 확인 → 여럿이면 그 시점 등록 이름으로
    가림 → 그래도 없으면 EDGAR 이름 검색(유사도 ≥ 0.85) 보조. 출력 match.csv (name, period, cik, status, how).
    """
    sw = _sw()
    _, ua = _edgar()
    strict, loose = load_name_index()
    pairs = []
    for p in sorted(HOLD.glob("*.csv")):
        fund, period = p.stem.split("_", 1)
        d = pd.read_csv(p)
        pairs += [(n, period) for n in d.loc[d["class"] == "주식", "name"].dropna()]
    pairs = sorted(set(pairs))
    subs: dict[int, dict] = {}
    rows_cache: dict[int, pd.DataFrame] = {}
    failed: list[int] = []

    def load(cik: int) -> None:
        if cik not in rows_cache:
            try:
                js = sw.fetch_submissions(cik, ua)
                subs[cik], rows_cache[cik] = js[0], sw.submissions_rows(js)
            except Exception:  # noqa: BLE001 — 실패 CIK는 모아 보고
                failed.append(cik)
                subs[cik], rows_cache[cik] = {}, pd.DataFrame(columns=["form", "filing_date"])

    memo: dict[tuple[str, int, str], bool] = {}

    dates: dict[int, "np.ndarray"] = {}

    def fdates(cik: int):
        # CIK별 10-K·10-Q 제출일 정렬 배열 (alive_at·ending_at과 같은 판정을 빠르게)
        if cik not in dates:
            load(cik)
            r = rows_cache[cik]
            dates[cik] = np.sort(pd.to_datetime(r.loc[r["form"].isin(ACTIVE_FORMS), "filing_date"]).values)                 if len(r) else np.array([], dtype="datetime64[ns]")
        return dates[cik]

    def active(cik: int, rep: str) -> bool:
        if ("a", cik, rep) not in memo:
            d, r = fdates(cik), np.datetime64(pd.Timestamp(rep))
            i = np.searchsorted(d, r, side="right")
            memo[("a", cik, rep)] = bool(i < len(d) and d[i] <= r + np.timedelta64(ALIVE_AFTER_DAYS, "D"))
        return memo[("a", cik, rep)]

    def stale(cik: int, rep: str) -> bool:
        if ("s", cik, rep) not in memo:
            load(cik)
            memo[("s", cik, rep)] = active_near(rows_cache[cik], rep)
        return memo[("s", cik, rep)]

    def gone(cik: int, rep: str) -> bool:
        if ("g", cik, rep) not in memo:
            load(cik)
            memo[("g", cik, rep)] = gone_before(rows_cache[cik], rep)
        return memo[("g", cik, rep)]

    def ending(cik: int, rep: str) -> bool:
        if ("e", cik, rep) not in memo:
            d, r = fdates(cik), np.datetime64(pd.Timestamp(rep))
            i = np.searchsorted(d, r, side="right")
            memo[("e", cik, rep)] = bool(i == len(d) and i > 0 and d[i - 1] >= r - np.timedelta64(ENDING_BEFORE_DAYS, "D"))
        return memo[("e", cik, rep)]

    overrides = load_overrides()

    def cands_of(name: str) -> tuple[set[int], str]:
        c = strict.get(match_key(name), set())
        return (c, "엄격 키") if c else (loose.get(loose_key(name), set()), "느슨한 키")

    prefetch_submissions({c for n, _ in pairs for c in cands_of(n)[0]}, ua)
    out = []
    for i, (name, rep) in enumerate(pairs):
        cands, how = cands_of(name)

        def same_name_then(cik: int, d: str, name=name) -> bool:
            load(cik)
            then = name_at(subs[cik], d)
            return bool(then) and (match_key(then) == match_key(name) or loose_key(then) == loose_key(name))

        ov = override_for(overrides, name, rep)
        if ov is not None:
            out.append({"name": name, "period": rep, "key": match_key(name), "cik": ov, "status": "확정(대조표)",
                        "how": "대조표"})
            continue
        def same_full_then(cik: int, d: str, name=name) -> bool:
            load(cik)
            return full_key(name_at(subs[cik], d) or "") == full_key(name)

        def is_partnership(cik: int, d: str) -> bool:
            load(cik)
            return bool(PARTNERSHIP_RE.search(name_at(subs[cik], d) or ""))

        cik, status = resolve(cands, rep, active, same_name_then, ending, is_partnership, ending_if_none_alive=False,
                              gone=gone, same_full_then=same_full_then)
        if cik is None and status != "모호":
            hits = [c for c, n in sw.name_search(clean_fund_name(name), ua) if sw.name_sim(n, clean_fund_name(name)) >= 0.85]
            hits = [c for c in hits if same_name_then(c, rep)]  # 이름 검색은 그 시점 이름이 맞을 때만 (SEACOR Marine 오매칭)
            c2, s2 = resolve(set(hits), rep, active, same_name_then, None, is_partnership, same_full_then=same_full_then)
            if c2 is not None:
                cik, status, how = c2, s2, "이름 검색"
            else:  # 살아 있는 후보가 어디에도 없을 때만 인수 직전 인정
                c3, s3 = resolve(cands | set(hits), rep, active, same_name_then, ending, is_partnership, gone=gone,
                                 stale=stale, same_full_then=same_full_then)
                if c3 is not None or status == "후보 없음":
                    cik, status = c3, s3
        out.append({"name": name, "period": rep, "key": match_key(name), "cik": cik, "status": status, "how": how})
        if i % 2000 == 0:
            print(i, len(pairs))
    m = fill_by_cusip(pd.DataFrame(out))
    m.to_csv(MATCH_CSV, index=False, encoding="utf-8-sig")
    print(m["status"].value_counts().to_dict(), m["how"].value_counts().to_dict())
    if failed or sw.SEARCH_FAILED:
        print("submissions 실패 CIK:", failed[:50], "이름 검색 실패:", sw.SEARCH_FAILED[:50])


# ── CUSIP 독립 검증 (NPORT) ─────────────────────────────────────────────────

CUSIP_CSV = CACHE / "cusip_check.csv"


def cusip_consistency(rows: pd.DataFrame) -> pd.DataFrame:
    """NPORT 행(cusip, period, name, cik) → CUSIP별 CIK 연속성 표 (순수).

    출력 열: cusip, n_periods, ciks(';'), n_cik, n_null, names(';'), first, last, verdict
    verdict: 일치 | 일부 기간 미확정 | 전 기간 미확정 | 같은 이름 다른 CIK(오매칭 의심) | 이름·CIK 함께 바뀜(재편·합병 또는 오매칭)
    """
    out = []
    valid = rows["cusip"].astype(str).str.fullmatch(r"[0-9A-Z]{9}") & (rows["cusip"].astype(str) != "000000000")
    for cu, g in rows[valid].groupby("cusip"):  # '000000000'·'N/A' 등 자리표시는 검사 못 함 (외국 법인 등)
        g = g.sort_values("period")
        ciks = [int(c) for c in g["cik"].dropna().unique()]
        n_null = int(g["cik"].isna().sum())
        keys = {match_key(n) for n in g["name"]}
        if len(ciks) == 1:
            v = "일치" if n_null == 0 else "일부 기간 미확정"
        elif not ciks:
            v = "전 기간 미확정"
        elif len(keys) == 1:
            v = "같은 이름 다른 CIK(오매칭 의심)"
        else:
            v = "이름·CIK 함께 바뀜(재편·합병 또는 오매칭)"
        out.append({"cusip": cu, "n_periods": g["period"].nunique(), "ciks": ";".join(map(str, ciks)), "n_cik": len(ciks),
                    "n_null": n_null, "names": ";".join(dict.fromkeys(g["name"])), "first": g["period"].min(),
                    "last": g["period"].max(), "verdict": v})
    return pd.DataFrame(out)


def step_cusip() -> None:
    """NPORT 보유 표의 CUSIP을 키로, 기준일마다 붙은 CIK가 같은지 확인 → cusip_check.csv."""
    match = pd.read_csv(MATCH_CSV)
    rows = []
    for p in sorted(HOLD.glob("*.csv")):
        fund, period = p.stem.split("_", 1)
        d = pd.read_csv(p)
        if "cusip" not in d.columns:
            continue
        d = d[d["class"] == "주식"]
        rows.append(d[["name", "cusip"]].assign(fund=fund, period=period))
    r = pd.concat(rows, ignore_index=True).merge(match[["name", "period", "cik", "status"]], on=["name", "period"], how="left")
    r.loc[r["status"] == "확정(CUSIP)", "cik"] = None  # CUSIP으로 채운 행은 독립 검증에서 뺀다
    r.loc[~r["status"].fillna("").str.startswith("확정"), "cik"] = None
    c = cusip_consistency(r)
    c.to_csv(CUSIP_CSV, index=False, encoding="utf-8-sig")
    print(c["verdict"].value_counts().to_dict())
    same_cik_multi = r.dropna(subset=["cik"]).groupby("cik")["cusip"].nunique()
    print("한 CIK에 CUSIP 2개 이상:", int((same_cik_multi > 1).sum()), "(주식 종류 2개·CUSIP 변경)")
    for _, x in c[c["n_cik"] > 1].iterrows():
        print(" ", x["cusip"], x["ciks"], x["names"][:120], x["first"], x["last"])


# ── 편입·편출 기간 ──────────────────────────────────────────────────────────

INTERVALS_CSV = CACHE / "membership_intervals.csv"


def presence_runs(present: list[bool], max_gap: int = 1) -> list[tuple[int, int, int]]:
    """보고 순서별 있음/없음 → (시작 위치, 끝 위치, 메운 빈칸 수) 목록 (순수).

    사이에 빠진 보고가 max_gap개 이하이면 한 구간으로 잇는다 (한 펀드의 리밸런싱 시차·파싱 누락 흡수).
    """
    idx = [i for i, p in enumerate(present) if p]
    if not idx:
        return []
    runs, s, prev, filled = [], idx[0], idx[0], 0
    for i in idx[1:]:
        if i - prev - 1 <= max_gap:
            filled += i - prev - 1
        else:
            runs.append((s, prev, filled))
            s, filled = i, 0
        prev = i
    runs.append((s, prev, filled))
    return runs


def intervals_for(dates: list[str], present: list[bool]) -> list[dict]:
    """한 종목의 보고 날짜·있음 → 편입·편출 날짜 범위 (순수).

    편입: (앞 보고 날짜 다음날 ~ 처음 보인 날짜). 첫 보고부터 있으면 편입 가장 이른 날 = None(관찰 전).
    편출: (마지막 보인 날짜 ~ 다음 보고 날짜 전날). 마지막 보고까지 있으면 편출 가장 늦은 날 = None(관찰 끝).
    """
    out = []
    for s, e, filled in presence_runs(present):
        d = lambda k: pd.Timestamp(dates[k])  # noqa: E731
        out.append({"entry_earliest": None if s == 0 else (d(s - 1) + pd.Timedelta(days=1)).date().isoformat(),
                    "entry_latest": dates[s], "exit_earliest": dates[e],
                    "exit_latest": None if e == len(dates) - 1 else (d(e + 1) - pd.Timedelta(days=1)).date().isoformat(),
                    "n_reports": e - s + 1 - filled, "gaps_filled": filled})
    return out


def step_intervals() -> None:
    """두 펀드 계열 보고를 합쳐 지수별·종목별 편입·편출 범위 표 → membership_intervals.csv."""
    match = pd.read_csv(MATCH_CSV)
    recs = []
    for p in sorted(HOLD.glob("*.csv")):
        fund, period = p.stem.split("_", 1)
        if not (FIRST_PERIOD <= period <= LAST_PERIOD):
            continue
        index = [x for x in FUNDS if x[0] == fund][0][1]
        d = pd.read_csv(p)
        lo, hi = SIZE_OK[index]
        d = d[d["class"] == "주식"]
        if not lo <= len(d) <= hi:
            continue
        d = d.assign(period=period).merge(match[["name", "period", "cik"]], on=["name", "period"], how="left")
        d["id"] = [f"cik:{int(c)}" if pd.notna(c) else f"key:{match_key(n)}" for c, n in zip(d["cik"], d["name"])]
        recs.append(d[["id", "cik", "name"]].assign(index=index, fund=fund, period=period))
    r = pd.concat(recs, ignore_index=True)
    out = []
    for index, g in r.groupby("index"):
        dates = sorted(g["period"].unique())
        pres = g.groupby("id")["period"].apply(set)
        names = g.groupby("id")["name"].apply(lambda s: ";".join(list(dict.fromkeys(s))[:3]))
        for i, ps in pres.items():
            for iv in intervals_for(dates, [x in ps for x in dates]):
                out.append({"index": index, "id": i, "cik": i[4:] if i.startswith("cik:") else None,
                            "names": names[i], **iv})
        print(index, "보고 날짜", len(dates), dates[0], "~", dates[-1])
    iv = pd.DataFrame(out)
    iv.to_csv(INTERVALS_CSV, index=False, encoding="utf-8-sig")
    for index, g in iv.groupby("index"):
        e_w = (pd.to_datetime(g["entry_latest"]) - pd.to_datetime(g["entry_earliest"])).dt.days.dropna()
        x_w = (pd.to_datetime(g["exit_latest"]) - pd.to_datetime(g["exit_earliest"])).dt.days.dropna()
        print(index, "구간", len(g), "종목", g["id"].nunique(), "CIK 미확정 구간", int(g["cik"].isna().sum()),
              "1회만 보임", int((g["n_reports"] == 1).sum()), "빈칸 메움", int((g["gaps_filled"] > 0).sum()),
              "편입 범위 폭(일) 중앙값·최대", e_w.median(), e_w.max(), "편출", x_w.median(), x_w.max(),
              "관찰 전부터", int(g["entry_earliest"].isna().sum()), "끝까지", int(g["exit_latest"].isna().sum()))


# ── 알려진 공백 ──────────────────────────────────────────────────────────────

GAPS_CSV = CACHE / "known_gaps.csv"
FOREIGN_FORMS = {"20-F", "40-F", "6-K", "20-F/A", "40-F/A"}


def gap_reason(name: str, status: str, foreign: bool) -> str:
    """CIK 미확정 종목의 공백 이유 (순수)."""
    if foreign:
        return "외국 공시(20-F·40-F, 10-K 없음)"
    if re.search(r"\bBank\b|\bBancorp\b|Bancshares", name, re.I) and status in ("활동 없음", "후보 없음"):
        return "은행(규제기관 제출 추정, EDGAR 10-K 없음)"
    if re.search(r"escrow|contra|cvr|rights?\b", name, re.I):
        return "주식 아님(Escrow·CVR)"
    return {"모호": "같은 이름 등록인 여럿(가리지 못함)", "활동 없음": "후보는 있으나 그 시점 10-K·10-Q 없음",
            "후보 없음": "EDGAR 이름 후보 없음"}.get(status, status)


def step_gaps() -> None:
    """확정 못 한 (이름, 기준일)을 종목별로 묶어 알려진 공백 목록 → known_gaps.csv."""
    sw = _sw()
    strict, loose = load_name_index()
    m = pd.read_csv(MATCH_CSV)
    bad = m[~m["status"].str.startswith("확정")]
    fund_of = {}
    for p in HOLD.glob("*.csv"):
        fund, period = p.stem.split("_", 1)
        for n in pd.read_csv(p)["name"].dropna():
            fund_of.setdefault((n, period), set()).add([x for x in FUNDS if x[0] == fund][0][1])
    out = []
    for key, g in bad.groupby("key"):
        name = g["name"].iloc[0]
        cands = strict.get(key, set()) or loose.get(" ".join(sorted(str(key).split())), set())
        foreign = False
        for c in list(cands)[:10]:
            p = sw.SUB_CACHE / f"CIK{int(c):010d}.json"
            if p.exists():
                rows = sw.submissions_rows([json_load(p)])
                foreign |= bool(rows["form"].isin(FOREIGN_FORMS).any())
        idxs = sorted({i for n, per in zip(g["name"], g["period"]) for i in fund_of.get((n, per), set())})
        status = g["status"].mode().iloc[0]
        out.append({"key": key, "name": name, "index": ";".join(idxs), "first": g["period"].min(), "last": g["period"].max(),
                    "n_reports": g["period"].nunique(), "status": status, "reason": gap_reason(name, status, foreign),
                    "candidates": ";".join(map(str, sorted(cands)[:5]))})
    gp = pd.DataFrame(out).sort_values(["reason", "name"])
    gp.to_csv(GAPS_CSV, index=False, encoding="utf-8-sig")
    print(gp["reason"].value_counts().to_dict())


def json_load(p: Path) -> dict:
    """JSON 파일 읽기 (파일)."""
    import json

    return json.loads(p.read_text(encoding="utf-8"))


# ── 보고 ───────────────────────────────────────────────────────────────────

SW1_CIK = ROOT / "data" / "cache" / "sw1" / "cik.csv"
# (지수, 보고 시점 이름, iShares 기준일, Vanguard 기준일)
PAIRS = [("sp400", "2021-12", "2021-12-31", "2021-11-30"), ("sp600", "2021-12", "2021-12-31", "2021-11-30"),
         ("sp400", "2019-12", "2019-12-31", "2019-11-30"), ("sp600", "2019-12", "2019-12-31", "2019-11-30"),
         ("sp400", "2016-12", "2016-12-31", "2016-11-30"), ("sp600", "2016-12", "2016-12-31", "2016-11-30"),
         ("sp400", "2011-12", "2011-12-31", "2011-11-30"), ("sp600", "2011-12", "2011-12-31", "2011-11-30")]
FUND_OF = {("sp400", "i"): "IJH", ("sp600", "i"): "IJR", ("sp400", "v"): "IVOO", ("sp600", "v"): "VIOO"}


def load_snapshot(fund: str, period: str, match: pd.DataFrame) -> pd.DataFrame | None:
    """펀드·기준일 보유 표 + CIK (주식만 id 열: CIK가 확정이면 'cik:N', 아니면 'key:이름키')."""
    p = HOLD / f"{fund}_{period}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d.assign(period=period).merge(match[["name", "period", "cik", "status"]], on=["name", "period"], how="left")
    d["id"] = [f"cik:{int(c)}" if pd.notna(c) else f"key:{match_key(n)}" for c, n in zip(d["cik"], d["name"])]
    return d


def diff_reason(item: str, other_prev: set[str] | None, other_next: set[str] | None) -> str:
    """한 펀드에만 있는 종목의 차이 원인 (순수).

    다른 펀드의 앞·뒤 보고에 있으면 기준일 차이(그 사이 편입·편출), id가 이름 키면 CIK 미확정으로 인한 비교 실패 가능.
    """
    if other_next is not None and item in other_next:
        return "기준일 차이(뒤 보고에 있음)"
    if other_prev is not None and item in other_prev:
        return "기준일 차이(앞 보고에 있음)"
    if item.startswith("key:"):
        return "CIK 미확정(이름 표기 차이 가능)"
    return "설명 안 됨"


def big_members_near(d: str, months: int = 3) -> set[int]:
    """기준일 앞뒤 months개월 안에 나스닥100 ∪ S&P 100에 속한 적 있는 CIK (SW1 캐시, 정책 1장 경계 규칙)."""
    c = pd.read_csv(SW1_CIK, parse_dates=["first", "last"])
    t = pd.Timestamp(d)
    lo, hi = t - pd.DateOffset(months=months), t + pd.DateOffset(months=months)
    hit = c[(c["first"] <= hi) & (c["last"] >= lo) & c["cik"].notna()]
    return set(hit["cik"].astype(int))


def step_report() -> None:
    """종목 수·CIK 매칭률·펀드 간 차이 원인·대형 겹침 제외 수를 출력 (문서에 옮김)."""
    match = pd.read_csv(MATCH_CSV)
    allp = {p.stem: p for p in HOLD.glob("*.csv")}
    print("## 보고별 구성 (주식 줄 / 그 밖)")
    for stem in sorted(allp):
        fund, period = stem.split("_", 1)
        d = load_snapshot(fund, period, match)
        st = d[d["class"] == "주식"]
        other = d.loc[d["class"] != "주식", "class"].value_counts().to_dict()
        s = st["status"].value_counts().to_dict()
        n_cik = st["cik"].nunique()
        print(f"{fund} {period} 줄 {len(d)} 주식 {len(st)} 회사(CIK) {n_cik} 매칭 {s} 그 밖 {other}")
    print("\n## iShares vs Vanguard 차이")
    for index, label, di, dv in PAIRS:
        fi, fv = FUND_OF[(index, "i")], FUND_OF[(index, "v")]
        a, b = load_snapshot(fi, di, match), load_snapshot(fv, dv, match)
        if a is None or b is None:
            print(index, label, "보고 없음")
            continue
        ia, ib = set(a.loc[a["class"] == "주식", "id"]), set(b.loc[b["class"] == "주식", "id"])

        def ids(fund, per):
            x = load_snapshot(fund, per, match)
            return None if x is None else set(x.loc[x["class"] == "주식", "id"])
        # 앞·뒤 보고 (NPORT 구간만 있음)
        prev_v, next_v = ids(fv, str((pd.Timestamp(dv) - pd.offsets.MonthEnd(3)).date())), ids(fv, str((pd.Timestamp(dv) + pd.offsets.MonthEnd(3)).date()))
        prev_i, next_i = ids(fi, str((pd.Timestamp(di) - pd.offsets.MonthEnd(3)).date())), ids(fi, str((pd.Timestamp(di) + pd.offsets.MonthEnd(3)).date()))
        ra = pd.Series([diff_reason(x, prev_v, next_v) for x in ia - ib]).value_counts().to_dict()
        rb = pd.Series([diff_reason(x, prev_i, next_i) for x in ib - ia]).value_counts().to_dict()
        print(f"{index} {label}: 공통 {len(ia & ib)} / {fi}만 {len(ia - ib)} {ra} / {fv}만 {len(ib - ia)} {rb}")
    print("\n## 대형(나스닥100 ∪ S&P 100, 앞뒤 3개월) 겹침 제외")
    for label, di in [("2012-01", "2011-12-31"), ("2017-01", "2016-12-31"), ("2021-12", "2021-12-31")]:
        big = big_members_near(di)
        out = []
        for fund in ("IJH", "IJR"):
            d = load_snapshot(fund, di, match)
            st = d[d["class"] == "주식"]
            ciks = set(st["cik"].dropna().astype(int))
            out.append(f"{fund} 회사 {len(ciks)} 대형 겹침 {len(ciks & big)} → 남음 {len(ciks - big)} (CIK 미확정 {int(st['cik'].isna().sum())})")
        print(label, " / ".join(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["filings", "nport", "nq", "reports", "match", "cusip", "intervals", "gaps", "report"])
    a = ap.parse_args()
    {"filings": step_filings, "nport": step_nport, "match": step_match, "nq": step_nq, "reports": step_reports, "cusip": step_cusip, "intervals": step_intervals, "gaps": step_gaps, "report": step_report}[a.step]()


if __name__ == "__main__":
    main()
