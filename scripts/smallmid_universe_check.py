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
        for form in ("NPORT-P", "N-Q"):
            url = (f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={sid}&type={form}"
                   f"&dateb=&owner=include&count=100&output=atom")
            atom = cached_get(url, RAW / "atom" / f"{sid}_{form}.xml")
            for f in parse_atom_filings(atom):
                if f["form"] != form:  # N-Q/A 등 정정 공시는 따로 표시만
                    f["amended"] = True
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
    soi = [i for i, x in enumerate(lines) if re.match(r"(?i)^schedule of investments", x)] + [len(lines)]
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
        if re.match(r"(?i)^common\s+stocks", x):
            started, buf = True, []
        elif started and NQ_END.match(x):
            break
        elif not started:
            pass
        elif "%" in x or header.match(x) or re.search(NQ_HEAD["sp400"] + "|" + NQ_HEAD["sp600"], x, re.I):
            buf = []  # 업종 제목·쪽 머리: 이름이 아님
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
    """펀드 보고서 종목 이름에서 매칭에 방해되는 꼬리 제거 (순수).

    '/WV', '/The', 'Class A', 'Cl A', '- A' 같은 주 등록지·주식 종류 표기를 뗀다.
    """
    x = str(name)
    x = re.sub(r"/[A-Za-z .]{1,12}/?$", " ", x)            # United Bankshares Inc/WV, Kroger Co/The
    x = re.sub(r"[,\s-]+(?:class|cl)\.?\s+[a-z]\b.*$", " ", x, flags=re.I)
    x = re.sub(r"\s+-\s+[a-z]$", " ", x, flags=re.I)
    return x.strip()


def match_key(name: str) -> str:
    """매칭 키 = 꼬리 제거 뒤 SW1·S&P 500 점검과 같은 norm_name (순수)."""
    return _sw().norm_name(clean_fund_name(name))


def load_name_index() -> dict[str, set[int]]:
    """EDGAR cik-lookup-data(옛 이름 포함 모든 등록인) → {키: {cik}} (네트워크 1회·파일 캐시)."""
    text = cached_get(CIK_LOOKUP_URL, RAW / "cik-lookup-data.txt")
    idx: dict[str, set[int]] = {}
    for line in text.splitlines():
        m = re.match(r"^(.*):(\d{10}):\s*$", line)
        if not m:
            continue
        k = match_key(m.group(1))
        if k:
            idx.setdefault(k, set()).add(int(m.group(2)))
    return idx


def active_near(rows: pd.DataFrame, rep_date: str, days: int = 400) -> bool:
    """submissions 행 표에 rep_date ± days 안에 10-K·10-Q 제출이 있는지 (순수)."""
    if rows.empty:
        return False
    d = pd.to_datetime(rows.loc[rows["form"].isin(ACTIVE_FORMS), "filing_date"])
    r = pd.Timestamp(rep_date)
    return bool(((d >= r - pd.Timedelta(days=days)) & (d <= r + pd.Timedelta(days=days))).any())


def resolve(cands: set[int], rep_date: str, active) -> tuple[int | None, str]:
    """후보 CIK들 → (CIK, 상태). active(cik, rep_date) → bool. 상태: 확정 | 모호 | 활동 없음 | 후보 없음 (순수)."""
    if not cands:
        return None, "후보 없음"
    act = sorted(c for c in cands if active(c, rep_date))
    if len(act) == 1:
        return act[0], "확정"
    if len(act) > 1:
        return None, "모호"
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


def step_match() -> None:
    """보유 표의 주식 이름 → CIK. 이름 키 일치 → 그 시점 10-K·10-Q 제출로 확인, 없으면 EDGAR 이름 검색 보조."""
    sw = _sw()
    _, ua = _edgar()
    idx = load_name_index()
    hold = []
    for p in sorted(HOLD.glob("*.csv")):
        fund, period = p.stem.split("_", 1)
        d = pd.read_csv(p)
        d = d[d["class"] == "주식"]
        hold.append(pd.DataFrame({"fund": fund, "period": period, "name": d["name"], "cusip": d.get("cusip")}))
    h = pd.concat(hold, ignore_index=True)
    rows_cache: dict[int, pd.DataFrame] = {}
    failed: list[int] = []

    def active(cik: int, rep: str) -> bool:
        if cik not in rows_cache:
            try:
                rows_cache[cik] = sw.submissions_rows(sw.fetch_submissions(cik, ua))
            except Exception:  # noqa: BLE001 — 실패 CIK는 모아 보고
                failed.append(cik)
                rows_cache[cik] = pd.DataFrame(columns=["form", "filing_date"])
        return active_near(rows_cache[cik], rep)

    # 이름마다 한 번: 그 이름이 처음 나온 기준일로 확인
    first = h.groupby("name")["period"].min()
    prefetch_submissions({c for n in first.index for c in idx.get(match_key(n), set())}, ua)
    out = []
    for i, (name, rep) in enumerate(first.items()):
        k = match_key(name)
        cik, status = resolve(idx.get(k, set()), rep, active)
        how = "이름 키"
        if cik is None:
            hits = [c for c, n in sw.name_search(clean_fund_name(name), ua) if sw.name_sim(n, clean_fund_name(name)) >= 0.85]
            c2, s2 = resolve(set(hits), rep, active)
            if c2 is not None:
                cik, status, how = c2, s2, "이름 검색"
            elif status == "후보 없음":
                status = s2
        out.append({"name": name, "key": k, "first_period": rep, "cik": cik, "status": status, "how": how})
        if i % 200 == 0:
            print(i, len(first))
    m = pd.DataFrame(out)
    m.to_csv(MATCH_CSV, index=False, encoding="utf-8-sig")
    print(m["status"].value_counts().to_dict(), m["how"].value_counts().to_dict())
    if failed or sw.SEARCH_FAILED:
        print("submissions 실패 CIK:", failed[:50], "이름 검색 실패:", sw.SEARCH_FAILED[:50])


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
    d = d.merge(match[["name", "cik", "status"]], on="name", how="left")
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
    ap.add_argument("--step", required=True, choices=["filings", "nport", "nq", "match", "report"])
    a = ap.parse_args()
    {"filings": step_filings, "nport": step_nport, "match": step_match, "nq": step_nq, "report": step_report}[a.step]()


if __name__ == "__main__":
    main()
