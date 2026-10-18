"""arXiv OAI-PMH(metadataPrefix=arXiv) 응답 해석 — 순수 함수 (신기술 레이더 G0).

네트워크·파일은 data/radar/arxiv_oai.py가 맡고, 여기는 받은 XML 바이트를 레코드로 바꾸기만 한다.
날짜 기준은 arXiv 메타데이터의 <created>(= v1 제출일)다. <updated>(수정 버전 날짜)와 OAI <datestamp>는 쓰지 않는다.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "a": "http://arxiv.org/OAI/arXiv/"}
LAST_V1 = "2021-12-31"
_WS = re.compile(r"\s+")


def _text(el) -> str:
    return _WS.sub(" ", el.text).strip() if el is not None and el.text else ""


def parse_list_records(xml: bytes) -> tuple[list[dict], str | None, dict]:
    """ListRecords 응답 → (레코드 목록, 다음 resumption token(없으면 None), 정보).

    레코드: {id, v1_date(YYYY-MM-DD), categories(공백 구분), primary(첫 분야), title, abstract}.
    삭제된 레코드(header status="deleted")는 건너뛴다. 정보: {n_records, n_deleted, complete_size, cursor, error}.
    """
    root = ET.fromstring(xml)
    err = root.find("oai:error", NS)
    info = {"n_records": 0, "n_deleted": 0, "complete_size": None, "cursor": None,
            "error": (err.get("code"), _text(err)) if err is not None else None}
    recs = []
    lr = root.find("oai:ListRecords", NS)
    if lr is None:
        return recs, None, info
    for r in lr.findall("oai:record", NS):
        h = r.find("oai:header", NS)
        if h is not None and h.get("status") == "deleted":
            info["n_deleted"] += 1
            continue
        a = r.find("oai:metadata/a:arXiv", NS)
        if a is None:
            continue
        cats = _text(a.find("a:categories", NS))
        recs.append({"id": _text(a.find("a:id", NS)), "v1_date": _text(a.find("a:created", NS)),
                     "categories": cats, "primary": cats.split(" ")[0] if cats else "",
                     "title": _text(a.find("a:title", NS)), "abstract": _text(a.find("a:abstract", NS))})
    info["n_records"] = len(recs)
    tok = lr.find("oai:resumptionToken", NS)
    token = None
    if tok is not None:
        token = (tok.text or "").strip() or None
        info["complete_size"] = int(tok.get("completeListSize")) if tok.get("completeListSize") else None
        info["cursor"] = int(tok.get("cursor")) if tok.get("cursor") else None
    return recs, token, info


def keep_until(recs: list[dict], last: str = LAST_V1) -> list[dict]:
    """v1 제출일 ≤ last(기본 2021-12-31)인 레코드만. v1 날짜가 비면 버린다."""
    return [r for r in recs if r["v1_date"] and r["v1_date"] <= last]
