"""arXiv 메타데이터 수집기 (신기술 레이더 G0) — 공식 OAI-PMH, 이어받기·재시도.

    python -m data.radar.arxiv_oai harvest [초]       # 처음부터 또는 저장된 token에서 이어받기 (초: 그 시간 뒤 쪽 사이에서 멈춤)
    python -m data.radar.arxiv_oai window 2015-03-02 2015-03-03 out_dir   # 재현성 확인용 datestamp 구간 수집

- 주소: export.arxiv.org/oai2 (지시) → 지금은 oaipmh.arxiv.org/oai로 301 이동한다(같은 공식 서비스). 이동한 주소로 직접 부른다.
- OAI-PMH는 datestamp(메타데이터 수정일)로만 고를 수 있어 전체를 받고, v1 제출일(<created>) ≤ 2021-12-31만 남긴다.
- 요청 간격 3초 이상, 실패하면 30·60·120·300·600초 기다려 다시(503 Retry-After는 그 값과 3초 중 큰 값).
- 저장(data/cache/arxiv, git 제외): raw/page_NNNNN.xml.gz(원래 응답), parsed/page_NNNNN.parquet(id·v1 날짜·분야·제목·요약),
  state.json(다음 token·쪽 수), runs.jsonl(실행 기록: run_id·시작·끝·코드 버전·건수).
- 가격·수익 캐시는 열지 않는다(data.radar.guard).
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.radar import arxiv_collect as ac  # noqa: E402
from data.radar import guard  # noqa: E402

BASE = "https://oaipmh.arxiv.org/oai"
OUT = ROOT / "data" / "cache" / "arxiv"
MIN_GAP = 3.0
BACKOFF = [30, 60, 120, 300, 600]
UA = "nasdaq100-screener radar-g0 (research; OAI-PMH metadata harvest)"


def code_version() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    d = subprocess.run(["git", "status", "--porcelain", "--", "core/radar", "data/radar"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() + ("+dirty" if d.stdout.strip() else "")


class Fetcher:
    """요청 간격·재시도를 지키는 GET."""

    def __init__(self):
        import requests

        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.last = 0.0

    def get(self, params: dict) -> bytes:
        for k, wait in enumerate([0] + BACKOFF):
            if wait:
                print(f"  재시도 {k}/{len(BACKOFF)}: {wait}초 기다림", flush=True)
                time.sleep(wait)
            gap = time.monotonic() - self.last
            if gap < MIN_GAP:
                time.sleep(MIN_GAP - gap)
            try:
                r = self.s.get(BASE, params=params, timeout=120)
            except Exception as exc:  # noqa: BLE001 — 네트워크 실패는 재시도, 끝내 실패하면 멈춤
                self.last = time.monotonic()
                print(f"  요청 실패: {type(exc).__name__}", flush=True)
                continue
            self.last = time.monotonic()
            if r.status_code == 503:
                ra = r.headers.get("Retry-After")
                time.sleep(max(MIN_GAP, float(ra) if ra and ra.isdigit() else 30))
                continue
            if r.status_code == 200:
                return r.content
            print(f"  HTTP {r.status_code}", flush=True)
        raise RuntimeError("arXiv OAI 요청이 재시도 끝에 실패했습니다 — state.json에서 이어받을 수 있습니다")


def _save_page(n: int, raw: bytes, recs: list[dict]) -> None:
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    (OUT / "parsed").mkdir(parents=True, exist_ok=True)
    (OUT / "raw" / f"page_{n:05d}.xml.gz").write_bytes(gzip.compress(raw))
    pd.DataFrame(recs, columns=["id", "v1_date", "categories", "primary", "title", "abstract"]).to_parquet(
        OUT / "parsed" / f"page_{n:05d}.parquet", index=False)


def harvest(max_seconds: float | None = None) -> None:
    """max_seconds가 있으면 그 시간이 지난 뒤 쪽 사이에서 깨끗이 멈춘다(다음 실행이 이어받음)."""
    guard.install()
    OUT.mkdir(parents=True, exist_ok=True)
    state_p = OUT / "state.json"
    st = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}
    if st.get("done"):
        print("이미 끝까지 받았습니다 (state.json done)")
        return
    run = {"run_id": uuid.uuid4().hex[:12], "started": datetime.now(timezone.utc).isoformat(), "code_version": code_version(),
           "resumed_from_page": st.get("pages", 0), "pages": 0, "records": 0, "kept": 0, "deleted": 0}
    f = Fetcher()
    page = st.get("pages", 0)
    token = st.get("token")
    t0 = time.time()
    while True:
        params = {"verb": "ListRecords", "resumptionToken": token} if token else {"verb": "ListRecords", "metadataPrefix": "arXiv"}
        raw = f.get(params)
        recs, nxt, info = ac.parse_list_records(raw)
        if info["error"]:
            raise RuntimeError(f"OAI 오류 {info['error']}")
        kept = ac.keep_until(recs)
        page += 1
        _save_page(page, raw, kept)
        run["pages"] += 1
        run["records"] += info["n_records"]
        run["kept"] += len(kept)
        run["deleted"] += info["n_deleted"]
        st = {"pages": page, "token": nxt, "complete_size": info["complete_size"], "cursor": info["cursor"],
              "done": nxt is None, "run_id": run["run_id"]}
        state_p.write_text(json.dumps(st), encoding="utf-8")
        if page % 20 == 0 or nxt is None:
            print(f"쪽 {page} · 이번 실행 레코드 {run['records']:,} · 남김 {run['kept']:,} · 전체 {info['complete_size']} · "
                  f"{time.time() - t0:.0f}초", flush=True)
        if nxt is None:
            break
        token = nxt
        if max_seconds is not None and time.time() - t0 > max_seconds:
            print(f"시간 제한 {max_seconds:.0f}초 — 쪽 {page}에서 멈춤(이어받기 가능)", flush=True)
            break
    run["ended"] = datetime.now(timezone.utc).isoformat()
    with open(OUT / "runs.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(run, ensure_ascii=False) + "\n")
    print(f"끝: {run}")


def window(frm: str, until: str, out_dir: str) -> None:
    """재현성 확인: datestamp 구간을 따로 받아 out_dir(data/cache/arxiv 아래)에 parsed parquet로 저장."""
    guard.install()
    out = OUT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    f = Fetcher()
    params = {"verb": "ListRecords", "metadataPrefix": "arXiv", "from": frm, "until": until}
    recs_all, n = [], 0
    while True:
        raw = f.get(params)
        recs, nxt, info = ac.parse_list_records(raw)
        if info["error"]:
            if info["error"][0] == "noRecordsMatch":
                break
            raise RuntimeError(f"OAI 오류 {info['error']}")
        recs_all += recs
        n += 1
        if nxt is None:
            break
        params = {"verb": "ListRecords", "resumptionToken": nxt}
    df = pd.DataFrame(recs_all, columns=["id", "v1_date", "categories", "primary", "title", "abstract"])
    df.to_parquet(out / "records.parquet", index=False)
    print(f"구간 {frm}~{until}: 쪽 {n}, 레코드 {len(df):,}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    if sys.argv[1] == "harvest":
        harvest(float(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif sys.argv[1] == "window":
        window(sys.argv[2], sys.argv[3], sys.argv[4])
