"""검색량 실적 예고 저장소 — 가상 장부 DB(data/virtual_books.db, 드라이브 동기화)의 표들 (docs/prereg/search_forecast.md).

- search_trends: 검색어별 주간 값 원값 그대로(받은 시각마다 따로 — 요청마다 0~100 기준이 달라 섞지 않는다)
- search_fetch_log: 검색어·위키 문서별 수집 성공·실패(실패는 기록만, 다음 주에 다시)
- wiki_views: 위키백과 월간 조회수 원값
- search_predictions: 점수·예측 행과 잠금 해시·시각 (종목·방향은 이 DB에만 — 공개 저장소에는 해시만)
- search_scores: 채점(매출 가속 정답·QQQ 대비 초과수익·유효 여부)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "virtual_books.db"  # store.virtual_books.VBOOKS_DB_PATH와 같은 파일

SCHEMA = {
    "search_trends": "ticker TEXT, term TEXT, week TEXT, value REAL, collected_at TEXT, PRIMARY KEY (term, collected_at, week)",
    "search_fetch_log": "kind TEXT, key TEXT, collected_at TEXT, ok INTEGER, rows INTEGER, error TEXT",
    "wiki_views": "ticker TEXT, article TEXT, month TEXT, views INTEGER, collected_at TEXT, PRIMARY KEY (article, collected_at, month)",
    "search_predictions": ("ticker TEXT, fiscal_end TEXT, season TEXT, quarter_start TEXT, yoy REAL, baseline REAL, score REAL, "
                           "base_n INTEGER, computed_at TEXT, score_lock_hash TEXT, prediction TEXT, pred_locked_at TEXT, "
                           "pred_lock_hash TEXT, PRIMARY KEY (ticker, fiscal_end)"),
    "search_scores": ("ticker TEXT, fiscal_end TEXT, season TEXT, prediction TEXT, announced TEXT, valid INTEGER, "
                      "valid_pred_lock INTEGER, actual_end TEXT, accelerated INTEGER, ex_1d REAL, ex_1m REAL, ex_6m REAL, "
                      "scored_at TEXT, PRIMARY KEY (ticker, fiscal_end)"),
}
PRED_COLS = ["ticker", "fiscal_end", "season", "quarter_start", "yoy", "baseline", "score", "base_n", "computed_at",
             "score_lock_hash", "prediction", "pred_locked_at", "pred_lock_hash"]
SCORE_COLS = ["ticker", "fiscal_end", "season", "prediction", "announced", "valid", "valid_pred_lock", "actual_end",
              "accelerated", "ex_1d", "ex_1m", "ex_6m", "scored_at"]


def connect(path=None) -> sqlite3.Connection:
    """DB 연결(표가 없으면 만든다)."""
    p = Path(path) if path is not None else DEFAULT_DB
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    for t, cols in SCHEMA.items():
        conn.execute(f"CREATE TABLE IF NOT EXISTS {t} ({cols})")
    return conn


def add_trends(conn, ticker: str, term: str, weekly: pd.Series, collected_at: str) -> int:
    rows = [(ticker, term, pd.Timestamp(d).date().isoformat(), float(v), collected_at) for d, v in weekly.dropna().items()]
    conn.executemany("INSERT OR REPLACE INTO search_trends VALUES (?,?,?,?,?)", rows)
    return len(rows)


def add_wiki(conn, ticker: str, article: str, monthly: pd.Series, collected_at: str) -> int:
    rows = [(ticker, article, pd.Timestamp(d).strftime("%Y-%m"), int(v), collected_at) for d, v in monthly.dropna().items()]
    conn.executemany("INSERT OR REPLACE INTO wiki_views VALUES (?,?,?,?,?)", rows)
    return len(rows)


def log_fetch(conn, kind: str, key: str, collected_at: str, ok: bool, rows: int = 0, error: str = "") -> None:
    conn.execute("INSERT INTO search_fetch_log VALUES (?,?,?,?,?,?)", (kind, key, collected_at, int(ok), rows, error[:300]))


def latest_pull(conn, term: str) -> pd.Series:
    """검색어 하나의 가장 최근 수집(한 요청) 값만. 없으면 빈 Series. 인덱스 = 주 시작일."""
    r = conn.execute("SELECT MAX(collected_at) FROM search_trends WHERE term = ?", (term,)).fetchone()
    if not r or r[0] is None:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    df = pd.read_sql_query("SELECT week, value FROM search_trends WHERE term = ? AND collected_at = ? ORDER BY week",
                           conn, params=(term, r[0]))
    return pd.Series(df["value"].to_numpy(dtype=float), index=pd.to_datetime(df["week"]))


def predictions(conn) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT {', '.join(PRED_COLS)} FROM search_predictions ORDER BY season, ticker", conn)


def add_prediction(conn, row: dict) -> None:
    conn.execute(f"INSERT INTO search_predictions ({', '.join(PRED_COLS)}) VALUES ({','.join('?' * len(PRED_COLS))})",
                 [row.get(c) for c in PRED_COLS])


def set_season_predictions(conn, labels: dict[tuple[str, str], str], locked_at: str, digest: str) -> None:
    conn.executemany("UPDATE search_predictions SET prediction = ?, pred_locked_at = ?, pred_lock_hash = ? WHERE ticker = ? AND fiscal_end = ?",
                     [(lab, locked_at, digest, t, e) for (t, e), lab in labels.items()])


def scores(conn) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT {', '.join(SCORE_COLS)} FROM search_scores", conn)


def upsert_score(conn, row: dict) -> None:
    conn.execute(f"INSERT OR REPLACE INTO search_scores ({', '.join(SCORE_COLS)}) VALUES ({','.join('?' * len(SCORE_COLS))})",
                 [row.get(c) for c in SCORE_COLS])


def letter_counts(path=None) -> dict | None:
    """월요일 편지 한 줄용 집계(읽기 전용 — 파일·표가 없으면 None).

    출력: {locked: 예측(가속·감속)이 잠긴 행 수, scored: 매출 정답이 나온 행 수, hit: 유효 '매출 가속' 중 실제 가속 비율(없으면 None)}.
    """
    p = Path(path) if path is not None else DEFAULT_DB
    if not p.exists():
        return None
    conn = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    try:
        locked = conn.execute("SELECT COUNT(*) FROM search_predictions WHERE pred_locked_at IS NOT NULL "
                              "AND prediction IN ('매출 가속', '매출 감속')").fetchone()[0]
        scored = conn.execute("SELECT COUNT(*) FROM search_scores WHERE accelerated IS NOT NULL").fetchone()[0]
        h = conn.execute("SELECT COUNT(*), SUM(accelerated) FROM search_scores WHERE prediction = '매출 가속' "
                         "AND valid = 1 AND accelerated IS NOT NULL").fetchone()
    except sqlite3.OperationalError:  # 표가 아직 없음
        return None
    finally:
        conn.close()
    return {"locked": int(locked), "scored": int(scored), "hit": (h[1] / h[0]) if h[0] else None}
