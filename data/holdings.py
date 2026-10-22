"""사용자 시트(「전략공장」) 「QQQM 모으기」 탭 읽기 (2026-10-22 새 출발 — 읽기 전용, 프로그램은 어떤 탭도 쓰지 않는다).

칸은 2행(머리글 줄) 이름으로 찾고 값은 3행에서 읽는다: "보유 주식 수"·"평균단가($)"·"평균 환율(원/$)"·"올해 실현 이익(원)"
(지금 배치: B3·C3·D3·K3). 이름·탭은 config holdings에서 읽는다. 빈칸은 None. 숫자에 쉼표·원·$가 섞여 있어도 읽는다.
"""

from __future__ import annotations

import re

DEFAULTS = {"tab": "QQQM 모으기", "data_row": 3,
            "headers": {"qty": "보유 주식 수", "avg_price": "평균단가($)", "avg_fx": "평균 환율(원/$)",
                        "realized_krw": "올해 실현 이익(원)"}}


def settings(cfg: dict | None) -> dict:
    """config holdings를 기본값과 합친다."""
    c = dict(DEFAULTS)
    c.update((cfg or {}).get("holdings") or {})
    c["headers"] = {**DEFAULTS["headers"], **(c.get("headers") or {})}
    return c


def _num(v) -> float | None:
    s = re.sub(r"[,\s원$₩]", "", str(v if v is not None else ""))
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _cell(values: list[list], r: int, c: int):
    return values[r][c] if r < len(values) and c < len(values[r]) else ""


def parse_holdings(values: list[list], cfg: dict | None = None) -> dict:
    """탭 값(get_all_values 2차원 목록) → {qty, avg_price, avg_fx, realized_krw, warnings}.

    머리글은 2행, 값은 data_row행(기본 3행). 못 찾은 머리글은 None + 경고.
    """
    s = settings(cfg)
    out = {"qty": None, "avg_price": None, "avg_fx": None, "realized_krw": None, "warnings": []}
    head = [str(x).strip() for x in (values[1] if len(values) > 1 else [])]
    row = int(s["data_row"]) - 1
    for key, name in s["headers"].items():
        if name not in head:
            out["warnings"].append(f"{s['tab']} 2행에 「{name}」 머리글이 없습니다")
            continue
        out[key] = _num(_cell(values, row, head.index(name)))
    return out


def read_holdings(cfg: dict | None = None, client=None) -> dict:
    """사용자 시트(GOOGLE_SHEETS_ID) 「QQQM 모으기」을 읽기만 한다. 예외: 인증·네트워크 오류는 그대로(호출부가 잡는다)."""
    from data import sheets

    client = client or sheets.get_client()
    values = client.open_by_key(sheets._sheet_id()).worksheet(settings(cfg)["tab"]).get_all_values()
    return parse_holdings(values, cfg)
