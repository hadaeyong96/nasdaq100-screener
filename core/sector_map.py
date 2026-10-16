"""EDGAR SIC 코드 → 11개 업종 대분류 (탐색 B 「강한 업종 속 강한 종목, 첫 눌림 매수」 0단계).

순수 함수만 둔다(네트워크·파일 접근 없음). 매핑표는 configs/sector_pullback_sic_map.csv
(결과 보기 전 고정)를 호출부가 읽어 DataFrame으로 넘긴다.

규칙: SIC가 들어가는 범위(양 끝 포함) 중 가장 좁은 범위의 업종. 어느 범위에도 없거나 SIC가
없으면 None(= 매핑 실패, 보고 대상).
"""

from __future__ import annotations

import pandas as pd

SECTORS = ("기술", "통신", "소비재량", "필수소비", "헬스", "금융", "산업", "에너지", "소재", "유틸리티", "부동산")
UNMAPPED = "매핑 실패"


def validate_rules(rules: pd.DataFrame) -> None:
    """매핑표 점검. 입력: DataFrame(sic_from, sic_to, sector) / 출력: 없음, 잘못되면 ValueError."""
    bad = rules[~rules["sector"].isin(SECTORS)]
    if len(bad):
        raise ValueError(f"11개 대분류에 없는 업종: {sorted(set(bad['sector']))}")
    if (rules["sic_from"] > rules["sic_to"]).any():
        raise ValueError("sic_from > sic_to 인 줄이 있습니다")
    width = rules["sic_to"] - rules["sic_from"]
    dup = rules.assign(w=width).duplicated(["sic_from", "sic_to"], keep=False)
    if dup.any():
        raise ValueError("같은 범위가 두 번 있습니다 — 가장 좁은 범위가 하나로 정해지지 않습니다")


def sic_to_sector(sic, rules: pd.DataFrame) -> str | None:
    """SIC 하나 → 업종. 입력: sic(int·str·None), rules(sic_from, sic_to, sector) / 출력: 업종 또는 None."""
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return None
    hit = rules[(rules["sic_from"] <= code) & (rules["sic_to"] >= code)]
    if hit.empty:
        return None
    return str(hit.loc[(hit["sic_to"] - hit["sic_from"]).idxmin(), "sector"])


def monthly_sector_counts(members: dict[pd.Timestamp, set[str]], sector_of: dict[str, str | None]) -> pd.DataFrame:
    """달별 업종 종목 수. 입력: members {달(명단을 쓰는 달): 티커 집합}, sector_of {티커: 업종 또는 None}
    출력: DataFrame(인덱스 달, 열 SECTORS + UNMAPPED, 값 종목 수)."""
    rows = {}
    for month, tickers in sorted(members.items()):
        row = dict.fromkeys((*SECTORS, UNMAPPED), 0)
        for t in tickers:
            row[sector_of.get(t) or UNMAPPED] += 1
        rows[month] = row
    return pd.DataFrame.from_dict(rows, orient="index")[[*SECTORS, UNMAPPED]]


def summarize_counts(counts: pd.DataFrame, min_members: int) -> pd.DataFrame:
    """업종별 월 평균·최소·최대 종목 수와 min_members 미만인 달 수.
    입력: monthly_sector_counts 결과, min_members / 출력: DataFrame(인덱스 업종)."""
    return pd.DataFrame({
        "월 평균": counts.mean().round(1),
        "최소": counts.min(),
        "최대": counts.max(),
        f"{min_members}종목 미만 달": (counts < min_members).sum(),
    })
