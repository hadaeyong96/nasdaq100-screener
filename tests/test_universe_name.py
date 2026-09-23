"""P1.2: nasdaq_official 종목명 꼬리 제거 테스트. 네트워크 없이 문자열만 검사한다."""

from __future__ import annotations

import pytest

from data.universe import shorten_company_name

CASES = [
    ("Apple Inc. Common Stock", "Apple Inc."),
    ("Alphabet Inc. Class A Common Stock", "Alphabet Inc."),
    ("Alphabet Inc. Class C Capital Stock", "Alphabet Inc."),
    ("Marvell Technology, Inc. Common Stock", "Marvell Technology, Inc."),
    ("Cisco Systems, Inc. Common Stock (DE)", "Cisco Systems, Inc."),
    ("Seagate Technology Holdings PLC Ordinary Shares (Ireland)", "Seagate Technology Holdings PLC"),
    ("PDD Holdings Inc. American Depositary Shares", "PDD Holdings Inc."),
    ("ASML Holding N.V. New York Registry Shares", "ASML Holding N.V."),
    ("Shopify Inc. Class A Subordinate Voting Shares", "Shopify Inc."),
    ("Strategy Inc Common Stock Class A", "Strategy Inc"),
    ("Warner Bros. Discovery, Inc. Series A Common Stock ", "Warner Bros. Discovery, Inc."),
    ("Thomson Reuters Corporation Common Shares", "Thomson Reuters Corporation"),
    ("Nebius Group N.V. Class A Ordinary Shares", "Nebius Group N.V."),
    ("Monster Beverage Corporation", "Monster Beverage Corporation"),  # 뗄 꼬리가 없는 경우
]


@pytest.mark.parametrize("raw,expected", CASES)
def test_shorten_company_name(raw: str, expected: str):
    assert shorten_company_name(raw) == expected


def test_never_returns_empty():
    """이름 전체가 꼬리뿐이면 원본을 그대로 돌려준다 (빈 이름 방지)."""
    assert shorten_company_name("Common Stock") == "Common Stock"
