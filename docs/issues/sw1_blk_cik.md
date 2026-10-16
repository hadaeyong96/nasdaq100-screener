# SW1 BLK CIK 오류 (기록만 — 고치지 않음, 판정 불변)

- 발견: 2026-10-16, 탐색 B(research/sector-pullback) 0단계 업종 매핑 중. 사용자 결정 5: 고치지 말고 내용·영향 범위만 기록.

## 내용

- SW1 `data/cache/sw1/cik.csv`의 BLK CIK = **1370755** (출처 "S&P 500 점검 identity.csv").
- EDGAR에서 CIK 1370755는 **BlackRock TCP Capital Corp.**(옛 이름 TCP Capital Corp., Special Value Continuation Fund) — BlackRock, Inc.가 아닌 다른 회사(사업개발회사).
- 올바른 CIK: BlackRock, Inc. **1364742** (2024년 지주회사 개편 전 법인, 지금 이름 BlackRock Finance, Inc.). 개편 뒤 새 지주회사는 2012383.
- 이름 유사도 점검으로는 걸리지 않았다: "BlackRock TCP Capital"이 "BlackRock"으로 시작해 유사도 1.0. EDGAR 현재 SIC가 비어 있어서 드러났다.

## 영향 범위 (SW1, 2012~2021 탐색·판정)

| 항목 | BLK 건수 | 비고 |
| --- | --- | --- |
| `events.csv` 제출 기록 | 82 (8-K 2.02 43 · 10-Q 29 · 10-K 10) | 2012-05-14 ~ 2022-02-24, 모두 TCP Capital의 제출 |
| `announcements.csv` 실적 발표 | 24 (판정 24) | 2016 ~ 2021 (BLK가 S&P 100에 있던 기간) |
| 신호 (636건 중) | **0** | `signal` · `signal_literal` 모두 0 |
| `signals_eps_ref.csv` · `sample20_eps.csv` | 0 | |

- 매매·신호에는 들어가지 않았다. 다만 TCP Capital의 발표일 24건이 BLK(BlackRock) 가격으로 "신호 없음" 판정을 받았으므로,
  "신호 없음" 쪽을 쓰는 대조(예: 같은 주 발표·신호 없음 대조군)에 최대 24건이 잘못된 날짜로 섞였을 수 있다.
- 봉인 ⓐ(`data/cache/sealed_a/sw1/cik.csv`)는 BLK CIK 2012383(BlackRock, Inc.)을 썼다 → 영향 없음.

## 하지 않은 것

- SW1 캐시·결과·판정은 바꾸지 않았다. 탐색 B는 자기 표(`configs/sector_pullback_cik_candidates.csv`)에서 1364742로 바로잡아 쓴다.
