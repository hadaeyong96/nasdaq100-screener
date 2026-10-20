# 신기술 레이더 G0-보완 표본 검수 (2026-10-20, 사용자·Claude)

대상: `docs/research/radar_g0b_terms_sample50.csv`, `docs/research/radar_g0b_link_sample50.csv` (시드 20261019 무작위 50개씩).

## 판정

- 연결 표본 50 정확도 **22/50 = 44%** < 50%(검수 전 정한 기준) → 투자 신호 연구(G1~G3) **중단**, 관심도 대시보드로만 유지.

## 단어 표본 50

**기술 23 · 조각 5 · 학문 7 · 상투어 15 · 사건 0** (산업 기술 46%, G0 36%). 줄별 분류는 받지 않아 CSV 분류 칸은 비워 둔다.

## 연결 표본 50

**맞음 22 · 무관 28** — 링크 CSV 분류 칸에 줄마다 "맞음"/"무관"으로 넣었다.

맞음: ADNT, NOC, SANM, MSTR, EGHT, CEVA×3, EXPO, APTV, MU, MDRX(language understanding), NUAN, JNPR, QCOM, QRVO, ALOG, XRX, VIAV×2, CIEN, FARO, DSPG.
(표본 CSV의 CEVA 줄은 2개(computer vision, speech recognition)라, 줄 단위로 넣으면 맞음 22가 된다 — 받은 합계 22와 같다.)

## 발견 (새 가설 후보 — 이번 데이터로 재시도 금지)

- 진짜 기술 용어 행만 보면 약 **22/27** 맞음. 틀린 연결 대부분은 상투어·일반 영어 단어("near term", "key component", "challenging due" 등)에서 나왔다.
- 새 가설 후보: **"결과 전에 고정한 기술 용어 사전으로 단어 선택"**. 이 표본을 보고 사전을 만들면 같은 데이터에 맞춘 것이 되므로, 같은 데이터로 재시도하지 않는다.
