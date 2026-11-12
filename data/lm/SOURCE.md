# Loughran-McDonald 감성 단어 목록 — 출처

- 원본: Loughran-McDonald Master Dictionary w/ Sentiment Word Lists, 1993–2025 (Notre Dame SRAF, 2026-03 갱신)
  - 안내 페이지: https://sraf.nd.edu/loughranmcdonald-master-dictionary/
  - CSV 받은 곳: https://drive.google.com/file/d/1iq2RUf8qGFEAk1g8wQntP3habOnR3fXF/view (안내 페이지의 CSV 링크)
  - 받은 날: 2026-11-12, 86,553단어, 9,093,460바이트
  - 원 CSV sha256: `e2d1328682bab7d2187684fb9f5420bb730401c9eefc00daf835edd203f4859d`
- 이용 조건(안내 페이지): 학술 연구 무료. 상업 이용은 저자에게 별도 허가. 이 저장소는 개인 기록용 꼬리표에만 쓴다.
- `lm_sentiment_words.csv`: 원 CSV에서 `Positive`·`Negative`·`Uncertainty` 열 값이 **양수**(그 목록에 넣은 연도)인 단어만 뽑아 0/1로 적은 것. 음수(그 연도에 목록에서 뺌)는 0. 단어는 대문자. 원 파일은 크기(9MB) 때문에 저장소에 두지 않는다.
  - 긍정 347 · 부정 2,345 · 불확실성 297 (합집합 2,949단어)
  - sha256: `e58ff745c447fe37d71d5df64f9bb96bf035bfe73a849dbc0621b8c612cdc955`
- 정의: `docs/prereg/pead_watch_tags.md` 4절.
