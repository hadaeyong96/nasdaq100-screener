# 규칙 문서(moat_paper.md) 해시 변경 기록

`scripts/moat_paper.py`는 시작 파일(`moat_paper_start.json`)의 `rules_doc_hash`와 지금 문서 해시가
다르면 경고한다. 아래 표에 "이전 해시 → 현재 해시" 쌍으로 적힌 변경은 내용이 검토된 알려진 변경이라
경고 대신 "알려진 변경(사유: hash_notes.md)" 한 줄만 출력한다. 표에 없는 불일치는 지금처럼 경고한다.

| 기록일 | 해시 변경 | 원인 | 규칙 내용 변경 |
|---|---|---|---|
| 2026-10-02 | `102f78f88e2e7e07` → `51ff52148b926add` | 줄바꿈 차이(LF → CRLF). 시작 파일 해시는 커밋 09317a2(= 현재 HEAD) 문서를 LF로 읽은 값이고, 지금 작업 사본은 `core.autocrlf=true`라 CRLF로 체크아웃돼 바이트가 다르다. 확인: `git show 09317a2:docs/design/moat_paper.md`의 해시가 LF 그대로 `102f78f8…`, CRLF로 바꾸면 `51ff5214…`. 처음에는 09317a2 커밋(설정 키 경로 `moat.` → `moat_backtest.`)이 원인으로 보였으나, 시작 파일은 이미 09317a2 이후 내용으로 만들어졌다(09317a2 이전 8cea71a 문서의 해시는 `0bb5b2ccf7c86b04`) | 없음 |
