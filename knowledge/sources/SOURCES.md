# RAG 원문 출처 아카이브 — 다운로드 목록 & 매니페스트

> 목적: RAG 지식베이스의 **원문(1차 출처)을 이 폴더에 보관**해 근거 추적성을 확보한다.
> `knowledge/raw/`(요약·발췌 노트, 임베딩됨) ↔ `knowledge/sources/`(원문 PDF, 보관용).
> ✅ **아카이브 완료 (2026-07-23): S1~S12 전 12건 원문을 로컬 브라우저/curl로 확보해 이 폴더에 반영.** PDF 9건은 git-lfs로 추적.
> 저장 규칙: 아래 "저장 파일명" 그대로 `knowledge/sources/`에 저장하면 매니페스트와 일치.

---

## A. 복합 결함(2차 MVP) 핵심 출처 — ✅ 확보 완료

| # | 문서(기준명) | 발행처·연도 | 원문 링크 | 저장 파일명 | 인용 RAG 문서 |
|---|---|---|---|---|---|
| S1 | 시설물의 안전 및 유지관리 실시 세부지침 (안전점검·진단 편) | 국토안전관리원 · 2021.12 | http://www.assi.or.kr/download/law/시설물의 안전 및 유지관리 실시 세부지침(안전점검·진단편).pdf | `세부지침_2021_안전점검진단편.pdf` | 12·13·14·15·16·17 |
| S2 | 안전점검 및 정밀안전진단 세부지침 (2017) | 국토안전관리원 · 2017.01 | https://www.codil.or.kr/filebank/original/RK/OTKCRK180849/OTKCRK180849.pdf | `세부지침_2017.pdf` | 12·13·14·15 (보조) |
| S3 | 건축물 안전점검 및 정밀안전진단 세부지침 | 건설교통부/시설안전공단 · 2003 | https://www.codil.or.kr/filebank/moct2014/200402/MOCT737_2.PDF?nserialno=737 (구 링크는 오류, `?nserialno=737` 필요) | `세부지침_2003_건축물_MOCT737.pdf` | 03·08·12·13·14·15 |
| S4 | 콘크리트구조 내구성 설계기준 (KDS 14 20 40 계열) | 국토교통부/한국콘크리트학회 | https://www.kira.or.kr/bbs/download.do?fId=118651&dFlag=1 | `콘크리트구조_내구성설계기준.pdf` | 12·14 (부식·수분) |
| S5 | 안전점검 및 정밀안전진단 세부지침 해설서(교량) | 국토안전관리원 | https://www.codil.or.kr/filebank/original/MA/OTKCMA150046/OTKCMA150046.pdf | `세부지침_해설서_교량.pdf` | 16·17 (강재·도장) |
| S6 | '14~'15 정밀안전진단 실시결과 평가사례집 | 국토안전관리원(KALIS) | https://www.kalis.or.kr/www/brd/m_435/down.do?brd_id=tech0207&seq=41&data_tp=A&file_seq=1 | `정밀안전진단_평가사례집_14-15.pdf` | 15 (복합 사례) |

## B. 균열(1차 MVP) 출처 — 기존 반영본 (참고·보관)

| # | 문서 | 발행처 | 원문 링크 | 저장 파일명 | 인용 RAG 문서 |
|---|---|---|---|---|---|
| S7 | 콘크리트구조 설계기준(허용균열폭) | 한국콘크리트학회 | https://www.codil.or.kr/filebank/construction/DC/CIGCDC410001/CIGCDC410001.pdf | `콘크리트구조_설계기준.pdf` | 01·07 |
| S8 | 콘크리트 균열보수재료·공법 선정방법 연구 | CODIL | https://www.codil.or.kr/filebank/original/RK/OTKNRK500317/OTKNRK500317.pdf | `균열보수공법_연구.pdf` | 02·05 |
| S9 | "콘크리트 균열폭 0.3mm 이하 허용" | 대한전문건설신문 | http://www.koscaj.com/news/articleView.html?idxno=54369 | `koscaj_허용균열폭_54369.html` | 01 |
| S10 | "콘크리트 균열 하자기준 0.3mm 통일" | 대한전문건설신문 | https://www.koscaj.com/news/articleView.html?idxno=214761 | `koscaj_하자기준_214761.html` | 10 |
| S11 | 콘크리트균열보수 전문시방서 (LHCS 14 20 10 25:2020) | 한국토지주택공사(LH)/KCSC | https://www.kcsc.re.kr/standardCode/viewer/LHCS%2014%2020%2010%2025:2020-12-09 (KCSC 로그인 후 PDF 다운로드) | `LH_균열보수_시방서_LHCS14201025.pdf` | 11 |
| S12 | 공동주택 하자의 조사·보수비용 산정 및 하자판정 기준 (고시 2016-1048) | 국토교통부 | https://www.law.go.kr/LSW/admRulInfoP.do?admRulSeq=2100000072850 (첨부파일 전문, **HWP 원본**) | `국토부고시_2016-1048_하자판정기준.hwp` | 10 |

---

## 사용 방법

1. 위 표의 링크에서 각 문서를 받아 **"저장 파일명" 그대로** 이 폴더(`knowledge/sources/`)에 저장.
2. HTML 기사(S9·S10)는 브라우저 "다른 이름으로 저장(웹페이지, HTML)"로 저장하거나 PDF 인쇄로 보관.
3. PDF는 저장소 `.gitignore`에서 `*.pdf`로 제외되므로 **git-lfs로 관리**한다.
   - `.gitattributes`: `knowledge/sources/*.pdf filter=lfs diff=lfs merge=lfs -text`
   - `.gitignore`: `!knowledge/sources/*.pdf` 예외 추가로 이 폴더 PDF만 추적 허용.
4. 저장 후 커밋: `git add knowledge/sources/ && git commit -m "docs: RAG 원문 출처 아카이브" && git push` (LFS 객체 함께 업로드됨).

> 인용 원칙: `knowledge/raw/`의 각 문서 @meta에 기준명·발행처·URL이 있고, 이 매니페스트가 원문 파일과 1:1로 연결한다. 보고서·앱은 @meta의 기준명·URL을 인용하며, 원문 검증이 필요하면 이 폴더의 원문을 참조한다.
