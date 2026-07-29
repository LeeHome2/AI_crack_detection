# AI 시설물 균열 안전점검 시스템

콘크리트 구조물의 균열 및 결함을 AI로 탐지하고, 위험도를 자동 평가하여 안전점검 보고서를 생성하는 Streamlit 웹 애플리케이션입니다.

## 주요 기능

### 1. 결함 탐지 (YOLOv8)
- **6종 복합 결함 탐지**: 균열, 박리/박락, 백태/누수, 철근노출, 강재손상, 도장손상
- **하이브리드 탐지**: 균열 전용 모델 + 복합 결함 모델 조합으로 정확도 향상
- **타일링 추론**: 640px 타일 슬라이스로 고해상도 이미지 처리
- **세그멘테이션**: 균열 정밀 마스크 생성 (선택적)

### 2. 비전 트리아지 (Claude Vision)
- **1차 게이트**: 근접/원거리/흐림/비균열 판정
- **ROI 추출**: 원거리 사진에서 결함 영역만 분리하여 오탐 감소
- **메타데이터 추출**: 구조부재, 재질, 방향 등 보고서용 정보 자동 수집

### 3. 위험도 평가 (Rule Engine)
- **점수 산정**: 결함 유형, 신뢰도, 개수, 크기 등 복합 고려
- **등급 분류**: 정상(A~B), 주의(C), 위험(D), 긴급(E)
- **RAG 연동**: 안전기준 문서 기반 가점 반영

### 4. 보고서 생성 (Claude LLM)
- **7섹션 구조**: 기본현황, 점검결과, 안전등급, 종합의견, 권고사항, 판단근거, 유의사항
- **대화형 정보 수집**: 채팅으로 시설물명, 위치, 점검자 등 입력
- **PDF 내보내기**: 한글 지원, 원본/탐지 이미지 포함

### 5. RAG (Retrieval-Augmented Generation)
- **안전기준 검색**: ChromaDB + Solar 임베딩
- **문서 인용**: 법령, 설계기준, 세부지침 등 출처 명시

## 설치

### 요구사항
- Python 3.10+
- CUDA (GPU 가속, 선택적)

### 환경 설정
```bash
# 가상환경 생성
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 환경변수 (.env)
```env
# 필수: Anthropic API (보고서 생성, 비전 트리아지)
ANTHROPIC_API_KEY=sk-ant-...

# 필수: Upstage API (RAG 임베딩)
UPSTAGE_API_KEY=up_...

# 선택: 기능 토글
HYBRID_DETECT_ENABLED=1      # 하이브리드 탐지 (균열+복합)
SEG_HYBRID_ENABLED=0         # 세그멘테이션 하이브리드
ROI_TRIAGE_ENABLED=1         # ROI 기반 2단계 탐지
MULTITURN_ENABLED=0          # 멀티턴 정보 수집
```

## 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

## 프로젝트 구조

```
crack_detection/
├── app.py                    # Streamlit 메인 앱
├── config.py                 # 전역 설정 (모델 경로, 임계값)
├── schemas.py                # 데이터 구조 (Detection, Report 등)
├── pipeline/
│   ├── orchestrator.py       # 파이프라인 조율
│   ├── detector.py           # YOLOv8 탐지
│   ├── segmenter.py          # 균열 세그멘테이션
│   ├── features.py           # 특징 추출 (스켈레톤, 길이)
│   ├── rules.py              # 위험도 규칙 엔진
│   ├── triage.py             # 비전 트리아지 (1차)
│   ├── roi_triage.py         # ROI 기반 트리아지 (2차)
│   ├── rag.py                # RAG 검색
│   ├── embedder.py           # 임베딩 (Solar/BGE)
│   ├── report.py             # 보고서 생성 (LLM)
│   ├── chat_agent.py         # 대화형 정보 수집
│   ├── pdf_export.py         # PDF 내보내기
│   ├── multiturn.py          # 멀티턴 대화 관리
│   └── postprocess.py        # 후처리 (NMS, 이음새 필터)
├── knowledge/
│   ├── sources/              # 안전기준 문서 (txt)
│   ├── chroma/               # ChromaDB 벡터 저장소
│   └── build_index.py        # 인덱스 빌드 스크립트
├── models/                   # 학습된 가중치 (.pt)
├── runs/                     # 학습 실험 결과
└── docs/                     # 문서 (실험 보고서, 세션 기록)
```

## 모델

### 사용 가중치
| 모델 | 용도 | 성능 (mAP50) |
|------|------|-------------|
| yolov8s_crack_tiled_best.pt | 균열 전용 | 17.9% |
| yolov8s_defect6_tiled_best.pt | 6종 복합 | 13.1% |
| yolov8s_seg_crack_tiled_best.pt | 균열 세그멘테이션 | 72.2% (mask) |

### 탐지 클래스
1. crack (균열)
2. spalling (박리/박락)
3. efflorescence (백태/누수)
4. rebar_exposure (철근노출)
5. steel_defect (강재손상)
6. paint_damage (도장손상)

## 주요 설정 (config.py)

| 설정 | 기본값 | 설명 |
|------|--------|------|
| TILE | 640 | 타일 크기 (px) |
| OVERLAP | 0.2 | 타일 겹침 비율 |
| CONF | 0.05 | 탐지 신뢰도 임계값 |
| RULE_CONF_STRONG | 0.10 | 강한 탐지 임계값 |
| RAG_MATCH_MIN_SCORE | 0.30 | RAG 유사도 임계값 |

## API 제공자

### Claude (Anthropic)
- 보고서 생성 LLM
- 비전 트리아지 (이미지 분석)
- 대화형 정보 수집

### Solar (Upstage)
- RAG 임베딩 (solar-embedding-1-large)
- 보고서 LLM 폴백 (solar-pro2)

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 개발되었습니다.

## 기여자

- 이호민 (201933505)

## 참고 문서

- [앱 아키텍처 설계](앱_아키텍처_설계.md)
- [배포 CD 설계 가이드](배포_CD_설계_가이드.md)
- [위험도 산정 규칙](위험도_산정_규칙.md)
- [LLM 프롬프트 정리](LLM_프롬프트_정리.md)
