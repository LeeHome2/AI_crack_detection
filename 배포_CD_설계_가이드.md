# 배포 · CD 설계 가이드 — AI 균열 안전점검 시스템 (GCE)

> 작성 2026-07-20 (Cowork 원격 세션). FinBrief(업스테이지 팀8) CD 파이프라인을 이 프로젝트에 맞게 **린(lean) 버전**으로 이식하기 위한 설계 문서. **구현은 RAG 실인덱스·보고서(API) 안정화 이후 착수.** 지금은 사양·비용·단계·초안 설정만 정리한다.

---

## 0. 한눈에 — 전체 워크플로우

```
[데스크탑 RTX 3070 Ti]  GPU 학습 (YOLO 타일)  ──┐  best.pt 커밋
[노트북]               RAG·보고서·features CPU 작업 ─┤  코드 push
                                                    ▼
                                          GitHub (main, 중앙 저장소)
                                                    ▼  push 트리거
                                    GitHub Actions CD (빌드→GHCR→GCE 배포)
                                                    ▼
                                      GCE 인스턴스 (Docker, Streamlit :8501)
                                                    ▼
                                     📱 폰으로 URL 접속해 실사진 테스트·개선
```

핵심: GitHub이 중앙(이미 구축 완료), **학습만 데스크탑 GPU**, 나머지는 어느 기기든, 배포는 push마다 자동. 배포 URL이 생기면 기존 "폰 실사용 테스트" 루프가 데스크탑 없이도 돌아간다.

---

## 1. FinBrief에서 그대로 가져올 것 vs 바꿀 것

이호민 님이 팀원이었던 FinBrief의 `cd.yml`은 이미 검증된 골격이다. 재사용하되 균열 앱 특성에 맞게 조정한다.

| 항목 | FinBrief | 균열 앱(이 프로젝트) |
|---|---|---|
| 앱 형태 | FastAPI API (+Discord bot, scheduler) | **Streamlit 단일 앱** |
| 포트 | 8000 | **8501** |
| 헬스체크 | `/api/v1/health` | **`/_stcore/health`** (Streamlit 내장) |
| 서비스 수 | 3개(api·bot·scheduler) | **1개** (단일 서비스) |
| 무거운 런타임 의존성 | 스텁 모드로 경량 | **BGE-m3(2GB+)·torch·YOLO·ChromaDB** |
| GPU | 불필요 | 추론은 **CPU로 충분**(학습만 데스크탑) |
| 배포 방식 | GHCR 이미지 → GCE SSH → compose up → 헬스 → 자동 롤백 | **동일 골격 재사용** |

**린 버전 원칙**: 자동 롤백·`.current_image`/`.previous_image` 보존·배포 전 `docker prune`(디스크 확보)은 FinBrief 것을 그대로 유지(검증된 안전장치라 가치 큼). 대신 멀티서비스·시크릿 매트릭스·CI 테스트 잡은 최소화한다.

---

## 2. 인스턴스 사양 & 비용 (Solar 임베딩 기준)

**임베딩을 Upstage Solar API로 돌리기로 결정 →** BGE-m3(2.3GB 모델 + RAM 2GB)가 통째로 빠진다. 런타임에 남는 무거운 요소는 **YOLO CPU 추론(torch+ultralytics)**뿐이고, YOLOv8s는 작아서(~50MB) RAM 부담이 훨씬 작다. 그만큼 인스턴스를 낮출 수 있다.

| 인스턴스 | vCPU/RAM | 적합성 | 월 비용(24/7) |
|---|---|---|---|
| e2-micro | 2/1GB | ⚠️ torch 로드엔 빠듯 | ~$7 (프리티어) |
| **e2-small** | **2/2GB** | ✅ **권장(Solar 기준 하한)** | ~$14 |
| e2-medium | 2/4GB | ✅ 여유·추론 안정 | ~$28 |
| e2-standard-2 | 2/8GB | 로컬 BGE로 돌릴 때만 필요 | ~$49 |

- 디스크: **표준 PD 20~30GB**(torch 이미지 포함). ~$1/월.
- **비용 절감 핵심**: 항상 켜둘 필요 없음. 테스트할 때만 `gcloud compute instances start`, 끝나면 `stop`. 정지 중엔 디스크 비용(~$1)만.
- **더 줄이려면(선택)**: `best.pt`를 ONNX로 export하고 detector를 onnxruntime로 바꾸면 torch까지 제거 → 이미지 수백MB, e2-micro도 가능. 지금은 불필요, 여력 되면.
- ⚠️ Solar는 외부 API라 런타임에 **네트워크 + 유효한 Upstage API 키**가 필요(8번 주의 참조).

---

## 3. Dockerfile 초안 (Streamlit · CPU · Solar 임베딩)

> torch/ultralytics는 **YOLO 추론 때문에 유지**, sentence-transformers·BGE-m3는 **제거**(Solar API 사용). opencv는 `opencv-python-headless`(서버엔 GUI 불필요). 인덱스는 이미지에 굽지 말고 **볼륨 + 최초 기동 시 빌드**.

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    EMBED_PROVIDER=solar

WORKDIR /app

# OpenCV 런타임 라이브러리 + 헬스체크용 curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# ⚠️ requirements.txt에서 opencv-python → opencv-python-headless 로 교체 권장
# ⚠️ CPU torch(YOLO용): --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .                        # best.pt 포함(수십MB, 문제없음)

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

# entrypoint: chroma 인덱스 없으면 최초 1회 build_index(Solar API로 임베딩) 후 앱 기동
#   → UPSTAGE_API_KEY 가 .env로 주입돼 있어야 최초 인덱싱 성공
ENTRYPOINT ["bash", "-c", "\
    [ -d knowledge/chroma ] && [ \"$(ls -A knowledge/chroma 2>/dev/null)\" ] || python knowledge/build_index.py; \
    exec streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true"]
```

**인덱스 전략**: `knowledge/chroma`를 **named volume**에 두면 최초 기동에서 Solar API로 한 번만 인덱싱하고, 재배포(이미지 교체)에도 볼륨이 유지돼 빠르게 뜬다. Solar는 로컬 모델이 없으니 HF 캐시 볼륨은 불필요.

---

## 4. docker-compose 초안 (GCE)

```yaml
name: crack

services:
  crack-app:
    image: ${APP_IMAGE:?APP_IMAGE required}   # ghcr.io/leehome2/ai_crack_detection:sha-xxxx
    container_name: crack-app
    restart: unless-stopped
    env_file: [.env]                          # ANTHROPIC_API_KEY 등 (CD가 시크릿으로 생성)
    ports: ["${SERVICE_PORT:-8501}:8501"]
    volumes:
      - crack_chroma:/app/knowledge/chroma    # RAG 인덱스 영속 (Solar면 HF 캐시 불필요)
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://127.0.0.1:8501/_stcore/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s

volumes:
  crack_chroma:
```

---

## 5. 린 CD 워크플로우 초안 (`.github/workflows/cd.yml`)

FinBrief의 `cd.yml`을 **거의 그대로** 쓰되, 서비스 1개·시크릿 축소·헬스 경로만 바꾼다. 구조:

1. **resolve** — 배포/롤백 모드 결정 (FinBrief 것 그대로 가능)
2. **build** — `docker buildx build --push` → GHCR (`ghcr.io/leehome2/ai_crack_detection`)
3. **deploy** — GCE에 SSH → `docker login` → `.env` 생성(시크릿) → `docker pull` → `docker compose up -d --force-recreate` → **`/_stcore/health` 대기** → 실패 시 `.previous_image`로 자동 롤백
4. 배포 전 `docker image prune -af`로 **디스크 확보**(FinBrief에서 겪은 "no space left" 방지 — 그대로 유지 권장)

**필요한 GitHub Secrets** (FinBrief보다 훨씬 적음):

| 시크릿 | 용도 |
|---|---|
| `GCE_HOST` | 인스턴스 외부 IP |
| `GCE_USERNAME` | SSH 계정명 |
| `GCE_SSH_KEY` | 배포용 SSH 개인키 |
| `UPSTAGE_API_KEY` | **Solar 임베딩 API** (RAG 인덱스·쿼리) |
| `ANTHROPIC_API_KEY` | 보고서 생성 (Claude) |
| `ANTHROPIC_MODEL` | 모델 ID (예: claude-sonnet-4-5-…) |
| `EMBED_PROVIDER` | (선택) 기본 `solar`. 로컬 폴백 시 `bge` |
| `RAG_MATCH_MIN_SCORE` | (선택) 임계값 오버라이드 |

트리거: `on: push: branches: [main]` (+ `workflow_dispatch`로 수동 배포/롤백). CI 테스트 잡은 나중에 pytest 붙일 때 추가.

---

## 6. GCE 최초 셋업 단계 (구현 시)

1. **VM 생성** — `e2-standard-2`, Ubuntu 22.04 LTS, 부팅 디스크 30GB, "HTTP 트래픽 허용" 체크.
2. **Docker 설치** — `curl -fsSL https://get.docker.com | sh` + compose 플러그인.
3. **방화벽** — Streamlit 포트 8501 오픈(방화벽 규칙 + 네트워크 태그). 데모용이면 충분. 운영급이면 nginx :80 리버스 프록시 앞단 권장.
4. **배포용 SSH 키** — `ssh-keygen`으로 키쌍 생성 → 공개키는 VM `~/.ssh/authorized_keys`, 개인키는 GitHub Secret `GCE_SSH_KEY`.
5. **GHCR 권한** — 워크플로우 `permissions: packages: write` (FinBrief 그대로). GHCR 패키지 첫 push 후 가시성 설정.
6. **첫 배포** — main에 push → CD 자동 실행. 최초엔 BGE-m3 다운로드+인덱싱 때문에 컨테이너 기동이 1~2분 걸림(볼륨 캐시 후엔 빠름).

**보안 한 줄**: Streamlit을 인터넷에 열면 누구나 접근 가능. 데모라면 방화벽 소스 IP 제한이나 간단한 비밀번호(Streamlit `secrets`) 정도는 걸어두는 걸 권장.

---

## 7. 착수 순서 (선행조건)

배포는 아래가 끝난 뒤 시작하는 게 맞다 (멘토 우선순위: RAG·보고서 > 배포):

- [ ] **선행 1 — RAG 실인덱스(Solar)**: 데스크탑/노트북에서 `UPSTAGE_API_KEY` 설정 + `EMBED_PROVIDER=solar`로 `build_index.py` 실행·검증. **임베더를 BGE→Solar로 바꿨으니 유사도 스케일이 달라짐 → `RAG_MATCH_MIN_SCORE`를 Solar 분포로 재보정.** (배포 이미지의 인덱스 빌드가 이걸 그대로 재현)
- [ ] **선행 2 — 보고서 API**: `ANTHROPIC_MODEL` 실 모델 ID + `ANTHROPIC_API_KEY` 연동 확인(목업→실제 Claude). 배포 시크릿으로 그대로 주입.
- [ ] 그 다음 — 위 3~6의 Dockerfile·compose·cd.yml 확정 및 GCE 셋업.

**재학습 반영 루프**: 데스크탑에서 재학습 → 새 `best.pt` 커밋·push → CD가 자동으로 이미지 재빌드·재배포. (`best.pt`는 이미 git 강제 포함 상태라 그대로 동작)

---

## 8. 솔직한 총평

방향은 좋다. (1) 멘토의 "서버에 올려 동작하는 UI로 점검" 요구를 충족하고, (2) 검증된 FinBrief CD를 재사용할 수 있고, (3) 폰 실사용 테스트 루프가 데스크탑 없이도 돌아간다. **Solar 임베딩 채택으로 인스턴스가 e2-standard-2 → e2-small급으로 내려가 비용·복잡도도 줄었다.** 다만 이 프로젝트 배점엔 배포가 직접 항목이 아니므로(AI 활용성·RAG·보고서가 핵심), **풀 CD로 과투자하지 말고 린 버전 + 필요할 때만 인스턴스 켜기**로 가는 게 노력 대비 최선이다. LLMOps 실습·포트폴리오 가치는 덤.

**Solar 임베딩 주의점**: (a) RAG가 런타임에 외부 API에 의존 → Upstage 장애·키 만료 시 근거 검색이 빈다(단, 보고서도 이미 Anthropic API 의존이라 외부 의존 자체는 설계에 이미 존재). (b) **부트캠프로 받은 Solar 키가 7/30 발표 시점까지 유효한지 확인 필요.** 만료 위험이 있으면 `EMBED_PROVIDER=bge` 로컬 폴백 경로를 그대로 남겨뒀으니 데스크탑에선 그걸로 대체 가능. (c) 문서·쿼리는 비대칭(passage/query)이지만 동일 공간이라 그대로 비교된다 — 코드가 이미 그렇게 호출한다.
