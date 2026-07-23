# 2차 MVP 배포 런북 — 복합 결함 모델 go-live (8501)

> 목적: 6종 재학습 완료 후 **복합 결함 진단을 8501(고도화·CD 자동 라인)에 올리는** 절차.
> 안전판: **8502(크랙 MVP)는 건드리지 않는다** — 배포/모델이 잘못돼도 8502로 데모 가능.
> ⚠️ 가장 놓치기 쉬운 것: **RAG 인덱스 재구축**(Phase 3). 안 하면 데모에서 철근노출·박리 근거가 안 나옴.

---

## 0. 구조 요약 (왜 이렇게 하나)

- **8501** = `docker-compose.yml`, CD(main 머지) 자동 배포 → **복합(6종) 라인**.
- **8502** = `docker-compose.dev.yml`, 수동 → **크랙 안전판(동결)**.
- 모델은 이미지에 굽는다(Dockerfile `COPY . .`) → best.pt가 **repo에 있어야** 서버 컨테이너에 들어감. (크랙 모델도 이미 이 방식으로 커밋돼 있음.)
- `config.YOLO_WEIGHTS`는 **env 경로가 실제 존재할 때만** 6종을 쓰고, 없으면 크랙으로 폴백 → 6종 미커밋 상태로 배포해도 안 죽음.
- RAG 인덱스는 **영속 볼륨**(`crack_chroma`)이라 재배포해도 유지됨 → 지식 문서가 바뀌면 **명시적으로 재구축**해야 반영됨.

---

## Phase 0 — 배포 전 로컬 검증 (데스크탑)

학습 끝난 직후, 서버에 올리기 전에 로컬에서 먼저 확인.

```powershell
cd D:\crack_detection
$env:YOLO_WEIGHTS = "D:\crack_detection\runs\detect\defect6_v2\weights\best.pt"
streamlit run app.py
```

- [ ] 폰/PC로 **복합 사진**(균열+철근노출 등) 업로드 → 결함별 색상 박스·범례 칩이 뜨는가
- [ ] 위험도 카드에 복합 가점(철근노출 등)이 반영되는가
- [ ] 보고서 "점검 결과"에 복합 결함 표가 나오는가
- [ ] val 지표(mAP·클래스별 AP) 확인 — 특정 클래스가 심하게 낮으면 신뢰도 임계값 보정 고려

RAG 인덱스도 로컬에서 15문서로 미리 재구축해 확인(Solar 키 필요):
```powershell
python knowledge\build_index.py    # "15개 문서 → NN개 청크" 확인
```

---

## Phase 1 — 6종 모델을 repo에 커밋

서버 컨테이너가 모델을 가지려면 best.pt가 git에 있어야 한다.

```bash
cd D:\crack_detection
git add runs/detect/defect6_v2/weights/best.pt
git commit -m "model: 6종 복합 결함 YOLOv8s 가중치(defect6_v2)"
git push origin main
```

- yolov8s ≈ 22MB → git 직접 커밋 OK(크랙 모델과 동일 패턴, `.dockerignore`가 `runs/` 제외 안 함).
- ⚠️ **주의**: cd.yml 경로 필터에 `runs/**`가 없어서 **모델만 커밋하면 CD가 안 돌 수 있음**. 아래 Phase 2에서 compose/config 변경과 함께 푸시하거나 `workflow_dispatch` 수동 실행. (서버 배포는 `git reset --hard origin/main`이라 일단 CD가 돌면 모델은 따라 들어감.)

---

## Phase 2 — 배포 (자동 CD 또는 수동)

compose·config에 이미 반영돼 있음(main): `YOLO_WEIGHTS` 기본값=6종 경로, `RAG_MATCH_MIN_SCORE=0.20`.

**A) 자동(CD)** — main에 (모델 포함) 푸시되면 GitHub Actions가 GCE에 SSH → `git reset --hard` → `docker compose up -d --build` → 헬스체크. (GCE Secrets 설정돼 있을 때만.)

**B) 수동(권장 — 데모 통제)** — 서버에서 직접:
```bash
ssh <user>@<GCE_HOST>
cd ~/AI_crack_detection
git fetch origin && git reset --hard origin/main
docker compose up -d --build          # 8501 재빌드(6종 모델 포함)
```

---

## Phase 3 — RAG 인덱스 재구축 (버전 마커로 자동)

**이제 자동이다.** `knowledge/KB_VERSION`(repo) ↔ 볼륨의 `.kb_version`을 entrypoint가 비교해,
**다르면 재배포 시 자동으로 재빌드**한다. 옛 볼륨엔 마커가 없으니(불일치) → 다음 배포에서
L2 인덱스가 코사인으로 자동 갱신됨. 지식 문서/스키마를 바꾸면 **`KB_VERSION` 값만 올리면**
다음 배포에서 자동 재빌드된다. (배포 로그에 `인덱스 재빌드 필요 (repo=... vs 인덱스=...)` 확인)

수동으로 강제하려면(옵션):
```bash
# A) 컨테이너 안에서 즉시 재빌드(무중단) — build_index가 컬렉션 삭제 후 코사인으로 재생성
docker compose exec crack-app python knowledge/build_index.py   # "17개 문서 → 98개 청크" 확인
# B) 클린 재기동
docker compose down && docker volume rm crack_crack_chroma && docker compose up -d --build
```

> ⚠️ **거리 메트릭 = 코사인**(build_index가 `hnsw:space=cosine`으로 생성). `RAG_MATCH_MIN_SCORE=0.30`.
> 옛 L2 인덱스가 남아 있으면 `score=1-거리`가 음수로 나와 결함 가점 게이팅이 무의미해지므로,
> 이 자동 재빌드가 반드시 한 번은 돌아야 함.

---

## Phase 4 — 검증

```bash
curl -fsS http://127.0.0.1:8501/_stcore/health     # ok 나오면 기동 성공
docker compose logs --tail 40                       # entrypoint 인덱스 로그 확인
```

- [ ] 폰에서 배포 URL(HTTPS 터널) 접속 → 복합 사진으로 결함별 색상·근거·보고서 확인
- [ ] 사이드바 "탐지 모델 🟢", "RAG 지식베이스 🟢" 초록인지
- [ ] RAG 근거에 **철근노출/박리/백태 문서**가 출처(기준명·URL)와 함께 뜨는지 ← Phase 3 성공 여부

---

## Phase 5 — 로컬 데모 폴백 (7/27 안전판)

HTTPS 터널·서버가 불안하면, 데스크탑 로컬로 데모(카메라는 localhost에서도 됨):
```powershell
$env:YOLO_WEIGHTS = "D:\crack_detection\runs\detect\defect6_v2\weights\best.pt"
streamlit run app.py     # http://localhost:8501
```
- 데모 구간은 **미리 한 번 화면 녹화**해 두기(라이브 실패 대비).
- 최악의 경우 **8502 크랙 MVP**로 균열 데모 → "복합은 코드 완료·재학습 완료, 배포만 조정 중"으로 정직하게.

---

## 롤백

6종/RAG가 이상하면 즉시 크랙으로 되돌리기:
```bash
# 8501에서 6종 끄기 → 크랙 폴백 (env만 비우면 config가 크랙으로)
YOLO_WEIGHTS= docker compose up -d
# 또는 8502(크랙 안전판)로 데모 전환
```

---

## 한눈 체크리스트

1. [ ] 로컬 검증(복합 렌더·RAG 15문서) — Phase 0
2. [ ] best.pt repo 커밋 — Phase 1
3. [ ] 배포(수동 `git reset --hard` + `compose up --build`) — Phase 2
4. [ ] **RAG 인덱스 재구축**(`exec build_index`) — Phase 3 ★
5. [ ] 헬스체크 + 폰 복합 사진 확인 — Phase 4
6. [ ] 로컬 폴백·데모 사전녹화 — Phase 5
