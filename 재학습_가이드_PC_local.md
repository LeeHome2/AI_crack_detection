# PC 재학습 가이드 — 2차 MVP 복합 결함 모델

> 대상: 데스크탑(RTX 3070 Ti 8GB, Windows, `D:\crack_detection` venv)
> 목표: 162 데이터셋 6종 결함으로 YOLOv8 다중클래스 모델을 학습해 **복합 결함 진단(2차 MVP)** 활성화.
> 최신화 7/22 · 코드는 `main`에 병합 완료(`git pull`로 최신 반영).

트랙은 둘이다. **트랙 1(6종 bbox)** 가 지금 바로 돌릴 수 있는 핵심이고, **트랙 2(균열 seg)** 는 고도화용 병렬 작업이다. 트랙 1부터 끝내면 학교측 점검(7/27) 데모에 필요한 복합 결함 진단이 완성된다.

---

## 0. 준비 (공통, 5분)

먼저 최신 코드를 받고 학습 환경을 확인한다.

```powershell
cd D:\crack_detection
git pull                        # convert_aihub_6class_to_yolo.py, train_6class.py 포함
.\venv\Scripts\activate
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`CUDA: True NVIDIA GeForce RTX 3070 Ti` 가 나오면 GPU 학습 준비 완료. `ultralytics`가 없으면 `pip install ultralytics`.

---

## 트랙 1 — 6종 결함 bbox 재학습 (핵심, 지금)

162 데이터셋(건물 균열 탐지드론)에는 6종 결함이 모두 bbox로 라벨돼 있다: **균열·박리(박락)·백태(누수)·철근노출·강재손상·도장손상**. 이걸 전부 뽑아 다중클래스 detection 모델을 만든다.

### 1-1. 데이터 위치 확인

변환기(`convert_aihub_6class_to_yolo.py`) 상단 경로가 실제 데이터와 맞는지 본다.

```python
BASE = r"D:\AIHub_dataset\112.건물_균열_탐지드론_개발을_위한_이미지\01.데이터"
IMAGE_DIR = os.path.join(BASE, r"1.Training\원천데이터")   # 원본 이미지(tiff)
LABEL_DIR = BASE                                          # json 라벨(재귀 탐색)
OUTPUT_DIR = r"D:\crack_detection\dataset_6class"         # 변환 결과
```

폴더 구조가 다르면 `BASE`/`IMAGE_DIR`만 실제 경로로 고친다. 라벨 json은 `LABEL_DIR` 아래를 재귀로 다 찾으니 하위 폴더 위치는 신경 안 써도 된다.

### 1-2. 변환 실행

```powershell
python convert_aihub_6class_to_yolo.py
```

끝나면 이런 요약이 나온다 — **클래스별 객체 수를 꼭 확인**한다.

```
===== 6종 변환 완료 =====
변환 성공: N개 · 이미지없음 .. · 결함없음 .. · 오류 ..
클래스별 객체 수(라벨 인스턴스):
  crack: ....
  spalling: ....
  efflorescence: ....
  rebar_exposure: ....
  steel_defect: ....
  paint_damage: ....
```

- **`매핑 안 된 class 문자열` 경고가 뜨면** 알려줘 — 데이터셋 표기가 예상과 달라 CLASS_MAP을 보정해야 한다(스크립트 안 CLASS_MAP은 대소문자·철자변형 대응돼 있음).
- 특정 클래스 수가 0이거나 극단적으로 적으면(예: steel_defect 수십 개) 그 클래스는 학습이 잘 안 된다 → 검증 후 판단(아래 1-5).

### 1-3. 학습 실행

```powershell
python train_6class.py
```

기본 설정: `yolov8s.pt` 전이학습 · `imgsz=640` · `batch=16` · `epochs=100`(patience 25 조기종료) · `cos_lr` · 막판 mosaic off. 결과는 `runs/detect/defect6/train/`에 저장된다.

- **`CUDA out of memory` 나면**: `train_6class.py`에서 `batch=16 → 8 → 4`로 낮춘다.
- **가는 균열 recall이 아쉬우면**: `imgsz=640 → 960`(VRAM 여유 시 1280). 단 batch를 함께 낮춰야 한다. (근본 해결은 트랙 2의 seg 모델이 균열을 맡는 하이브리드)
- 예상 시간: 3070 Ti·데이터 규모에 따라 대략 **2~5시간**(밤샘 걸어두면 편함).

### 1-4. 학습 모니터링 / 결과

학습 중·후 확인할 것:

- `runs/detect/defect6/train/results.png` — mAP·loss 곡선(수렴 여부)
- `runs/detect/defect6/train/confusion_matrix.png` — 어떤 결함끼리 헷갈리는지
- 콘솔 마지막의 **클래스별 AP50-95** — 데이터 적은 클래스가 낮게 나온다

가이드 예시 신뢰도(0.92~0.81)는 **면적 결함(철근노출·박락)** 에서 자연스럽게 나올 가능성이 높다(박스가 blob을 꽉 채워서). 균열은 여전히 낮게 나오는 게 정상 — 이건 탐지 방식의 구조적 한계라 정직하게 보고하면 된다.

### 1-5. 데이터 추가 판단

클래스별 AP가 심하게 불균형이면(예: 도장손상만 유독 낮음):

1. 그 클래스 인스턴스 수가 절대적으로 적은지 확인(1-2 요약).
2. 적으면 → 해당 결함이 많은 데이터를 더 받거나, `rules.py`의 `DEFECT_CONF_MIN[label]`을 높여 저품질 탐지를 위험 산정에서 배제(오탐 방지).
3. 균열만 유독 낮으면 → 정상. 트랙 2(seg)로 균열을 분리 담당시키는 하이브리드가 정답.

### 1-6. 앱에 적용

학습된 가중치를 서비스가 쓰게 한다. `config.py`의 `YOLO_WEIGHTS`는 이제 **환경변수로 덮어쓸 수 있다.**

- **로컬 테스트**: PowerShell에서
  ```powershell
  $env:YOLO_WEIGHTS = "D:\crack_detection\runs\detect\defect6\train\weights\best.pt"
  streamlit run app.py
  ```
  복합 결함이 탐지되면 위험도 카드에 철근노출·박락 등이 가점으로 잡히고, 보고서 "점검 결과"에 복합 결함 표가 나온다.

- **배포(2차 MVP·8501 컨테이너)**: `best.pt`를 서버로 올리고 8501 컨테이너의 env에 `YOLO_WEIGHTS=/app/runs/detect/defect6/train/weights/best.pt`를 지정. **8502(1차 MVP·균열 전용)는 기본값 유지** → 두 컨테이너가 다른 모델을 쓰며 8502가 안전판으로 동결된다.

> 코드는 이미 6종을 소화하도록 배선돼 있다(detector 클래스별 NMS·라벨, features 균열/면적결함 분리, 복합 Rule, 복합 보고서). **가중치만 바꾸면 즉시 복합 진단이 켜진다.**

---

## 트랙 2 — 균열 seg 재학습 (고도화, 병렬)

균열은 가늘고 긴 대각선이라 사각형 박스로는 한계가 크다. 균열 중심선/영역이 라벨링된 데이터(71769 폴리라인, 567 폴리곤)로 **YOLOv8-seg**를 학습하면 균열을 마스크로 감싸 폭·중심선을 정밀 분석할 수 있다. 최종 하이브리드는 **균열=seg 모델 + 면적 결함=트랙 1의 bbox 모델**을 orchestrator에서 병합하는 구조다.

절차 개요:

1. 71769(폴리라인+px 폭)·567(폴리곤+심각도) → YOLOv8-seg 폴리곤 포맷으로 변환.
2. `yolov8s-seg.pt` 전이학습(`task=segment`), imgsz 640, 균열은 다운스케일에 약하니 필요 시 타일.
3. seg 가중치를 하이브리드 detector에 연결(#16) — orchestrator가 두 모델 결과를 합침.

> ⚠️ 이전 세션에서 만든 seg 변환 스크립트(`convert_71769_seg_to_yolo.py`, `convert_567_seg_to_yolo.py`)가 현재 저장소에 남아 있지 않다(세션 리셋으로 유실). **트랙 2를 시작할 때 알려주면 두 변환기 + `train_seg.py`를 다시 만들어 커밋할게.** 트랙 1이 학교 점검(7/27) 데모의 핵심이니 트랙 1을 먼저 끝내는 걸 권장.

---

## 트러블슈팅 요약

| 증상 | 원인 | 대응 |
|---|---|---|
| `CUDA out of memory` | VRAM 8GB 초과 | `batch` 16→8→4, `imgsz` 낮추기 |
| `매핑 안 된 class` 경고 | 데이터셋 표기 상이 | 경고 문자열 공유 → CLASS_MAP 보정 |
| `이미지없음` 다수 | file_name 불일치/경로 | `IMAGE_DIR` 확인, tiff 확장자 확인 |
| 특정 클래스 AP 0 | 인스턴스 부족 | 데이터 추가 or 신뢰도 하한 상향 |
| 균열 AP만 낮음 | bbox 구조적 한계 | 정상 — 트랙 2(seg)로 분리 |

## 한눈 요약

```
git pull → convert_aihub_6class_to_yolo.py → (클래스별 수 확인) → train_6class.py
  → results.png/AP 확인 → YOLO_WEIGHTS로 best.pt 지정 → streamlit run app.py
```
