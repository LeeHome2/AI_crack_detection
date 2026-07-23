# 6종 결함 탐지 프로젝트 - 세션 핸드오버

## 프로젝트 개요
- **목표**: 건물/구조물 6종 결함 탐지 모델 (YOLOv8)
- **데모 일정**: 2025-07-27 (학교 발표)
- **작업 경로**: `D:\crack_detection`
- **문서 갱신일**: 2025-07-23

---

## 1. 현재 진행 상태

### 1.1 BBox (6종 결함 탐지) - 학습 일시 중단
| 단계 | 상태 | 비고 |
|------|------|------|
| 데이터 다운로드 | ✅ 완료 | AI Hub 112번, ~142GB |
| 압축 해제 | ✅ 완료 | 6종 라벨 + 원천 이미지 |
| YOLO 변환 | ✅ 완료 | 20,838개 이미지, 64,675개 객체 |
| 모델 학습 | ⏸️ 일시 중단 | **epoch 43/100, mAP 9.9%** |

**학습 출력 경로**: `D:\crack_detection\runs\detect\defect6_v2`

### 1.2 학습 재개 방법 (중요!)
```powershell
cd D:\crack_detection
.\venv\Scripts\Activate
yolo detect train resume=True model=runs/detect/defect6_v2/weights/last.pt
```

### 1.3 현재 학습 성과 (Epoch 43 기준)
| 지표 | Epoch 1 | Epoch 43 | 변화 |
|------|---------|----------|------|
| mAP50 | 2.8% | **9.9%** | +7.1% |
| mAP50-95 | 0.8% | 3.6% | +2.8% |
| Precision | 12.1% | 18.6% | +6.5% |
| Recall | 6.4% | 18.1% | +11.7% |

**남은 학습**: 57 epochs (~12시간 예상)

### 1.2 Seg (세그멘테이션) - 대기
| 단계 | 상태 | 비고 |
|------|------|------|
| 라벨 다운로드 | ✅ 완료 | 71769, 567 라벨 |
| 원천 ZIP 다운로드 | ✅ 완료 | 24개 ZIP 파일 |
| ZIP 압축 해제 | ❌ 미완료 | 186개만 해제됨 |
| YOLO 변환 | ⏸️ 대기 | 압축 해제 후 진행 |
| 모델 학습 | ⏸️ 대기 | |

---

## 2. 재부팅 후 작업 순서

### Step 0: BBox 학습 재개 (최우선!)
```powershell
cd D:\crack_detection
.\venv\Scripts\Activate

# 학습 재개 (epoch 44부터 이어서)
yolo detect train resume=True model=runs/detect/defect6_v2/weights/last.pt
```
- 현재: epoch 43/100 완료, mAP 9.9%
- 남은 시간: ~12시간 (57 epochs)
- 완료되면 자동 종료됨

### Step 1: BBox 학습 결과 확인 (학습 완료 후)
```powershell
cd D:\crack_detection

# 학습 결과 확인
type runs\detect\defect6_v2\results.csv

# Best 모델 확인
dir runs\detect\defect6_v2\weights\best.pt
```

**예상 결과**:
- `best.pt` 파일 생성됨
- mAP50 15-25% 예상 (현재 9.9%에서 상승)

### Step 2: BBox 모델 테스트 (선택)
```powershell
# 가상환경 활성화
.\venv\Scripts\Activate

# 테스트 이미지로 추론
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/defect6_v2/weights/best.pt'); model.predict('test_image.jpg', save=True)"
```

### Step 3: Seg 압축 해제 (HDD 속도 조절)
```powershell
# 가상환경 활성화
.\venv\Scripts\Activate

# Throttle 모드로 압축 해제 (PC 반응성 유지)
python seg_step1b_extract.py --throttle
```

**소요 시간**: 약 1-2시간 (throttle 모드)

### Step 4: Seg YOLO 변환
```powershell
python seg_step2_convert.py
```

### Step 5: Seg 모델 학습
```powershell
python seg_step3_train.py
```

---

## 3. 데이터셋 정보

### 3.1 BBox 데이터셋 (6종 결함)
- **소스**: AI Hub 112번 - 건물 균열 탐지드론 개발을 위한 이미지
- **경로**: `D:\AIHub_dataset\112.건물_균열_탐지드론_개발을_위한_이미지`
- **YOLO 변환 결과**: `D:\crack_detection\dataset_6class`

#### 클래스 분포 (균형 잡힘!)
| ID | 클래스명 | 한글명 | 객체 수 | 비율 |
|----|----------|--------|---------|------|
| 0 | crack | 균열 | 13,101개 | 20.3% |
| 1 | spalling | 박리/박락 | 8,522개 | 13.2% |
| 2 | efflorescence | 백태/누수 | 12,139개 | 18.8% |
| 3 | rebar_exposure | 철근노출 | 9,568개 | 14.8% |
| 4 | steel_defect | 강재손상 | 9,429개 | 14.6% |
| 5 | paint_damage | 도장손상 | 11,916개 | 18.4% |
| | **합계** | | **64,675개** | |

### 3.2 Seg 데이터셋
- **71769**: SOC 시설물 균열패턴 (도로교량, 터널 등)
- **567**: 서울시 노후주택 균열
- **경로**: `D:\AIHub_dataset\075.건물_균열_탐지_이미지_고도화_SOC_시설물_균열패턴_이미지_데이터`

---

## 4. 학습 설정

### 4.1 BBox 학습 (step3_train.py)
```yaml
MODEL: yolov8s.pt          # YOLOv8 Small
IMGSZ: 640
BATCH: 16
EPOCHS: 100
PATIENCE: 25               # 조기 종료
OPTIMIZER: AdamW
LR0: 0.001
MOSAIC: 1.0
MIXUP: 0.1
CACHE: ram                 # HDD 부하 감소
```

### 4.2 data.yaml
```yaml
path: D:\crack_detection\dataset_6class
train: images/train
val: images/val

names:
  0: crack
  1: spalling
  2: efflorescence
  3: rebar_exposure
  4: steel_defect
  5: paint_damage
```

---

## 5. 스크립트 목록

### 5.1 BBox 파이프라인
```
D:\crack_detection\
├── step1_unzip_all.py        # 원천데이터 압축 해제
├── step2_convert_to_yolo.py  # YOLO bbox 포맷 변환
└── step3_train.py            # YOLOv8 학습
```

### 5.2 Seg 파이프라인
```
D:\crack_detection\
├── seg_step0_list_datasets.py  # AI Hub filekey 조회
├── seg_step1_download.py       # 다운로드
├── seg_step1b_extract.py       # tar/zip 압축 해제 (--throttle 옵션)
├── seg_step2_convert.py        # YOLO seg 포맷 변환
└── seg_step3_train.py          # YOLOv8-seg 학습
```

### 5.3 HDD 속도 조절 옵션
```powershell
# 압축 해제 시 HDD 100% 방지
python seg_step1b_extract.py --throttle
```

---

## 6. 환경 정보

```
OS: Windows 10/11
GPU: NVIDIA GeForce RTX 3070 Ti (8GB VRAM)
Python: 3.11.9
PyTorch: 2.x + CUDA 12.4
Ultralytics: 8.x
venv: D:\crack_detection\venv
```

### 가상환경 활성화
```powershell
cd D:\crack_detection
.\venv\Scripts\Activate
```

---

## 7. 학습 이력

### 1차 시도 (실패)
- 데이터: 균열만 91.7%, 나머지 8.3%
- 결과: mAP 5.28% (클래스 불균형으로 실패)

### 2차 시도 (현재 - 일시 중단)
- 데이터: 6종 균형 (13~20% 분포)
- **Epoch 43/100 완료 후 중단**
- 현재 mAP50: **9.9%** (1차 대비 +4.6%)
- 남은 학습: 57 epochs (~12시간)
- `last.pt`에서 재개 가능

---

## 8. 트러블슈팅 기록

### 해결된 문제들
1. **WinError 193**: aihubshell이 bash 스크립트 → Git Bash로 실행
2. **cp949 codec error**: subprocess에 `encoding="utf-8", errors="replace"` 추가
3. **seg_step2_convert.py 무응답**: 전체 디렉토리 스캔 → 특정 경로만 스캔
4. **PyTorch CPU 버전**: CUDA 12.4 버전으로 재설치
5. **HDD 100% 사용**: `cache='ram'` 옵션 추가, `--throttle` 모드 구현

---

## 9. 중요 파일 경로

| 항목 | 경로 |
|------|------|
| BBox 데이터셋 | `D:\crack_detection\dataset_6class` |
| BBox Best 모델 | `D:\crack_detection\runs\detect\defect6_v2\weights\best.pt` |
| BBox Best 백업 (epoch 43) | `D:\crack_detection\runs\detect\defect6_v2\weights\best_epoch43.pt` |
| 학습 로그 | `D:\crack_detection\runs\detect\defect6_v2\results.csv` |
| Seg 원본 데이터 | `D:\AIHub_dataset\075.건물_균열_탐지_이미지_고도화_SOC_시설물_균열패턴_이미지_데이터` |
| 이 문서 | `D:\crack_detection\SESSION_HANDOVER.md` |

---

## 10. 모델 백업 이력

| 파일명 | Epoch | mAP50 | 백업일 | 용도 |
|--------|-------|-------|--------|------|
| `best_epoch43.pt` | 43 | 9.9% | 2025-07-23 | 서비스 테스트용 |

---

*생성일: 2025-07-23*
*세션 핸드오버용 문서*
