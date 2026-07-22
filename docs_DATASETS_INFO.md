# AIHub 균열 탐지 데이터셋 정보

## 개요

균열 탐지 모델 학습을 위한 3개 AIHub 데이터셋 정보.

| 데이터셋 키 | 이름 | 라벨 포맷 | 이미지 포맷 |
|------------|------|----------|------------|
| 162 | 건물 균열 탐지드론 개발을 위한 이미지 | COCO (bbox) | TIFF |
| 567 | 서울시 노후 주택 균열 데이터 | Polygon | JPG |
| 71769 | SOC 시설물 균열패턴 이미지 데이터 | Polyline | JPG |

---

## 1. 데이터셋 162 - 건물 균열 탐지드론

### 기본 정보
- **AIHub 키**: 162
- **폴더명**: `112.건물_균열_탐지드론_개발을_위한_이미지`
- **이미지 해상도**: 2560x1440 (TIFF)
- **라벨 포맷**: COCO JSON (bbox)

### 라벨 구조
```json
{
  "info": {
    "name": "건물 균열 탐지드론 개발을 위한 이미지",
    "version": "1.0"
  },
  "images": [
    {
      "id": 1,
      "file_name": "101_0008af00-579e-43ef-a959-d22c3fc591bd.tiff",
      "width": 2560,
      "height": 1440
    }
  ],
  "categories": [
    {"id": 1, "name": "건물결함유형"}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "attributes": {
        "class": "ConcreteCrack",
        "facility": "Other"
      },
      "bbox": [324, 726, 420, 66]  // [x, y, width, height]
    }
  ]
}
```

### 클래스 정보
- `ConcreteCrack`: 콘크리트 균열

### 파일 구조
```
112.건물_균열_탐지드론_개발을_위한_이미지/
├── 01.데이터/
│   ├── 1.Training/
│   │   ├── 원천데이터/           # TIFF 이미지
│   │   └── 라벨링데이터_240326_add/  # JSON 라벨
│   └── 2.Validation/
│       └── 라벨링데이터_240326_add/
```

---

## 2. 데이터셋 567 - 서울시 노후 주택 균열

### 기본 정보
- **AIHub 키**: 567
- **폴더명**: `189.서울시_노후_주택_균열_데이터`
- **이미지 해상도**: 1440x1080 (JPG)
- **라벨 포맷**: Polygon (flat array)

### 라벨 구조
```json
{
  "Raw_Data_Info": {
    "Date": "2021-08-25",
    "Structure": "M",
    "Resolution": [1440, 1080]
  },
  "Source_Data_Info": {
    "File_Extension": "jpg",
    "Source_Data_ID": "S-210825_M_C_2_R_29616682007-11"
  },
  "Learning_Data_Info": {
    "Annotations": [
      {
        "Class_ID": "2",
        "Type": "polygon",
        "polygon": [886, 482, 891, 482, 891, 484, 884, 485, 883, 482, 882, 480]
        // [x1, y1, x2, y2, x3, y3, ...] flat array
      }
    ]
  }
}
```

### 클래스 정보 (Class_ID)
- `1`: 미세 균열
- `2`: 보통 균열
- `3`: 심한 균열

### 파일 구조
```
189.서울시_노후_주택_균열_데이터/
├── 01.데이터/
│   ├── 1.Training/
│   │   ├── 원천데이터/     # JPG 이미지 (.egg 압축)
│   │   └── 라벨링데이터/   # JSON 라벨 (.egg/.zip)
│   └── 2.Validation/
│       └── 라벨링데이터/
```

### 하위 카테고리
- 다세대주택, 단독주택, 비주거용주택, 아파트, 연립주택

---

## 3. 데이터셋 71769 - SOC 시설물 균열패턴

### 기본 정보
- **AIHub 키**: 71769
- **폴더명**: `075.건물_균열_탐지_이미지_고도화_SOC_시설물_균열패턴_이미지_데이터`
- **이미지 해상도**: 1920x1080 (JPG)
- **라벨 포맷**: Polyline (nested array)

### 라벨 구조
```json
{
  "task": {
    "name": "지하",
    "loc": "도로터널",
    "mode": "annotation"
  },
  "image": {
    "name": "URT001_2021_01_03_001_1462.jpg",
    "width": 1920,
    "height": 1080,
    "object_included": "Y"
  },
  "annotations": [
    {
      "label": "reticular crack",
      "labelNum": 0,
      "points": [[345, 697], [367, 682], [388, 654], [397, 632]],
      // [[x1, y1], [x2, y2], ...] nested array
      "shape": "Polyline",
      "px": 1
    }
  ]
}
```

### 클래스 정보 (label)
- `reticular crack`: 망상균열
- `linear crack`: 선형균열
- `complex crack`: 복합균열

### 파일 구조
```
075.건물_균열_탐지_이미지_고도화_SOC_시설물_균열패턴_이미지_데이터/
├── 3.개방데이터/
│   └── 1.데이터/
│       ├── Training/
│       │   ├── 01.원천데이터/   # JPG 이미지
│       │   └── 02.라벨링데이터/ # JSON 라벨
│       └── Validation/
```

### 하위 카테고리
- 지상시설물: 댐(벽체)
- 지하시설물: 도로터널, 수로터널

---

## YOLO 변환 시 주의사항

### 162 (bbox)
```python
# bbox: [x, y, width, height] -> YOLO: [x_center, y_center, w, h] (normalized)
x_center = (x + width/2) / img_width
y_center = (y + height/2) / img_height
w = width / img_width
h = height / img_height
```

### 567 (polygon -> bbox)
```python
# polygon flat array -> bbox
xs = polygon[0::2]  # 짝수 인덱스 = x
ys = polygon[1::2]  # 홀수 인덱스 = y
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)
```

### 71769 (polyline -> bbox)
```python
# points nested array -> bbox
xs = [p[0] for p in points]
ys = [p[1] for p in points]
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)
```

---

## 샘플 파일 위치

```
D:/AIHub_dataset/samples/
├── DATASETS_INFO.md                              # 이 문서
├── 162/
│   ├── 101_0008af00-579e-43ef-a959-d22c3fc591bd.json   # 라벨
│   └── 101_0008af00-579e-43ef-a959-d22c3fc591bd.tiff   # 이미지 (12MB, 2560x1440)
├── 567/
│   ├── S-210828_N_C_2_R_66166895040-11.json      # 라벨 (polygon)
│   └── S-210828_N_C_2_R_66166895040-11.jpg       # 이미지 (1.1MB, 1440x1080)
└── 71769/
    ├── UWT015_2021_01_01_001_10380.json          # 라벨 (polyline)
    └── UWT015_2021_01_01_001_10380.jpg           # 이미지 (204KB, 1920x1080)
```

## AIHub 다운로드 정보

- **API 키**: AIHub 마이페이지 → 회원정보관리에서 발급
- **aihubshell 사용법**:
  ```bash
  # 파일 목록 조회
  aihubshell -mode l -datasetkey <데이터셋키>

  # 파일 다운로드
  aihubshell -mode d -datasetkey <데이터셋키> -filekey <파일키> -aihubapikey '<API키>'
  ```
