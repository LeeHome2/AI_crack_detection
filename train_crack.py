"""
AI HUB 건물 균열 데이터로 YOLOv8 학습
- convert_aihub_to_yolo.py 로 변환 완료 후 실행
- RTX 3070 Ti (8GB) 기준 설정
"""
from ultralytics import YOLO

def main():
    # 사전학습 가중치에서 시작 (전이학습)
    model = YOLO("yolov8s.pt")

    # 학습
    results = model.train(
        data=r"D:\crack_detection\dataset\data.yaml",
        epochs=50,            # 50이면 충분, 시간 부족하면 30
        imgsz=640,            # 입력 크기
        batch=8,              # 8GB VRAM 기준. 메모리 부족 에러(CUDA out of memory) 나면 4로
        device=0,             # GPU 사용
        patience=15,          # 15에폭 개선 없으면 조기 종료
        project="runs/crack", # 결과 저장 위치
        name="train_v1",      # 실험 이름
        # 데이터 증강 (균열 특성에 맞게)
        degrees=10,           # 약간의 회전
        fliplr=0.5,           # 좌우 반전
        flipud=0.3,           # 상하 반전 (균열은 방향 다양)
        hsv_v=0.4,            # 밝기 변화 (조도 대응)
    )

    # 검증
    metrics = model.val()
    print("\n===== 학습 완료 =====")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"모델 저장: runs/crack/train_v1/weights/best.pt")

if __name__ == "__main__":
    main()
