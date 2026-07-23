"""
Seg Step 0: AI Hub segmentation 데이터셋 filekey 목록 조회
- 71769: 도로 노면 균열 (대용량, 50만장급)
- 567: 서울시 노후주택 균열 (중소용량)

데스크탑에서 실행 후 출력 결과를 확인하고 다운로드할 filekey를 선택하세요.
"""
import subprocess
import sys

API_KEY = "48042AF3-7981-4796-B0D7-63E87FE0FCDB"
AIHUB_SHELL = r"C:\Users\user\aihubshell"
BASH_PATH = r"C:\Program Files\Git\bin\bash.exe"  # Git Bash 사용

DATASETS = [
    {"key": 71769, "name": "도로 노면 균열 탐지 (seg)", "note": "대용량 - 라벨 전부 + 원천 1개만 권장"},
    {"key": 567, "name": "서울시 노후주택 균열 (seg)", "note": "중소용량 - 전부 다운로드 가능"},
]

def list_dataset(dataset_key):
    """데이터셋의 파일 목록 조회"""
    # Git Bash를 통해 aihubshell 실행
    cmd = [
        BASH_PATH,
        AIHUB_SHELL,
        "-mode", "l",
        "-datasetkey", str(dataset_key),
        "-aihubapikey", API_KEY,
    ]
    print(f"\n명령어: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        return result.stdout
    except subprocess.TimeoutExpired:
        print("타임아웃 (60초)")
        return None
    except Exception as e:
        print(f"오류: {e}")
        return None

def main():
    print("=" * 70)
    print("AI Hub Segmentation 데이터셋 파일 목록 조회")
    print("=" * 70)

    results = {}

    for ds in DATASETS:
        print(f"\n{'='*70}")
        print(f"[{ds['key']}] {ds['name']}")
        print(f"참고: {ds['note']}")
        print("=" * 70)

        output = list_dataset(ds["key"])
        results[ds["key"]] = output

    print("\n" + "=" * 70)
    print("조회 완료!")
    print("=" * 70)
    print("\n위 출력에서 filekey를 확인한 후:")
    print("1. seg_step1_download.py 의 FILEKEYS 설정을 수정하거나")
    print("2. 아래 명령어로 직접 다운로드:")
    print(f"""
# 71769 (라벨 전부 + 원천 1개):
aihubshell -mode d -datasetkey 71769 -filekey <라벨키들>,<원천_01키> -aihubapikey '{API_KEY}'

# 567 (전부):
aihubshell -mode d -datasetkey 567 -filekey <라벨키>,<원천키> -aihubapikey '{API_KEY}'
""")

if __name__ == "__main__":
    main()
