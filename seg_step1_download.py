"""
Seg Step 1: AI Hub segmentation 데이터셋 다운로드

71769: SOC 시설물 균열패턴 (도로교량, 터널 등)
567: 서울시 노후주택 균열

사용법:
  python seg_step1_download.py           # 둘 다 다운로드
  python seg_step1_download.py --71769   # 71769만
  python seg_step1_download.py --567     # 567만
"""
import subprocess
import sys
import os
import argparse

API_KEY = "48042AF3-7981-4796-B0D7-63E87FE0FCDB"
AIHUB_SHELL = r"C:\Users\user\aihubshell"
BASH_PATH = r"C:\Program Files\Git\bin\bash.exe"  # Git Bash 사용
OUTPUT_DIR = r"C:\temp_seg"  # SSD로 변경 (HDD 병목 방지)

# ================== 71769: SOC 시설물 균열 ==================
# 전략: 라벨 전부 + 원천 일부 (대용량이므로)
DS_71769 = {
    "name": "SOC 시설물 균열패턴 (seg)",
    "total_source": "~115GB (전체)",

    # Training 라벨 (전부, 총 ~333MB)
    "train_labels": [
        521301,  # TL_지상시설물_댐(벽체).zip | 9 MB
        521302,  # TL_지상시설물_도로교량_교각.zip | 40 MB
        521303,  # TL_지상시설물_도로교량_상부구조.zip | 26 MB
        521304,  # TL_지상시설물_옹벽.zip | 15 MB
        521305,  # TL_지상시설물_철도교량_교각.zip | 5 MB
        521306,  # TL_지상시설물_철도교량_상부구조.zip | 38 MB
        521307,  # TL_지하시설물_도로터널.zip | 96 MB
        521308,  # TL_지하시설물_수로터널.zip | 1 MB
        521309,  # TL_지하시설물_지하차도.zip | 4 MB
        521310,  # TL_지하시설물_지하철.zip | 90 MB
        521311,  # TL_지하시설물_철도터널.zip | 9 MB
    ],

    # Validation 라벨 (전부, 총 ~41MB)
    "val_labels": [
        521323,  # VL_지상시설물_댐(벽체).zip | 1 MB
        521324,  # VL_지상시설물_도로교량_교각.zip | 5 MB
        521325,  # VL_지상시설물_도로교량_상부구조.zip | 3 MB
        521326,  # VL_지상시설물_옹벽.zip | 2 MB
        521327,  # VL_지상시설물_철도교량_교각.zip | 652 KB
        521328,  # VL_지상시설물_철도교량_상부구조.zip | 5 MB
        521329,  # VL_지하시설물_도로터널.zip | 12 MB
        521330,  # VL_지하시설물_수로터널.zip | 160 KB
        521331,  # VL_지하시설물_지하차도.zip | 470 KB
        521332,  # VL_지하시설물_지하철.zip | 11 MB
        521333,  # VL_지하시설물_철도터널.zip | 1 MB
    ],

    # Training 원천 - 서브셋 (도로교량, 터널만 먼저)
    "train_sources_subset": [
        521291,  # TS_지상시설물_도로교량_교각.zip | 10 GB
        521296,  # TS_지하시설물_도로터널.zip | 24 GB
    ],

    # Validation 원천 (작아서 전부)
    "val_sources": [
        521312,  # VS_지상시설물_댐(벽체).zip | 1 GB
        521313,  # VS_지상시설물_도로교량_교각.zip | 1 GB
        521314,  # VS_지상시설물_도로교량_상부구조.zip | 1 GB
        521315,  # VS_지상시설물_옹벽.zip | 1 GB
        521316,  # VS_지상시설물_철도교량_교각.zip | 195 MB
        521317,  # VS_지상시설물_철도교량_상부구조.zip | 2 GB
        521318,  # VS_지하시설물_도로터널.zip | 3 GB
        521319,  # VS_지하시설물_수로터널.zip | 28 MB
        521320,  # VS_지하시설물_지하차도.zip | 131 MB
        521321,  # VS_지하시설물_지하철.zip | 3 GB
        521322,  # VS_지하시설물_철도터널.zip | 277 MB
    ],
}

# ================== 567: 서울시 노후주택 ==================
# 전략: 231023_add (추가 데이터, 작음) 먼저 + 원본 중 비주거용만
DS_567 = {
    "name": "서울시 노후주택 균열 (seg)",
    "total_source": "~460GB (전체) - 너무 큼!",

    # 231023_add 라벨 (총 ~38MB)
    "labels_add": [
        400977,  # Tl_다세대주택.zip | 14 MB
        400978,  # Tl_단독주택.zip | 12 MB
        400979,  # Tl_비주거용주택.zip | 4 MB
        400980,  # Tl_아파트.zip | 4 MB
        400981,  # Tl_연립주택.zip | 4 MB
    ],

    # 231023_add 원천 (총 ~22GB) - 적당한 크기
    "sources_add": [
        400982,  # Ts_다세대주택.zip | 8 GB
        400983,  # Ts_단독주택.zip | 7 GB
        400984,  # Ts_비주거용주택.zip | 2 GB
        400985,  # Ts_아파트.zip | 3 GB
        400986,  # Ts_연립주택.zip | 2 GB
    ],

    # Validation 라벨 (총 ~68MB)
    "val_labels": [
        61226,  # VL_다세대주택.zip | 23 MB
        61227,  # VL_단독주택.zip | 23 MB
        61228,  # VL_비주거용주택.zip | 6 MB
        61229,  # VL_아파트.zip | 9 MB
        61230,  # VL_연립주택.zip | 7 MB
    ],

    # 원본 라벨 (원본 이미지 쓸 때만)
    # 61214-61218 (.egg 포맷)

    # 원본 원천 - 너무 큼 (54~75GB each)
    # 61219-61225, 61231-61235 (.egg 포맷)
}

def download(dataset_key, filekeys, desc):
    """다운로드 실행"""
    if not filekeys:
        print(f"  [{desc}] filekey 없음, 스킵")
        return

    filekey_str = ",".join(str(k) for k in filekeys)
    print(f"\n  [{desc}] {len(filekeys)}개 파일")
    print(f"    filekeys: {filekey_str[:80]}{'...' if len(filekey_str) > 80 else ''}")

    # Git Bash를 통해 aihubshell 실행 (Windows에서 bash 스크립트 실행)
    cmd = [
        BASH_PATH,
        AIHUB_SHELL,
        "-mode", "d",
        "-datasetkey", str(dataset_key),
        "-filekey", filekey_str,
        "-aihubapikey", API_KEY,
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=OUTPUT_DIR,
            encoding="utf-8",
            errors="replace",  # 디코딩 오류 시 대체 문자 사용
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()
        return process.returncode == 0

    except Exception as e:
        print(f"    오류: {e}")
        return False

def download_71769():
    """71769 다운로드 (SOC 시설물)"""
    print("\n" + "=" * 60)
    print(f"[71769] {DS_71769['name']}")
    print(f"참고: {DS_71769['total_source']}")
    print("전략: 라벨 전부 + 원천 서브셋(도로교량,도로터널) + Validation 전부")
    print("=" * 60)

    # 1. Training 라벨 (전부)
    download(71769, DS_71769["train_labels"], "Training 라벨")

    # 2. Validation 라벨 (전부)
    download(71769, DS_71769["val_labels"], "Validation 라벨")

    # 3. Training 원천 (서브셋: 도로교량 + 도로터널 = ~34GB)
    download(71769, DS_71769["train_sources_subset"], "Training 원천 (서브셋)")

    # 4. Validation 원천 (전부, ~13GB)
    download(71769, DS_71769["val_sources"], "Validation 원천")

def download_567():
    """567 다운로드 (서울시 노후주택)"""
    print("\n" + "=" * 60)
    print(f"[567] {DS_567['name']}")
    print(f"참고: {DS_567['total_source']}")
    print("전략: 231023_add 전부 (~22GB) + Validation 라벨")
    print("=" * 60)

    # 1. 231023_add 라벨 (전부)
    download(567, DS_567["labels_add"], "라벨링데이터_231023_add")

    # 2. 231023_add 원천 (전부, ~22GB)
    download(567, DS_567["sources_add"], "원천데이터_231023_add")

    # 3. Validation 라벨 (전부)
    download(567, DS_567["val_labels"], "Validation 라벨")

def main():
    parser = argparse.ArgumentParser(description="Seg 데이터셋 다운로드")
    parser.add_argument("--71769", dest="ds_71769", action="store_true", help="71769만 다운로드")
    parser.add_argument("--567", dest="ds_567", action="store_true", help="567만 다운로드")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Seg 데이터셋 다운로드")
    print(f"저장 위치: {OUTPUT_DIR}")
    print("=" * 60)

    # 둘 다 지정 안 하면 둘 다 다운로드
    if not args.ds_71769 and not args.ds_567:
        args.ds_71769 = True
        args.ds_567 = True

    if args.ds_71769:
        download_71769()

    if args.ds_567:
        download_567()

    print("\n" + "=" * 60)
    print("다운로드 완료!")
    print("=" * 60)
    print("\n예상 다운로드 용량:")
    print("  71769: ~47GB (라벨 374MB + 원천 34GB + Val 13GB)")
    print("  567:   ~22GB (add 22GB + Val 라벨 68MB)")
    print("\n다음 단계:")
    print("  1. 압축 해제")
    print("  2. pip install shapely (71769 seg 변환에 필요)")
    print("  3. seg_step2_convert.py (변환)")

if __name__ == "__main__":
    main()
