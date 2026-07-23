"""
Seg Step 1b: 다운로드된 tar 파일 압축 해제 + 분할 파일 병합

다운로드 후 실행:
  python seg_step1b_extract.py

HDD 사용률 조절:
  python seg_step1b_extract.py --throttle
"""
import os
import glob
import tarfile
import subprocess
import time
import argparse
from pathlib import Path
from tqdm import tqdm

# ================== 설정 ==================
BASE_DIR = r"D:\AIHub_dataset"

# HDD 속도 조절 (throttle 모드에서 사용)
THROTTLE_DELAY = 0.01       # 파일 간 대기 시간 (초)
BATCH_SIZE = 100            # N개 파일마다 짧은 휴식
BATCH_PAUSE = 0.5           # 배치 간 휴식 시간 (초)
# =====================================================

# 전역 throttle 플래그
THROTTLE_MODE = False

def extract_tar_files():
    """tar 파일들 압축 해제"""
    tar_files = glob.glob(os.path.join(BASE_DIR, "*.tar"))

    if not tar_files:
        print("tar 파일이 없습니다.")
        return []

    print(f"\n[1/3] tar 파일 압축 해제 ({len(tar_files)}개)")
    print("="*60)

    extracted = []
    for tar_path in tar_files:
        name = os.path.basename(tar_path)
        size_gb = os.path.getsize(tar_path) / (1024**3)
        print(f"\n{name} ({size_gb:.2f} GB)")

        try:
            with tarfile.open(tar_path, 'r') as tf:
                members = tf.getnames()
                print(f"  파일 수: {len(members)}개")

                for i, member in enumerate(tqdm(members, desc="  압축 해제", unit="파일")):
                    tf.extract(member, BASE_DIR)

                    # HDD 속도 조절
                    if THROTTLE_MODE:
                        time.sleep(THROTTLE_DELAY)
                        if (i + 1) % BATCH_SIZE == 0:
                            time.sleep(BATCH_PAUSE)

                extracted.append(tar_path)
                print(f"  완료!")
        except Exception as e:
            print(f"  오류: {e}")

    return extracted

def find_part_files():
    """분할 파일 그룹 찾기"""
    part_files = glob.glob(os.path.join(BASE_DIR, "**", "*.part0"), recursive=True)

    # 그룹화 (base name 기준)
    groups = {}
    for p0 in part_files:
        base = p0.replace(".part0", "")
        parts = sorted(glob.glob(base + ".part*"))
        if parts:
            groups[base] = parts

    return groups

def merge_parts(groups):
    """분할 파일 병합"""
    if not groups:
        print("병합할 분할 파일이 없습니다.")
        return

    print(f"\n[2/3] 분할 파일 병합 ({len(groups)}개 그룹)")
    print("="*60)

    for base, parts in groups.items():
        target = base  # .zip 파일명
        name = os.path.basename(target)

        if os.path.exists(target):
            existing_size = os.path.getsize(target) / (1024**2)
            print(f"\n{name}: 이미 존재 ({existing_size:.1f} MB) - 스킵")
            continue

        total_size = sum(os.path.getsize(p) for p in parts) / (1024**2)
        print(f"\n{name}: {len(parts)}개 파트 ({total_size:.1f} MB)")

        try:
            with open(target, 'wb') as outfile:
                for part in tqdm(parts, desc="  병합", unit="파트"):
                    with open(part, 'rb') as infile:
                        chunk_count = 0
                        while True:
                            chunk = infile.read(1024*1024*10)  # 10MB chunks
                            if not chunk:
                                break
                            outfile.write(chunk)

                            # HDD 속도 조절
                            if THROTTLE_MODE:
                                chunk_count += 1
                                if chunk_count % 5 == 0:  # 50MB마다 짧은 대기
                                    time.sleep(0.1)

            # 병합 성공 시 part 파일 삭제
            for part in parts:
                os.remove(part)
            print(f"  완료: {target}")

        except Exception as e:
            print(f"  오류: {e}")
            if os.path.exists(target):
                os.remove(target)

def extract_zips():
    """병합된 zip 파일 압축 해제"""
    # Seg 데이터셋 폴더들
    search_dirs = [
        os.path.join(BASE_DIR, "075.건물_균열_탐지_이미지_고도화_SOC_시설물_균열패턴_이미지_데이터"),
        os.path.join(BASE_DIR, "189.서울시_노후_주택_균열_데이터"),
    ]

    zip_files = []
    for d in search_dirs:
        if os.path.isdir(d):
            zip_files.extend(glob.glob(os.path.join(d, "**", "*.zip"), recursive=True))

    if not zip_files:
        print("압축 해제할 zip 파일이 없습니다.")
        return

    print(f"\n[3/3] ZIP 파일 압축 해제 ({len(zip_files)}개)")
    print("="*60)

    for zip_path in zip_files:
        name = os.path.basename(zip_path)
        extract_dir = zip_path.replace(".zip", "")

        # 이미 해제되었는지 확인
        if os.path.isdir(extract_dir):
            file_count = sum(len(files) for _, _, files in os.walk(extract_dir))
            if file_count > 10:
                print(f"\n{name}: 이미 해제됨 ({file_count}개 파일) - 스킵")
                continue

        size_mb = os.path.getsize(zip_path) / (1024**2)
        print(f"\n{name} ({size_mb:.1f} MB)")

        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zf:
                members = zf.namelist()
                print(f"  파일 수: {len(members)}개")

                os.makedirs(extract_dir, exist_ok=True)
                for i, member in enumerate(tqdm(members, desc="  압축 해제", unit="파일")):
                    zf.extract(member, extract_dir)

                    # HDD 속도 조절
                    if THROTTLE_MODE:
                        time.sleep(THROTTLE_DELAY)
                        if (i + 1) % BATCH_SIZE == 0:
                            time.sleep(BATCH_PAUSE)

                print(f"  완료: {extract_dir}")
        except Exception as e:
            print(f"  오류: {e}")

def cleanup_tar_files(extracted):
    """압축 해제 완료된 tar 파일 삭제 (선택)"""
    if not extracted:
        return

    print(f"\n압축 해제 완료된 tar 파일 {len(extracted)}개:")
    for t in extracted:
        size_gb = os.path.getsize(t) / (1024**3)
        print(f"  {os.path.basename(t)} ({size_gb:.2f} GB)")

    print("\n이 파일들을 삭제하시겠습니까? (디스크 공간 확보)")
    print("수동 삭제: del D:\\AIHub_dataset\\*.tar")

def main():
    global THROTTLE_MODE

    parser = argparse.ArgumentParser(description="Seg 데이터 압축 해제")
    parser.add_argument("--throttle", action="store_true",
                        help="HDD 사용률 조절 모드 (느리지만 PC 반응성 유지)")
    args = parser.parse_args()

    THROTTLE_MODE = args.throttle

    print("="*60)
    print("Seg Step 1b: tar 압축 해제 + 분할 파일 병합")
    if THROTTLE_MODE:
        print("** HDD 속도 조절 모드 활성화 **")
    print("="*60)

    # 1. tar 파일 압축 해제
    extracted = extract_tar_files()

    # 2. 분할 파일 병합
    groups = find_part_files()
    merge_parts(groups)

    # 3. zip 파일 압축 해제
    extract_zips()

    # 4. 정리 안내
    cleanup_tar_files(extracted)

    print("\n" + "="*60)
    print("완료!")
    print("="*60)
    print("\n다음 단계:")
    print("  python seg_step2_convert.py")

if __name__ == "__main__":
    main()
