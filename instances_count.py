from pathlib import Path
from collections import Counter



'''
***** class ******
0: crosswalk
1: green
2: parking
3: red
4: straightz
5: turn
6: yellow
'''


label_path = Path("./dataset_split/valid/labels/")
counter = Counter()

for file in label_path.rglob("*.txt"):
    with open(file) as f:
        for line in f:
            cls = int(line.strip().split()[0])
            counter[cls] += 1

print("클래스별 객체 수:")
for cls_id, count in counter.items():
    print(f"클래스 {cls_id}: {count}개")



# from pathlib import Path
# from sklearn.model_selection import train_test_split
# import shutil

# # 1. 원본 데이터셋 경로 설정 (train, valid, test 다 합친 경로)
# image_dirs = [Path("./data/car20250926_2/train/images"), Path("./data/car20250926_2/valid/images"), Path("./data/car20250926_2/test/images")]
# label_dirs = [Path("./data/car20250926_2/train/labels"), Path("./data/car20250926_2/valid/labels"), Path("./data/car20250926_2/test/labels")]

# # 2. 이미지 이름과 해당 클래스(주 클래스)를 저장할 리스트
# img_list = []
# cls_list = []

# # 3. 모든 이미지/라벨 폴더를 순회하며 파일 모으기
# for img_dir, lbl_dir in zip(image_dirs, label_dirs):
#     for label_file in lbl_dir.glob("*.txt"):
#         img_name = label_file.stem  # 확장자 없는 파일명
#         # 라벨 파일에서 첫 번째 클래스만 추출 (주 클래스)
#         with open(label_file) as f:
#             lines = f.readlines()
#             if not lines:
#                 continue
#             first_class = int(lines[0].strip().split()[0])
#         img_list.append(img_name)
#         cls_list.append(first_class)

# print(f"총 이미지 개수: {len(img_list)}")

# # 4. stratified split: train 70%, temp 30%
# train_imgs, temp_imgs, train_cls, temp_cls = train_test_split(
#     img_list, cls_list, test_size=0.4, stratify=cls_list, random_state=42)

# # 5. temp를 valid 50%, test 50%로 나눔 (각각 15%씩)
# valid_imgs, test_imgs, valid_cls, test_cls = train_test_split(
#     temp_imgs, temp_cls, test_size=0.4, stratify=temp_cls, random_state=42)

# print(f"Train: {len(train_imgs)}, Valid: {len(valid_imgs)}, Test: {len(test_imgs)}")

# # 6. 복사할 새 폴더 경로 지정
# output_base = Path("./dataset_split")
# train_img_out = output_base / "train/images"
# train_lbl_out = output_base / "train/labels"
# valid_img_out = output_base / "valid/images"
# valid_lbl_out = output_base / "valid/labels"
# test_img_out = output_base / "test/images"
# test_lbl_out = output_base / "test/labels"

# # 7. 복사 함수 정의
# def copy_files(img_names, dest_img_dir, dest_lbl_dir):
#     dest_img_dir.mkdir(parents=True, exist_ok=True)
#     dest_lbl_dir.mkdir(parents=True, exist_ok=True)
#     for img_name in img_names:
#         # 원본 이미지/라벨 경로 찾기 (train, valid, test 폴더 중에서)
#         src_img_path = None
#         src_lbl_path = None
#         for img_dir, lbl_dir in zip(image_dirs, label_dirs):
#             if (img_dir / f"{img_name}.jpg").exists():
#                 src_img_path = img_dir / f"{img_name}.jpg"
#                 src_lbl_path = lbl_dir / f"{img_name}.txt"
#                 break
#         if src_img_path and src_lbl_path:
#             shutil.copy(src_img_path, dest_img_dir / f"{img_name}.jpg")
#             shutil.copy(src_lbl_path, dest_lbl_dir / f"{img_name}.txt")
#         else:
#             print(f"Warning: {img_name} 이미지 또는 라벨을 찾을 수 없습니다.")

# # 8. 실제 복사 실행
# copy_files(train_imgs, train_img_out, train_lbl_out)
# copy_files(valid_imgs, valid_img_out, valid_lbl_out)
# copy_files(test_imgs, test_img_out, test_lbl_out)

# print("데이터셋 재분할 및 복사 완료!")