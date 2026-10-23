
from pathlib import Path
from collections import Counter

'''功能实现：
1. 检查图片与标签文件是否一一对应。
2. 检查六个类别是否都存在，类别编号是否正确。
3. 检查 YOLO 坐标是否在合法范围内。
4. 检查每张图片是否有对应的有效目标框。
'''

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT.parent / "datasets" / "NEU-DET-redownload"

SPLITS = ["train", "validation"]

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_split(split):
    split_dir = DATASET_ROOT / split
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    print(f"\n{'=' * 15} {split} {'=' * 15}")

    if not image_dir.exists() or not label_dir.exists():
        print("[错误] 找不到 images 或 labels 目录")
        return

    images = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    labels = sorted(label_dir.glob("*.txt"))

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print(f"图片数量: {len(images)}")
    print(f"标签数量: {len(labels)}")
    print(f"缺少标签的图片: {len(missing_labels)}")
    print(f"没有对应图片的标签: {len(missing_images)}")
    print("缺少标签的图片文件名：", sorted(missing_labels))
    print("缺少图片的标签文件名：", sorted(missing_images))

    class_counts = Counter()
    empty_labels = 0
    invalid_lines = 0
    total_boxes = 0

    for label_path in labels:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        if not lines:
            empty_labels += 1
            continue

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                invalid_lines += 1
                print(f"[格式错误] {label_path.name}: {line}")
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                invalid_lines += 1
                print(f"[数值错误] {label_path.name}: {line}")
                continue

            if not 0 <= class_id < len(CLASS_NAMES):
                invalid_lines += 1
                print(f"[类别编号错误] {label_path.name}: {class_id}")
                continue

            if not all(map(lambda v: 0 <= v <= 1, [x, y, w, h])):
                invalid_lines += 1
                print(f"[坐标越界] {label_path.name}: {line}")
                continue

            if w <= 0 or h <= 0:
                invalid_lines += 1
                print(f"[框尺寸错误] {label_path.name}: {line}")
                continue

            # 检查边界框是否完整落在图像范围内
            if x - w / 2 < -1e-6 or x + w / 2 > 1 + 1e-6:
                invalid_lines += 1
                print(f"[框超出图像] {label_path.name}: {line}")
                continue

            if y - h / 2 < -1e-6 or y + h / 2 > 1 + 1e-6:
                invalid_lines += 1
                print(f"[框超出图像] {label_path.name}: {line}")
                continue

            class_counts[class_id] += 1
            total_boxes += 1

    print(f"空标签文件: {empty_labels}")
    print(f"无效标注行: {invalid_lines}")
    print(f"有效目标框: {total_boxes}")

    print("\n各类别目标框数量:")
    for class_id, name in enumerate(CLASS_NAMES):
        print(f"  {class_id}: {name:<18} {class_counts[class_id]}")

    if missing_labels or missing_images or invalid_lines:
        print("\n[提示] 存在需要进一步排查的数据问题")
    else:
        print("\n[初步通过] 文件匹配及标签格式检查通过")


def main():
    for split in SPLITS:
        check_split(split)


if __name__ == "__main__":
    main()
