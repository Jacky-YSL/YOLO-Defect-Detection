from pathlib import Path
import shutil

'''功能实现：
从 NEU-DET-redownload 读取图片和 YOLO 标签。
将图片复制到统一的训练集、验证集目录。
检查每张图片是否有对应标签。
生成 data.yaml。
'''

# ==================== 路径配置 ====================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 已经修复并检查过的数据集
SOURCE_ROOT = PROJECT_ROOT.parent / "datasets" / "NEU-DET-redownload"

# 最终 YOLO 数据集目录
OUTPUT_ROOT = PROJECT_ROOT.parent / "datasets" / "NEU-DET-YOLO"

CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(split):
    """递归查找指定划分下的图片。"""
    image_root = SOURCE_ROOT / split / "images"

    return sorted(
        p for p in image_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_labels(split):
    """递归查找指定划分下的 YOLO 标签。"""
    label_root = SOURCE_ROOT / split / "labels"

    return sorted(label_root.rglob("*.txt"))


def validate_label(label_path):
    """检查 YOLO 标签的基本格式和数值范围。"""
    with label_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"标签格式错误：{label_path}，第 {line_number} 行"
            )

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(
                f"类别编号错误：{label_path}，第 {line_number} 行"
            )

        if not all(
            0 <= value <= 1
            for value in (x_center, y_center, width, height)
        ):
            raise ValueError(
                f"坐标超出 [0, 1]：{label_path}，第 {line_number} 行"
            )

        if width <= 0 or height <= 0:
            raise ValueError(
                f"目标框宽高不合法：{label_path}，第 {line_number} 行"
            )

    return len(lines)


def prepare_split(source_split, output_split):
    """复制图片和标签，统一为 YOLO 标准目录。"""
    images = find_images(source_split)
    labels = find_labels(source_split)

    image_map = {p.stem: p for p in images}
    label_map = {p.stem: p for p in labels}

    if len(image_map) != len(images):
        raise ValueError(
            f"{source_split} 中存在同名图片，请先排查文件名冲突。"
        )

    if len(label_map) != len(labels):
        raise ValueError(
            f"{source_split} 中存在同名标签，请先排查文件名冲突。"
        )

    image_stems = set(image_map)
    label_stems = set(label_map)

    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)

    print(f"\n========== {source_split} ==========")
    print(f"图片数量：{len(images)}")
    print(f"标签数量：{len(labels)}")
    print(f"缺少标签的图片：{missing_labels}")
    print(f"缺少图片的标签：{missing_images}")

    if missing_labels or missing_images:
        raise ValueError(
            f"{source_split} 的图片与标签不匹配，停止整理。"
        )

    output_images = OUTPUT_ROOT / "images" / output_split
    output_labels = OUTPUT_ROOT / "labels" / output_split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    total_boxes = 0

    for stem, image_path in image_map.items():
        label_path = label_map[stem]

        # 检查标签格式
        total_boxes += validate_label(label_path)

        # 扁平化文件结构，复制图片与标签
        shutil.copy2(
            image_path,
            output_images / image_path.name
        )

        shutil.copy2(
            label_path,
            output_labels / label_path.name
        )

    print(f"整理后图片数量：{len(images)}")
    print(f"整理后标签数量：{len(labels)}")
    print(f"目标框总数：{total_boxes}")

    return len(images), total_boxes


def write_data_yaml():
    """生成 YOLO 数据集配置文件。"""
    yaml_path = OUTPUT_ROOT / "data.yaml"

    lines = [
        f"path: {OUTPUT_ROOT.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]

    for class_id, class_name in enumerate(CLASS_NAMES):
        lines.append(f"  {class_id}: {class_name}")

    yaml_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"\n配置文件已生成：{yaml_path}")


def main():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"找不到数据集目录：{SOURCE_ROOT}"
        )

    if not (SOURCE_ROOT / "train" / "images").exists():
        raise FileNotFoundError("训练集图片目录不存在。")

    if not (SOURCE_ROOT / "validation" / "images").exists():
        raise FileNotFoundError("验证集图片目录不存在。")

    if not (SOURCE_ROOT / "train" / "labels").exists():
        raise FileNotFoundError(
            "训练集标签目录不存在，请确认已经生成 YOLO TXT 标签。"
        )

    if not (SOURCE_ROOT / "validation" / "labels").exists():
        raise FileNotFoundError(
            "验证集标签目录不存在，请确认已经生成 YOLO TXT 标签。"
        )

    # 避免覆盖旧数据
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"输出目录已存在且不为空：{OUTPUT_ROOT}\n"
            "请先检查该目录，不要直接覆盖。"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    train_count, train_boxes = prepare_split("train", "train")
    val_count, val_boxes = prepare_split("validation", "val")

    write_data_yaml()

    print("\n========== 全部完成 ==========")
    print(f"训练集图片：{train_count}")
    print(f"验证集图片：{val_count}")
    print(f"训练集目标框：{train_boxes}")
    print(f"验证集目标框：{val_boxes}")
    print(f"输出目录：{OUTPUT_ROOT}")


if __name__ == "__main__":
    main()