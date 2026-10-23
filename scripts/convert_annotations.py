from pathlib import Path
import xml.etree.ElementTree as ET

'''功能实现：
1. 读取 XML 文件中的图片尺寸、类别和目标框坐标。
2. 把 Pascal VOC 坐标转换成 YOLO 归一化坐标。
3. 将转换结果写入对应的 TXT 文件。
4. 检查未知类别、非法边界框和 XML 格式错误。
'''
# =========================
# 1. 路径配置
# =========================

PROJECT_ROOT = Path(__file__).parent.parent

DATASET_ROOT = PROJECT_ROOT.parent / "datasets" / "NEU-DET-redownload"

# XML 类别名称与 YOLO 类别编号的映射
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# 需要转换的数据集划分
SPLITS = ["train", "validation"]


def convert_box(size, box):
    """
    将 Pascal VOC 坐标转换为 YOLO 归一化坐标。

    size: (width, height)
    box: (xmin, ymin, xmax, ymax)

    返回：
        x_center, y_center, width, height
    """

    image_width, image_height = size
    xmin, ymin, xmax, ymax = box

    if image_width <= 0 or image_height <= 0:
        raise ValueError("图像宽高必须大于 0")

    if not (0 <= xmin < xmax <= image_width and 0 <= ymin < ymax <= image_height):
        raise ValueError( f"非法边界框: {box}, 图像尺寸: {size}")

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    box_width = xmax - xmin
    box_height = ymax - ymin

    # YOLO 要求使用归一化坐标
    x_center /= image_width
    y_center /= image_height
    box_width /= image_width
    box_height /= image_height

    return (
        x_center,
        y_center,
        box_width,
        box_height,
    )

def convert_xml(xml_path, output_path):
    """
    将单个 XML 文件转换成 YOLO TXT 文件。
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_node = root.find("size")

    if size_node is None:
        raise ValueError(f"缺少图像尺寸信息: {xml_path}")

    image_width = int(size_node.findtext("width", "0"))
    image_height = int(size_node.findtext("height", "0"))

    size = (image_width, image_height)

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name","").strip()

        if class_name not in CLASS_TO_ID:
            raise ValueError(f"未知类别: '{class_name}', 文件：'{xml_path}'")

        # difficult=1 的目标也先保留，避免未经分析就丢弃数据
        box_node = obj.find("bndbox")
        if box_node is None:
            raise ValueError(f"缺少边界框信息: {xml_path}")

        xmin = float(box_node.findtext("xmin", "0"))
        ymin = float(box_node.findtext("ymin", "0"))
        xmax = float(box_node.findtext("xmax", "0"))
        ymax = float(box_node.findtext("ymax", "0"))

        box = (xmin, ymin, xmax, ymax)

        x_center, y_center, width, height = convert_box(size, box)

        class_id = CLASS_TO_ID[class_name]

        line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        yolo_lines.append(line)

    # 创建输出目录
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入 YOLO TXT 文件
    output_path.write_text("\n".join(yolo_lines), encoding="utf-8")

def main():

    total_converted = 0
    total_errors = 0

    for split in SPLITS:

        annotation_dir = (
            DATASET_ROOT / split / "annotations"
        )

        if not annotation_dir.exists():
            print(f"[跳过] 找不到目录: {annotation_dir}")
            continue

        # 转换结果保存在每个划分自己的 labels 目录
        output_dir = (
            DATASET_ROOT / split / "labels"
        )

        xml_files = sorted(
            annotation_dir.glob("*.xml")
        )

        print(f"\n数据集划分: {split}")
        print(f"XML 文件数量: {len(xml_files)}")

        for xml_path in xml_files:

            try:
                output_path = (
                    output_dir / f"{xml_path.stem}.txt"
                )

                convert_xml(
                    xml_path,
                    output_path
                )

                total_converted += 1

            except (
                ET.ParseError,
                OSError,
                ValueError,
                AttributeError,
                TypeError,
            ) as error:

                total_errors += 1

                print(
                    f"[错误] {xml_path.name}: {error}"
                )

    print("\n========== 转换完成 ==========")
    print(f"成功转换: {total_converted}")
    print(f"转换失败: {total_errors}")


if __name__ == "__main__":
    main()