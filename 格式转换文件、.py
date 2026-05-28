import os
import shutil
import xml.etree.ElementTree as ET

# ================= 1. 配置区域 =================

# 数据集根目录
root_path = r"D:\Kaggle_Cache\datasets\kaustubhdikshit\neu-surface-defect-database"

# 原文件夹名
source_folder_name = "NEU-DET"

# 新文件夹名 (脚本会自动创建，存放转换好的数据)
target_folder_name = "NEU-DET"

# 需要处理的子集 (训练集和验证集)
sets = ["train", "validation"]

# 缺陷类别名称 (顺序千万不能错)
classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


# ===============================================


def convert(size, box):
    """计算坐标转换 (XML -> YOLO)."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def make_dir(path):
    """创建文件夹."""
    if not os.path.exists(path):
        os.makedirs(path)


def get_image_map(search_dir):
    """关键函数：地毯式搜索！ 去 search_dir 文件夹及其所有子文件夹里找图片。 返回一个字典：{'文件名(无后缀)': '文件的完整绝对路径'}.
    """
    img_map = {}
    # os.walk 会一层层深入所有子文件夹
    for root, dirs, files in os.walk(search_dir):
        for f in files:
            # 获取文件名（不带后缀），比如 crazing_1
            name_no_ext = os.path.splitext(f)[0]
            # 记录完整路径
            img_map[name_no_ext] = os.path.join(root, f)
    return img_map


def run():
    print("🚀 开始处理，正在深入搜索图片...\n")

    base_source_path = os.path.join(root_path, source_folder_name)
    base_target_path = os.path.join(root_path, target_folder_name)

    for subset in sets:
        print(f"--- 正在处理子集: {subset} ---")

        # 1. 定义源路径
        src_img_dir = os.path.join(base_source_path, subset, "images")
        src_xml_dir = os.path.join(base_source_path, subset, "annotations")

        # 2. 定义目标路径 (YOLO格式)
        dst_img_dir = os.path.join(base_target_path, "images", subset)
        dst_label_dir = os.path.join(base_target_path, "labels", subset)

        make_dir(dst_img_dir)
        make_dir(dst_label_dir)

        # --- 第一步：建立图片地图 ---
        if os.path.exists(src_img_dir):
            print(f"  正在扫描 {src_img_dir} 下的所有子文件夹...")
            img_map = get_image_map(src_img_dir)
            print(f"  -> 找到了 {len(img_map)} 张图片。")
        else:
            print(f"  [错误] 找不到图片文件夹: {src_img_dir}，跳过该子集。")
            continue

        # --- 第二步：处理 XML 并匹配图片 ---
        if not os.path.exists(src_xml_dir):
            print(f"  [错误] 找不到标注文件夹: {src_xml_dir}")
            continue

        xml_files = [f for f in os.listdir(src_xml_dir) if f.endswith(".xml")]
        count = 0
        missing_count = 0

        for filename in xml_files:
            file_id = filename[:-4]  # 去掉 .xml 后缀

            # 1. 转换标签 (XML -> TXT)
            in_file_path = os.path.join(src_xml_dir, filename)
            out_file_path = os.path.join(dst_label_dir, file_id + ".txt")

            try:
                in_file = open(in_file_path, encoding="utf-8")
                tree = ET.parse(in_file)
                root = tree.getroot()

                size = root.find("size")
                w = int(size.find("width").text)
                h = int(size.find("height").text)

                if w == 0 or h == 0:
                    print(f"  [警告] {filename} 宽高为0，跳过")
                    in_file.close()
                    continue

                out_file = open(out_file_path, "w", encoding="utf-8")
                for obj in root.iter("object"):
                    cls = obj.find("name").text
                    if cls not in classes:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find("bndbox")
                    b = (
                        float(xmlbox.find("xmin").text),
                        float(xmlbox.find("xmax").text),
                        float(xmlbox.find("ymin").text),
                        float(xmlbox.find("ymax").text),
                    )
                    bb = convert((w, h), b)
                    out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")

                out_file.close()
                in_file.close()

                # 2. 复制图片 (利用之前的地图)
                if file_id in img_map:
                    src_img_path = img_map[file_id]  # 获取真实的图片路径（在某个子文件夹里）
                    # 复制时我们把图片都“平铺”放在一起，方便YOLO读取
                    # 这里自动保留原后缀（不管是 .jpg 还是 .bmp）
                    ext = os.path.splitext(src_img_path)[1]
                    dst_img_path = os.path.join(dst_img_dir, file_id + ext)

                    shutil.copy(src_img_path, dst_img_path)
                    count += 1
                else:
                    print(f"  [缺失] 有XML但找不到图片: {file_id}")
                    missing_count += 1

            except Exception as e:
                print(f"  处理 {filename} 发生错误: {e}")

        print(f"  -> {subset} 搞定！成功转换并复制了 {count} 组数据。")
        if missing_count > 0:
            print(f"  -> 有 {missing_count} 个XML没找到对应的图片。")
        print("-" * 30)

    print("\n✅ 全部完成！")
    print(f"新数据集保存在: {base_target_path}")
    print("现在的结构是 YOLO 标准格式 (images 和 labels 文件夹)，可以直接拿去训练了。")


if __name__ == "__main__":
    run()
