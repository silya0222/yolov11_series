import os
import sys
import shutil
import time
import threading
import itertools
import xml.etree.ElementTree as ET

# ================= 配置区域 =================
ROOT_DIR = r"D:\Kaggle_Cache\datasets"
# ===========================================

# --- 1. 强制引入 kaggle 库 ---
try:
    import kaggle
except ImportError:
    print("❌ 严重错误：Python 找不到 kaggle 库。")
    print("请确认在 Pycharm 的 Terminal 中运行了: pip install kaggle")
    sys.exit(1)
except OSError:
    print("❌ 配置错误：找不到 kaggle.json")
    print("请检查 C:\\Users\\你的用户名\\.kaggle\\kaggle.json 是否存在")
    sys.exit(1)

# --- 2. 动画工具 (因为官方API没有进度条，我们自己做一个转圈圈) ---
done_flag = False


def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done_flag:
            break
        sys.stdout.write(f'\r正在下载中，请稍候... {c}  ')
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\r下载完成！                 \n')


# ---------------------------------------------

def get_next_folder_name(root_path):
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    existing_dirs = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d.endswith('_k')]
    max_num = 0
    for d in existing_dirs:
        try:
            num = int(d.split('_')[0])
            if num > max_num: max_num = num
        except:
            continue
    return f"{max_num + 1:03d}_k"


def auto_detect_classes(xml_dir):
    classes_set = set()
    print("  正在扫描类别...", end="", flush=True)
    for root, dirs, files in os.walk(xml_dir):
        for file in files:
            if file.endswith('.xml'):
                try:
                    tree = ET.parse(os.path.join(root, file))
                    for obj in tree.getroot().iter('object'):
                        classes_set.add(obj.find('name').text)
                except:
                    pass
    print(" 完成！")
    return sorted(list(classes_set))


def convert_coordinates(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def find_images_recursive(search_path):
    img_map = {}
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    for root, dirs, files in os.walk(search_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                name_no_ext = os.path.splitext(f)[0]
                img_map[name_no_ext] = os.path.join(root, f)
    return img_map


def print_progress_bar(iteration, total, prefix='', suffix='', length=30, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()


def run_pipeline():
    global done_flag
    print("=" * 50)
    print("      Kaggle 数据集自动下载 & YOLO 转换 (V5 纯净版)")
    print("=" * 50)

    # 1. 获取输入 (增加去引号功能，防止你输入 "name" 导致报错)
    raw_name = input("\n请输入 Kaggle 数据集名称 (例如 liuxiaolong1/pcb-defect-detection-dataset):\n>>> ").strip()
    # 自动去掉用户可能手误输入的引号
    dataset_name = raw_name.replace('"', '').replace("'", "")

    if not dataset_name: return

    # 2. 准备路径
    folder_name = get_next_folder_name(ROOT_DIR)
    download_path = os.path.join(ROOT_DIR, folder_name)
    os.makedirs(download_path, exist_ok=True)
    print(f"\n[1/4] 保存路径: {download_path}")

    # 3. 直接调用 API 下载 (最稳妥的方式)
    print(f"[2/4] 连接 Kaggle 服务器...")

    # 启动等待动画线程
    done_flag = False
    t = threading.Thread(target=animate)
    t.start()

    try:
        # === 核心：直接使用 Python 函数下载，不走命令行 ===
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=download_path, unzip=True, quiet=False)
        # ============================================
        done_flag = True
        t.join()  # 等待动画结束
        print("✅ 下载成功！")

    except Exception as e:
        done_flag = True  # 停止动画
        print(f"\n❌ 下载失败: {e}")
        if "404" in str(e):
            print("提示：找不到该数据集，请检查名称是否拼写正确。")
        elif "403" in str(e):
            print("提示：权限拒绝，请检查你的 kaggle.json 是否有效，或者该数据集是否需要同意协议。")
        try:
            os.rmdir(download_path)
        except:
            pass
        return

    # 4. 检查 XML
    print("\n[3/4] 正在体检数据...")
    has_xml = False
    xml_files_list = []
    for root, dirs, files in os.walk(download_path):
        for f in files:
            if f.endswith('.xml'):
                has_xml = True
                xml_files_list.append(os.path.join(root, f))

    if not has_xml:
        print("ℹ️ 未发现 XML 文件，无需转换。任务结束。")
        return

    print(f"✅ 发现 {len(xml_files_list)} 个 XML 标注文件。")

    # 5. 转换 YOLO
    yolo_folder_path = download_path + "_yolo"
    classes = auto_detect_classes(download_path)
    print(f"   -> 识别类别: {classes}")

    images_save_dir = os.path.join(yolo_folder_path, 'images')
    labels_save_dir = os.path.join(yolo_folder_path, 'labels')
    os.makedirs(images_save_dir, exist_ok=True)
    os.makedirs(labels_save_dir, exist_ok=True)

    img_map = find_images_recursive(download_path)
    convert_count = 0

    print(f"[4/4] 开始转换格式 (共 {len(xml_files_list)} 个任务)")

    total_files = len(xml_files_list)
    for i, xml_full_path in enumerate(xml_files_list):
        # 转换进度条
        print_progress_bar(i + 1, total_files, prefix='转换进度:', suffix='完成', length=40)

        try:
            filename = os.path.basename(xml_full_path)
            file_id = filename[:-4]
            txt_path = os.path.join(labels_save_dir, file_id + '.txt')

            tree = ET.parse(xml_full_path)
            root_xml = tree.getroot()
            size = root_xml.find('size')
            if size is None: continue
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            if w == 0 or h == 0: continue

            yolo_lines = []
            for obj in root_xml.iter('object'):
                cls = obj.find('name').text
                if cls not in classes: continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_coordinates((w, h), b)
                yolo_lines.append(str(cls_id) + " " + " ".join([str(a) for a in bb]))

            if yolo_lines:
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))

                # 复制图片
                if file_id in img_map:
                    src_img = img_map[file_id]
                    ext = os.path.splitext(src_img)[1]
                    dst_img = os.path.join(images_save_dir, file_id + ext)
                    shutil.copy(src_img, dst_img)
                    convert_count += 1
        except Exception:
            pass

    with open(os.path.join(yolo_folder_path, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes))

    print("\n" + "=" * 50)
    print("🎉 大功告成！")
    print(f"📂 YOLO数据: {yolo_folder_path}")
    print(f"📊 成功转换: {convert_count} / {total_files}")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline()