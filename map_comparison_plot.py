import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ================= 配置区域 (可调整参数) =================

# 1. 设置实验数据的根目录
# 根据你的描述，CSV 文件都在这个 results 文件夹下
BASE_LOG_DIR = r"C:\Users\闫司隶\PycharmProjects\ultralytics\results"

# 2. 定义实验名称和对应的 文件名 或 文件夹名
# 脚本会自动判断这是一个文件夹还是一个 .csv 文件
# 请将 '001', '002' 修改为你实际的文件编号
experiment_configs = {
    "Baseline": "NEU-DET_baseline_002",  # 对应 NEU-DET_baseline_00x
    "Ours (All Modules)": "NEU-DET_all_modules_002",  # 对应 NEU-DET_all_modules_00x
}

# 3. 设置图片保存的完整路径 (包含文件名和后缀)
SAVE_IMAGE_PATH = (
    r"C:\Users\闫司隶\PycharmProjects\ultralytics\experiment_plots\map_comparison_plot\map_comparison-NEU-DET_002.png"
)


# ================= 绘图逻辑 (通常无需修改) =================
# 设置图表风格
sns.set_theme(style="whitegrid")  # 使用 Seaborn 的白色网格风格
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]  # 支持中文显示
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 创建画布
plt.figure(figsize=(10, 6))

# 构建完整的路径字典 (智能识别文件或文件夹)
experiments = {}
for name, file_or_folder in experiment_configs.items():
    # 拼接基础路径
    candidate_path = os.path.join(BASE_LOG_DIR, file_or_folder)

    full_path = ""
    # 逻辑判断：
    # 1. 如果路径本身是一个存在的 .csv 文件 -> 直接使用
    if os.path.isfile(candidate_path) and candidate_path.endswith(".csv"):
        full_path = candidate_path
    # 2. 如果路径加上 .csv 后存在 -> 是个文件，但配置没写后缀
    elif os.path.isfile(candidate_path + ".csv"):
        full_path = candidate_path + ".csv"
    # 3. 如果是一个文件夹 -> 找里面的 results.csv
    elif os.path.isdir(candidate_path):
        full_path = os.path.join(candidate_path, "results.csv")
    # 4. 默认回退 -> 假设是文件夹模式
    else:
        full_path = os.path.join(candidate_path, "results.csv")

    experiments[name] = full_path

# 数据读取与绘图
for label, csv_path in experiments.items():
    if not os.path.exists(csv_path):
        print(f"⚠️ 警告: 找不到文件 {csv_path}，请检查文件名或路径配置。")
        continue

    try:
        # 读取 CSV 数据
        df = pd.read_csv(csv_path)

        # 清理列名（Ultralytics 的 csv 列名通常带有空格）
        df.columns = [c.strip() for c in df.columns]

        # 确定 x 轴和 y 轴的列名
        x_col = "epoch"
        y_col = "metrics/mAP50(B)"  # 默认列名

        # 兼容性处理：如果找不到默认列名，尝试模糊搜索
        if y_col not in df.columns:
            possible_cols = [c for c in df.columns if "map50" in c.lower()]
            if possible_cols:
                y_col = possible_cols[0]
            else:
                print(f"❌ 错误: 在 {label} 中找不到 mAP50 数据列。现有列名: {df.columns}")
                continue

        # 绘制曲线
        plt.plot(df[x_col], df[y_col], label=label, linewidth=2)

        # 打印最终精度
        max_map = df[y_col].max()
        print(f"✅ {label}: 最佳 mAP@0.5 = {max_map:.4f}")

    except Exception as e:
        print(f"❌ 读取 {label} 时出错: {e}")

# ================= 图表美化 =================
plt.title("mAP@0.5 Comparison", fontsize=16, fontweight="bold")
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("mAP@0.5", fontsize=12)

# 设置图例
plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)

# 设置坐标轴范围
plt.xlim(0, None)
plt.ylim(0, 1.0)

# 添加网格
plt.grid(True, linestyle="--", alpha=0.7)

# ================= 保存与显示 =================
# 自动创建输出目录
output_dir = os.path.dirname(SAVE_IMAGE_PATH)
if output_dir and not os.path.exists(output_dir):
    try:
        os.makedirs(output_dir)
        print(f"📂 已创建目录: {output_dir}")
    except OSError as e:
        print(f"❌ 创建目录失败: {e}")

# 保存图片
try:
    plt.savefig(SAVE_IMAGE_PATH, dpi=300, bbox_inches="tight")
    print(f"\n🎉 图表已成功保存到: {SAVE_IMAGE_PATH}")
except Exception as e:
    print(f"\n❌ 保存图片失败: {e}")
    # 如果指定路径保存失败，尝试保存到当前目录作为备份
    fallback_path = "map_comparison_plot_fallback.png"
    plt.savefig(fallback_path, dpi=300, bbox_inches="tight")
    print(f"   已备份保存到当前目录: {fallback_path}")

# 显示图片
plt.show()
