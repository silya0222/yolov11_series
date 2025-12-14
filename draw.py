import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_map(recall, precision):
    """计算曲线下面积 (mAP 近似值)"""
    if len(recall) < 2:
        return 0.0
    # 按照 recall 排序，防止计算面积出错
    sorted_indices = np.argsort(recall)
    r = np.array(recall)[sorted_indices]
    p = np.array(precision)[sorted_indices]
    return np.trapz(p, r)

def find_file(filename, search_dirs):
    """在指定的多个目录中查找文件"""
    for d in search_dirs:
        try:
            p = Path(d) / filename
            if p.exists():
                return p
        except Exception:
            continue
    return None

def plot_pr_comparison():
    # ================= 配置区域 =================
    # 1. 设置保存路径
    save_dir = Path(r"C:\Users\闫司隶\PycharmProjects\ultralytics\experiment_plots")
    
    # 2. 定义文件名
    name_baseline = "all_classes_bBox.csv"
    name_improve  = "all_classes_iBox.csv"
    
    # 3. 样式配置 (在这里修改颜色、线宽、线型)
    CONFIG_LINE_WIDTH = 4       # 线条粗细 (数字越大越粗)
    CONFIG_LINE_STYLE = '-'     # 线条类型 ('-' 为实线, '--' 为虚线, ':' 为点线)
    
    # 颜色设置 (可以使用 'blue', 'red', 'green', 'black' 等，也可以用十六进制 '#FF5733')
    COLOR_BASELINE = '#FF5733'     # Baseline 曲线颜色
    COLOR_IMPROVE  = "#EBDE29"      # Improve 曲线颜色
    # ===========================================

    # 获取当前工作目录
    cwd = Path.cwd()
    print(f"\n📂 当前工作目录 (脚本运行位置): {cwd}")
    
    # 定义搜索路径列表
    search_candidates = [
        cwd, 
        cwd / "b",
        Path(r"C:\Users\闫司隶\PycharmProjects\ultralytics"),
        Path(r"C:\Users\闫司隶\PycharmProjects\ultralytics\b"),
    ]
    
    # 自动查找文件
    path_baseline = find_file(name_baseline, search_candidates)
    path_improve  = find_file(name_improve, search_candidates)

    files_map = {}
    
    # 检查并添加到绘图列表，应用配置的颜色和样式
    if path_baseline:
        print(f"✅ 成功找到 Baseline 文件: {path_baseline}")
        files_map['Baseline'] = {
            'path': path_baseline, 
            'color': COLOR_BASELINE, 
            'style': CONFIG_LINE_STYLE
        }
    else:
        print(f"❌ 失败: 找不到 Baseline 文件 ({name_baseline})")

    if path_improve:
        print(f"✅ 成功找到 Improve 文件: {path_improve}")
        files_map['Improve'] = {
            'path': path_improve, 
            'color': COLOR_IMPROVE, 
            'style': CONFIG_LINE_STYLE
        }
    else:
        print(f"❌ 失败: 找不到 Improve 文件 ({name_improve})")

    # 如果两个都没找到，退出
    if not files_map:
        print("\n⚠️ 错误: 没有任何文件被找到，无法绘图！")
        return

    # 创建保存目录
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / "comparison_baseline_vs_improve_final.png"

    # 创建画布
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)
    
    lines_plotted = 0

    for label_name, config in files_map.items():
        file_path = config['path']
        
        try:
            data = pd.read_csv(file_path, header=None)
            
            if data.empty:
                continue

            recall = data.iloc[:, 0].values
            precision = data.iloc[:, 1].values
            
            if len(recall) == 0:
                continue

            map_score = calculate_map(recall, precision)
            
            # 绘图 (使用配置的参数)
            ax.plot(recall, precision, 
                    linewidth=CONFIG_LINE_WIDTH,  # 使用配置的线宽
                    linestyle=config['style'],    # 使用配置的线型
                    color=config['color'],        # 使用配置的颜色
                    label=f"{label_name} (mAP≈{map_score:.3f})")
            
            lines_plotted += 1
            
        except Exception as e:
            print(f"❌ 读取或绘制 {label_name} 出错: {e}")

    if lines_plotted == 0:
        print("\n⚠️ 警告: 没有绘制任何线条。")
        plt.close(fig)
        return

    # 设置图表样式
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Precision-Recall Curve Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')

    # 保存并显示
    try:
        fig.savefig(output_file, dpi=300)
        print(f"\n🎉 绘图成功！图片已保存到: {output_file}")
    except Exception as e:
        print(f"\n❌ 保存图片失败: {e}")
        backup_file = cwd / "comparison_backup.png"
        fig.savefig(backup_file, dpi=300)
        print(f"   -> 已备份保存到当前目录: {backup_file}")

    plt.show()
    plt.close(fig)

if __name__ == "__main__":
    plot_pr_comparison()