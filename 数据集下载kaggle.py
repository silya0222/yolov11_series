import os
# 1. 设置全局缓存目录 (必须在 import kagglehub 之前设置)
# 所有的 kagglehub 下载都会保存在这个文件夹下
os.environ['KAGGLEHUB_CACHE'] = r"D:\Kaggle_Cache"
import kagglehub
import kagglehub

# 2. 下载
# 现在文件会下载到 D:\Kaggle_Cache\datasets\kaustubhdikshit\neu-surface-defect-database\...
path = kagglehub.dataset_download("kaustubhdikshit/neu-surface-defect-database")
print("Path to dataset files:", path)
# 查看下载的文件结构 (可选，帮助您确认数据格式)
print("\n数据集目录内容:")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = ' ' * 4 * (level)
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 4 * (level + 1)
    # 只打印前5个文件，避免刷屏
    for f in files[:5]:
        print(f'{subindent}{f}')
    if len(files) > 5:
        print(f'{subindent}... (共 {len(files)} 个文件)')