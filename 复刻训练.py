import warnings

warnings.filterwarnings('ignore')  # 忽略一些版本警告
from ultralytics import YOLO
import torch
import numpy as np
import random
import os


# ==============================================================================
# 1. 随机种子设置 (Ensuring Reproducibility)
# ==============================================================================
def set_seed(seed=21):
    """
    固定所有随机种子以确保实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Info] Random seed set to {seed}")


# 设置常数种子
SEED_VALUE = 42
set_seed(SEED_VALUE)

# ==============================================================================
# 2. 超参数配置 (Hyperparameters - Default Values)
# 这里列出了YOLO训练涉及的所有主要超参数，均设为默认值
# ==============================================================================
TRAIN_ARGS = {
    # --- 基础配置 ---
    'imgsz': 640,  # 训练图像大小
    'epochs': 100,  # 训练轮数
    'batch': 16,  # 批次大小 (-1 为自动)
    'device': '0',  # GPU设备索引 (e.g., '0' or '0,1,2,3' or 'cpu')
    'workers':1,  # 数据加载线程数
    'seed': SEED_VALUE,  # 随机种子
    'exist_ok': True,  # 是否覆盖同名实验文件夹
    'pretrained': True,  # 是否加载预训练权重 (对于微调通常为True，对于严格消融可设为False)
    'optimizer': 'auto',  # 优化器: 'SGD', 'Adam', 'AdamW', 'RMSProp'
    'verbose': True,  # 是否打印详细信息
    'deterministic': True,  # 强制确定性算法 (与 seed 配合)
    'single_cls': False,  # 是否作为单类别训练
    'rect': False,  # 是否使用矩形训练 (Rectangular training)
    'cos_lr': False,  # 是否使用余弦学习率调度器
    'close_mosaic': 10,  # 最后10个epoch关闭Mosaic增强
    'resume': False,  # 是否从断点恢复
    'amp': True,  # 是否使用自动混合精度 (Automatic Mixed Precision)
    'fraction': 1.0,  # 使用数据集的比例 (1.0 = 100%)
    'profile': False,  # 是否在训练期间分析 ONNX/TensorRT 速度
    'freeze': None,  # 冻结层索引列表 (e.g., [0] 冻结第一层)

    # --- 优化器与损失函数超参数 ---
    'lr0': 0.01,  # 初始学习率 (SGD=0.01, Adam=0.001)
    'lrf': 0.01,  # 最终学习率 (lr0 * lrf)
    'momentum': 0.937,  # SGD momentum/Adam beta1
    'weight_decay': 0.0005,  # 优化器权重衰减
    'warmup_epochs': 3.0,  # 预热轮数
    'warmup_momentum': 0.8,  # 预热初始动量
    'warmup_bias_lr': 0.1,  # 预热初始偏置学习率
    'box': 7.5,  # Box loss gain
    'cls': 0.5,  # Cls loss gain (scale with pixels)
    'dfl': 1.5,  # DFL loss gain
    'pose': 12.0,  # Pose loss gain (仅关键点任务)
    'kobj': 1.0,  # Keypoint obj loss gain (仅关键点任务)
    'label_smoothing': 0.0,  # 标签平滑
    'nbs': 64,  # Nominal batch size
    'overlap_mask': True,  # Masks overlap (仅分割任务)
    'mask_ratio': 4,  # Mask downsample ratio (仅分割任务)
    'dropout': 0.0,  # Dropout regularization (0.0 means no dropout)
    'val': True,  # 训练期间是否验证
}

# ==============================================================================
# 3. 数据增强配置 (Data Augmentation)
# 这里配置反转、旋转、平移、缩放、裁剪(通过mosaic隐含)、拼接等
# ==============================================================================
AUGMENTATION_ARGS = {
    # --- 几何变换 ---
    'degrees': 0.0,  # [旋转] 图像旋转 (+/- deg)
    'translate': 0.1,  # [平移] 图像平移 (+/- fraction)
    'scale': 0.5,  # [缩放] 图像缩放增益 (+/- gain)
    'shear': 0.0,  # [剪切] 图像剪切 (+/- deg)
    'perspective': 0.0,  # [透视] 图像透视变换 (+/- fraction), range 0-0.001

    # --- 翻转 ---
    'flipud': 0.0,  # [反转] 垂直翻转概率 (0.0 - 1.0)
    'fliplr': 0.5,  # [反转] 水平翻转概率 (0.0 - 1.0)

    # --- 拼接与混合 (Splicing) ---
    'mosaic': 1.0,  # [拼接] Mosaic 增强概率 (0.0 - 1.0) - 将4张图拼成1张
    'mixup': 0.0,  # [混合] Mixup 增强概率 (0.0 - 1.0) - 图像混合
    'copy_paste': 0.0,  # [复制粘贴] Segment copy-paste 概率 (0.0 - 1.0, 需分割数据)

    # --- 颜色与噪声 (Color & Noise) ---
    # YOLO 没有直接的 "gaussian_noise" 参数，但通过 HSV 变换引入颜色噪声
    # 也可以通过 albumentations 引入高斯噪声，这里使用原生参数模拟
    'hsv_h': 0.015,  # [噪声] 色调 (Hue) 变换分数
    'hsv_s': 0.7,  # [噪声] 饱和度 (Saturation) 变换分数
    'hsv_v': 0.4,  # [噪声] 亮度 (Value) 变换分数

    # --- 自动增强 ---
    'auto_augment': 'randaugment',  # 自动增强策略 (randaugment, autoaugment, augmix)
    'erasing': 0.4,  # [擦除] Random erasing 增强概率 (0.0 - 1.0)
    'crop_fraction': 1.0,  # [裁剪] 图像裁剪比例 (1.0 means no random crop during eval)
}

# 合并所有参数
FULL_ARGS = {**TRAIN_ARGS, **AUGMENTATION_ARGS}


# ==============================================================================
# 4. 自动命名工具函数
# ==============================================================================
def get_next_experiment_name(project_dir, base_name):
    """
    检查 project_dir 下已有的文件夹，生成 base_name_001, base_name_002 等递增名称
    """
    if not os.path.exists(project_dir):
        return f"{base_name}_001"

    existing_dirs = [d for d in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, d))]
    max_idx = 0

    for d in existing_dirs:
        # 检查文件夹是否以 base_name 开头，并且后续是数字
        if d.startswith(base_name):
            try:
                # 尝试解析后缀: name_001 -> 001
                suffix = d.replace(base_name + '_', '')
                if suffix.isdigit():
                    idx = int(suffix)
                    if idx > max_idx:
                        max_idx = idx
            except Exception:
                continue

    next_idx = max_idx + 1
    return f"{base_name}_{next_idx:03d}"


# ==============================================================================
# 5. 训练执行函数
# ==============================================================================
def run_training(model_yaml, project_name, dataset_yaml):
    """
    执行单次训练任务，自动生成实验名称
    """
    # 1. 提取文件名作为基础名称 (去除路径和后缀)
    # 例如: 'yolov11n.yaml' -> 'yolov11n'
    base_name = os.path.splitext(os.path.basename(model_yaml))[0]

    # 2. 生成递增的实验名称 (e.g., yolov11n_001)
    experiment_name = get_next_experiment_name(project_name, base_name)

    print(f"\n{'=' * 40}")
    print(f"Starting Experiment: {experiment_name}")
    print(f"Model Config: {model_yaml}")
    print(f"Save Path: {os.path.join(project_name, experiment_name)}")
    print(f"{'=' * 40}\n")

    try:
        # 3. 加载模型
        if model_yaml.endswith('.yaml'):
            model = YOLO(model_yaml)
        else:
            model = YOLO(model_yaml)

            # 4. 开始训练
        results = model.train(
            data=dataset_yaml,
            project=project_name,
            name=experiment_name,
            **FULL_ARGS  # 传入上面定义的所有参数
        )

        print(f"Experiment {experiment_name} finished successfully.")
        return results

    except Exception as e:
        print(f"Error during training {experiment_name}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # ==========================================================================
    # 配置区
    # ==========================================================================
    DATASET_CFG = 'NEU-DET.yaml'  # 数据集配置文件
    PROJECT_DIR = 'results'  # 结果保存根目录

    # 定义两个模型文件
    BASELINE_MODEL = 'yolov11n_baseline.yaml'  # 请确保该文件存在
    IMPROVED_MODEL = 'yolov11n_improve.yaml'  # 请确保该文件存在

    # ==========================================================================
    # 任务 1: 训练 Baseline
    # ==========================================================================
    # 自动保存为 result/yolov11n_001 (如果已存在则为 _002, _003...)
    #run_training(
        #model_yaml=BASELINE_MODEL,
        #project_name=PROJECT_DIR,
        #dataset_yaml=DATASET_CFG
    #)

    # ==========================================================================
    # 任务 2: 训练 Improved Model
    # ==========================================================================
    # 自动保存为 results/yolo11n_improved_v2_001 (如果已存在则为 _002...)
    # 注意：运行改进模型前，请确保已经修改了 ultralytics 源码
    run_training(
        model_yaml=IMPROVED_MODEL,
        project_name=PROJECT_DIR,
        dataset_yaml=DATASET_CFG
    )

    print("\nAll experiments completed.")