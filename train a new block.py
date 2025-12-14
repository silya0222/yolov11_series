import warnings

warnings.filterwarnings('ignore')  # 忽略一些不必要的警告
from ultralytics import YOLO


def main():
    # ---------------------------------------------------
    # 1. 路径配置
    # ---------------------------------------------------
    # 指向你刚才创建的魔改配置文件 (请确保文件路径正确)
    model_yaml = 'models/yolo11_cbam.yaml'

    # 指向官方的预训练权重文件 (用于迁移学习，加速收敛)
    # 如果没有自动下载，可以手动去官网下载 yolo11n.pt 放在根目录
    pretrained_weights = 'yolo11n.pt'

    # ---------------------------------------------------
    # 2. 模型初始化
    # ---------------------------------------------------
    # 这一步会根据你的 yaml 文件构建网络结构
    print(f"正在加载模型结构: {model_yaml} ...")
    model = YOLO(model_yaml)

    # 加载预训练权重
    # 注意：由于我们添加了 CBAM 模块，这部分权重的形状与官方权重不匹配
    # 所以控制台会提示 "transferred ... items, missing ..."，这是正常的！
    print(f"正在加载预训练权重: {pretrained_weights} ...")
    try:
        model.load(pretrained_weights)
    except Exception as e:
        print(f"权重加载警告 (正常现象): {e}")

    # ---------------------------------------------------
    # 3. 开始训练 (Training)
    # ---------------------------------------------------
    print("开始训练...")
    results = model.train(
        # === 数据集设置 ===
        data='km.yaml',  # 这里换成你自己的数据集 yaml 路径，例如 'data/my_dataset.yaml'

        # === 训练超参数 ===
        epochs=100,  # 训练轮次：建议至少 100-300 轮
        imgsz=640,  # 图片大小：YOLOv11 默认 640
        batch=16,  # 批次大小：根据你显存大小调整 (显存小就设为 8 或 4)

        # === 硬件与性能 ===
        device='0',  # 设备：'0' 表示第一块 GPU，'cpu' 表示使用 CPU
        workers=1,  # 数据加载线程：Windows下建议设为 0-4，Linux 可以设高一点

        # === 优化与保存 ===
        project='results',  # 项目保存的主目录 (已修改为 results)
        name='yolo11_cbam_v1',  # 本次实验的名称 (结果会保存在 results/yolo11_cbam_v1)
        exist_ok=True,  # 如果文件夹已存在，是否覆盖 (True=覆盖/继续写，False=新建 v2, v3)
        optimizer='auto',  # 优化器：'auto' 会自动选择 SGD 或 AdamW
        verbose=True,  # 是否打印详细信息

        # === 魔改调试常用 ===
        amp=True,  # 混合精度训练 (True=快且省显存，如果报错设为 False)
        plots=True  # 训练结束后自动画出混淆矩阵等图表
    )

    # ---------------------------------------------------
    # 4. 验证与导出 (可选)
    # ---------------------------------------------------
    # print("训练完成，开始在验证集上评估...")
    # metrics = model.val() # 在验证集上跑一遍

    # print("导出 ONNX 模型用于部署...")
    # path = model.export(format="onnx")


if __name__ == '__main__':
    # 必须加这个判断，否则在 Windows 下多线程会报错
    main()