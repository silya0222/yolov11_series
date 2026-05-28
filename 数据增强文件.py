import cv2
import numpy as np


class DataAugmentor:
    def __init__(self):
        pass

    def augment_flip(self, image, mode=1):
        """
        【1. 翻转 (Flip)】
        :param image: 输入图像
        :param mode: 翻转模式。1=水平翻转(左右), 0=垂直翻转(上下), -1=水平垂直同时翻转
        :return: 翻转后的图像.
        """
        # cv2.flip 直接进行翻转
        return cv2.flip(image, mode)

    def augment_rotate(self, image, angle=15):
        """
        【2. 旋转 (Rotation)】
        :param image: 输入图像
        :param angle: 旋转角度（正数逆时针，负数顺时针）
        :return: 旋转后的图像，边缘填充黑色(0).
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 获取旋转矩阵：指定中心、角度、缩放因子(这里保持1.0)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 进行仿射变换
        # borderMode=cv2.BORDER_CONSTANT, borderValue=0 实现了"边界外为常数0/黑色"的假设
        rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return rotated

    def augment_shift(self, image, shift_x=0.1, shift_y=0.1):
        """
        【3. 移位/平移 (Shift/Translation)】
        :param image: 输入图像
        :param shift_x: 水平移动比例 (例如 0.1 代表移动宽度的10%)
        :param shift_y: 垂直移动比例
        :return: 平移后的图像，空出区域填充黑色.
        """
        h, w = image.shape[:2]
        tx = shift_x * w
        ty = shift_y * h

        # 定义平移矩阵: [[1, 0, tx], [0, 1, ty]]
        M = np.float32([[1, 0, tx], [0, 1, ty]])

        # 仿射变换，同样填充黑色背景
        shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return shifted

    def augment_scale(self, image, scale_factor=0.8):
        """
        【4. 缩放比例 (Scaling)】
        此实现保持原图画布大小不变。
        - 如果 scale < 1.0: 缩小图像，周围填充黑色。
        - 如果 scale > 1.0: 放大图像，并裁剪掉超出画布的部分（中心裁剪）。
        :param image: 输入图像
        :param scale_factor: 缩放因子
        :return: 缩放并保持原尺寸的图像.
        """
        h, w = image.shape[:2]

        # 按照比例resize
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建一个全黑的画布，大小与原图一致
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if scale_factor < 1.0:
            # 缩小模式：计算中心位置并贴图
            pad_x = (w - new_w) // 2
            pad_y = (h - new_h) // 2
            canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
            return canvas
        else:
            # 放大模式：裁剪中心区域
            crop_x = (new_w - w) // 2
            crop_y = (new_h - h) // 2
            cropped = resized[crop_y : crop_y + h, crop_x : crop_x + w]
            return cropped

    def augment_crop(self, image, crop_ratio=0.8):
        """
        【5. 裁剪 (Cropping)】
        这里模拟"中心放大裁剪"，即保留中间部分，拉伸回原图大小。
        也可以理解为 Zoom In。
        :param image: 输入图像
        :param crop_ratio: 裁剪保留的比例 (例如0.8表示保留中间80%的区域)
        :return: 裁剪并Resize回原大小的图像.
        """
        h, w = image.shape[:2]

        # 计算裁剪区域的大小
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)

        # 计算起始坐标（中心裁剪）
        y1 = (h - crop_h) // 2
        x1 = (w - crop_w) // 2

        # 执行切片裁剪
        cropped = image[y1 : y1 + crop_h, x1 : x1 + crop_w]

        # 重新Resize回原图大小，以便输入神经网络
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return resized

    def augment_gaussian_noise(self, image, mean=0, var=0.01):
        """
        【6. 高斯噪声 (Gaussian Noise)】
        :param image: 输入图像
        :param mean: 噪声均值
        :param var: 噪声方差 (决定噪声的强度)
        :return: 添加噪声后的图像.
        """
        # 将图像归一化到 [0, 1] 方便计算
        img_float = image.astype(np.float32) / 255.0

        # 生成高斯噪声
        sigma = var**0.5
        noise = np.random.normal(mean, sigma, img_float.shape)

        # 叠加噪声
        noisy_img = img_float + noise

        # 截断超出范围的值并恢复到 [0, 255]
        noisy_img = np.clip(noisy_img, 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)

        return noisy_img


# ==========================================
# 使用示例 (Usage)
# ==========================================
if __name__ == "__main__":
    # 1. 读取一张本地图片 (请替换为你自己的路径)
    # 如果没有图片，这行代码会报错，请确保路径正确
    # img_path = "dataset/images/train/crazing_1.jpg"
    # img = cv2.imread(img_path)

    # 为了演示，我们生成一张灰色测试图
    img = np.zeros((640, 640, 3), dtype=np.uint8) + 100
    cv2.rectangle(img, (200, 200), (440, 440), (0, 255, 0), 5)  # 画个框方便看效果

    aug = DataAugmentor()

    # --- 测试各个功能 ---

    # 1. 翻转 (水平)
    res_flip = aug.augment_flip(img, mode=1)

    # 2. 旋转 (逆时针30度)
    res_rot = aug.augment_rotate(img, angle=30)

    # 3. 移位 (右移20%，下移10%)
    res_shift = aug.augment_shift(img, shift_x=0.2, shift_y=0.1)

    # 4. 缩放 (缩小为0.6倍，周围补黑)
    res_scale = aug.augment_scale(img, scale_factor=0.6)

    # 5. 裁剪 (保留中间70%区域并拉伸)
    res_crop = aug.augment_crop(img, crop_ratio=0.7)

    # 6. 高斯噪声 (方差0.05)
    res_noise = aug.augment_gaussian_noise(img, var=0.05)

    # --- 显示结果 (按任意键关闭窗口) ---
    # 拼接显示：上排原图+旋转+移位，下排缩放+裁剪+噪声
    top_row = np.hstack((img, res_rot, res_shift))
    bottom_row = np.hstack((res_scale, res_crop, res_noise))
    final_show = np.vstack((top_row, bottom_row))

    # 缩放一下以便在屏幕上显示
    view_img = cv2.resize(final_show, (0, 0), fx=0.5, fy=0.5)

    cv2.imshow("Data Augmentation Demo", view_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
