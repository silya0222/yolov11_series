#ASFF (Adaptive Spatial Feature Fusion) 检测头
#原理：自适应空间特征融合。
#作用：解决 FPN 多尺度融合时，不同层级特征（高层语义与低层细节）之间的冲突问题。它通过学习权重参数，自适应地融合 Level 1, 2, 3 的特征。
#论文位置：位于检测头（Head）之前。
#YOLOv11 适配：这是改动最大的部分。需要修改 YOLOv11 的 Detect 头，或者在进入 Detect 之前增加 ASFF 层来融合 P3, P4, P5 的特征。
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, Autopad


class ASFF(nn.Module):
    """
    Adaptively Spatial Feature Fusion (ASFF) for YOLO
    Args:
        c1 (list): List of input channels from P3, P4, P5 [256, 512, 1024]
        c2 (int): Output channels for this specific level
        level (int): 0 for P3, 1 for P4, 2 for P5
        channel_list (list): Same as c1, explicit list of channels [256, 512, 1024]
    """

    def __init__(self, c1, c2, level=0, channel_list=[256, 512, 1024]):
        super().__init__()
        self.level = level
        # 如果 tasks.py 传入的 c1 是列表，则直接使用；否则使用 yaml 里的 channel_list
        self.dim = c1 if isinstance(c1, list) else channel_list
        self.inter_dim = self.dim[self.level]  # 当前层级的通道数

        # 压缩通道用于计算权重 (通常压缩到 16 或根据比例)
        compress_c = 8 if self.inter_dim < 48 else 16

        # 权重计算层
        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, 1, 1, 0)

        # 特征对齐层 (Stride / Resize)
        self.stride_level_0 = nn.ModuleList()  # P3
        self.stride_level_1 = nn.ModuleList()  # P4
        self.stride_level_2 = nn.ModuleList()  # P5

        # === 构建 P3, P4, P5 到当前 Level 的转换路径 ===
        # 这里的逻辑是：将所有其他层级调整到当前 self.level 的分辨率和通道数

        # Input: Level 0 (P3) -> Target: Self.Level
        if self.level == 0:
            self.stride_level_0.add_module('conv', nn.Identity())
        elif self.level == 1:  # P3 -> P4 (Downsample)
            self.stride_level_0.add_module('conv', Conv(self.dim[0], self.inter_dim, 3, 2))
        elif self.level == 2:  # P3 -> P5 (Downsample x4)
            self.stride_level_0.add_module('conv', Conv(self.dim[0], self.inter_dim, 3, 2))
            self.stride_level_0.add_module('pool', nn.MaxPool2d(3, 2, padding=1))

        # Input: Level 1 (P4) -> Target: Self.Level
        if self.level == 0:  # P4 -> P3 (Upsample)
            self.stride_level_1.add_module('conv', Conv(self.dim[1], self.inter_dim, 1, 1))
            self.stride_level_1.add_module('up', nn.Upsample(scale_factor=2, mode='nearest'))
        elif self.level == 1:
            self.stride_level_1.add_module('conv', nn.Identity())
        elif self.level == 2:  # P4 -> P5 (Downsample)
            self.stride_level_1.add_module('conv', Conv(self.dim[1], self.inter_dim, 3, 2))

        # Input: Level 2 (P5) -> Target: Self.Level
        if self.level == 0:  # P5 -> P3 (Upsample x4)
            self.stride_level_2.add_module('conv', Conv(self.dim[2], self.inter_dim, 1, 1))
            self.stride_level_2.add_module('up', nn.Upsample(scale_factor=4, mode='nearest'))
        elif self.level == 1:  # P5 -> P4 (Upsample)
            self.stride_level_2.add_module('conv', Conv(self.dim[2], self.inter_dim, 1, 1))
            self.stride_level_2.add_module('up', nn.Upsample(scale_factor=2, mode='nearest'))
        elif self.level == 2:
            self.stride_level_2.add_module('conv', nn.Identity())

    def forward(self, x):
        # x 是一个包含 [P3, P4, P5] 特征图的列表
        level_0_resized = x[0]
        level_1_resized = x[1]
        level_2_resized = x[2]

        # 1. 将所有特征图调整到当前层级的分辨率和通道数
        # Process Level 0
        for layer in self.stride_level_0:
            level_0_resized = layer(level_0_resized)

        # Process Level 1
        for layer in self.stride_level_1:
            level_1_resized = layer(level_1_resized)

        # Process Level 2
        for layer in self.stride_level_2:
            level_2_resized = layer(level_2_resized)

        # 2. 计算权重 (Weights)
        # 现在的 level_0/1/2_resized 形状都是一样的 [B, C, H, W]
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        # 拼接权重特征并在通道维度上计算 Softmax
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)  # [B, 3, H, W]

        # 3. 自适应融合 (Weighted Sum)
        # Split weights for each level
        fused_out = level_0_resized * levels_weight[:, 0:1, :, :] + \
                    level_1_resized * levels_weight[:, 1:2, :, :] + \
                    level_2_resized * levels_weight[:, 2:3, :, :]

        return fused_out


```

### 3. 如何集成（关键步骤）

#### 第一步：放入代码
将上面的代码复制粘贴到
`ultralytics / nn / modules / block.py`
文件的末尾。

#### 第二步：注册模块
打开
`ultralytics / nn / modules / __init__.py`，在
`
from .block import (

...)` 这一行里，把
`ASFF`
加进去。
```python
# ultralytics/nn/modules/__init__.py
from .block import (

..., ASFF)  # <--- 添加 ASFF
```

#### 第三步：确保 tasks.py 导入
检查你之前修改的
`ultralytics / nn / tasks.py`，确保顶部导入了
`ASFF`。

### 4. 对应你的 YAML 分析
你的
YAML
写法：
```yaml
- [[15, 18, 21], 1, ASFF, [0, [256, 512, 1024]]]