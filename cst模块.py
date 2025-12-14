class CST(nn.Module):
    """
    Contextual Spatial Transformer (CST) Module.
    使用大核深度卷积 (Depthwise Conv) 来捕捉长距离的空间依赖关系，
    模拟 Transformer 的空间注意力机制，但计算量更低。
    """

    def __init__(self, c1, c2, k=7, s=1):
        """
        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            k (int): 空间感知卷积核大小 (建议 7, 11 等大核)
            s (int): 步长
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # 1x1 卷积调整通道
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False)

        # 大核深度卷积 (Depthwise Convolution)
        # padding = k // 2 保证尺寸不变
        self.spatial_conv = nn.Conv2d(
            c2, c2, kernel_size=k, stride=s, padding=k // 2, groups=c2, bias=False
        )

        self.act = nn.SiLU()

        # 1x1 卷积融合特征
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.spatial_conv(x)
        x = self.act(x)
        x = self.conv2(x)

        # 如果输入输出通道一致且尺寸未变，使用残差连接
        if self.c1 == self.c2 and x.shape[2:] == identity.shape[2:]:
            return x + identity
        return x
