class CSE(nn.Module):
    """
    Contextual Squeeze-and-Excitation (CSE) Module.
    通过全局平均池化捕捉通道上下文信息，增强有效特征，抑制无效特征。
    """

    def __init__(self, c1, r=16):
        """
        Args:
            c1 (int): 输入通道数
            r (int): 缩减比率 (Reduction ratio)
        """
        super().__init__()
        # 确保缩减后的通道数至少为 1
        mid_channels = max(1, c1 // r)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, c1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        # attn: [B, C, 1, 1]
        attn = self.fc(self.avg_pool(x))
        return x * attn