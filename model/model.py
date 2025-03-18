import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from base import BaseModel


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ECABlock(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=4, stride=1):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_residual = in_channels == out_channels and stride == 1

        layers = []
        if expansion != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Mish()
            ])

        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Mish(),
            ECABlock(hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)
        self.drop_path = DropPath(0.2) if self.use_residual else nn.Identity()

    def forward(self, x):
        if self.use_residual:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)


class EfficientTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, reduction_ratio=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction_ratio, 1),
            Mish(),
            nn.Conv2d(embed_dim // reduction_ratio, embed_dim, 1)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.local_window = 7  # 修改为可配置参数

    def _window_partition(self, x):
        """
        输入: [B, C, H, W]
        输出: [B*num_windows, window_size², C]
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.local_window, self.local_window, W // self.local_window, self.local_window)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, self.local_window, self.local_window)
        windows = windows.view(-1, C, self.local_window * self.local_window).permute(0, 2, 1)
        return windows

    def _window_reverse(self, windows, H, W):
        """
        输入: [B*num_windows, window_size², C]
        输出: [B, C, H, W]
        """
        B = int(windows.shape[0] / (H * W / self.local_window / self.local_window))
        x = windows.view(B, H // self.local_window, W // self.local_window, self.local_window, self.local_window, -1)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        # 自动调整窗口尺寸
        if H < self.local_window or W < self.local_window:
            self.local_window = min(H, W)

        try:
            # Local window attention
            x_windows = self._window_partition(x)  # [B*num_windows, window_size², C]
            attn_windows, _ = self.attn(x_windows, x_windows, x_windows)
            attn_out = self._window_reverse(attn_windows, H, W)
        except:
            # Fallback机制：当尺寸不匹配时使用全局注意力
            x_flat = x.view(B, C, H * W).permute(0, 2, 1)
            attn_out, _ = self.attn(x_flat, x_flat, x_flat)
            attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # 残差连接
        x = x + attn_out
        x = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 卷积前馈网络
        conv_out = self.conv(x)
        x = x + conv_out
        x = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class CNN_Stork(BaseModel):
    def __init__(self, num_classes=40):
        super().__init__()
        # 输入形状: [B, 3, 300, 300]
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),  # [B,32,150,150]
            ECABlock(32),
            nn.MaxPool2d(2)  # [B,32,75,75]
        )

        self.mobile_block = nn.Sequential(
            MBConvBlock(32, 64, expansion=2),
            MBConvBlock(64, 128, expansion=4),
            nn.MaxPool2d(2)  # [B,128,37,37]
        )

        self.res_blocks = nn.Sequential(
            MBConvBlock(128, 256, expansion=4),
            ECABlock(256),
            nn.MaxPool2d(2)  # [B,256,18,18]
        )

        self.transformer = nn.Sequential(
            EfficientTransformerBlock(256, num_heads=8),
            nn.AdaptiveAvgPool2d(1)  # [B,256,1,1]
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(0.5),
            Mish(),
            nn.Linear(512, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)  # [B,32,75,75]
        x = self.mobile_block(x)  # [B,128,37,37]
        x = self.res_blocks(x)  # [B,256,18,18]
        x = self.transformer(x)  # [B,256,1,1]
        x = x.flatten(1)  # [B,256]
        return self.classifier(x)

