"""
PyTorch 分类模型 — 轻量 CNN，适配 MFCC/Mel 频谱输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import config


class CryClassifier(nn.Module):
    """4 层卷积 + 全局平均池化 + 全连接"""

    def __init__(self, input_channels: int = 1, input_height: int = 40):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: (1, H, W) -> (32, H/2, W/2)
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: (32, H/2, W/2) -> (64, H/4, W/4)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: (64, H/4, W/4) -> (128, H/8, W/8)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4: (128, H/8, W/8) -> (256, H/16, W/16)
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 固定输出 (256, 4, 4)
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, config.num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def save_model(model: nn.Module, path: str) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: str) -> nn.Module:
    model = CryClassifier()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
