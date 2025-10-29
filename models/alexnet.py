import torch.nn as nn
import torch
from models.layers.conv2d import ConvBlock

class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes, input_size=28):
        super().__init__()
        max_pool_idx = [1, 3, 7]
        layers = []
        inp = in_channels
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }
        for layer_idx in range(8):
            if layer_idx in max_pool_idx:
                layers.append(nn.MaxPool2d(2, 2))
            else:
                k = kp[layer_idx][0]
                p = kp[layer_idx][1]
                layers.append(ConvBlock(inp, oups[layer_idx], k, 1, p))
                inp = oups[layer_idx]

        self.features = nn.Sequential(*layers)
        # 动态推断flatten后的特征数
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            feat = self.features(dummy)
            flatten_dim = feat.view(1, -1).shape[1]
        self.classifier = nn.Linear(flatten_dim, num_classes)

    def forward(self, x):
        for m in self.features:
            x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 为了与resnet的导入方式兼容
def alexnet(num_classes, in_channels, input_size):
    """创建AlexNet模型的工厂函数"""
    return AlexNet(in_channels=in_channels, num_classes=num_classes, input_size=input_size)