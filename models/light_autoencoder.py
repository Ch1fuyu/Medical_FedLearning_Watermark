from torch import nn

class LightAutoencoder(nn.Module):
    def __init__(self, input_channels=3):
        super(LightAutoencoder, self).__init__()
        self.input_channels = input_channels
        
        # 编码器：支持可变输入通道数
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, stride=2, padding=1),  # 224x224 → 112x112
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # 112x112 → 56x56
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=4)  # 56x56 → 53x53
        )
        
        # 解码器：输出与输入相同的通道数
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4),  # 53x53 → 56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # 56x56 → 112x112
            nn.ReLU(True),
            nn.ConvTranspose2d(8, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1)  # 112x112 → 224x224
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded