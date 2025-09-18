import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import time

from models.light_autoencoder import LightAutoencoder

# 设置中文字体（跨平台）
plt.rcParams["font.family"] = ["SimHei"]

# 参数定义
batch_size = 128
learning_rate = 0.005
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重保存路径
weights_dir = './save/autoencoder'
os.makedirs(weights_dir, exist_ok=True)
final_model_path = os.path.join(weights_dir, 'autoencoder.pth')

# 加载数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                         batch_size=batch_size)


# 工具函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """保存模型参数"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存至：{path}")
    
    # 分别保存编码器和解码器参数
    encoder_path = os.path.join(weights_dir, 'encoder.pth')
    decoder_path = os.path.join(weights_dir, 'decoder.pth')
    torch.save(model.encoder.state_dict(), encoder_path)
    torch.save(model.decoder.state_dict(), decoder_path)
    print(f"编码器已保存至：{encoder_path}")
    print(f"解码器已保存至：{decoder_path}")


def load_model(model, path):
    """加载模型参数"""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint)
    print(f"已加载模型：{path}")
    return model


def train(model, loader, criterion, optimizer, epochs, device):
    """训练自编码器"""
    model.train()
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        for data, _ in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1:02d}/{epochs}, Loss: {avg_loss:.6f}, Time: {time.time() - start_time:.2f}s")
    return model


def test_and_visualize(model, loader, device, n=8):
    """测试并可视化重建结果"""
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(loader))
        data = data.to(device)
        output = model(data)
        
        # 创建保存目录
        os.makedirs('./save/image', exist_ok=True)
        
        plt.figure(figsize=(16, 4))
        for i in range(n):
            plt.subplot(2, n, i + 1)
            plt.imshow(data[i].cpu().squeeze(), cmap='gray')
            plt.title("原始图像")
            plt.axis('off')

            plt.subplot(2, n, i + 1 + n)
            plt.imshow(output[i].cpu().squeeze(), cmap='gray')
            plt.title("重建图像")
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('./save/image/reconstruction_results.png')
        plt.show()


def main():
    model = LightAutoencoder().to(device)
    print(f"参数总量: {count_parameters(model):,}")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists(final_model_path):
        model = load_model(model, final_model_path)
    else:
        model = train(model, train_loader, criterion, optimizer, num_epochs, device)
        save_model(model, final_model_path)

    test_and_visualize(model, test_loader, device)


if __name__ == '__main__':
    main()