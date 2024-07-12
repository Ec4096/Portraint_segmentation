import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import model
import dataset
import torch.nn.functional as F


# def train():
#     # 初始化模型、数据集、优化器等
#     # 进行训练循环，更新模型参数
#     pass

# if __name__ == "__main__":
#     train()


def train(num_epochs=10, batch_size=4, learning_rate=0.001):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    net = model.UNet(n_class=1).to(device)

    # 准备数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256))
    ])
    train_dataset = dataset.SegmentationDataset(image_dir="picture/source360", mask_dir="picture/mask360", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = net(images)
            # 调整 outputs 的尺寸以匹配 masks 的尺寸
            outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(f"\rEpoch {epoch+1}/{num_epochs}, Batch {i+1}/{total_batches}, Loss: {running_loss/(i+1)}", end='')
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    
        # 保存模型
    torch.save(net.state_dict(), "model.pth")
    print("Model saved to model.pth")
    print("Finished Training")

if __name__ == "__main__":
    train()