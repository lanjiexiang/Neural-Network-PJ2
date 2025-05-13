import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# 定义数据集
class CIFAR10(Dataset):
    def __init__(self, data_dir, batches=[1,2,3,4,5], transform=None):
        data_list = []
        label_list = []
        for b in batches:
            path = os.path.join(data_dir, f'data_batch_{b}')
            with open(path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = batch[b'labels']
            data_list.append(data)
            label_list += labels

        self.data = np.vstack(data_list).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.labels = np.array(label_list, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 自定义 residual connection block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # output = F(x) + x, 跳跃连接直接加上输入本身
        return F.relu(out)

# 训练网络结构定义
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # ResidualBlock 1
        self.resblock1 = ResidualBlock(32)
        # 卷积层2
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 卷积层3
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # 卷积层4
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # dropout层
        self.dropout = nn.Dropout(0.5)
        # 全连接层
        self.fc = nn.Linear(256 * 2 * 2, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.resblock1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# 标签平滑交叉熵
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss   = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss= -log_probs.mean(dim=-1)
        return (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

def main():
    # 设备与路径
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    data_dir = '/mnt/disk2/jiexiang.lan/Blue/PJ2_codes-DL&NN/VGG_BatchNorm/data/cifar-10-batches-py'
    model_dir= '/mnt/disk2/jiexiang.lan/Blue/PJ2_codes-DL&NN/Basic network/models'
    os.makedirs(model_dir, exist_ok=True)

    # 数据准备
    full_ds    = CIFAR10(data_dir)
    total      = len(full_ds)
    val_size   = int(0.1 * total)
    train_size = total - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=4)

    # 模型、损失、优化器
    model     = CIFAR10Net().to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # 训练超参
    num_epochs      = 10
    patience        = 3
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    best_val_loss   = float('inf')
    wait            = 0
    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # —— 训练 —— 
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss = loss.mean() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        train_loss = running_loss / train_size

        # —— 验证 —— 
        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss = loss.mean()  
                running_val += loss.item() * imgs.size(0)
        val_loss = running_val / val_size

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f'Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 提前停止 & 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'  Best model updated (val_loss: {best_val_loss:.4f})')
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered')
                break

    # 绘制并保存损失曲线
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1),   val_losses,   label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'train_loss_curve.png'))

if __name__ == '__main__':
    main()
