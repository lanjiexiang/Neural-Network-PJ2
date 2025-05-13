import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns  # for nicer heatmap

# CIFAR-10 类别标签
CIFAR10_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# 定义测试数据集
class CIFAR10Test(Dataset):
    def __init__(self, batch_path, transform=None):
        with open(batch_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            data = batch[b'data']
            labels = batch[b'labels']
        self.data = data.reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

#同训练网络结构定义
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        identity = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)
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


def main():
    # 加载数据和训练后的模型
    model_path = '/mnt/disk2/jiexiang.lan/Blue/PJ2_codes-DL&NN/Basic network/models/best_model.pth'
    test_batch = '/mnt/disk2/jiexiang.lan/Blue/PJ2_codes-DL&NN/VGG_BatchNorm/data/cifar-10-batches-py/test_batch'
    save_cm_path = '/mnt/disk2/jiexiang.lan/Blue/PJ2_codes-DL&NN/Basic network/models/test_confusion_matrix.png'
    test_ds = CIFAR10Test(test_batch)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = CIFAR10Net().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 收集预测结果
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # 计算准确率
    acc = accuracy_score(all_labels, all_preds)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CIFAR10_LABELS, yticklabels=CIFAR10_LABELS)
    plt.title(f'CIFAR-10 Confusion Matrix (Accuracy: {acc:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_cm_path)
    plt.show()
if __name__ == '__main__':
    main()