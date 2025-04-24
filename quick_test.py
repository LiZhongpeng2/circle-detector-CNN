import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from models.network import CircleCenterNet  # 使用你已有的模型

# 数据集类 (与你的train_test.py中相同)
class CircleDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        with open(label_file, 'r') as f:
            self.labels = json.load(f)
        self.transform = transform
        self.img_list = [f for f in os.listdir(img_dir) if f in self.labels]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # 直接读取预处理好的灰度图
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # 不需要再调整大小，因为已经是256×256了
        
        # 获取标签并生成热力图
        center = self.labels[img_name]
        # 由于图像已经是256×256，不需要再计算缩放因子
        # x_scale 和 y_scale 可以省略
        scaled_x = int(center[0])  # 直接使用原坐标
        scaled_y = int(center[1])  # 直接使用原坐标
        
        heatmap = np.zeros((256, 256), dtype=np.float32)
        cv2.circle(heatmap, (scaled_x, scaled_y), 5, 1, -1)
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 3)
        
        # 转换为tensor
        if self.transform:
            image = self.transform(image)
        heatmap = torch.from_numpy(heatmap)
        
        return image, heatmap

def quick_test():
    # 1. 快速检查一张图片和标签
    dataset = CircleDataset(
        img_dir='data/train',
        label_file='data/result_data/center_circle_centers.json',
        transform=transforms.Compose([
            transforms.ToTensor(),  # 将图像从[0,255]转换为[0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1,1]
        ])
    )
    
    print(f"\n数据集大小: {len(dataset)} 张图片")
    
    # 只检查第一张图片
    image, heatmap = dataset[0]
    print(f"图片尺寸: {image.shape}")
    print(f"热力图尺寸: {heatmap.shape}")
    
    # 2. 快速测试GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory/1024/1024:.0f}MB")
    
    # 3. 快速训练测试（只训练2个batch）
    model = CircleCenterNet().to(device)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\n开始简单训练测试...")
    model.train()
    
    # 只测试前两个batch
    for i, (images, targets) in enumerate(loader):
        if i >= 2:  # 只测试2个batch
            break
            
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
        
        print(f"Batch {i+1} Loss: {loss.item():.6f}")
    
    print("\n测试完成！")
    print("如果你看到这条消息，且上面没有错误信息，说明基本流程正常。")

if __name__ == "__main__":
    quick_test()
