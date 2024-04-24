import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# 이미지 텐서
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 학습 , 연습 데이터 
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 데이터 64 배치 사이즈
train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 숫자 레이블 변환 
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 이미지 레이블
image, label = train_dataset[0]

# 시각화
plt.figure(figsize=(4, 4))
plt.imshow(image.squeeze(), cmap='gray') # 1차원 제거
plt.title(class_labels[label])
plt.axis('off')    # x, y축 제거
plt.show()

