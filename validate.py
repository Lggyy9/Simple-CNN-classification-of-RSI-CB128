import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from rsi_cb128_classification_pytorch import CustomCNN
from tqdm import tqdm


# 加载已训练好的模型
def load_model(model_path):
    model = CustomCNN(num_classes=45)
    model.load_state_dict(torch.load(model_path))
    return model


# 验证模型
def validate_model(model, dataloader):
    model.eval()  # 设置模型为评估模式

    correct = 0
    total = 0

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)

            # 统计正确预测的数量
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算准确率
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')


# 加载数据集
data_dir = 'your_validation_data_path'  # 验证集路径，请自行设置
batch_size = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
dataset = ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 打印验证集中图片的数量和验证所使用的设备
print(f'Total images loaded: {len(dataset)}')
print(f'The validating is running on device: {device}')

# 加载模型
model_path = 'trained_model.pth'
model = load_model(model_path)
model.to(device)

# 验证模型
validate_model(model, dataloader)
