# import torch
# from torchvision import transforms
# from PIL import Image
# import model

# def predict(image_path, model_path):
#     # 加载模型，进行预测
#     pass


import torch
from torchvision import transforms
from PIL import Image
import model

def predict(image_path, model_path):
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    net = model.Net().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()  # 设置为评估模式
    
    # 图像转换：根据训练时的设置调整
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 假设使用224x224像素
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet的均值和标准差
    ])
    
    # 加载图像并应用转换
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)  # 增加一个批次维度并转移到设备上
    
    # 进行预测
    with torch.no_grad():  # 不计算梯度
        output = net(image)
        _, predicted = torch.max(output, 1)  # 获取最大概率的类别
    
    # 返回类别索引
    return predicted.item()