import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.functional import interpolate


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=None, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()

        vgg16 = models.vgg16(pretrained=True).features
        self.device = device

        # 选择用来计算感知损失的层
        if layers is None:
            layers = ['0', '5', '10', '19']  # 选择conv1_2, conv2_2, conv3_3, conv4_3等

        self.selected_layers = nn.ModuleList()
        for i, layer in enumerate(vgg16):
            self.selected_layers.append(layer)
            if str(i) in layers:
                break

        for param in self.parameters():
            param.requires_grad = False  # 冻结VGG的参数

    def forward(self, x, y):
        """计算感知损失"""
        # 对输入的图像进行VGG的标准化
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x = normalize(x)
        y = normalize(y)

        x_features = self.extract_features(x)
        y_features = self.extract_features(y)

        # 计算VGG感知损失（特征L2距离）
        loss = 0
        for xf, yf in zip(x_features, y_features):
            loss += torch.nn.functional.mse_loss(xf, yf)

        return loss

    def extract_features(self, x):
        """提取VGG网络的中间层特征"""
        features = []
        for layer in self.selected_layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features
