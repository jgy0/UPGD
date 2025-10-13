import math
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
#from skimage.measure.simple_metrics import compare_psnr
from torchvision import models
import cv2
from skimage import filters
from skimage.color import rgb2hsv
from skimage.measure import shannon_entropy
from skimage import feature
from torchvision.models import vgg16
from pytorch_msssim import ssim

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        # 加载预训练的VGG16模型
        vgg = vgg16(weights='DEFAULT')
        # 提取VGG16的前31层（即卷积层和池化层，不包含全连接层）
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # 冻结VGG16的参数，使其在训练过程中不更新
        for param in loss_network.parameters():
            param.requires_grad = False
        # 将VGG16的前31层作为损失网络
        self.loss_network = loss_network
        # 定义均方误差损失函数
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        # 计算生成图像和目标图像在VGG16特征空间中的感知损失
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    #out = image
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))
    #return out





def uciqe(img):
    """
    Calculate UCIQE (Universal Image Quality Index) for the image.
    """
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Brightness
    brightness = np.mean(gray_img)

    # Contrast (Variance)
    contrast = np.var(gray_img)

    # Edge Information (Gradient Magnitude)
    edges = filters.sobel(gray_img)
    edge_info = np.sum(edges)

    # Saturation
    hsv_img = rgb2hsv(img)
    saturation = np.mean(hsv_img[:, :, 1])

    # UCIQE Calculation
    uciqe_value = (0.5 * brightness + 0.3 * contrast + 0.1 * edge_info + 0.1 * saturation)
    return uciqe_value


def uiqm(img):
    """
    Calculate UIQM (Underwater Image Quality Measure) for the image.
    """
    # Convert to HSV for saturation
    hsv_img = rgb2hsv(img)
    saturation = np.mean(hsv_img[:, :, 1])

    # Edge sharpness using Canny Edge detector
    edges = feature.canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    edge_sharpness = np.sum(edges)

    # Image clarity using entropy
    clarity = shannon_entropy(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))

    # Noise (Standard deviation of pixel values)
    noise = np.std(img)

    # UIQM Calculation (simplified version)
    uiqm_value = (0.4 * saturation + 0.3 * edge_sharpness + 0.2 * clarity + 0.1 * noise)
    return uiqm_value


class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, pred, target):
        # 计算 SSIM，值越大表示越相似
        ssim_value = ssim(pred, target, data_range=1.0, size_average=True)
        # 返回 1 - SSIM，作为损失函数（越小越好）
        return 1 - ssim_value