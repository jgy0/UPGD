import os
import random
import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_ssim(img1, img2):
    """
    计算两幅图像的 SSIM（结构相似性）
    - 输入：img1, img2（numpy.ndarray, 范围 0-255, uint8）
    - 返回：SSIM 值（越接近 1 越相似）
    """
    return ssim(img1, img2, data_range=255, channel_axis=-1 if img1.ndim == 3 else None)

def calculate_psnr(img1, img2):
    """
    计算两幅图像的 PSNR（峰值信噪比）
    - 输入：img1, img2（numpy.ndarray, 范围 0-255, uint8）
    - 返回：PSNR 值（越高表示质量越好，通常 20-40 dB）
    """
    return psnr(img1, img2, data_range=255)



def batch_ssim(batch1, batch2, data_range=1.0):
    """
    计算 Batch SSIM（适用于 PyTorch Tensor）
    - 输入：batch1, batch2（形状 [B, C, H, W] 或 [B, H, W, C]）
    - data_range: 1.0（输入范围 [0, 1]）或 255（输入范围 [0, 255]）
    """
    ssim_scores = []
    for img1, img2 in zip(batch1, batch2):
        # 转为 NumPy 数组 + CPU（如果 Tensor 在 GPU 上）
        img1_np = img1.detach().cpu().numpy().transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
        img2_np = img2.detach().cpu().numpy().transpose(1, 2, 0)
        # 计算 SSIM
        ssim_val = ssim(img1_np, img2_np, data_range=data_range, channel_axis=-1)
        ssim_scores.append(ssim_val)
    return np.mean(ssim_scores)

def batch_psnr(batch1, batch2, data_range=1.0):
    """
    计算 Batch PSNR（适用于 PyTorch Tensor）
    - 输入：batch1, batch2（形状 [B, C, H, W] 或 [B, H, W, C]）
    - data_range: 1.0（输入范围 [0, 1]）或 255（输入范围 [0, 255]）
    """
    psnr_scores = []
    for img1, img2 in zip(batch1, batch2):
        # 转为 NumPy 数组 + CPU（如果 Tensor 在 GPU 上）
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()
        # 计算 PSNR
        psnr_val = psnr(img1_np, img2_np, data_range=data_range)
        psnr_scores.append(psnr_val)
    return np.mean(psnr_scores)

def get_patch(img_in, img_tar, img_aop, img_dop, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_in.size

    patch_mult = scale
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    img_aop_patch = img_aop.crop((iy, ix, iy + ip, ix + ip))  # 与 input 相同区域
    img_dop_patch = img_dop.crop((iy, ix, iy + ip, ix + ip))  # 与 input 相同区域

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_aop_patch, img_dop_patch, info_patch

def augment(img_in, img_tar, aop, dop, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        aop = ImageOps.flip(aop)
        dop = ImageOps.flip(dop)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            aop = ImageOps.mirror(aop)
            dop = ImageOps.mirror(dop)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            aop = img_in.rotate(180)
            dop = img_in.rotate(180)
            info_aug['trans'] = True

    return img_in, img_tar, aop, dop, info_aug