from random import random
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import torchvision.utils as utils
import pytorch_ssim
from torch.optim import lr_scheduler
import torch.utils.data as dataf
from torch.autograd import Variable
from net.Ushape_Trans import *
from net.utils import *
import cv2
from loss.LAB import *
import re
import torchvision.transforms as transforms
from PIL import Image
from pytorch_msssim import ms_ssim
from loss.LCH import *
import time as time
import datetime
import sys
from torchvision.utils import save_image
import csv
import random
from net.utils import *
from torch.utils.data import Dataset, DataLoader
import torchvision
from utility.BM3D import *
from thop import profile
from utils import *
from piq import vif_p
import torch.nn.functional as F
import numpy as np

torch.cuda.empty_cache()
dtype = 'float32'
torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择卡1

print(f"Using device: {torch.cuda.current_device()}")
print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

checkpoint_interval = 1  # 多少epoch保存一次模型
epochs = 0
n_epochs = 25  # 这两个要一起修改
test_epoch = 24  # 这两个要一起修改
sample_interval = 338  # batch size=6时设为57，batch size=1时设为338
batch_si = 1


def split(img):
    output = []
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output


def _save_heatmap(tensor_2d, save_path):
    """Save a single 2D attention map as a heatmap image. Falls back to tensor save if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        attn = tensor_2d.detach().cpu().float().numpy()
        plt.figure(figsize=(8, 8))
        plt.imshow(attn, cmap='viridis')
        plt.axis('off')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()
    except Exception:
        # fallback: normalize and save as grayscale image via torchvision
        t = tensor_2d.detach().cpu()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        t = t.unsqueeze(0).unsqueeze(0)  # 1x1xH xW
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(t, save_path)


def _reshape_attention_to_spatial(attn_tensor, target_size=256):
    """
    Reshape attention tensor from token space back to spatial image space.

    Args:
        attn_tensor: attention tensor of shape (B, H, N, M) or (B, N, M)
        target_size: target spatial size (default 256 for 256x256)

    Returns:
        spatial_attn: attention map in spatial coordinates (H, W)
    """
    if attn_tensor.dim() == 4:
        # (B, H, N, M) -> average over heads and batch
        attn = attn_tensor.mean(0).mean(0)  # (N, M)
    else:
        # (B, N, M) -> average over batch
        attn = attn_tensor.mean(0)  # (N, M)

    # For agent attention: reshape to spatial
    if attn.shape[0] == 49:  # agent_num = 49 = 7x7
        # Reshape agent dimension to 7x7
        agent_attn = attn.view(7, 7, -1)  # (7, 7, M)
        # Average over agents to get spatial attention
        spatial_attn = agent_attn.mean(0).mean(0)  # (M,)
    elif attn.shape[1] == 49:  # (N, 49)
        # Reshape agent dimension to 7x7
        agent_attn = attn.view(-1, 7, 7)  # (N, 7, 7)
        # Average over agents to get spatial attention
        spatial_attn = agent_attn.mean(1).mean(1)  # (N,)
    else:
        # For other cases, just take the mean
        spatial_attn = attn.mean(0) if attn.dim() == 2 else attn

    # Reshape to square if possible
    seq_len = spatial_attn.shape[0]
    if seq_len == 256:  # 16x16 tokens
        spatial_attn = spatial_attn.view(16, 16)
    elif seq_len == 64:  # 8x8 tokens
        spatial_attn = spatial_attn.view(8, 8)
    elif seq_len == 16:  # 4x4 tokens
        spatial_attn = spatial_attn.view(4, 4)
    else:
        # For other sizes, try to make it square
        side_len = int(seq_len ** 0.5)
        if side_len * side_len == seq_len:
            spatial_attn = spatial_attn.view(side_len, side_len)
        else:
            # Pad to make it square
            side_len = int(seq_len ** 0.5) + 1
            padded = torch.zeros(side_len * side_len, device=spatial_attn.device)
            padded[:seq_len] = spatial_attn
            spatial_attn = padded.view(side_len, side_len)

    # Upsample to target size
    spatial_attn = spatial_attn.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    spatial_attn = F.interpolate(spatial_attn, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return spatial_attn.squeeze()  # (H, W)


def save_attention_maps_for_batch(generator, real_A, real_P, real_B, batch_idx, epoch, save_dir="attention_maps"):
    """
    为每个batch的每张图片保存注意力图
    """
    generator.eval()
    with torch.no_grad():
        out = generator(real_A, real_P, device, return_vis=True)
        if isinstance(out, tuple) and len(out) == 3:
            fake_B, _, vis_dict = out
        else:
            fake_B, _ = out
            vis_dict = None

    if vis_dict is None:
        return

    # 为batch中的每张图片保存注意力图
    batch_size = real_A.size(0)
    for img_idx in range(batch_size):
        # 创建每张图片的保存目录
        img_save_dir = os.path.join(save_dir, f"epoch_{epoch:03d}", f"batch_{batch_idx:04d}", f"img_{img_idx:02d}")
        os.makedirs(img_save_dir, exist_ok=True)

        # 保存原始输入图像作为参考
        torchvision.utils.save_image(real_A[img_idx], os.path.join(img_save_dir, "input_rgb.png"), normalize=True)
        torchvision.utils.save_image(real_P[img_idx, :3], os.path.join(img_save_dir, "input_aop.png"), normalize=True)
        torchvision.utils.save_image(real_P[img_idx, 3:], os.path.join(img_save_dir, "input_dop.png"), normalize=True)
        if real_B is not None:
            torchvision.utils.save_image(real_B[img_idx], os.path.join(img_save_dir, "gt.png"), normalize=True)
            torchvision.utils.save_image(fake_B[3][img_idx], os.path.join(img_save_dir, "output.png"), normalize=True)

        # 保存注意力图
        for key in ["aop_agent_attn", "aop_q_attn", "dop_agent_attn", "dop_q_attn"]:
            att = vis_dict.get(key, None)
            if att is None:
                continue
            try:
                # 取当前图片的注意力图
                if att.dim() == 4:  # (B, H, N, M)
                    att_img = att[img_idx]  # (H, N, M)
                else:  # (B, N, M)
                    att_img = att[img_idx:img_idx + 1]  # (1, N, M)

                # 重塑为空间坐标
                spatial_attn = _reshape_attention_to_spatial(att_img)
                _save_heatmap(spatial_attn, os.path.join(img_save_dir, f"{key}_spatial.png"))
            except Exception as e:
                print(f"Failed to save {key} for img {img_idx}: {e}")

        # 保存通道注意力图
        if "channel_attn_weights" in vis_dict and vis_dict["channel_attn_weights"]:
            for layer_idx, weights_per_scale in enumerate(vis_dict["channel_attn_weights"]):
                if weights_per_scale is None:
                    continue
                for scale_idx, w in enumerate(weights_per_scale):
                    if w is None:
                        continue
                    try:
                        # 取当前图片的注意力图
                        if w.dim() == 3:  # (B, N, M)
                            w_img = w[img_idx:img_idx + 1]  # (1, N, M)
                        else:  # (N, M)
                            w_img = w.unsqueeze(0)  # (1, N, M)

                        spatial_w = _reshape_attention_to_spatial(w_img)
                        _save_heatmap(spatial_w,
                                      os.path.join(img_save_dir, f"channel_attn_L{layer_idx}_S{scale_idx}.png"))
                    except Exception as e:
                        print(f"Failed to save channel attention L{layer_idx}_S{scale_idx} for img {img_idx}: {e}")

        # 保存融合权重
        try:
            fusion_weights = []
            for layer_idx, layer in enumerate(generator.transformer.layers):
                if hasattr(layer, "prompt") and hasattr(layer.prompt, "weight1") and hasattr(layer.prompt, "weight2"):
                    alpha = layer.prompt.weight1.detach().cpu().item()
                    beta = layer.prompt.weight2.detach().cpu().item()
                    fusion_weights.append(f"Layer_{layer_idx}: alpha(AOP)={alpha:.4f}, beta(DOP)={beta:.4f}")

            if fusion_weights:
                with open(os.path.join(img_save_dir, "fusion_weights.txt"), "w") as f:
                    f.write("PPM Fusion Weights (AOP vs DOP):\n")
                    f.write("=" * 40 + "\n")
                    for weight_info in fusion_weights:
                        f.write(weight_info + "\n")
        except Exception as e:
            print(f"Failed to save fusion weights for img {img_idx}: {e}")


def sample_images(generator, test_loader, device, batches_done, save_dir="images/results", nrow=3):
    """
    从测试集中随机生成样本并保存
    :param generator: 生成器模型
    :param test_loader: 测试集的DataLoader
    :param device: 设备（CPU或GPU）
    :param batches_done: 当前批次编号
    :param save_dir: 保存图像的目录
    :param nrow: 每行显示的图像数量
    """
    generator.eval()  # 设置模型为评估模式

    # 随机选择一个批次
    random_batch_idx = random.randint(0, len(test_loader) - 1)  # 随机选择一个批次的索引
    for i, batch in enumerate(test_loader):
        if i == random_batch_idx:
            real_A, real_B, real_P = batch
            break

    real_A = real_A.to(device)
    real_B = real_B.to(device)
    real_P = real_P.to(device)

    # 生成图像，同时请求可视化字典
    with torch.no_grad():
        out = generator(real_A, real_P, device, return_vis=True)
        if isinstance(out, tuple) and len(out) == 3:
            fake_B, _, vis_dict = out
        else:
            fake_B, _ = out
            vis_dict = None

    # 使用最后一层的输出
    fake_B = fake_B[3].data  # 假设 fake_B 是一个列表，取最后一层的输出
    fake_B = torch.clamp(fake_B, 0., 1.)  # 将像素值限制在 [0, 1] 范围内

    # 将生成图像和真实图像拼接在一起
    img_sample = torch.cat([fake_B, real_B], dim=-2)  # 在高度维度上拼接

    # 保存图像
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    save_path = os.path.join(save_dir, f"{batches_done}.png")
    torchvision.utils.save_image(img_sample, save_path, nrow=nrow, normalize=True)

    # 保存可视化结果（注意力图/融合权重）
    if vis_dict is not None:
        attn_dir = os.path.join(save_dir, f"vis_{batches_done}")
        os.makedirs(attn_dir, exist_ok=True)

        # Channel transformer weights: nested list [layer][scale] -> tensor
        if "channel_attn_weights" in vis_dict and vis_dict["channel_attn_weights"]:
            for layer_idx, weights_per_scale in enumerate(vis_dict["channel_attn_weights"]):
                if weights_per_scale is None:
                    continue
                for scale_idx, w in enumerate(weights_per_scale):
                    if w is None:
                        continue
                    try:
                        # Reshape to spatial and save
                        spatial_w = _reshape_attention_to_spatial(w[0])
                        _save_heatmap(spatial_w, os.path.join(attn_dir, f"channel_attn_L{layer_idx}_S{scale_idx}.png"))
                    except Exception as e:
                        print(f"Failed to save channel attention L{layer_idx}_S{scale_idx}: {e}")
                        pass

        # Agent attention maps - reshape to 256x256 spatial maps
        for key in ["aop_agent_attn", "aop_q_attn", "dop_agent_attn", "dop_q_attn"]:
            att = vis_dict.get(key, None)
            if att is None:
                continue
            try:
                # Reshape attention to spatial coordinates
                spatial_attn = _reshape_attention_to_spatial(att)
                _save_heatmap(spatial_attn, os.path.join(attn_dir, f"{key}_spatial.png"))
            except Exception as e:
                print(f"Failed to save {key}: {e}")
                pass

        # Save fusion weights (alpha and beta) from Prompt_block modules
        try:
            fusion_weights = []
            for layer_idx, layer in enumerate(generator.transformer.layers):
                if hasattr(layer, "prompt") and hasattr(layer.prompt, "weight1") and hasattr(layer.prompt, "weight2"):
                    alpha = layer.prompt.weight1.detach().cpu().item()
                    beta = layer.prompt.weight2.detach().cpu().item()
                    fusion_weights.append(f"Layer_{layer_idx}: alpha(AOP)={alpha:.4f}, beta(DOP)={beta:.4f}")

            if fusion_weights:
                with open(os.path.join(attn_dir, "fusion_weights.txt"), "w") as f:
                    f.write("PPM Fusion Weights (AOP vs DOP):\n")
                    f.write("=" * 40 + "\n")
                    for weight_info in fusion_weights:
                        f.write(weight_info + "\n")
                print(f"Saved fusion weights to {os.path.join(attn_dir, 'fusion_weights.txt')}")
        except Exception as e:
            print(f"Failed to save fusion weights: {e}")

    print(f"Saved sample images to {save_path}")


def evaluate_test_set(generator, test_loader, device):
    """
    评估测试集，计算平均PSNR和SSIM
    :param generator: 生成器模型
    :param test_loader: 测试集的DataLoader
    :param device: 设备（CPU或GPU）
    :return: 平均PSNR和SSIM
    """
    generator.eval()  # 设置模型为评估模式
    psnr_test_list = []
    ssim_test_list = []

    with torch.no_grad():  # 评估时不计算梯度
        for batch in test_loader:
            real_A_test, real_B_test, real_P_test = batch
            real_A_test = real_A_test.to(device)
            real_B_test = real_B_test.to(device)
            real_P_test = real_P_test.to(device)

            # 生成测试图像
            fake_B_test, _ = generator(real_A_test, real_P_test, device)

            # 使用最后一层的输出
            out_test = torch.clamp(fake_B_test[3], 0., 1.)

            # 计算PSNR和SSIM
            psnr_test = batch_PSNR(out_test, real_B_test, 1.)
            ssim_test = SSIM(out_test, real_B_test)
            # psnr_test = batch_psnr(out_test, real_B_test)
            # ssim_test = batch_ssim(out_test, real_B_test)

            psnr_test_list.append(psnr_test)
            ssim_test_list.append(ssim_test.item())

    # 计算每个epoch的平均PSNR和SSIM
    # psnr_test_list=psnr_test_list[:-1]
    # ssim_test_list=ssim_test_list[:-1]
    avg_psnr_test = np.mean(psnr_test_list)
    avg_ssim_test = np.mean(ssim_test_list)

    return avg_psnr_test, avg_ssim_test


class PolarDataset(Dataset):
    def __init__(self, rgb_path, gt_path, aop_path, dop_path, resize=(256, 256), dtype='float32',
                 data_augmentation=False):
        """
        初始化数据集
        :param rgb_path: RGB图像的路径
        :param gt_path: 真实标签图像的路径
        :param aop_path: 偏振角（AOP）图像的路径
        :param dop_path: 偏振度（DOP）图像的路径
        :param resize: 图像调整大小
        :param dtype: 数据类型
        """
        self.rgb_path = rgb_path
        self.gt_path = gt_path
        self.aop_path = aop_path
        self.dop_path = dop_path
        self.resize = resize
        self.dtype = dtype
        self.gt_path_bool = gt_path
        self.augment = data_augmentation

        # 获取文件列表并排序
        self.rgb_list = sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0]))
        if self.gt_path != None:
            self.gt_list = sorted(os.listdir(gt_path), key=lambda x: int(x.split('.')[0]))
        self.aop_list = sorted(os.listdir(aop_path), key=lambda x: int(x.split('.')[0]))
        self.dop_list = sorted(os.listdir(dop_path), key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        """返回数据集的大小"""
        return len(self.rgb_list)

    def __getitem__(self, idx):
        """根据索引加载并返回一个样本"""
        # 加载RGB图像
        rgb_img = cv2.imread(os.path.join(self.rgb_path, self.rgb_list[idx]))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, self.resize)
        # 加载真实标签图像
        if self.gt_path != None:
            gt_img = cv2.imread(os.path.join(self.gt_path, self.gt_list[idx]))
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = cv2.resize(gt_img, self.resize)

        # 加载偏振角（AOP）和偏振度（DOP）图像
        aop_img = cv2.imread(os.path.join(self.aop_path, self.aop_list[idx]))
        dop_img = cv2.imread(os.path.join(self.dop_path, self.dop_list[idx]))
        aop_img = cv2.cvtColor(aop_img, cv2.COLOR_BGR2RGB)
        dop_img = cv2.cvtColor(dop_img, cv2.COLOR_BGR2RGB)
        aop_img = cv2.resize(aop_img, self.resize)
        dop_img = cv2.resize(dop_img, self.resize)

        # 数据增强
        if self.gt_path != None and self.augment == True:
            rgb_img = Image.fromarray(rgb_img)
            aop_img = Image.fromarray(aop_img)
            dop_img = Image.fromarray(dop_img)
            gt_img = Image.fromarray(gt_img)
            rgb_img, gt_img, aop_img, dop_img, _ = augment(rgb_img, gt_img, aop_img, dop_img)
            rgb_img = np.array(rgb_img)
            aop_img = np.array(aop_img)
            dop_img = np.array(dop_img)
            gt_img = np.array(gt_img)
        # 合并偏振图像
        polar_img = np.concatenate([aop_img, dop_img], axis=-1)

        # 转换为Tensor并归一化
        rgb_img = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float() / 255.0
        if self.gt_path != None:
            gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float() / 255.0
        polar_img = torch.from_numpy(polar_img.transpose(2, 0, 1)).float() / 255.0
        if self.gt_path != None:
            return rgb_img, gt_img, polar_img
        else:
            return rgb_img, polar_img


class MixedPolarDataset(Dataset):
    def __init__(self, sim_data, real_data, real_ratio=0.2):
        self.sim_dataset = sim_data  # 原模拟数据集
        self.real_dataset = real_data  # 新增真实数据集（无需GT）
        self.real_ratio = real_ratio  # 每个batch中真实数据占比

    def __len__(self):
        return max(len(self.sim_dataset), len(self.real_dataset))

    def __getitem__(self, idx):
        if random.random() < self.real_ratio:
            # 加载真实数据（无GT）
            real_rgb, real_polar = self.real_dataset[idx % len(self.real_dataset)]

            return {
                'rgb': real_rgb,

                'polar': real_polar,
                'is_real': 1  # 域标签：1表示真实数据
            }
        else:
            # 加载模拟数据
            sim_rgb, sim_gt, sim_polar = self.sim_dataset[idx % len(self.sim_dataset)]

            return {
                'rgb': sim_rgb,
                'gt': sim_gt,
                'polar': sim_polar,
                'is_real': 0  # 域标签：0表示模拟数据
            }


# 定义路径
train_rgb_path = '/home/guangyao/code/data/polar_split/train_rgb'
train_gt_path = '/home/guangyao/code/data/polar_split/train_gt'
train_aop_path = '/home/guangyao/code/data/polar_split/train_p/AOP'
train_dop_path = '/home/guangyao/code/data/polar_split/train_p/DOP'

test_rgb_path = '/home/guangyao/code/data/polar_split/test_rgb'
test_gt_path = '/home/guangyao/code/data/polar_split/test_gt'
test_aop_path = '/home/guangyao/code/data/polar_split/test_p/AOP'
test_dop_path = '/home/guangyao/code/data/polar_split/test_p/DOP'

# 原代码中的数据集定义
train_dataset = PolarDataset(train_rgb_path, train_gt_path, train_aop_path, train_dop_path)
test_dataset = PolarDataset(test_rgb_path, test_gt_path, test_aop_path, test_dop_path)

# 新增真实数据集路径（假设真实数据放在以下目录）
real_rgb_path = '/home/guangyao/code/data/real_underwater_deal/input'
real_aop_path = '/home/guangyao/code/data/real_underwater_deal/p/AOP'
real_dop_path = '/home/guangyao/code/data/real_underwater_deal/p/DOP'

# 创建真实数据集（无GT）
real_dataset = PolarDataset(real_rgb_path, None, real_aop_path, real_dop_path)

# 创建混合数据集
# 创建DataLoader

num_workers = 0  # 根据CPU核心数调整
mixed_train_dataset = MixedPolarDataset(train_dataset, real_dataset, real_ratio=0.2)
mixed_train_loader = DataLoader(mixed_train_dataset, batch_size=batch_si, shuffle=True, num_workers=num_workers)
train_loader = DataLoader(train_dataset, batch_size=batch_si, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_si, shuffle=False, num_workers=num_workers)

# 注意力图保存设置
save_attention_every_batch = True  # 是否每个batch都保存注意力图
attention_save_interval = 10  # 如果save_attention_every_batch=False，则每N个batch保存一次

criterion_GAN = nn.MSELoss(reduction='sum')
l1_loss = nn.L1Loss(reduction='sum')  # l1 loss
l2_loss = nn.MSELoss(reduction='sum')  # l2 loss
SSIM = pytorch_ssim.SSIM()  # SSIMLoss()
L_vgg = VGG19_PercepLoss()  # PerceptionLoss()
L_lab = lab_Loss()
L_lch = lch_Loss()
# 新增域分类损失
domain_criterion = nn.BCELoss()

criterion_pixelwise = l2_loss.to(device)  # 切换l1 loss和 l2 loss
L_vgg = L_vgg.to(device)
l2_loss = l2_loss.to(device)
SSIM = SSIM.to(device)
criterion_GAN = criterion_GAN.to(device)
L_lab = L_lab.to(device)
L_lch = L_lch.to(device)

patch = (1, 256 // 2 ** 5, 256 // 2 ** 5)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

use_pretrain = True

# 仅在需要时初始化 missing_key
missing_key = []

if use_pretrain:
    # 加载预训练模型
    generator_state_dict = torch.load("saved_models/G/generator_795.pth")
    discriminator_state_dict = torch.load("saved_models/D/discriminator_795.pth")

    # 创建新的字典来保存修改后的权重
    modified_dict = {}

    # 对预训练模型的每个参数键进行处理
    for key in generator_state_dict:
        # 替换 'transformer.net.X' 为 'transformer.layers.X'
        if 'transformer.net.0' in key or 'transformer.net.1' in key:
            new_key = key.replace('transformer.net', 'transformer')
            new_key = 'transformer.layers.0.' + new_key
            modified_dict[new_key] = generator_state_dict[key]
        elif 'transformer.net.2' in key or 'transformer.net.3' in key:
            new_key = key.replace('transformer.net', 'transformer')
            new_key = 'transformer.layers.1.' + new_key
            if 'transformer.2' in new_key:
                new_key = new_key.replace('transformer.2', 'transformer.0')
            else:
                new_key = new_key.replace('transformer.3', 'transformer.1')
            modified_dict[new_key] = generator_state_dict[key]
        elif 'transformer.net.4' in key or 'transformer.net.5' in key:
            new_key = key.replace('transformer.net', 'transformer')
            new_key = 'transformer.layers.2.' + new_key
            if 'transformer.4' in new_key:
                new_key = new_key.replace('transformer.4', 'transformer.0')
            else:
                new_key = new_key.replace('transformer.5', 'transformer.1')
            modified_dict[new_key] = generator_state_dict[key]
        elif 'transformer.net.6' in key or 'transformer.net.7' in key:
            new_key = key.replace('transformer.net', 'transformer')
            new_key = 'transformer.layers.3.' + new_key
            if 'transformer.6' in new_key:
                new_key = new_key.replace('transformer.6', 'transformer.0')
            else:
                new_key = new_key.replace('transformer.7', 'transformer.1')
            modified_dict[new_key] = generator_state_dict[key]
        else:
            # 保持原样
            modified_dict[key] = generator_state_dict[key]

    # 加载修改后的字典到新的模型

    # 加载生成器的状态字典，并获取 missing_key
    missing_key, _ = generator.load_state_dict(modified_dict, strict=False)
    discriminator.load_state_dict(discriminator_state_dict, strict=False)

    print("成功加载预训练模型")
    print("[missing_key]", *missing_key, sep='\n')
    print("[unexpected_key]", *_, sep='\n')
else:
    print('No pretrain model found, training will start from scratch！')

# # 冻结所有参数
# for param in generator.parameters():
#     param.requires_grad = False

# # 只解冻 missing_key 中的参数
# for name, param in generator.named_parameters():
#     if name in missing_key:
#         param.requires_grad = True

# # 打印 generator 的可训练参数
# print("可训练的参数：")
# for name, param in generator.named_parameters():
#     if param.requires_grad:
#         print(name)


LR = 0.0005
# 修改优化器（包含域分类器参数）
optimizer_G = torch.optim.Adam(
    list(generator.parameters()) + list(generator.domain_classifier.parameters()),
    lr=LR, betas=(0.5, 0.999)
)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
scheduler_G = lr_scheduler.StepLR(optimizer_G, step_size=3, gamma=0.8)  # step_size=4
scheduler_D = lr_scheduler.StepLR(optimizer_D, step_size=3, gamma=0.8)  # step_size=4

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
# dummy_inputs = (
#     torch.randn(4, 3, 256, 256).to(device),  # Input
#     torch.randn(4, 6, 256, 256).to(device),  # aop
#
# )
#
# flops, params = profile(generator, inputs=(dummy_inputs[0], dummy_inputs[1],device))
# print('flops:{}').format(flops)
# print('params:{}').format(params)
#


lambda_pixel = 1    # pixel loss
lambda_lab = 0.001
lambda_lch = 1
lambda_con = 1     #  VGGloss
lambda_ssim = 1    #  ssim loss
# lambda_msssim = 100  # MS-SSIM loss 的权重
lambda_domain = 0.1  # 域对抗损失权重
real_ratio = 0.3  # 真实数据比例


# lambda_vif = 50  # 新增VIF损失权重
# 确保输入张量的值范围在 [0, 1]
def normalize_tensor(tensor):
    # 将张量归一化到 [0, 1]
    tensor_min = tensor.min()
    tensor_max = tensor.max()
    if tensor_max > tensor_min:
        return (tensor - tensor_min) / (tensor_max - tensor_min)
    else:
        # 如果所有值相同，则返回全零张量
        return torch.zeros_like(tensor)


# 在损失计算中添加VIF(Visual Information Fidelity)损失
def vif_loss(pred, target):
    return 1 - vif_p(pred, target, data_range=1.0)


f1_train = open('train_psnr.csv', 'w', encoding='utf-8')
csv_writer1_train = csv.writer(f1_train)
f2_train = open('train_SSIM.csv', 'w', encoding='utf-8')
csv_writer2_train = csv.writer(f2_train)

prev_time = time.time()
log_file = open("training_log.txt", "w")  # 日志文件
for epoch in range(epochs, n_epochs):
    ssim_list = []
    psnr_list = []
    for i, batch in enumerate(mixed_train_loader):
        # 获取域标签
        is_real = batch['is_real'].unsqueeze(1).to(device).float()  # 来自MixedDataset
        flag = is_real[0].int()
        # Model inputs
        if flag < 1:
            real_A = batch['rgb'].to(device)  # x
            real_B = batch['gt'].to(device)  # y
            real_P = batch['polar'].to(device)  # p

            real_A1 = split(real_A)
            real_B1 = split(real_B)
            real_P1 = split(real_P)
        else:
            real_A = batch['rgb'].to(device)  # x
            real_P = batch['polar'].to(device)  # p

            real_A1 = split(real_A)
            real_P1 = split(real_P)

        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False).to(device)  # 全1
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False).to(device)  # 全0

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B, domain_pred = generator(real_A, real_P, device)  #
        # 计算域分类损失
        domain_loss = domain_criterion(domain_pred, is_real)
        if 'gt' in batch:  # 模拟数据
            pred_fake = discriminator(fake_B, real_A1)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = (criterion_pixelwise(fake_B[0], real_B1[0]) + \
                          criterion_pixelwise(fake_B[1], real_B1[1]) + \
                          criterion_pixelwise(fake_B[2], real_B1[2]) + \
                          criterion_pixelwise(fake_B[3], real_B1[3])) / 4.0
            loss_ssim = -(SSIM(fake_B[0], real_B1[0]) + \
                          SSIM(fake_B[1], real_B1[1]) + \
                          SSIM(fake_B[2], real_B1[2]) + \
                          SSIM(fake_B[3], real_B1[3])) / 4.0
            ssim_value = -loss_ssim.item()
            loss_con = (L_vgg(fake_B[0], real_B1[0]) + \
                        L_vgg(fake_B[1], real_B1[1]) + \
                        L_vgg(fake_B[2], real_B1[2]) + \
                        L_vgg(fake_B[3], real_B1[3])) / 4.0
            loss_lab = (L_lab(fake_B[0], real_B1[0], device) + \
                        L_lab(fake_B[1], real_B1[1], device) + \
                        L_lab(fake_B[2], real_B1[2], device) + \
                        L_lab(fake_B[3], real_B1[3], device)) / 4.0
            loss_lch = (L_lch(fake_B[0], real_B1[0]) + \
                        L_lch(fake_B[1], real_B1[1]) + \
                        L_lch(fake_B[2], real_B1[2]) + \
                        L_lch(fake_B[3], real_B1[3])) / 4.0

            # msssim = ms_ssim(fake_B[3], real_B1[3], data_range=1.0)
            # loss_ms_ssim=1-msssim
            # pred = normalize_tensor(fake_B[3])
            # target = normalize_tensor(real_B1[3])
            # vif_losses= vif_loss(pred,target )
            # Total loss
            loss_G = (loss_GAN + lambda_pixel * loss_pixel + lambda_ssim * loss_ssim + \
                      lambda_con * loss_con + lambda_lab * loss_lab + lambda_lch * loss_lch
                      )  # lambda_msssim * loss_ms_ssim +lambda_vif *vif_losses
        else:
            loss_G = 0.1 * domain_loss  # 仅优化域损失

        loss_G.backward(retain_graph=True)
        # 在优化器更新之前添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        if 'gt' in batch:
            pred_real = discriminator(real_B1, real_A1)
            loss_real = criterion_GAN(pred_real, valid)
        else:
            # 没有真实标签时，跳过判别器的真实样本训练
            loss_real = 0

        fake_B_detached = [x.detach() for x in fake_B]
        pred_fake1 = discriminator(fake_B_detached, real_A1)
        # pred_fake1 = discriminator(fake_B, real_A1)
        loss_fake = criterion_GAN(pred_fake1, fake)

        # Total loss
        if 'gt' in batch:
            loss_D = 0.5 * (loss_real + loss_fake)
        else:
            loss_D = loss_fake

        loss_D.backward(retain_graph=True)
        # 在 optimizer_D.step() 之前
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10.0)
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        i = i
        # Determine approximate time left
        batches_done = epoch * len(mixed_train_loader) + i + 1  # i+1  +   (epoch-1)* len(loader)
        batches_left = n_epochs * len(mixed_train_loader) - batches_done
        if 'gt' in batch:
            out_train = torch.clamp(fake_B[3], 0., 1.)
            psnr_train = batch_PSNR(out_train, real_B, 1.)
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # 保存注意力图（仅对模拟数据）
            if save_attention_every_batch or (batches_done % attention_save_interval == 0):
                save_attention_maps_for_batch(generator, real_A, real_P, real_B, i, epoch)

            # Print log
            # sys.stdout.write(
            #     "\r[Epoch %d/%d] [Batch %d/%d][PSNR: %f] [SSIM: %f] [D loss: %f] [G loss: %f],[lab: %f],[lch: %f], [pixel: %f],[VGG_loss: %f], [adv: %f] ETA: %s"
            #     % (
            #         epoch,
            #         n_epochs-1,
            #         i+1,
            #         len(mixed_train_loader),
            #         psnr_train,
            #         ssim_value,
            #         loss_D.item(),
            #         loss_G.item(),
            #         0.001 * loss_lab.item(),
            #         1 * loss_lch.item(),
            #         0.1 * loss_pixel.item(),
            #         100 * loss_con.item(),
            #         loss_GAN.item(),
            #         time_left,
            #     )
            # )
            progress_line = (
                    "[Epoch %d/%d] [Batch %d/%d][PSNR: %f] [SSIM: %f] [D loss: %f] [G loss: %f],"
                    "[lab: %f],[lch: %f], [pixel: %f],[ssim_loss: %f],[VGG_loss: %f], [loss_GAN: %f] ETA: %s" % (
                        epoch,
                        n_epochs - 1,
                        i + 1,
                        len(mixed_train_loader),
                        psnr_train,
                        ssim_value,
                        loss_D.item(),
                        loss_G.item(),
                        loss_lab.item(),
                        loss_lch.item(),
                        loss_pixel.item(),
                        loss_ssim.item(),
                        loss_con.item(),
                        loss_GAN.item(),
                        time_left,
                    )
            )
            # 控制台显示（带覆盖效果）
            sys.stdout.write("\r" + progress_line)
            sys.stdout.flush()

            # 写入日志文件
            # log_file.write(progress_line + "\n")
            # log_file.flush()

            csv_writer1_train.writerow([str(psnr_train)])
            csv_writer2_train.writerow([str(ssim_value)])

            ssim_list.append(ssim_value)
            psnr_list.append(psnr_train)
        else:
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            progress_line = (
                    "[Epoch %d/%d] [Batch %d/%d][D loss: %f] [G loss: %f],"
                    " ETA: %s" % (
                        epoch,
                        n_epochs - 1,
                        i + 1,
                        len(mixed_train_loader),
                        loss_D.item(),
                        loss_G.item(),
                        time_left,
                    )
            )
            # 控制台显示（带覆盖效果）
            sys.stdout.write("\r" + progress_line)
            sys.stdout.flush()
        if batches_done % sample_interval == 0:
            sample_images(generator, test_loader, device, batches_done)
        # 输出真实真实水下损失
    ssim = sum(ssim_list) / len(ssim_list)
    psnr = sum(psnr_list) / len(psnr_list)
    # print(f"[Epoch {epoch + 1}/{n_epochs}] train PSNR: {psnr:.4f}, train SSIM: {ssim:.4f}\n")
    epoch_summary = f"[Epoch {epoch + 1}/{n_epochs}] train PSNR: {psnr:.4f}, train SSIM: {ssim:.4f}\n"
    print(epoch_summary)
    log_file.write(epoch_summary + "\n")
    log_file.flush()

    scheduler_G.step()
    scheduler_D.step()

    if epoch % checkpoint_interval == 0 or epoch == n_epochs - 1:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/G/generator_%d.pth" % (epoch + 1))
        torch.save(discriminator.state_dict(), "saved_models/D/discriminator_%d.pth" % (epoch + 1))

    # 测试集指标

    # if epoch >= 0:  # 每个epoch结束时计算一次测试集的PSNR和SSIM
    #     avg_psnr_test, avg_ssim_test = evaluate_test_set(generator, test_loader, device)
    #     print(f"[Epoch {epoch + 1}/{n_epochs}] Test PSNR: {avg_psnr_test:.4f}, Test SSIM: {avg_ssim_test:.4f}\n")
    if epoch >= 0:
        avg_psnr_test, avg_ssim_test = evaluate_test_set(generator, test_loader, device)
        test_summary = f"[Epoch {epoch + 1}/{n_epochs}] Test PSNR: {avg_psnr_test:.4f}, Test SSIM: {avg_ssim_test:.4f}\n"
        print(test_summary)
        log_file.write(test_summary + "\n")
        log_file.flush()

        # 推理时间统计：单张/整集
        generator.eval()
        num_images = 0
        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                real_A_b, _, real_P_b = batch
                real_A_b = real_A_b.to(device)
                real_P_b = real_P_b.to(device)
                _ = generator(real_A_b, real_P_b, device)
                num_images += real_A_b.size(0)
        elapsed = time.time() - start_time
        avg_time_per_image = elapsed / max(1, num_images)
        time_summary = f"[Epoch {epoch + 1}/{n_epochs}] Inference: {elapsed:.3f}s for {num_images} images, {avg_time_per_image * 1000:.2f} ms/img\n"
        print(time_summary)
        log_file.write(time_summary + "\n")
        log_file.flush()

# 在训练结束后，汇报 ViPT 中 Prompt_block 的 alpha(weight1) 和 beta(weight2)
try:
    alpha_beta_lines = []
    # 遍历 transformer 的各层，读取每层 Prompt_block 的可学习权重
    for li, layer in enumerate(generator.transformer.layers):
        prompt = layer["prompt"]
        alpha = prompt.weight1.detach().cpu().item()
        beta = prompt.weight2.detach().cpu().item()
        alpha_beta_lines.append(f"Layer {li}: alpha(weight1)={alpha:.6f}, beta(weight2)={beta:.6f}")

    header = "Learned alpha/beta (ViPT.Prompt_block weights):"
    print(header)
    log_file.write(header + "\n")
    for line in alpha_beta_lines:
        print(line)
        log_file.write(line + "\n")
    log_file.flush()
except Exception as e:
    print(f"Failed to read alpha/beta weights: {e}")

log_file.close()