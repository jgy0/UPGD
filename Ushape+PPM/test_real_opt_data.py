import numpy as np
from tqdm import tqdm
import pytorch_ssim
import torch.utils.data as dataf
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from net.Ushape_Trans import *
from net.utils import *
import cv2
import torchvision.utils as utils
from random import random
import matplotlib.pyplot as plt
from utility import plots as plots, ptcolor as ptcolor, ptutils as ptutils, data as data
from loss.LAB import *
from loss.LCH import *
dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择卡1
from torch.utils.data import Dataset, DataLoader

def apply_clahe_np(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return enhanced

def apply_gamma_np(img_rgb, gamma=1.5):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(img_rgb, table)

def apply_retinex_np(img_rgb, sigma=30):
    img_float = img_rgb.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
    retinex = np.log10(img_float) - np.log10(blur)
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255
    return np.uint8(retinex)

def preprocess_image(img_rgb, method='clahe'):
    if method == 'clahe':
        return apply_clahe_np(img_rgb)
    elif method == 'gamma':
        return apply_gamma_np(img_rgb, gamma=1.3)
    elif method == 'retinex':
        return apply_retinex_np(img_rgb)
    else:
        return img_rgb  # 不做预处理


def split(img):
    output=[]
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output


class PolarDataset(Dataset):
    def __init__(self, rgb_path, aop_path, dop_path, resize=(256, 256),method='retinex', dtype='float32'):
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
        self.aop_path = aop_path
        self.dop_path = dop_path
        self.resize = resize
        self.dtype = dtype
        self.preprocess_method=method

        # 获取文件列表并排序
        self.rgb_list = sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0]))
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
        rgb_img = preprocess_image(rgb_img, self.preprocess_method)

        # 加载偏振角（AOP）和偏振度（DOP）图像
        aop_img = cv2.imread(os.path.join(self.aop_path, self.aop_list[idx]))
        dop_img = cv2.imread(os.path.join(self.dop_path, self.dop_list[idx]))
        aop_img = cv2.cvtColor(aop_img, cv2.COLOR_BGR2RGB)
        dop_img = cv2.cvtColor(dop_img, cv2.COLOR_BGR2RGB)
        aop_img = cv2.resize(aop_img, self.resize)
        dop_img = cv2.resize(dop_img, self.resize)

        # 合并偏振图像
        polar_img = np.concatenate([aop_img, dop_img], axis=-1)

        # 转换为Tensor并归一化
        rgb_img = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float() / 255.0
        polar_img = torch.from_numpy(polar_img.transpose(2, 0, 1)).float() / 255.0

        return rgb_img,polar_img



test_rgb_path = '/home/guangyao/code/data/UCPD/input_resize1224'
test_aop_path = '/home/guangyao/code/data/UCPD/AOP'
test_dop_path = '/home/guangyao/code/data/UCPD/DOP'

preprocess_method = 'retinex'
test_dataset = PolarDataset(test_rgb_path, test_aop_path, test_dop_path,resize=(256, 256),method=preprocess_method)

# 创建DataLoader

num_workers = 4  # 根据CPU核心数调整
batch_si=1
test_loader = DataLoader(test_dataset, batch_size=batch_si, shuffle=False, num_workers=num_workers)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.load_state_dict(torch.load("/home/guangyao/code/ushape-transformer/result/new_result/ppm-u2/generator_20.pth"))#strict=False   #./result/baseline/model/generator_59.pth   # ./result/new_result/Utrans6-attention/model/generator_42.pth
discriminator.load_state_dict(torch.load("/home/guangyao/code/ushape-transformer/result/new_result/ppm-u2/discriminator_20.pth")) # ./result/baseline/model/discriminator_59.pth  # ./result/new_result/Utrans6-attention/model/discriminator_42.pth


generator.eval()  # 设置模型为评估模式
discriminator.eval()



# 初始化评价列表
uciqe_test_list = []
uiqm_test_list = []
nioe_test_list = []
musiq_test_list = []
uranker_test_list = []

with torch.no_grad():  # 评估时不计算梯度
    for i, batch in enumerate(test_loader):
        real_A, real_P = batch
        real_A = real_A.to(device)
        real_P = real_P.to(device)

        # 生成图像
        with torch.no_grad():
            fake_B, _ = generator(real_A, real_P, device)

        # 使用最后一层的输出
        fake_B = fake_B[3].data  # 假设 fake_B 是一个列表，取最后一层的输出
        out_test = torch.clamp(fake_B, 0., 1.)  # 将像素值限制在 [0, 1] 范围内
        utils.save_image(out_test, "./images/UCDP/%s.png" % (i + 1), nrow=5, normalize=True)  #













