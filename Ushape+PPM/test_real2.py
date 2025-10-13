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

# 亮度归一化方式：可选 'clahe' | 'gamma' | 'retinex' | None
preprocess_method = 'retinex'
test_x = []
path = '/home/guangyao/code/data/my_real/input'  # 要改
path_list = os.listdir(path)
path_list.sort(key=lambda x: int(x.split('.')[0]))
for item in path_list:
    impath = path + '/'+ item
    # print("开始处理"+impath)
    imgx = cv2.imread(impath)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = cv2.resize(imgx, (256, 256))
    imgx = preprocess_image(imgx, preprocess_method)
    test_x.append(imgx)

x_test = np.array(test_x)
x_test=x_test.astype(dtype)
x_test= torch.from_numpy(x_test)
x_test=x_test.permute(0,3,1,2)
x_test=x_test/255.0


test_x_polar = []
path_list_AOP = os.listdir('/home/guangyao/code/data/my_real/AOP')  # 偏振角
path_list_DOP = os.listdir('/home/guangyao/code/data/my_real/DOP')  # 偏振度


path_list_AOP.sort(key=lambda x:int(x.split('.')[0]))
path_list_DOP.sort(key=lambda x:int(x.split('.')[0]))

for i in range(len(path_list_AOP)):
    # 读取四个角度的偏振图像
    img_AOP = cv2.imread('/home/guangyao/code/data/my_real/AOP' +'/'+ path_list_AOP[i])
    img_DOP = cv2.imread('/home/guangyao/code/data/my_real/DOP' +'/'+ path_list_DOP[i])


    # 转换为RGB，并将图像大小调整为256x256
    img_AOP = cv2.cvtColor(img_AOP, cv2.COLOR_BGR2RGB)
    img_DOP = cv2.cvtColor(img_DOP, cv2.COLOR_BGR2RGB)


    img_AOP = cv2.resize(img_AOP, (256, 256))
    img_DOP = cv2.resize(img_DOP, (256, 256))

    # 可选：是否对偏振图像也预处理？
    # img_AOP = preprocess_image(img_AOP, preprocess_method)
    # img_DOP = preprocess_image(img_DOP, preprocess_method)

    # 将两个偏振图像作为一个输入样本
    img_combined = np.concatenate([img_AOP, img_DOP], axis=-1)
    test_x_polar.append(img_combined)

test_x = np.array(test_x_polar)
X_test_polar = test_x.astype(dtype)
X_test_polar = torch.from_numpy(X_test_polar)
X_test_polar = X_test_polar.permute(0, 3, 1, 2)  # 转换为 (B, C, H, W) 格式
X_test_polar = X_test_polar / 255.0  # 归一化到[0, 1]

generator = Generator().to(device)
discriminator = Discriminator().to(device)

generator.load_state_dict(torch.load("/home/guangyao/code/ushape-transformer/result/new_result/ppm-u2/generator_20.pth"))#strict=False
discriminator.load_state_dict(torch.load("/home/guangyao/code/ushape-transformer/result/new_result/ppm-u2/discriminator_20.pth")) #


generator.eval()  # 设置模型为评估模式
discriminator.eval()


def sample_images(i):
    """Saves a generated sample from the validation set"""

    real_A = x_test[i,:,:,:].to(device)
    real_P = X_test_polar[i,:,:,:].to(device)
    real_A=real_A.unsqueeze(0)
    real_P = real_P.unsqueeze(0)

    fake_B,_ = generator(real_A,real_P,device) #
    out_test = torch.clamp(fake_B[3], 0., 1.)



    # imgy=fake_B[3].data
    # imgx=real_A.data
    # x=imgx[:,:,:,:]
    # y=imgy[:,:,:,:]
    # img_sample = torch.cat((x,y), -2)

    utils.save_image(out_test, "images/tmp/ours/%s.png" % ( i+1), nrow=5, normalize=True)#要改



# 初始化评价列表
uciqe_test_list = []
uiqm_test_list = []
nioe_test_list = []
musiq_test_list = []
uranker_test_list = []

with torch.no_grad():  # 评估时不计算梯度
    for i in range(len(x_test)):  # 遍历测试集
        real_A_test = Variable(x_test[i:i + 1]).cuda()  # 获取测试图像
        real_P_test = Variable(X_test_polar[i:i + 1]).cuda()
        fake_B_test,_ = generator(real_A_test, real_P_test, device)  # 生成测试图像

        out_test = torch.clamp(fake_B_test[3], 0., 1.)
        out_test_np = out_test.cpu().numpy().squeeze().transpose(1, 2, 0)  # 转换为 (H, W, C) 格式

        # fake_B = (out_test * 255).astype(np.uint8)  # 转换为 [0, 255]
        # fake_B_adjusted = adjust_brightness(fake_B, alpha=1.2, beta=30)
        # 保存图像

        sample_images(i)
        # 计算PSNR和SSIM










