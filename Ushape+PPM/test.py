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

def split(img):
    output=[]
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output

test_x = []
path = '/home/guangyao/code/data/polar_split/test_rgb'  # 要改
path_list = os.listdir(path)
path_list.sort(key=lambda x: int(x.split('.')[0]))
for item in path_list:
    impath = path + '/'+ item
    # print("开始处理"+impath)
    imgx = cv2.imread(impath)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = cv2.resize(imgx, (256, 256))
    test_x.append(imgx)

x_test = np.array(test_x)
x_test=x_test.astype(dtype) #dtype = 'float32'
x_test= torch.from_numpy(x_test)
x_test=x_test.permute(0,3,1,2)
x_test=x_test/255.0

test_Y = []
path = '/home/guangyao/code/data/polar_split/test_gt'  # 要改
path_list = os.listdir(path)
path_list.sort(key=lambda x: int(x.split('.')[0]))
for item in path_list:
    impath = path +  '/' +item
    # print("开始处理"+impath)
    imgx = cv2.imread(impath)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx = cv2.resize(imgx, (256, 256))
    test_Y.append(imgx)


Y_test = np.array(test_Y)
Y_test=Y_test.astype(dtype) #dtype = 'float32'
Y_test= torch.from_numpy(Y_test)
Y_test=Y_test.permute(0,3,1,2)
Y_test=Y_test/255.0

test_x_polar = []
path_list_AOP = os.listdir('/home/guangyao/code/data/polar_split/test_p/AOP')  # 偏振角
path_list_DOP = os.listdir('/home/guangyao/code/data/polar_split/test_p/DOP')  # 偏振度


path_list_AOP.sort(key=lambda x:int(x.split('.')[0]))
path_list_DOP.sort(key=lambda x:int(x.split('.')[0]))

for i in range(len(path_list_AOP)):
    # 读取四个角度的偏振图像
    img_AOP = cv2.imread('/home/guangyao/code/data/polar_split/test_p/AOP' +'/'+ path_list_AOP[i])
    img_DOP = cv2.imread('/home/guangyao/code/data/polar_split/test_p/DOP' +'/'+ path_list_DOP[i])


    # 转换为RGB，并将图像大小调整为256x256
    img_AOP = cv2.cvtColor(img_AOP, cv2.COLOR_BGR2RGB)
    img_DOP = cv2.cvtColor(img_DOP, cv2.COLOR_BGR2RGB)


    img_AOP = cv2.resize(img_AOP, (256, 256))
    img_DOP = cv2.resize(img_DOP, (256, 256))

    # 将两个偏振图像作为一个输入样本
    img_combined = np.concatenate([img_AOP, img_DOP], axis=-1)
    test_x_polar.append(img_combined)

test_x = np.array(test_x_polar)
X_test_polar = test_x.astype(dtype)
X_test_polar = torch.from_numpy(X_test_polar)
X_test_polar = X_test_polar.permute(0, 3, 1, 2)  # 转换为 (B, C, H, W) 格式
X_test_polar = X_test_polar / 255.0  # 归一化到[0, 1]


dataset = dataf.TensorDataset(x_test,Y_test,X_test_polar)
loader = dataf.DataLoader(dataset, batch_size=6, shuffle=True,num_workers=4)

generator = Generator().cuda()
discriminator = Discriminator().cuda()

SSIM = pytorch_ssim.SSIM()

use_pretrain=True
if use_pretrain:
    missing_key,unexpected_key=generator.load_state_dict(torch.load("/home/guangyao/code/ushape-transformer/saved_models/G/generator_20.pth"),strict=True)
    discriminator.load_state_dict(torch.load("/home/guangyao/code/ushape-transformer/saved_models/D/discriminator_20.pth"),strict=True)
    print("成功加载预训练模型")
    print("[missing_key]",*missing_key,sep='\n')
    print("[unexpected_key]",*unexpected_key,sep='\n')
else:
    print('No pretrain model found, training will start from scratch！')


def sample_images(i):
    """Saves a generated sample from the validation set"""
    generator.eval()
    real_A = x_test[i,:,:,:].to(device)
    real_B = Y_test[i,:,:,:].to(device)
    real_P = X_test_polar[i,:,:,:].to(device)
    real_A=real_A.unsqueeze(0)
    real_B=real_B.unsqueeze(0)
    real_P = real_P.unsqueeze(0)
    fake_B,_ = generator(real_A,real_P,device) #
    #print(fake_B.shape)
    imgx=fake_B[3].data
    imgy=real_B.data
    x=imgx[:,:,:,:]
    # y=imgy[:,:,:,:]
    # img_sample = torch.cat((x,y), -2)
    utils.save_image(x, "/home/guangyao/code/ushape-transformer/result/aba/agents/proxy-qkv/%s.png" % ( i), nrow=5, normalize=True)#



generator.eval()  # 设置模型为评估模式
psnr_test_list = []
ssim_test_list = []


with torch.no_grad():  # 评估时不计算梯度
    for i in range(len(x_test)):  # 遍历测试集
        real_A_test = Variable(x_test[i:i + 1]).cuda()  # 获取测试图像
        real_B_test = Variable(Y_test[i:i + 1]).cuda()  # 获取测试标签图像
        real_P_test = Variable(X_test_polar[i:i + 1]).cuda()
        fake_B_test,_ = generator(real_A_test, real_P_test ,device)  # 生成测试图像

        sample_images(i)
        # 计算PSNR和SSIM
        out_test = torch.clamp(fake_B_test[3], 0., 1.)  # 使用最后一层的输出

        # 检查输出图像
        # plt.figure()
        # plt.hist(out_test.ravel(), bins=256, range=(0, 1))
        # plt.pause(0.1)
        # plt.show()
        # plt.savefig("./output/indoor/%s.png" % (i + 1))

        psnr_test = batch_PSNR(out_test, real_B_test, 1.)  # 计算PSNR
        ssim_test = SSIM(out_test, real_B_test)  # 计算SSIM
        # 计算无参考质量评估指标

        out_test_np = out_test.cpu().numpy().squeeze().transpose(1, 2, 0)  # 转换为 (H, W, C) 格式
        # 记录结果
        psnr_test_list.append(psnr_test)
        ssim_test_list.append(ssim_test.item())

    # 计算每个epoch的平均PSNR和SSIM
    psnr_test_list=psnr_test_list[:-1]
    ssim_test_list=ssim_test_list[:-1]
    avg_psnr_test = np.mean(psnr_test_list)
    avg_ssim_test = np.mean(ssim_test_list)

    # 打印并记录测试集的结果
    print(f"Test PSNR: {avg_psnr_test:.4f}, Test SSIM: {avg_ssim_test:.4f}\n")



