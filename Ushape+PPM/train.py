from random import random
import torchvision.utils as utils
import pytorch_ssim
from torch.optim import lr_scheduler
import torch.utils.data as dataf
from torch.autograd import Variable
from net.Ushape_Trans_no_domain import *
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
from thop import profile
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.empty_cache()
dtype = 'float32'

torch.set_default_tensor_type(torch.FloatTensor)
torch.cuda.set_device(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择卡1

print(f"Using device: {torch.cuda.current_device()}")
print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


checkpoint_interval = 1  # 多少epoch保存一次模型
epochs = 0
n_epochs = 50# 这两个要一起修改
test_epoch= 49   # 这两个要一起修改
sample_interval = 57  #  batch size=6时设为57，batch size=1时设为338
batch_si=1


def split(img):
    output=[]
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output


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

    # 生成图像
    with torch.no_grad():
        fake_B = generator(real_A, real_P, device)

    # 使用最后一层的输出
    fake_B = fake_B[3].data  # 假设 fake_B 是一个列表，取最后一层的输出
    fake_B = torch.clamp(fake_B, 0., 1.)  # 将像素值限制在 [0, 1] 范围内

    # 将生成图像和真实图像拼接在一起
    img_sample = torch.cat([fake_B, real_B], dim=-2)  # 在高度维度上拼接

    # 保存图像
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    save_path = os.path.join(save_dir, f"{batches_done}.png")
    torchvision.utils.save_image(img_sample, save_path, nrow=nrow, normalize=True)

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
            fake_B_test = generator(real_A_test, real_P_test, device)

            # 使用最后一层的输出
            out_test = torch.clamp(fake_B_test[3], 0., 1.)

            # 计算PSNR和SSIM
            psnr_test = batch_PSNR(out_test, real_B_test, 1.)
            ssim_test = SSIM(out_test, real_B_test)

            psnr_test_list.append(psnr_test)
            ssim_test_list.append(ssim_test.item())

    # 计算每个epoch的平均PSNR和SSIM
    avg_psnr_test = np.mean(psnr_test_list)
    avg_ssim_test = np.mean(ssim_test_list)

    return avg_psnr_test, avg_ssim_test

class PolarDataset(Dataset):
    def __init__(self, rgb_path, gt_path, aop_path, dop_path, resize=(256, 256), dtype='float32'):
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

        # 获取文件列表并排序
        self.rgb_list = sorted(os.listdir(rgb_path), key=lambda x: int(x.split('.')[0]))
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

        # 合并偏振图像
        polar_img = np.concatenate([aop_img, dop_img], axis=-1)

        # 转换为Tensor并归一化
        rgb_img = torch.from_numpy(rgb_img.transpose(2, 0, 1)).float() / 255.0
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float() / 255.0
        polar_img = torch.from_numpy(polar_img.transpose(2, 0, 1)).float() / 255.0

        return rgb_img, gt_img, polar_img

# 定义路径
train_rgb_path = '/home/guangyao/code/data/polar_split/train_rgb'
train_gt_path = '/home/guangyao/code/data/polar_split/train_gt'
train_aop_path = '/home/guangyao/code/data/polar_split/train_p/AOP'
train_dop_path = '/home/guangyao/code/data/polar_split/train_p/DOP'

test_rgb_path = '/home/guangyao/code/data/polar_split/test_rgb'
test_gt_path = '/home/guangyao/code/data/polar_split/test_gt'
test_aop_path = '/home/guangyao/code/data/polar_split/test_p/AOP'
test_dop_path = '/home/guangyao/code/data/polar_split/test_p/DOP'

# 创建训练和测试数据集
train_dataset = PolarDataset(train_rgb_path, train_gt_path, train_aop_path, train_dop_path)
test_dataset = PolarDataset(test_rgb_path, test_gt_path, test_aop_path, test_dop_path)

# 创建DataLoader

num_workers = 4  # 根据CPU核心数调整
train_loader = DataLoader(train_dataset, batch_size=batch_si, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_si, shuffle=False, num_workers=num_workers)

criterion_GAN=nn.MSELoss(reduction='sum')
l1_loss = nn.L1Loss(reduction='sum')  # l1 loss
l2_loss = nn.MSELoss(reduction='sum')# l2 loss
SSIM = pytorch_ssim.SSIM()#    SSIMLoss()
L_vgg =VGG19_PercepLoss()    # PerceptionLoss()
L_lab=lab_Loss()
L_lch=lch_Loss()


criterion_pixelwise=l2_loss.to(device)   # 切换l1 loss和 l2 loss
L_vgg=L_vgg.to(device)
l2_loss=l2_loss.to(device)
SSIM=SSIM.to(device)
criterion_GAN=criterion_GAN.to(device)
L_lab=L_lab.to(device)
L_lch=L_lch.to(device)


patch = (1, 256 // 2 ** 5, 256// 2 ** 5)

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
        if 'transformer.net.0' in key or 'transformer.net.1' in key :
            new_key = key.replace('transformer.net', 'transformer')
            new_key = 'transformer.layers.0.'+new_key
            modified_dict[new_key] = generator_state_dict[key]
        elif 'transformer.net.2' in key or 'transformer.net.3' in key :
            new_key = key.replace('transformer.net', 'transformer')
            new_key = 'transformer.layers.1.'+new_key
            if 'transformer.2' in new_key:
                new_key=new_key.replace('transformer.2', 'transformer.0')
            else:
                new_key=new_key.replace('transformer.3', 'transformer.1')
            modified_dict[new_key] = generator_state_dict[key]
        elif 'transformer.net.4' in key or 'transformer.net.5' in key :
            new_key = key.replace('transformer.net', 'transformer')
            new_key = 'transformer.layers.2.'+new_key
            if 'transformer.4' in new_key:
                new_key=new_key.replace('transformer.4', 'transformer.0')
            else:
                new_key=new_key.replace('transformer.5', 'transformer.1')
            modified_dict[new_key] = generator_state_dict[key]
        elif 'transformer.net.6' in key or 'transformer.net.7' in key :
            new_key = key.replace('transformer.net', 'transformer')
            new_key = 'transformer.layers.3.'+new_key
            if 'transformer.6' in new_key:
                new_key=new_key.replace('transformer.6', 'transformer.0')
            else:
                new_key=new_key.replace('transformer.7', 'transformer.1')
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


LR=0.0005
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR,  betas=(0.5, 0.999) , weight_decay=1e-4)#
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR,  betas=(0.5, 0.999))
scheduler_G=lr_scheduler.StepLR(optimizer_G,step_size=50,gamma=0.8)
scheduler_D=lr_scheduler.StepLR(optimizer_D,step_size=50,gamma=0.8)

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
# flops, params = profile(generator, inputs=(dummy_inputs[0],))
# print('flops:{}'.format(flops))
# print('params:{}'.format(params))



lambda_pixel=0.1
lambda_lab=0.001
lambda_lch=1
lambda_con = 100
lambda_ssim=100
lambda_msssim = 0  # MS-SSIM loss 的权重


f1_train = open('train_psnr.csv', 'w', encoding='utf-8')
csv_writer1_train = csv.writer(f1_train)
f2_train = open('train_SSIM.csv', 'w', encoding='utf-8')
csv_writer2_train = csv.writer(f2_train)

prev_time = time.time()
patience=1  # 早停策略
for epoch in range(epochs, n_epochs):
    ssim_list=[]
    psnr_list=[]
    for i, batch in enumerate(train_loader):

        # Model inputs
        real_A = batch[0].to(device)  # x
        real_B = batch[1].to(device)  # y
        real_P = batch[2].to(device)  # p

        real_A1 = split(real_A)
        real_B1 = split(real_B)
        real_P1  = split(real_P)


        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False).to(device)   # 全1
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False).to(device)   # 全0

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_B = generator(real_A,real_P,device) #

        # fake_B1 = list(map(lambda x: x.detach(), fake_B))
        pred_fake = discriminator(fake_B, real_A1)
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = (criterion_pixelwise(fake_B[0], real_B1[0]) + \
                      criterion_pixelwise(fake_B[1], real_B1[1]) + \
                      criterion_pixelwise(fake_B[2], real_B1[2]) + \
                      criterion_pixelwise(fake_B[3], real_B1[3])) / 4.0
        loss_ssim =  -(SSIM(fake_B[0], real_B1[0])+\
                      SSIM(fake_B[1], real_B1[1])+\
                      SSIM(fake_B[2], real_B1[2])+\
                      SSIM(fake_B[3], real_B1[3]))/4.0
        ssim_value = - loss_ssim.item()
        loss_con = (L_vgg(fake_B[0], real_B1[0]) + \
                    L_vgg(fake_B[1], real_B1[1]) + \
                    L_vgg(fake_B[2], real_B1[2]) + \
                    L_vgg(fake_B[3], real_B1[3])) / 4.0
        loss_lab = (L_lab(fake_B[0], real_B1[0],device) + \
                    L_lab(fake_B[1], real_B1[1],device) + \
                    L_lab(fake_B[2], real_B1[2],device) + \
                    L_lab(fake_B[3], real_B1[3],device)) / 4.0
        loss_lch = (L_lch(fake_B[0], real_B1[0]) + \
                    L_lch(fake_B[1], real_B1[1]) + \
                    L_lch(fake_B[2], real_B1[2]) + \
                    L_lch(fake_B[3], real_B1[3])) / 4.0

        msssim = ms_ssim(fake_B[3], real_B1[3], data_range=1.0)
        loss_ms_ssim=1-msssim
        # Total loss
        loss_G = (loss_GAN + lambda_pixel * loss_pixel + lambda_ssim * loss_ssim + \
                  lambda_con * loss_con + lambda_lab * loss_lab + lambda_lch * loss_lch+ \
                  lambda_msssim * loss_ms_ssim)


        loss_G.backward(retain_graph=True)
        # 在优化器更新之前添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(real_B1, real_A1)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        # fake_B[0] = fake_B[0].detach()
        # fake_B[1] = fake_B[1].detach()
        # fake_B[2] = fake_B[2].detach()
        # fake_B[3] = fake_B[3].detach()
        fake_B_detached = [x.detach() for x in fake_B]
        pred_fake1 = discriminator(fake_B_detached, real_A1)
        # pred_fake1 = discriminator(fake_B, real_A1)
        loss_fake = criterion_GAN(pred_fake1, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        # loss_D=loss_real

        loss_D.backward(retain_graph=True)
        # 在 optimizer_D.step() 之前
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=10.0)
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------
        i=i
        # Determine approximate time left
        batches_done =  epoch * len(train_loader)   +   i+1    # i+1  +   (epoch-1)* len(loader)
        batches_left = n_epochs * len(train_loader) - batches_done
        out_train = torch.clamp(fake_B[3], 0., 1.)
        psnr_train = batch_PSNR(out_train, real_B, 1.)
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d][PSNR: %f] [SSIM: %f] [D loss: %f] [G loss: %f],[lab: %f],[lch: %f], [pixel: %f],[VGG_loss: %f], [adv: %f] ETA: %s"
            % (
                epoch,
                n_epochs-1,
                i+1,
                len(train_loader),
                psnr_train,
                ssim_value,
                loss_D.item(),
                loss_G.item(),
                0.001 * loss_lab.item(),
                1 * loss_lch.item(),
                0.1 * loss_pixel.item(),
                100 * loss_con.item(),
                loss_GAN.item(),
                time_left,
            )
        )

        csv_writer1_train.writerow([str(psnr_train)])
        csv_writer2_train.writerow([str(ssim_value)])

        ssim_list.append(ssim_value)
        psnr_list.append(psnr_train)

        if batches_done % sample_interval == 0:
            sample_images(generator, test_loader, device, batches_done)

    ssim= sum(ssim_list) / len(ssim_list)
    psnr=sum(psnr_list) / len(psnr_list)
    print(f"[Epoch {epoch + 1}/{n_epochs}] train PSNR: {psnr:.4f}, train SSIM: {ssim:.4f}\n")

    scheduler_G.step()
    scheduler_D.step()


    if  epoch % checkpoint_interval == 0 or epoch == n_epochs-1 :
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/G/generator_%d.pth" % (epoch+1))
        torch.save(discriminator.state_dict(), "saved_models/D/discriminator_%d.pth" % (epoch+1))


# 测试集指标

    if epoch >= 0:  # 每个epoch结束时计算一次测试集的PSNR和SSIM
        avg_psnr_test, avg_ssim_test = evaluate_test_set(generator, test_loader, device)
        print(f"[Epoch {epoch + 1}/{n_epochs}] Test PSNR: {avg_psnr_test:.4f}, Test SSIM: {avg_ssim_test:.4f}\n")

