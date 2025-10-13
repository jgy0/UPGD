import torch
import torch.nn as nn
from torchvision.models.vgg import vgg16
from torch.distributions import kl
from utils import  ResBlock, ConvBlock, Up, Compute_z
from PPM import *

class Encoder(nn.Module):
    def __init__(self, ch):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(ch_in=ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=64)
        self.conv3 = ConvBlock(ch_in=64, ch_out=64)
        self.conv4 = ResBlock(64)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x2 = self.conv2(x2)
        x3 = self.pool2(x2)
        x3 = self.conv3(x3)
        x4 = self.pool3(x3)
        x4 = self.conv4(x4)
        return x1, x2, x3, x4


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.pr_encoder = Encoder(3)
        self.po_encoder = Encoder(6)

        self.pr_conv = ResBlock(64)
        self.po_conv = ResBlock(64)

        self.pr_Up3 = Up()
        self.pr_UpConv3 = ConvBlock(ch_in=128, ch_out=64)
        self.pr_Up2 = Up()
        self.pr_UpConv2 = ConvBlock(ch_in=128, ch_out=64)
        self.pr_Up1 = Up()
        self.pr_UpConv1 = ConvBlock(ch_in=128, ch_out=64)

        self.po_Up3 = Up()
        self.po_UpConv3 = ConvBlock(ch_in=128, ch_out=64)
        self.po_Up2 = Up()
        self.po_UpConv2 = ConvBlock(ch_in=128, ch_out=64)

        out_conv = []
        out_conv.append(ResBlock(64))
        out_conv.append(ResBlock(64))
        out_conv.append(nn.Conv2d(64, 3, kernel_size=1, padding=0))
        self.out_conv = nn.Sequential(*out_conv)

        z = 20

        self.compute_z_pr = Compute_z(z)
        self.compute_z_po = Compute_z(z)

        self.conv_u = nn.Conv2d(z, 128, kernel_size=1, padding=0)
        self.conv_s = nn.Conv2d(z, 128, kernel_size=1, padding=0)

        self.insnorm = nn.InstanceNorm2d(128)
        self.sigmoid = nn.Sigmoid()
        """
        """
        # 新增：AOP/DOP编码器和PPM模块
        self.aop_encoder = Encoder(3)  # AOP为3通道
        self.dop_encoder = Encoder(3)  # DOP为3通道
        # self.ppm = Prompt_block(inplanes=64, hide_channel=8)  # 初始化PPM
        self.ppm_x1 = Prompt_block(inplanes=64, hide_channel=8)  # PPM for x1
        self.ppm_x2 = Prompt_block(inplanes=64, hide_channel=8)  # PPM for x2
        self.ppm_x3 = Prompt_block(inplanes=64, hide_channel=8)  # PPM for x3
        self.ppm_x4 = Prompt_block(inplanes=64, hide_channel=8)  # PPM for x4


    def forward(self, Input, Target,aop,dop, training=True):

        pr_x1, pr_x2, pr_x3, pr_x4 = self.pr_encoder.forward(Input)
        aop_x1, aop_x2, aop_x3, aop_x4 = self.aop_encoder.forward(aop)
        dop_x1, dop_x2, dop_x3, dop_x4 = self.dop_encoder.forward(dop)

        if training:
            po_x1, po_x2, po_x3, po_x4 = self.po_encoder.forward(torch.cat((Input, Target), dim=1))
            # x4->x3
            # 在关键层（如x4）融合多模态特征
            # fused_feat = torch.cat([pr_x4, aop_x4, dop_x4], dim=1)  # 通道拼接
            # ppm_out, aop_attn, dop_attn = self.ppm(fused_feat)  # PPM处理

            # 将融合后的特征与原始特征结合
            # pr_x4 = self.pr_conv(pr_x4 + ppm_out)  # 残差连接
            # po_x4 = self.po_conv(po_x4)

            # ===== 多层级 PPM 融合 =====
            # x1 层融合
            fused_x1 = torch.cat([pr_x1, aop_x1, dop_x1], dim=1)
            ppm_x1_out, _, _ = self.ppm_x1(fused_x1)
            pr_x1 = pr_x1 + ppm_x1_out  # 残差连接

            # x2 层融合
            fused_x2 = torch.cat([pr_x2, aop_x2, dop_x2], dim=1)
            ppm_x2_out, _, _ = self.ppm_x2(fused_x2)
            pr_x2 = pr_x2 + ppm_x2_out

            # x3 层融合
            fused_x3 = torch.cat([pr_x3, aop_x3, dop_x3], dim=1)
            ppm_x3_out, _, _ = self.ppm_x3(fused_x3)
            pr_x3 = pr_x3 + ppm_x3_out

            # x4 层融合
            fused_x4 = torch.cat([pr_x4, aop_x4, dop_x4], dim=1)
            ppm_x4_out, aop_attn, dop_attn = self.ppm_x4(fused_x4)
            pr_x4 = pr_x4 + ppm_x4_out

            pr_x4 = self.pr_conv(pr_x4)
            po_x4 = self.po_conv(po_x4)
            pr_d3 = self.pr_Up3(pr_x4)
            po_d3 = self.po_Up3(po_x4)
            # x3->x2
            pr_d3 = torch.cat((pr_x3, pr_d3), dim=1)
            po_d3 = torch.cat((po_x3, po_d3), dim=1)
            pr_d3 = self.pr_UpConv3(pr_d3)
            po_d3 = self.po_UpConv3(po_d3)
            pr_d2 = self.pr_Up2(pr_d3)
            po_d2 = self.po_Up2(po_d3)
            # x2->x1
            pr_d2 = torch.cat((pr_x2, pr_d2), dim=1)
            po_d2 = torch.cat((po_x2, po_d2), dim=1)
            pr_d2 = self.pr_UpConv2(pr_d2)
            po_d2 = self.po_UpConv2(po_d2)
            pr_d1 = self.pr_Up1(pr_d2)
            po_d1 = self.pr_Up1(po_d2)
            # cat
            pr_d1 = torch.cat((pr_x1, pr_d1), dim=1)
            po_d1 = torch.cat((po_x1, po_d1), dim=1)
            # x1->dis
            pr_u_dist, pr_s_dist, _, _, _, _ = self.compute_z_pr(pr_d1)
            po_u_dist, po_s_dist, _, _, _, _ = self.compute_z_po(po_d1)
            po_latent_u = po_u_dist.rsample()
            po_latent_s = po_s_dist.rsample()
            po_latent_u = torch.unsqueeze(po_latent_u, -1)
            po_latent_u = torch.unsqueeze(po_latent_u, -1)
            po_latent_s = torch.unsqueeze(po_latent_s, -1)
            po_latent_s = torch.unsqueeze(po_latent_s, -1)
            po_u = self.conv_u(po_latent_u)
            po_s = self.conv_s(po_latent_s)
            pr_d1 = self.insnorm(pr_d1) * torch.abs(po_s) + po_u
            # x1->out
            pr_d1 = self.pr_UpConv1(pr_d1)
            out = self.out_conv(pr_d1)

            return out, pr_u_dist, pr_s_dist, po_u_dist, po_s_dist

        else:
            # x4->x3
            pr_x4 = self.pr_conv(pr_x4)
            pr_d3 = self.pr_Up3(pr_x4)
            # x3->x2
            pr_d3 = torch.cat((pr_x3, pr_d3), dim=1)
            pr_d3 = self.pr_UpConv3(pr_d3)
            pr_d2 = self.pr_Up2(pr_d3)
            # x2->x1
            pr_d2 = torch.cat((pr_x2, pr_d2), dim=1)
            pr_d2 = self.pr_UpConv2(pr_d2)
            pr_d1 = self.pr_Up1(pr_d2)
            # cat
            pr_d1 = torch.cat((pr_x1, pr_d1), dim=1)
            # x1->dis
            pr_u_dist, pr_s_dist, _, _, _, _ = self.compute_z_pr(pr_d1)
            pr_latent_u = pr_u_dist.rsample()
            pr_latent_s = pr_s_dist.rsample()
            pr_latent_u = torch.unsqueeze(pr_latent_u, -1)
            pr_latent_u = torch.unsqueeze(pr_latent_u, -1)
            pr_latent_s = torch.unsqueeze(pr_latent_s, -1)
            pr_latent_s = torch.unsqueeze(pr_latent_s, -1)
            pr_u = self.conv_u(pr_latent_u)
            pr_s = self.conv_s(pr_latent_s)
            pr_d1 = self.insnorm(pr_d1) * torch.abs(pr_s) + pr_u
            # x1->out
            pr_d1 = self.pr_UpConv1(pr_d1)
            out = self.out_conv(pr_d1)
            return out


class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        return perception_loss


class mynet(nn.Module):
    def __init__(self, opt):
        super(mynet, self).__init__()
        self.device = torch.device(opt.device)
        self.decoder = Decoder(device=self.device).to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        self.VGG16 = PerceptionLoss().to(self.device)

    def forward(self, Input, label,aop,dop, training=True):
        self.Input = Input
        self.label = label
        self.aop=aop
        self.dop=dop
        if training:
            self.out, self.pr_u, self.pr_s, self.po_u, self.po_s = self.decoder.forward(Input, label,aop,dop)

    def sample(self, testing=False):
        if testing:
            self.out = self.decoder.forward(self.Input, self.label,self.aop, self.dop, training=False)
            return self.out

    def kl_divergence(self, analytic=True):
        if analytic:
            kl_div_u = torch.mean(kl.kl_divergence(self.po_u, self.pr_u))
            kl_div_s = torch.mean(kl.kl_divergence(self.po_s, self.pr_s))
        return kl_div_u + kl_div_s

    def elbo(self, target, analytic_kl=True):
        self.kl_loss = self.kl_divergence(analytic=analytic_kl)
        self.reconstruction_loss = self.criterion(self.out, target)
        self.vgg16_loss = self.VGG16(self.out, target)
        return self.reconstruction_loss + self.vgg16_loss + self.kl_loss
