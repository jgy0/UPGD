import torch.nn as nn
import torch
from timm.models.layers import to_2tuple
from net.Agent_Attention import *
from net.channel_attention import *
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=512, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        # allow different input size
        # B, C, H, W = x.shape
        # _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        # _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)  # 1,3,256,256->1,512,16,16
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # 1,256,512 BCHW -> BNC
        x = self.norm(x)
        return x


def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x


'''
feature2token
'''
def feature2token(x):
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]  1,8,16,16
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w) # 1,8,256

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x) # 1,8,256
        output = mask * x  # 1,8,256
        output = output.contiguous().view(b, c, h, w)  # 1,8,16,16

        return output


# class Prompt_block(nn.Module, ):
#     def __init__(self, inplanes=512, hide_channel=8, smooth=False):
#         super(Prompt_block, self).__init__()
#         self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
#         self.fovea = Fovea(smooth=smooth)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, x):
#         """ Forward pass with input x. """
#         B, C, W, H = x.shape
#         x0 = x[:, 0:int(C/2), :, :].contiguous()
#         x0 = self.conv0_0(x0)
#         x1 = x[:, int(C/2):, :, :].contiguous()
#         x1 = self.conv0_1(x1)
#         x0 = self.fovea(x0) + x1
#
#         return self.conv1x1(x0)
"""
Utrans6（agent-attention）
"""

# class Prompt_block(nn.Module, ):
#     def __init__(self, inplanes=512, hide_channel=8, smooth=False):
#         super(Prompt_block, self).__init__()
#         self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv0_2 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
#         self.fovea = Fovea(smooth=smooth)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, x):
#         """ Forward pass with input x. """
#         B, C, W, H = x.shape #1 1536 16 16
#         if C==1536:
#             x0 = x[:, 0:int(C/3), :, :].contiguous()  # 1,512,16,16
#             x0 = self.conv0_0(x0)  # 1,8,16,16
#             x1 = x[:, int(C/3):int(2*C/3), :, :].contiguous() # 1,512,16,16   AOP
#             x1 = self.conv0_1(x1)  # 1,8,16,16
#             x2 = x[:, int(2*C/3):, :, :].contiguous()  #1,512,16,16 DOP
#             x2 = self.conv0_2(x2)#1,8,16,16
#             x0 = self.fovea(x0) + x1+x2
#         else:
#             B, C, W, H = x.shape
#             x0 = x[:, 0:int(C / 2), :, :].contiguous()
#             x0 = self.conv0_0(x0)
#             x1 = x[:, int(C / 2):, :, :].contiguous()
#             x1 = self.conv0_1(x1)
#             x0 = self.fovea(x0) + x1
#
#         return self.conv1x1(x0)

"""
Utrans8  VIPT改版   agent in prompt 
"""
class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=512, hide_channel=8, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_2 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)
        self.channel_attention1=SqueezeExcitation()
        self.channel_attention2 = SqueezeExcitation()

        # 添加可学习的权重
        self.weight1 = nn.Parameter(torch.ones(1))
        self.weight2 = nn.Parameter(torch.ones(1))

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape #1 2048 16 16
        x0 = x[:, 0:int(C/4), :, :].contiguous()  # 1,512,16,16
        x0 = self.conv0_0(x0)  # 1,8,16,16
        x1 = x[:, int(C/4):int(2*C/4), :, :].contiguous() #  init feature/ mix feature
        x1 = self.conv0_1(x1)  # 1,8,16,16
        x2 = x[:, int(2*C/4):int(3*C/4), :, :].contiguous()  # 1,512,16,16   AOP
        x2_later=self.channel_attention1(x2)
        x2_o=self.conv0_2(x2_later)
        # aop传到下一轮


        x3 = x[:, int(3*C/4):, :, :].contiguous()# 1,512,16,16 DOP
        x3_later=self.channel_attention2(x3)
        x3_o=self.conv0_2(x3_later)# 1，8，16，16
        # dop传到下一轮

        x0 = self.fovea(x0) + x1+ self.weight1 * x2_o + self.weight2 * x3_o
        # x0 = self.fovea(x0) + x1 +  x2_o +  x3_o
        # x0 = x0 + x1 + self.weight1 * x2_o + self.weight2 * x3_o
        return self.conv1x1(x0),x2_later,x3_later


"""
Utrans9  
"""
# class Prompt_block(nn.Module, ):
#     def __init__(self, inplanes=512, hide_channel=8, smooth=False):
#         super(Prompt_block, self).__init__()
#         self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv0_2 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv0_3 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
#         self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
#         self.conv1x2 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
#         self.conv1x3 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
#         self.fovea = Fovea(smooth=smooth)
#
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward(self, x):
#         """ Forward pass with input x. """
#         B, C, W, H = x.shape #1 2048 16 16
#         x0 = x[:, 0:int(C/4), :, :].contiguous()  # 1,512,16,16
#         x0 = self.conv0_0(x0)  # 1,8,16,16
#         x1 = x[:, int(C/4):int(2*C/4), :, :].contiguous() #  init feature/ mix feature
#         x1 = self.conv0_1(x1)  # 1,8,16,16
#         x2 = x[:, int(2*C/4):int(3*C/4), :, :].contiguous()  # 1,512,16,16   AOP
#         x2 = self.conv0_2(x2)  # 1,8,16,16
#         x3 = x[:, int(3*C/4):, :, :].contiguous()# 1,512,16,16 DOP
#         x3 = self.conv0_3(x3)  # 1,8,16,16
#
#         x0=self.fovea(x0)+x1+x2+x3
#         return self.conv1x1(x0),self.conv1x2(x2),self.conv1x3(x3)
