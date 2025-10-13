import torch.nn as nn
import torch
from timm.models.layers import to_2tuple
from net.Agent_Attention import *
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
"""
Utrans6（agent-attention）
"""

class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=480, hide_channel=8, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.fovea = Fovea(smooth=smooth)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape #1 1536 16 16
        x0 = x[:, 0:int(C / 2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C / 2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        x0 = self.fovea(x0) + x1

        return self.conv1x1(x0)