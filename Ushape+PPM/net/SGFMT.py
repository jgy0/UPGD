# -*- coding: utf-8 -*-
# @File    : SGFMT.py
# coding=utf-8
# Design based on the Vit
import torch
import torch.nn as nn
from net.IntmdSequential import IntermediateSequential
from net.ViPT import PatchEmbed,Prompt_block



#实现了自注意力机制，相当于unet的bottleneck层
class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


# class TransformerModel(nn.Module):
#     def __init__(
#         self,
#         dim,  #512
#         depth,  #4
#         heads,  #8
#         mlp_dim,  #4096
#         dropout_rate=0.1,
#         attn_dropout_rate=0.1,
#     ):
#         super().__init__()
#         layers = []
#         for _ in range(depth):
#             layers.extend(
#                 [
#                     Residual(
#                         PreNormDrop(
#                             dim,
#                             dropout_rate,
#                             SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
#                         )
#                     ),
#                     Residual(
#                         PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
#                     ),
#                 ]
#             )
#             # dim = dim / 2
#         self.net = IntermediateSequential(*layers)
#
#
#     def forward(self, x): # ,polar
#         # x_1=torch.cat([x, polar], dim=1)
#         # print(self.net)
#         return self.net(x)


"""
第一波修改
"""

# '''
# add token transfer to feature
# '''
# def token2feature(tokens):
#     B,L,D=tokens.shape
#     H=W=int(L**0.5)
#     x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
#     return x
#
#
# '''
# feature2token
# '''
# def feature2token(x):
#     B,C,W,H = x.shape
#     L = W*H
#     tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
#     return tokens
#
# class TransformerModel(nn.Module):
#     def __init__(
#             self,
#             dim,  # 512
#             depth,  # 4
#             heads,  # 8
#             mlp_dim,  # 4096
#              # The prompt module
#             dropout_rate=0.1,
#             attn_dropout_rate=0.1,
#             prompt_dropout_rate=0.2  # Dropout率，用于Prompt模块
#     ):
#         super().__init__()
#
#         self.layers = nn.ModuleList([
#             nn.ModuleDict({
#                 "transformer": nn.ModuleList([
#                     Residual(
#                         PreNormDrop(
#                             dim,
#                             dropout_rate,
#                             SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
#                         )
#                     ),
#                     Residual(
#                         PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
#                     )
#                 ]),
#                 "prompt": Prompt_block(),  # Assuming P is a class that processes the input
#             }) for _ in range(depth)
#         ])
#
#     def forward(self, x, polar):
#         """
#         x: Tensor of shape [B, N, D] (Main modality)
#         polar: Tensor of shape [B, N, D] (Auxiliary modality)
#         """
#         # Step 1: First layer initialization
#         # Convert tokens to features for both modalities
#         x_feat = token2feature(x)  # 1,512,16,16
#         polar_feat = token2feature(polar)  # 1,512,16,16
#
#         # Combine main and auxiliary modalities and process through the first prompt module
#         prompt_output = torch.cat([x_feat, polar_feat], dim=1)  # 1,1024,16,16 Concatenate main and auxiliary modalities
#         prompt_output = self.layers[0]["prompt"](prompt_output)  # 1,512,16,16 P1 processes the combined features
#
#         # Convert prompt output back to token format and combine with main modality
#         x = feature2token(prompt_output) # 1,256,512
#         x = x + feature2token(x_feat)  #1,256,512 Combine with the original main modality features
#
#         # Step 2: Process through the first Transformer layer
#         for transformer_layer in self.layers[0]["transformer"]:
#              x = transformer_layer(x)
#
#         # Step 3: Subsequent layers processing
#         for layer in self.layers[1:]:
#             prompt = layer["prompt"]
#
#             # Convert tokens back to features for prompt processing
#             x_feat = token2feature(x)
#
#             # Combine current features with previous prompt output and process through the prompt module
#             prompt_output = torch.cat([x_feat, prompt_output], dim=1)  # Combine current features and previous prompt
#             prompt_output = prompt(prompt_output)  # Process through P^L
#
#             # Convert prompt output back to token format
#             x = feature2token(prompt_output)+x
#
#             # Process through the Transformer layers
#             for transformer_layer in layer["transformer"]:
#                 x = transformer_layer(x)
#
#         # Return the final processed tokens
#         return x
"""
第二波修改+agent attention
"""

# '''
# add token transfer to feature
# '''
# def token2feature(tokens):
#     B,L,D=tokens.shape
#     H=W=int(L**0.5)
#     x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
#     return x
#
#
# '''
# feature2token
# '''
# def feature2token(x):
#     B,C,W,H = x.shape
#     L = W*H
#     tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
#     return tokens
#
# class TransformerModel(nn.Module):
#     def __init__(
#             self,
#             dim,  # 512
#             depth,  # 4
#             heads,  # 8
#             mlp_dim,  # 4096
#              # The prompt module
#             dropout_rate=0.1,
#             attn_dropout_rate=0.1,
#     ):
#         super().__init__()
#         self.layers = nn.ModuleList([
#             nn.ModuleDict({
#                 "transformer": nn.ModuleList([
#                     Residual(
#                         PreNormDrop(
#                             dim,
#                             dropout_rate,
#                             SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
#                         )
#                     ),
#                     Residual(
#                         PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
#                     )
#                 ]),
#                 "prompt": Prompt_block(),  # Assuming P is a class that processes the input
#             }) for _ in range(depth)
#         ])
#
#     def forward(self, x, polar_AOP,polar_DOP):
#         """
#         x: Tensor of shape [B, N, D] (Main modality)
#         polar: Tensor of shape [B, N, D] (Auxiliary modality)
#         """
#         # Step 1: First layer initialization
#         # Convert tokens to features for both modalities
#         x_feat = token2feature(x)  # 1,512,16,16
#         polar_feat_a = token2feature(polar_AOP)  # 1,512,16,16
#         polar_feat_d = token2feature(polar_DOP) #1,512,16,16
#
#         # Combine main and auxiliary modalities and process through the first prompt module
#         prompt_output = torch.cat([x_feat, polar_feat_a,polar_feat_d], dim=1)  # 1,1536,16,16 Concatenate main and auxiliary modalities
#         prompt_output = self.layers[0]["prompt"](prompt_output)  # 1,512,16,16 P1 processes the combined features
#
#         # Convert prompt output back to token format and combine with main modality
#         x = feature2token(prompt_output) # 1,256,512
#         x = x + feature2token(x_feat)  #1,256,512 Combine with the original main modality features
#
#         # Step 2: Process through the first Transformer layer
#         for transformer_layer in self.layers[0]["transformer"]:
#              x = transformer_layer(x)
#
#         # Step 3: Subsequent layers processing
#         for layer in self.layers[1:]:
#             prompt = layer["prompt"]
#
#             # Convert tokens back to features for prompt processing
#             x_feat = token2feature(x)
#
#             # Combine current features with previous prompt output and process through the prompt module
#             prompt_output = torch.cat([x_feat, prompt_output], dim=1)  #1,1024,16,16 Combine current features and previous prompt
#             prompt_output = prompt(prompt_output)  # Process through P^L  1,512,16,16
#
#             # Convert prompt output back to token format
#             x = feature2token(prompt_output)+x
#
#             # Process through the Transformer layers
#             for transformer_layer in layer["transformer"]:
#                 x = transformer_layer(x)
#
#         # Return the final processed tokens
#         return x
"""
第三次修改           -四进三出
"""

'''
add token transfer to feature
'''
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

class TransformerModel(nn.Module):
    def __init__(
            self,
            dim,  # 512
            depth,  # 4
            heads,  # 8
            mlp_dim,  # 4096
             # The prompt module
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        # Initialize a random parameter with the same shape as AOP (1, 256, 512)
        self.random_param = nn.Parameter(torch.randn(6, 256, 512))  # 6和bach size一致
        # 改为 register_buffer：固定随机值，不参与训练
        # self.register_buffer("random_param", torch.randn(1, 256, 512))  # 形状不变，但不可训练
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "transformer": nn.ModuleList([
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    )
                ]),
                "prompt": Prompt_block(),  # Assuming P is a class that processes the input
            }) for _ in range(depth)
        ])

    def forward(self, x, polar_AOP,polar_DOP,device):
        """
        x: Tensor of shape [B, N, D] (Main modality)
        polar: Tensor of shape [B, N, D] (Auxiliary modality)
        """
        # Step 1: First layer initialization
        # Convert tokens to features for both modalities
        x_feat = token2feature(x)  # 1,512,16,16
        polar_feat_a = token2feature(polar_AOP)  # 1,512,16,16
        polar_feat_d = token2feature(polar_DOP) #1,512,16,16


        # Combine main and auxiliary modalities and process through the first prompt module
        prompt_output = torch.cat([x_feat, polar_feat_a,polar_feat_d], dim=1)  # 1,1536,16,16 Concatenate main and auxiliary modalities

        # Add the random parameter to the prompt output
        random_feat = token2feature(self.random_param)  # Convert random parameter to feature format
        if prompt_output.shape[0] == random_feat.shape[0]:
            prompt_output = torch.cat([prompt_output, random_feat], dim=1)  # 1,2048,16,16 Concatenate with random parameter
        else:
            random_feat=token2feature(nn.Parameter(torch.randn(prompt_output.shape[0], 256, 512)))
            random_feat = random_feat.to(device)
            prompt_output = torch.cat([prompt_output, random_feat], dim=1)
        prompt_output, prompt_a,prompt_d= self.layers[0]["prompt"](prompt_output)  # 1,512,16,16 ×3

        # Convert prompt output back to token format and combine with main modality
        x = feature2token(prompt_output) # 1,256,512
        x = x + feature2token(x_feat)  #1,256,512 Combine with the original main modality features

        # Step 2: Process through the first Transformer layer
        for transformer_layer in self.layers[0]["transformer"]:
             x = transformer_layer(x)

        # Step 3: Subsequent layers processing
        for layer in self.layers[1:]:
            prompt = layer["prompt"]

            # Convert tokens back to features for prompt processing
            x_feat = token2feature(x)

            # Combine current features with previous prompt output and process through the prompt module
            prompt_output = torch.cat([x_feat, prompt_output,prompt_a,prompt_d], dim=1)  #1,1024,16,16 Combine current features and previous prompt
            prompt_output,prompt_a,prompt_d = prompt(prompt_output)  # Process through P^L  1,512,16,16

            # Convert prompt output back to token format
            x = feature2token(prompt_output)+x

            # Process through the Transformer layers
            for transformer_layer in layer["transformer"]:
                x = transformer_layer(x)

        # Return the final processed tokens
        return x
