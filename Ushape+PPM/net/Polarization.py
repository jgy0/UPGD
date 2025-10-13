# -*- coding: utf-8 -*-
# @Author  : guangyao Ju
# @File    : Polarization.py
# coding=utf-8
# Design based on the CTrans
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair



class PolarizationPreprocess(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PolarizationPreprocess, self).__init__()
        # 使用卷积层提取特征
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, polarization_info):
        # polarization_info 的维度: (batch_size, 6, 256, 256)
        x = self.relu(self.conv1(polarization_info))
        x = self.relu(self.conv2(x))
        return x  # 输出维度: (batch_size, out_channels, 256, 256)

class PolarizationAttention(nn.Module):
    def __init__(self, channels):
        super(PolarizationAttention, self).__init__()
        self.channels = channels
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, polarization_info):
        # polarization_info 的维度: (batch_size, 6, 256, 256)
        attention = self.channel_attention(polarization_info)
        return polarization_info * attention  # 增强后的偏振信息

class PolarizationMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PolarizationMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, polarization_info):
        # polarization_info 的维度: (batch_size, 6, 256, 256)
        return self.mlp(polarization_info)  # 输出维度: (batch_size, out_channels)


import torch
import torch.nn as nn


class PolarizationPreprocessCombined(nn.Module):
    def __init__(self, in_channels=6, out_channels=64, token_dim=480):
        super(PolarizationPreprocessCombined, self).__init__()
        self.token_dim = token_dim

        # 卷积预处理
        self.conv_preprocess = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # MLP 全局编码
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(out_channels, token_dim),
            nn.ReLU(),
            nn.Linear(token_dim, token_dim))

        # 线性层调整维度
        self.final_proj = nn.Linear(256 * 256, 480)  # 将空间维度压缩为 64

    def forward(self, polarization_info):
        # 输入 polarization_info 的维度: (batch_size, 6, 256, 256)
        batch_size = polarization_info.size(0)

        # 1. 卷积提取局部特征
        x = self.conv_preprocess(polarization_info)  # (batch_size, 64, 256, 256)

        # 2. 注意力机制增强特征
        attention = self.attention(x)  # (batch_size, 64, 1, 1)
        x = x * attention  # (batch_size, 64, 256, 256)

        # 3. MLP 全局编码
        global_feature = self.mlp(x)  # (batch_size, 480)
        global_feature = global_feature.unsqueeze(1).expand(-1, 64, -1)  # (batch_size, 64, 480)

        # 4. 将局部特征的空间维度压缩为 64
        x = x.view(batch_size, 64, -1)  # (batch_size, 64, 256*256)
        x = self.final_proj(x)  # (batch_size, 64, 480)

        # 5. 将局部特征和全局特征结合
        x=x+global_feature

        return x  # 输出维度: (batch_size, 64, 480)