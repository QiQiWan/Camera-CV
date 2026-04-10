# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(out))
        return x * attention


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class CannyEdgeModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.in_channels = in_channels

    def forward(self, x):
        _batch_size, channels, _h, _w = x.shape
        edge_maps = []
        for c in range(channels):
            x_c = x[:, c:c + 1, :, :]
            grad_x = F.conv2d(x_c, self.sobel_x, padding=1)
            grad_y = F.conv2d(x_c, self.sobel_y, padding=1)
            edge_c = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            edge_maps.append(edge_c)
        edge_magnitude = torch.cat(edge_maps, dim=1)
        edge_magnitude = edge_magnitude / (edge_magnitude.max() + 1e-6)
        enhanced = x + self.fusion_weight * edge_magnitude
        return enhanced


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, use_attention=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.attention = SpatialAttention() if use_attention else nn.Identity()
        self.channel_attention = ChannelAttention(out_ch) if use_attention else nn.Identity()
        self.use_attention = use_attention
        self.skip_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip_conv(x)
        x = self.conv(x)
        if self.use_attention:
            x = self.attention(x)
            x = self.channel_attention(x)
        return x + identity


class ThinCrackUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super().__init__()
        features = features or [32, 64, 128, 256]
        self.features = features
        self.enc1 = DoubleConv(in_channels, features[0])
        self.enc2 = DoubleConv(features[0], features[1])
        self.enc3 = DoubleConv(features[1], features[2])
        self.enc4 = DoubleConv(features[2], features[3])
        self.edge_enhance = CannyEdgeModule(features[3])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(features[3], features[2])
        self.up3 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(features[2], features[1])
        self.up2 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(features[1], features[0])
        fusion_channels = features[0] + features[0] + features[1] + features[2]
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(features[0], 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        original_size = x.shape[2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e4_enhanced = self.edge_enhance(e4)
        d4 = self.up4(e4_enhanced)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e3], dim=1)
        d4 = self.dec4(d4)
        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        d3_up = F.interpolate(d3, size=original_size, mode='bilinear', align_corners=False)
        d4_up = F.interpolate(d4, size=original_size, mode='bilinear', align_corners=False)
        multi_scale = torch.cat([e1, d2, d3_up, d4_up], dim=1)
        fused = self.multi_scale_fusion(multi_scale)
        return self.final(fused)
