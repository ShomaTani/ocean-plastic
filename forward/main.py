import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.append(nn.Dropout2d(dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):

    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class AttentionGate(nn.Module):

    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(self.relu(g1 + x1))  # (B,1,H,W)
        return x * alpha


class UpSample(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.attn = AttentionGate(g_ch=in_ch // 2, x_ch=skip_ch, inter_ch=skip_ch // 2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x, skip):
        x = self.up(x)                   # 解像度2倍 = decoderのgating signal
        skip = self.attn(g=x, x=skip)     # skip connectionにattentionをかける
        x = torch.cat([x, skip], dim=1)   # channel方向に結合
        return self.conv(x)


class AttentionUNet(nn.Module):
    """down x4 -> ConvBlock(bottleneck) -> up x4(各段でAttentionGate)
    -> 1x1 Convで出力チャンネルを1に落とす。
    loss functionによる柔軟性確保のためforwardの戻り値はロジット(生の値、活性化なし)。
    """

    def __init__(self, in_ch=1, out_ch=1, base_ch=32, dropout_p=0.0):
        super().__init__()
        self.down1 = DownSample(in_ch, base_ch, dropout_p=dropout_p)
        self.down2 = DownSample(base_ch, base_ch * 2, dropout_p=dropout_p)
        self.down3 = DownSample(base_ch * 2, base_ch * 4, dropout_p=dropout_p)
        self.down4 = DownSample(base_ch * 4, base_ch * 8, dropout_p=dropout_p)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16, dropout_p=dropout_p)

        self.up4 = UpSample(base_ch * 16, base_ch * 8, base_ch * 8, dropout_p=dropout_p)
        self.up3 = UpSample(base_ch * 8, base_ch * 4, base_ch * 4, dropout_p=dropout_p)
        self.up2 = UpSample(base_ch * 4, base_ch * 2, base_ch * 2, dropout_p=dropout_p)
        self.up1 = UpSample(base_ch * 2, base_ch, base_ch, dropout_p=dropout_p)

        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down1(x)   # 128 -> 64
        x, skip2 = self.down2(x)   # 64 -> 32
        x, skip3 = self.down3(x)   # 32 -> 16
        x, skip4 = self.down4(x)   # 16 -> 8

        x = self.bottleneck(x)     # 8x8のまま、channelだけ増やす

        x = self.up4(x, skip4)     # 8 -> 16
        x = self.up3(x, skip3)     # 16 -> 32
        x = self.up2(x, skip2)     # 32 -> 64
        x = self.up1(x, skip1)     # 64 -> 128

        return self.out_conv(x)    # (B, 1, 128, 128) ロジット

