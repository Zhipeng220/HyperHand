import torch
import torch.nn as nn
import numpy as np
import math
from torch.autograd import Variable


# ---------------------------------------------------------------------------------
# (A) 辅助模块 (从 net/ctrgcn.py 复制)
# ---------------------------------------------------------------------------------

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=False)  # [ MODIFIED ]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y = y + self.down(x)  # [ MODIFIED ] (Avoided inplace +=)
        y = self.relu(y)
        return y


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # [ MODIFIED ]
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


# ---------------------------------------------------------------------------------
# (B) MAE 解码器 (Decoder) - 已修正
# ---------------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, A, kernel_size=5, stride=1, adaptive=True):
        super().__init__()
        self.up_tcn = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0),
            stride=(2, 1),
            output_padding=(1, 0)
        )
        self.bn_t = nn.BatchNorm2d(in_channels)
        self.gcn = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.residual = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(2, 1),
                output_padding=(1, 0)
            ),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=False)  # [ MODIFIED ]

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.bn_t(self.up_tcn(x)))
        x = self.gcn(x)
        x = x + res  # [ MODIFIED ] (Avoided inplace +=)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels=256, out_channels=3, A=None, adaptive=True, num_person=2):
        super().__init__()

        if A is None:
            raise ValueError("Decoder 必须接收一个图 A")

        self.num_person = num_person
        self.out_channels = out_channels

        self.l1 = DecoderLayer(in_channels, in_channels // 2, A, adaptive=adaptive)
        self.l2 = DecoderLayer(in_channels // 2, in_channels // 4, A, adaptive=adaptive)

        self.final_proj = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.normal_(self.mask_token, std=.02)

    def forward(self, z, mask):
        N_M, C_feat, T_feat, V = z.shape

        mask_expanded = mask.view(1, 1, 1, V)
        z_masked = z * (~mask_expanded)

        mask_tokens = self.mask_token.expand(N_M, C_feat, T_feat, V)
        mask_tokens = mask_tokens * mask_expanded

        z_full = z_masked + mask_tokens

        d1 = self.l1(z_full)
        d2 = self.l2(d1)

        x_hat_flat = self.final_proj(d2)

        T_out = x_hat_flat.shape[2]
        V_out = x_hat_flat.shape[3]

        N_loader = N_M // self.num_person

        x_hat = x_hat_flat.view(N_loader, self.num_person, self.out_channels, T_out, V_out)

        x_hat = x_hat.permute(0, 2, 3, 4, 1)

        return x_hat