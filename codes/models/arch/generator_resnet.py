import math
import torch
import torch.nn as nn
import torchvision
from . import block as B
import functools
import torch.nn.functional as F


class ResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1):
        super(DENOISEDResNet, self).__init__()

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        NOISY_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)
        GT_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        GT_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)
        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, NOISY_conv)), GT_conv0, GT_conv1)

    def forward(self, x):
        x = self.model(x)
        return x


class ResBlock(nn.Module):
    def __init__(self):  # no normalization is it ok?
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        res = x
        x = F.relu(self.conv0(x), inplace=True)
        x = self.conv1(x)
        return res + x



class ResBlock_128(nn.Module):
    def __init__(self):  # no normalization is it ok?
        super(ResBlock_128, self).__init__()
        self.conv0 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        res = x
        x = F.relu(self.conv0(x), inplace=True)
        x = self.conv1(x)
        return res + x

class ResNet_no_Condition(nn.Module):
    def __init__(self):
        super(ResNet_no_Condition, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        resnet_branch = []
        for i in range(16):
            resnet_branch.append(ResBlock())
        resnet_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.resnet_branch = nn.Sequential(*resnet_branch)

        self.Final_stage = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        # x[0]: img; x[1]: seg
        fea = self.conv0(x[0])
        res = self.resnet_branch(fea)
        fea = fea + res
        out = self.GT_branch(fea)
        # out = out + x[0]
        # out = F.relu(out)
        return out



class ResNet_Condition(nn.Module):
    def __init__(self):
        super(ResNet_Condition, self).__init__()
        self.conv0 = nn.Conv2d(10, 64, 3, 1, 1)

        resnet_branch = []
        for i in range(16):
            resnet_branch.append(ResBlock())
        resnet_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.resnet_branch = nn.Sequential(*resnet_branch)

        # self.CondNet = nn.Sequential(
        #     # nn.Conv2d(7, 128, 4, 4), #CHANGED from 8 to 7
        #     nn.Conv2d(7, 128, 3, 1, 1), #CHANGED from 8 to 7
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(128, 128, 1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(128, 128, 1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(128, 128, 1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(128, 32, 1)
        # )

        # self.after_concat = nn.Sequential( # this is to control the the parameters for the ablation
        #     nn.Conv2d(10, 64, 3, 1, 1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(128, 128, 3, 1, 1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(128, 128, 3, 1, 1)
        # )

        self.Final_stage = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.ReLU(True)
        )

    def forward(self, x):
        # x[0]: img; x[1]: seg
        # fea = self.conv0(x[0])
        # cond = self.CondNet(x[1]) 
        fea = x[0]
        # fea1 = self.conv0(fea)
        cond = x[1]
        concat = torch.cat((fea,cond) , 1)
        concat = self.conv0(concat)
        # concat = self.after_concat(concat)
        res = self.resnet_branch(concat)
        concat = concat + res
        out = self.Final_stage(concat)
        # out = out + x[0]
        # out = F.relu(out)
        return out