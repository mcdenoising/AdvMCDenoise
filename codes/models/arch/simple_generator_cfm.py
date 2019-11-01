'''
architecture for CFM
'''
import torch.nn as nn
import torch.nn.functional as F


class CFMLayer(nn.Module):
    def __init__(self):
        super(CFMLayer, self).__init__()
        self.CFM_scale_conv0 = nn.Conv2d(32, 32, 1)
        self.CFM_scale_conv1 = nn.Conv2d(32, 64, 1)
        self.CFM_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.CFM_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.CFM_scale_conv1(F.leaky_relu(self.CFM_scale_conv0(x[1]), 0.1, inplace=True))
        shift = self.CFM_shift_conv1(F.leaky_relu(self.CFM_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (scale + 1) + shift


class ResBlock_CFM(nn.Module):
    def __init__(self):
        super(ResBlock_CFM, self).__init__()
        self.CFM0 = CFMLayer()
        self.conv0 = nn.Conv2d(64, 64, 3, 1, 1)
        self.CFM1 = CFMLayer()
        self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.CFM0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.CFM1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions

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

class CFM_Network(nn.Module):
    def __init__(self):
        super(CFM_Network, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        resnet_branch = []
        for i in range(15):
            resnet_branch.append(ResBlock())
        self.resnet_branch  = nn.Sequential(*resnet_branch)    
        CFM_branch = []
        CFM_branch.append(ResBlock_CFM())    
        CFM_branch.append(CFMLayer())
        CFM_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.CFM_branch = nn.Sequential(*CFM_branch)

        self.GT_branch = nn.Sequential(

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.ReLU(True)
        )


        self.CondNet = nn.Sequential(
            # nn.Conv2d(7, 128, 4, 4), #CHANGED from 8 to 7
            nn.Conv2d(8, 128, 3, 1, 1), #CHANGED from 8 to 7
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 32, 1)
        )



    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1]) #CHANGED
        fea = self.conv0(x[0])
        res = self.resnet_branch(fea)
        res = self.CFM_branch((res, cond))
        fea = fea + res
        out = self.GT_branch(fea)
        return out
