import torch.nn as nn
import torch.nn.functional as F


class CFMLayer(nn.Module):
    def __init__(self):
        super(CFMLayer, self).__init__()
        self.CFM_shift_conv0 = nn.Conv2d(32, 32, 1)
        self.CFM_shift_conv1 = nn.Conv2d(32, 64, 1)

    def forward(self, x):
        # # x[0]: fea; x[1]: cond
        shift = self.CFM_shift_conv1(F.leaky_relu(self.CFM_shift_conv0(x[1]), 0.1, inplace=True))
        return x[0] * (1 + 1) + shift


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


class CFM_Network(nn.Module):
    def __init__(self):
        super(CFM_Network, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        CFM_branch = []
        for i in range(16):
            CFM_branch.append(ResBlock_CFM())
        CFM_branch.append(CFMLayer())
        CFM_branch.append(nn.Conv2d(64, 64, 3, 1, 1))
        self.CFM_branch = nn.Sequential(*CFM_branch)

        self.Final_stage = nn.Sequential(
    
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.ReLU(True)
        )


        self.Condition_process = nn.Sequential(
            # nn.Conv2d(7, 128, 4, 4), #CHANGED from 8 to 7
            nn.Conv2d(7, 128, 3, 1, 1), #CHANGED from 8 to 7
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
        cond = self.Condition_process(x[1]) #CHANGED
        fea = self.conv0(x[0])
        res = self.CFM_branch((fea, cond))
        fea = fea + res
        out = self.Final_stage(fea)
        return out

