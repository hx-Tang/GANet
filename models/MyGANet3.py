import torch
import torch.nn as nn
import torch.nn.init as init
from libs.GANet.modules.GANet import DisparityRegression, GetCostVolume
from libs.GANet.modules.GANet import MyNormalize
from libs.GANet.modules.GANet import SGA
from libs.GANet.modules.GANet import LGA, LGA2, LGA3
# from libs.sync_bn.modules.sync_bn import BatchNorm2d, BatchNorm3d
import torch.nn.functional as F
import apex
from torch.autograd import Variable
import numpy as np
class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
#        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = apex.parallel.SyncBatchNorm(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = apex.parallel.SyncBatchNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x

class Guidance(nn.Module):
    def __init__(self):
        super(Guidance, self).__init__()

        self.conv0 = BasicConv(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(16, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)

        self.conv11 = nn.Sequential(BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                                    BasicConv(48, 48, kernel_size=3, padding=1))
        self.conv12 = BasicConv(48, 48, kernel_size=3, padding=1)

        self.weight_sg1 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg2 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_sg11 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg12 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1) ,bias=False))
        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1) ,bias=False))

    def forward(self, x):
        x = self.conv0(x)
        rem = x
        x = self.conv1(x)
        sg1 = self.weight_sg1(x)
        x = self.conv2(x)
        sg2 = self.weight_sg2(x)

        x = self.conv11(x)
        sg11 = self.weight_sg11(x)
        x = self.conv12(x)
        sg12 = self.weight_sg12(x)

        lg1 = self.weight_lg1(rem)
        lg2 = self.weight_lg2(rem)

        return dict([
            ('sg1', sg1),
            ('sg2', sg2),
            ('sg11', sg11),
            ('sg12', sg12),
            ('lg1', lg1),
            ('lg2', lg2)])

class Disp(nn.Module):

    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=int(self.maxdisp))
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        self.conv32x1 = nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def forward(self, x):
        x = self.conv32x1(x)
        # x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)
        return self.disparity(x)

class DispAgg(nn.Module):

    def __init__(self, maxdisp=192):
        super(DispAgg, self).__init__()
        self.maxdisp = maxdisp
        self.LGA3 = LGA3(radius=2)
        self.LGA2 = LGA2(radius=2)
        self.LGA = LGA(radius=2)
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=int(self.maxdisp))
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        self.conv32x1=nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def lga(self, x, g):
        g = F.normalize(g, p=1, dim=1)
        x = self.LGA2(x, g)
        return x

    def forward(self, x, lg1, lg2):
        x = self.conv32x1(x)
        # x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        assert(lg1.size() == lg2.size())
        x = self.lga(x, lg1)
        x = self.softmax(x)
        x = self.lga(x, lg2)
        x = F.normalize(x, p=1, dim=1)
        return self.disparity(x)

class SGABlock(nn.Module):
    def __init__(self, channels=32, refine=False):
        super(SGABlock, self).__init__()
        self.refine = refine
        if self.refine:
            self.bn_relu = nn.Sequential(apex.parallel.SyncBatchNorm(channels),
                                         nn.ReLU(inplace=True))
            self.conv_refine = BasicConv(channels, channels, is_3d=True, kernel_size=3, padding=1, relu=False)
#            self.conv_refine1 = BasicConv(8, 8, is_3d=True, kernel_size=1, padding=1)
        else:
            self.bn = apex.parallel.SyncBatchNorm(channels)
        self.SGA=SGA()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, g):
        rem = x
        k1, k2, k3, k4 = torch.split(g, (x.size()[1]*5, x.size()[1]*5, x.size()[1]*5, x.size()[1]*5), 1)
        k1 = F.normalize(k1.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k2 = F.normalize(k2.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k3 = F.normalize(k3.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k4 = F.normalize(k4.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        x = self.SGA(x, k1, k2, k3, k4)
        if self.refine:
            x = self.bn_relu(x)
            x = self.conv_refine(x)
        else:
            x = self.bn(x)
        assert(x.size() == rem.size())
        x += rem
        return self.relu(x)
#        return self.bn_relu(x)


class CostAggregation(nn.Module):
    def __init__(self, maxdisp=192):
        super(CostAggregation, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1, relu=False)

        self.conv1a = BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1)
#        self.conv3a = BasicConv(64, 96, is_3d=True, kernel_size=3, stride=2, padding=1)

        self.deconv1a = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2a = Conv2x(64, 48, deconv=True, is_3d=True)
#        self.deconv0a = Conv2x(8, 8, deconv=True, is_3d=True)

        self.sga1 = SGABlock(refine=True)
        self.sga2 = SGABlock(refine=True)

        self.sga11 = SGABlock(channels=48, refine=True)
        self.sga12 = SGABlock(channels=48, refine=True)

        self.disp0 = Disp(self.maxdisp)
        self.disp1 = DispAgg(self.maxdisp)


    def forward(self, x, g):

        x = self.conv_start(x)
        x = self.sga1(x, g['sg1'])
        rem0 = x

        if self.training:
            disp0 = self.disp0(x)

        x = self.conv1a(x)
        x = self.sga11(x, g['sg11'])
        rem1 = x
        x = self.conv2a(x)
        x = self.deconv2a(x, rem1)
        x = self.sga12(x, g['sg12'])
        x = self.deconv1a(x, rem0)
        x = self.sga2(x, g['sg2'])
        disp1 = self.disp1(x, g['lg1'], g['lg2'])

        if self.training:
            return disp0, disp1
        else:
            return disp1

class ConvRes(nn.Module):
    def __init__(self, in_channel, out_channel, stride, padding, dilation, is_3d=False):
        super(ConvRes,self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel,relu=False, is_3d=is_3d, kernel_size=3, stride=stride, padding=padding, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        out = self.conv(x)
        out = x + out
        return out


class EdgeRefine(nn.Module):
    def __init__(self):
        super(EdgeRefine,self).__init__()
        self.conv2d_feature = BasicConv(4, 32, kernel_size=3, stride=1, padding=1)
        self.convRes_block = nn.Sequential(ConvRes(32, 32, stride=1, padding=2, dilation=2, is_3d=False),
                                           ConvRes(32, 32, stride=1, padding=4, dilation=4, is_3d=False),
                                           ConvRes(32, 32, stride=1, padding=8, dilation=8, is_3d=False),
                                           ConvRes(32, 32, stride=1, padding=1, dilation=1, is_3d=False),
                                           ConvRes(32, 32, stride=1, padding=1, dilation=1, is_3d=False)
                                           )
        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, disp):
        out = torch.cat([x, disp], 1)
        out = self.conv2d_feature(out)
        out = self.convRes_block(out)
        out = torch.squeeze(
            disp + self.conv2d_out(out), 1)
        out = nn.ReLU(inplace=True)(out)
        return out


class GANet(nn.Module):
    def __init__(self, maxdisp=192):
        super(GANet, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, stride=2, padding=1),
                                        BasicConv(16, 32, kernel_size=3, stride=2, padding=1),
                                        BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv_x = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_y = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1, 1), (1,1), bias=False)
        self.bn_relu = nn.Sequential(apex.parallel.SyncBatchNorm(32),
                                     nn.ReLU(inplace=True))
        self.feature = Feature()
        self.guidance = Guidance()
        self.cost_agg = CostAggregation(int(self.maxdisp/4))
        self.cv = GetCostVolume(int(self.maxdisp/4))
        self.edge_refine = EdgeRefine()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (apex.parallel.SyncBatchNorm, apex.parallel.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        rem0 = x

        g = self.conv_start(x)

        x = self.feature(x)
        rem = x
        y = self.feature(y)

        x = self.conv_x(x)
        y = self.conv_y(y)

        x = self.cv(x,y)

        x1 = self.conv_refine(rem)
        x1 = self.bn_relu(x1)
        g = torch.cat((g, x1), 1)
        g = self.guidance(g)

        disp = self.cost_agg(x, g)

        disp = F.interpolate(torch.unsqueeze(disp, 1), [x1.size()[2] * 4, x1.size()[3] * 4], mode='bilinear',
                              align_corners=False) * 4

        disp1 = self.edge_refine(rem0, disp)

        disp = torch.squeeze(disp, 1)

        # return disp, disp, disp1
        return disp1

        # if self.training:
        #     disp = list(disp)
        #     disp[0] = F.interpolate(torch.unsqueeze(disp[0],1), [x1.size()[2] * 4, x1.size()[3] * 4], mode='bilinear', align_corners=False)*4
        #     disp[0] = torch.squeeze(disp[0],1)
        #     disp[1] = F.interpolate(torch.unsqueeze(disp[1],1), [x1.size()[2] * 4, x1.size()[3] * 4], mode='bilinear', align_corners=False)*4
        #     # disp.append(self.edge_refine(rem0, disp[1]))
        #     disp[1] = torch.squeeze(disp[1], 1)
        #     disp.append(disp[1])
        # else:
        #     disp = F.interpolate(torch.unsqueeze(disp,1), [x1.size()[2] * 4, x1.size()[3] * 4], mode='bilinear',align_corners=False)*4
        #     disp0 = self.edge_refine(rem0, disp)
        #     disp = torch.squeeze(disp, 1)
        # return disp
