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
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
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
            BasicConv(3, 8, kernel_size=3, padding=1),
            BasicConv(8, 8, kernel_size=3, stride=2, padding=1),
            BasicConv(8, 8, kernel_size=3, padding=1))

        self.conv1a = BasicConv(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv5a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv6a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv6a = Conv2x(128, 96, deconv=True)
        self.deconv5a = Conv2x(96, 64, deconv=True)
        self.deconv4a = Conv2x(64, 48, deconv=True)
        self.deconv3a = Conv2x(48, 32, deconv=True)
        self.deconv2a = Conv2x(32, 16, deconv=True)
        self.deconv1a = Conv2x(16, 8, deconv=True)

        self.conv1b = Conv2x(8, 16)
        self.conv2b = Conv2x(16, 32)
        self.conv3b = Conv2x(32, 48)
        self.conv4b = Conv2x(48, 64)
        self.conv5b = Conv2x(64, 96)
        self.conv6b = Conv2x(96, 128)

        self.deconv6b = Conv2x(128, 96, deconv=True)
        self.deconv5b = Conv2x(96, 64, deconv=True)
        self.deconv4b = Conv2x(64, 48, deconv=True)
        self.deconv3b = Conv2x(48, 32, deconv=True)
        self.deconv2b = Conv2x(32, 16, deconv=True)
        self.deconv1b = Conv2x(16, 8, deconv=True)

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
        x = self.conv5a(x)
        rem5 = x
        x = self.conv6a(x)
        rem6 = x

        x = self.deconv6a(x, rem5)
        rem5 = x
        x = self.deconv5a(x, rem4)
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
        rem4 = x
        x = self.conv5b(x, rem5)
        rem5 = x
        x = self.conv6b(x, rem6)

        x = self.deconv6b(x, rem5)
        x = self.deconv5b(x, rem4)
        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        out1 = x
        x = self.deconv2b(x, rem1)
        out2 = x
        x = self.deconv1b(x, rem0)
        out3 = x

        return out1, out2, out3


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
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))
        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

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
        self.conv32x1 = nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def lga(self, x, g):
        g = F.normalize(g, p=1, dim=1)
        x = self.LGA2(x, g)
        return x

    def forward(self, x, lg1, lg2):
        x = self.conv32x1(x)
        # x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        assert (lg1.size() == lg2.size())
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
        self.SGA = SGA()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        rem = x
        k1, k2, k3, k4 = torch.split(g, (x.size()[1] * 5, x.size()[1] * 5, x.size()[1] * 5, x.size()[1] * 5), 1)
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
        assert (x.size() == rem.size())
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
        super(ConvRes, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, relu=False, is_3d=is_3d, kernel_size=3, stride=stride, padding=padding,
                      dilation=dilation),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

    def forward(self, x):
        out = self.conv(x)
        out = x + out
        return out


class EdgeRefine(nn.Module):
    def __init__(self):
        super(EdgeRefine, self).__init__()
        self.conv2d_feature = nn.Sequential(BasicConv(4, 32, kernel_size=3, stride=1, padding=1, relu=False),
                                            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.convRes_block = nn.Sequential(ConvRes(32, 32, stride=1, padding=2, dilation=2, is_3d=False),
                                           ConvRes(32, 32, stride=1, padding=4, dilation=4, is_3d=False),
                                           ConvRes(32, 32, stride=1, padding=8, dilation=8, is_3d=False),
                                           ConvRes(32, 32, stride=1, padding=1, dilation=1, is_3d=False),
                                           ConvRes(32, 32, stride=1, padding=1, dilation=1, is_3d=False)
                                           )
        self.conv2d_out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, disp):
        disp = torch.unsqueeze(disp,1)
        out = torch.cat([x, disp], 1)
        out = self.conv2d_feature(out)
        out = self.convRes_block(out)
        out = torch.squeeze(
            disp + self.conv2d_out(out), 1)
        # out = torch.squeeze(
        #     self.conv2d_out(out), 1)
        out = nn.ReLU(inplace=True)(out)
        return out

# concat 4D cost volume
class ResCostVolume(nn.Module):
    def __init__(self, maxdisp):
        super(ResCostVolume, self).__init__()
        self.maxdisp = maxdisp + 1

    def forward(self, x, y):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, self.maxdisp, height, width).zero_()
            for i in range(self.maxdisp):
                index = i - int(self.maxdisp - 1 / 2)
                if index < 0:
                    cost[:, :x.size()[1], i, :, :index] = x[:, :, :, :index]
                    cost[:, x.size()[1]:, i, :, :index] = y[:, :, :, -index:]
                if index > 0:
                    cost[:, :x.size()[1], i, :, index:] = x[:, :, :, index:]
                    cost[:, x.size()[1]:, i, :, index:] = y[:, :, :, :-index]
                else:
                    cost[:, i, :, :] = torch.norm(x-y, 1, 1)
            cost = cost.contiguous()
        return cost

# warping
class RebuildCostVolume(nn.Module):
    def __init__(self, maxdisp):
        super(RebuildCostVolume, self).__init__()
        self.cv = ResCostVolume(maxdisp)

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the disp
        x: [B, C, H, W] im2
        disp: [B, 1, H, W] disp
        """
        with torch.cuda.device_of(x):
            B, C, H, W = x.size()
            # mesh grid
            xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            vgrid = torch.cat((xx, yy), 1).float()

            vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp

            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

            vgrid = vgrid.permute(0, 2, 3, 1)
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        return output

    def forward(self, x, disp, y):
        y = self.warp(y, disp)
        cost = self.cv(x, y)
        return cost.contiguous()

# distance based cost volume
class RebuildCostVolume2(nn.Module):
    def __init__(self, maxdisp):
        super(RebuildCostVolume2, self).__init__()
        self.maxdisp = maxdisp

    def warp(self, x, disp):
        """
        warp an image/tensor (im2) back to im1, according to the disp
        x: [B, C, H, W] im2
        disp: [B, 1, H, W] disp
        """
        with torch.cuda.device_of(x):
            B, C, H, W = x.size()
            # mesh grid
            xx = torch.arange(0, W, device='cuda').view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H, device='cuda').view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            vgrid = torch.cat((xx, yy), 1).float()

            vgrid[:, :1, :, :] = vgrid[:, :1, :, :] - disp

            vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
            vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

            vgrid = vgrid.permute(0, 2, 3, 1)
            output = nn.functional.grid_sample(x, vgrid, align_corners=True)
        return output

    def forward(self, feat_l, feat_r, disp):
        size = feat_l.size()
        batch_disp = disp[:, None, :, :, :].repeat(1, self.maxdisp * 2 - 1, 1, 1, 1).view(-1, 1, size[-2], size[-1])
        batch_shift = torch.arange(-self.maxdisp + 1, self.maxdisp, device='cuda').repeat(size[0])[:, None, None, None]
        batch_disp = batch_disp - batch_shift.float()
        batch_feat_l = feat_l[:, None, :, :, :].repeat(1, self.maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2],
                                                                                         size[-1])
        batch_feat_r = feat_r[:, None, :, :, :].repeat(1, self.maxdisp * 2 - 1, 1, 1, 1).view(-1, size[-3], size[-2],
                                                                                         size[-1])
        cost = torch.norm(batch_feat_l - self.warp(batch_feat_r, batch_disp), 1, 1)
        cost = cost.view(size[0], -1, size[2], size[3])
        return cost.contiguous()


class ResidualPredition(nn.Module):
    def __init__(self, maxdisp, channel):
        super(ResidualPredition, self).__init__()
        self.maxdisp = maxdisp + 1
        self.conv3ds = nn.Sequential(BasicConv(1, channel, kernel_size=3, padding=1, is_3d=True),
                                     BasicConv(channel, channel, kernel_size=3, padding=1, is_3d=True),
                                     BasicConv(channel, channel, kernel_size=3, padding=1, is_3d=True),
                                     BasicConv(channel, channel, kernel_size=3, padding=1, is_3d=True),
                                     BasicConv(channel, channel, kernel_size=3, padding=1, is_3d=True),
                                     BasicConv(channel, 1, kernel_size=3, padding=1, is_3d=True)
                                     )
    def forward(self, cv,h,w):
        cv = torch.unsqueeze(cv, 1)
        cv = self.conv3ds(cv)
        cv = F.interpolate(cv, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
        x = torch.squeeze(cv, 1)
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = x.new().resize_(x.size()[0], self.maxdisp, x.size()[2], x.size()[3]).zero_()
            for i in range(self.maxdisp):
                disp[:, i, :, :] = i - int((self.maxdisp - 1) / 2)
            out = torch.sum(x * disp, 1)
        return out


class upsample(nn.Module):
    def __init__(self, h, w, keep_dim=False):
        super(upsample, self).__init__()
        self.h = int(h)
        self.w = int(w)
        self.keep_dim = keep_dim

    def forward(self, x):
        _, xh, __ = x.size()
        x = torch.unsqueeze(x, 1)
        x = F.interpolate(x, [self.h, self.w], mode='bilinear', align_corners=False)
        x = x * self.h / xh
        if self.keep_dim:
            return x
        return torch.squeeze(x, 1)


class GANet(nn.Module):
    def __init__(self, maxdisp=192):
        super(GANet, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, stride=2, padding=1),
                                        BasicConv(16, 32, kernel_size=3, stride=2, padding=1),
                                        BasicConv(32, 32, kernel_size=3, stride=2, padding=1),
                                        BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv_x = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_y = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu = nn.Sequential(apex.parallel.SyncBatchNorm(32),
                                     nn.ReLU(inplace=True))
        self.feature = Feature()
        self.guidance = Guidance()
        self.cost_agg = CostAggregation(int(self.maxdisp / 8))
        self.cv = GetCostVolume(int(self.maxdisp / 8))
        # self.Res_cv1 = RebuildCostVolume(4)
        self.Res_cv11 = RebuildCostVolume2(3)
        self.Res_Pred1 = ResidualPredition(4 * 4, 16)
        # self.Res_cv2 = RebuildCostVolume(4)
        self.Res_cv22 = RebuildCostVolume2(3)
        self.Res_Pred2 = ResidualPredition(4 * 2, 8)
        self.edge_refine = EdgeRefine()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (apex.parallel.SyncBatchNorm, apex.parallel.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        __, _, h, w = x.size()
        rem0 = x

        g = self.conv_start(x)

        x, x1, x2 = self.feature(x)
        rem = x
        y, y1, y2 = self.feature(y)

        x = self.conv_x(x)
        y = self.conv_y(y)
        x = self.cv(x, y)

        xg = self.conv_refine(rem)
        xg = self.bn_relu(xg)
        g = torch.cat((g, xg), 1)
        g = self.guidance(g)

        disp0 = self.cost_agg(x, g)
        # TODO out1
        if self.training:
            disp00 = upsample(h, w)(disp0[0])
            disp01 = upsample(h, w)(disp0[1])
            # return disp00, disp01
        else:
            disp01 = upsample(h, w)(disp0)
        return disp01

        disp1 = upsample(h / 4, w / 4, keep_dim=True)(disp01)
        res_cv1 = self.Res_cv11(x1, y1, disp1)
        disp1 = self.Res_Pred1(res_cv1, h, w)
        disp1 = disp01 + disp1
        # return disp1

        disp2 = upsample(h / 2, w / 2, keep_dim=True)(disp1)
        res_cv2 = self.Res_cv22(x2, y2, disp2)
        disp2 = self.Res_Pred2(res_cv2, h, w)
        disp2 = disp1 + disp2
        # return disp2
        # TODO out2
        # return disp00, disp01, disp1, disp2

        disp3 = self.edge_refine(rem0, disp2)
        # TODO out3
        if self.training:
            return disp00, disp01, disp1, disp2, disp3
        return disp3

