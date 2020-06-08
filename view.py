# # from models.MyGANet4 import GANet
# #
# # model = GANet()
# # for name, module in model.named_children():
# #     print(name)
#
# import torch
# import torch.nn as nn
#
# a = torch.randn(2, 3, 2, 2)   # 右图
# b = torch.ones(2, 1, 2, 2)   # disp
# print(a)
#
# def warp(x, disp):
#     """
#     warp an image/tensor (im2) back to im1, according to the optical flow
#     x: [B, C, H, W] (im2)
#     flo: [B, 2, H, W] flow
#     """
#     B, C, H, W = x.size()
#     # mesh grid
#     xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
#     yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
#     xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
#     yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
#     vgrid = torch.cat((xx, yy), 1).float()
#
#     # vgrid = Variable(grid)
#     vgrid[:, :1, :, :] = vgrid[:, :1, :, :] + disp
#
#     # scale grid to [-1,1]
#     vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
#     vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
#
#     vgrid = vgrid.permute(0, 2, 3, 1)
#     output = nn.functional.grid_sample(x, vgrid,align_corners=True)
#     return output
#
# o = warp(a,b)
#
# print(o)

from models.CasGANet10 import GANet
import numpy as np
import datetime
import torch

model = GANet()


print('parameters:{}'.format(np.sum([p.numel() for p in model.parameters()]).item()))

model = torch.nn.DataParallel(model).cuda()
model.eval()
input1 = torch.randn(1, 3, 384, 768).cuda()
input2 = torch.randn(1, 3, 384, 768).cuda()

t = 0.
for i in range(10):
    with torch.no_grad():
        start = datetime.datetime.now()
        out1 = model(input1, input2)
        end = datetime.datetime.now()
        t += (end - start).total_seconds()
print(t/10)

