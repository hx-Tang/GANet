from __future__ import print_function
import argparse

from libs.GANet.modules.GANet import MyLoss2
import sys
import shutil
import os
import time
import matplotlib.pyplot as plt
import torch

import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.nn.functional as F
from dataloader.data import get_training_set, get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--crop_height', type=int, required=True, help="crop height")
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--crop_width', type=int, required=True, help="crop width")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=4, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2048, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--shift', type=int, default=0, help='random shift of left image. Default=0')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--training_list', type=str, default='./lists/sceneflow_train.list', help="training list")
parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
parser.add_argument('--save_path', type=str, default='./checkpoint/', help="location to save models")
parser.add_argument('--model', type=str, default='GANet_deep', help="model to train")

opt = parser.parse_args()

print(opt)
if opt.model == 'GANet11':
    from models.GANet11 import GANet
elif opt.model == 'GANet_deep':
    from models.GANet_deep import GANet
elif opt.model == 'MyGANet':
    from models.tests.MyGANet import GANet
elif opt.model == 'MyGANet2':
    from models.tests.MyGANet2 import GANet
elif opt.model == 'MyGANet3':
    from models.tests.MyGANet3 import GANet
elif opt.model == 'MyGANet4':
    from models.tests.MyGANet4 import GANet
elif opt.model == 'MyGANet4_8' or opt.model == 'MyGANet4_8_t1':
    from models.tests.MyGANet4_8 import GANet
elif opt.model == 'MyGANet4_8_rf' or opt.model == 'MyGANet4_8_rf_t1':
    from models.tests.MyGANet4_8_rf import GANet
elif opt.model == 'MyGANet5' or opt.model == 'MyGANet5_t1':
    from models.tests.MyGANet5 import GANet
elif opt.model == 'MyGANet9'or opt.model == 'MyGANet9_t1' or opt.model == 'MyGANet9_t2'or opt.model == 'MyGANet9_t3':
    from models.MyGANet9 import GANet
elif opt.model == 'CasGANet10':
    from models.CasGANet10 import GANet
else:
    raise Exception("No suitable model found ...")

cuda = opt.cuda
# cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.data_path, opt.training_list, [opt.crop_height, opt.crop_width], opt.left_right,
                             opt.kitti, opt.kitti2015, opt.shift)
test_set = get_test_set(opt.data_path, opt.val_list, [256, 512], opt.left_right, opt.kitti, opt.kitti2015)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                                  drop_last=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = GANet(opt.max_disp)

# skip layers for Hierarchical training
if opt.model == 'MyGANet3' or opt.model == 'MyGANet4' or opt.model == 'MyGANet4_8'\
        or opt.model == 'MyGANet4_8_rf_t3'or opt.model == 'MyGANet9_t3':
    open_layers = ['edge_refine']
    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

if opt.model == 'MyGANet5' or opt.model == 'MyGANet4_8_rf':
    open_layers = ['Res_cv1', 'Res_cv2', 'Res_Pred11', 'Res_Pred22','conv_x11','conv_y11','conv_x22','conv_y22']
    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
    model.feature.deconv1b.train()
    model.feature.deconv1b.requires_grad = True
    model.feature.deconv2b.train()
    model.feature.deconv2b.requires_grad = True
if opt.model == 'MyGANet9_t2':
    open_layers = ['Res_cv1', 'Res_cv2', 'Res_Pred1', 'Res_Pred2']
    for name, module in model.named_children():
        if name in open_layers:
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
    model.feature.deconv1b.train()
    model.feature.deconv1b.requires_grad = True
    model.feature.deconv2b.train()
    model.feature.deconv2b.requires_grad = True

criterion = MyLoss2(thresh=3, alpha=2)
if cuda:
    model = torch.nn.DataParallel(model).cuda()

# optimizer=optim.Adam(model.parameters(), lr=opt.lr,betas=(0.9,0.999))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, betas=(0.9, 0.999))

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        # model_dict = model.state_dict()
        # checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        # model_dict.update(checkpoint)
        # model.load_state_dict(model_dict)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    #        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))


def train(epoch):
    epoch_time = time.time()
    epoch_loss = 0
    epoch_error0 = 0
    epoch_error1 = 0
    epoch_error2 = 0
    valid_iteration = 0

    if opt.model == 'MyGANet3' or opt.model == 'MyGANet4' or opt.model == 'MyGANet4_8' or opt.model == 'MyGANet5'\
            or opt.model == 'MyGANet4_8_rf'or opt.model == 'MyGANet9_t2'or opt.model == 'MyGANet9_t3':
        pass
    else:
        model.train()

    for iteration, batch in enumerate(training_data_loader):
        input1, input2, target = Variable(batch[0], requires_grad=True), Variable(batch[1],
                                                                                  requires_grad=True), Variable(
            batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
            target = target.cuda()

        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]

        start_full_time = time.time()
        if valid > 0:
            optimizer.zero_grad()
            # T1 train
            if opt.model == 'GANet11' or opt.model == 'MyGANet' or opt.model == 'MyGANet2' or opt.model == 'MyGANet4_8_t1' \
                    or opt.model == 'MyGANet5_t1' or opt.model == 'MyGANet4_8_rf_t1'or opt.model == 'MyGANet9_t1':
                disp1, disp2 = model(input1, input2)
                disp0 = (disp1 + disp2) / 2.
                if opt.kitti or opt.kitti2015:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * criterion(
                        disp2[mask], target[mask])
                else:
                    loss = 0.4 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + 1.2 * F.smooth_l1_loss(
                        disp2[mask], target[mask], reduction='mean')
            # T2 train
            elif opt.model == 'MyGANet5'or opt.model == 'MyGANet4_8_rf'or opt.model == 'MyGANet9_t2':
                disp0, disp1, disp2 = model(input1, input2)
                loss0 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
                if opt.kitti or opt.kitti2015:
                    loss = 0.4 * (0.9 - (loss0 - F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean'))) + \
                           1.2 * (0.9 - (loss0 - criterion(disp2[mask],target[mask])))
                else:
                    loss = 0.4 * (0.9 - (loss0 - F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean'))) + \
                           1.2 * (0.9 - (loss0 - F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')))
            # T3 train
            elif opt.model == 'MyGANet3' or opt.model == 'MyGANet4' or opt.model == 'MyGANet4_8'or opt.model == 'MyGANet9_t3':
                disp0, disp1, disp2 = model(input1, input2)
                loss1 = F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean')
                if opt.kitti or opt.kitti2015:
                    loss = 0.9-(loss1-criterion(disp2[mask], target[mask]))
                else:
                    loss = 0.9-(loss1-F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean'))
            elif opt.model == 'MyGANet9':
                disp00, disp0, disp11, disp1, disp2 = model(input1, input2)
                if opt.kitti or opt.kitti2015:
                    loss = 0.2 * F.smooth_l1_loss(disp00[mask], target[mask], reduction='mean') + \
                           0.4 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + \
                           0.6 * F.smooth_l1_loss(disp11[mask], target[mask], reduction='mean') + \
                           1 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + \
                           1 * criterion(disp2[mask], target[mask])
                else:
                    loss = 0.2 * F.smooth_l1_loss(disp00[mask], target[mask], reduction='mean') + \
                           0.4 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + \
                           0.6 * F.smooth_l1_loss(disp11[mask], target[mask], reduction='mean') + \
                           1 * F.smooth_l1_loss(disp1[mask], target[mask], reduction='mean') + \
                           1 * F.smooth_l1_loss(disp2[mask], target[mask], reduction='mean')
            elif opt.model == 'GANet_deep' or opt.model == 'CasGANet10':
                disp0, disp1, disp2 = model(input1, input2)
                if opt.kitti or opt.kitti2015:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(
                        disp1[mask], target[mask], reduction='mean') + criterion(disp2[mask], target[mask])
                else:
                    loss = 0.2 * F.smooth_l1_loss(disp0[mask], target[mask], reduction='mean') + 0.6 * F.smooth_l1_loss(
                        disp1[mask], target[mask], reduction='mean') + F.smooth_l1_loss(disp2[mask], target[mask],
                                                                                        reduction='mean')
            else:
                raise Exception("No suitable model found ...")

            loss.backward()
            optimizer.step()
            error0 = torch.mean(torch.abs(disp0[mask] - target[mask]))
            error1 = torch.mean(torch.abs(disp1[mask] - target[mask]))
            error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))

            epoch_loss += loss.item()
            valid_iteration += 1
            epoch_error0 += error0.item()
            epoch_error1 += error1.item()
            epoch_error2 += error2.item()
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}, Error: ({:.4f} {:.4f} {:.4f}), Time:{:.2f}s".format(epoch,
                                                                                                            iteration,
                                                                                                            len(
                                                                                                                training_data_loader),
                                                                                                            loss.item(),
                                                                                                            error0.item(),
                                                                                                            error1.item(),
                                                                                                            error2.item(),
                                                                                                            time.time() - start_full_time))
            sys.stdout.flush()

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}, Avg. Error: ({:.4f} {:.4f} {:.4f}), Time:{:.2f}min".format(epoch,
                                                                                                                 epoch_loss / valid_iteration,
                                                                                                                 epoch_error0 / valid_iteration,
                                                                                                                 epoch_error1 / valid_iteration,
                                                                                                                 epoch_error2 / valid_iteration,
                                                                                                                 (time.time() - epoch_time) / 60))
    return epoch_loss / valid_iteration

def val():
    epoch_error2 = 0
    epoch_rate2 = 0
    valid_iteration = 0
    model.eval()
    for iteration, batch in enumerate(testing_data_loader):
        input1, input2, target = Variable(batch[0],requires_grad=False), Variable(batch[1], requires_grad=False), Variable(batch[2], requires_grad=False)
        if cuda:
            input1 = input1.cuda()
            input2 = input2.cuda()
        target = target
        target = torch.squeeze(target, 1)
        mask = target < opt.max_disp
        mask.detach_()
        valid = target[mask].size()[0]
        if valid>0:
            with torch.no_grad():
                disp2 = model(input1,input2)
                disp2 = disp2.cpu()
                disp2 = disp2.detach()
                error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                rate2 = torch.sum(torch.abs(disp2[mask] - target[mask]) > 3.0).numpy() / torch.sum(mask).numpy()
                valid_iteration += 1
                epoch_error2 += error2.item()
                epoch_rate2 += rate2.item()
                print("===> Test({}/{}): EPE Error: ({:.4f}),Error Rate: {:.4f}  ".format(iteration, len(testing_data_loader), error2.item(), rate2.item()))

    print("===> Test: Avg. EPE Error: ({:.4f}),AVG Error Rate: {:.4f}".format(epoch_error2 / valid_iteration, epoch_rate2 / valid_iteration))
    return epoch_error2 / valid_iteration, epoch_rate2 / valid_iteration


def save_checkpoint(save_path, epoch, state, is_best):
    filename = save_path + "_epoch_{}.pth".format(epoch)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, save_path + '_best.pth')
    print("Checkpoint saved to {}".format(filename))


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = opt.lr
    else:
        lr = opt.lr * 0.1
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot_curve(name,list):
    epoch = len(list)
    x = range(0, epoch)
    y = list
    plt.plot(x, y, 'o-')
    plt.title('{} vs. {}epoches'.format(name, epoch))
    plt.xlabel('epoch')
    plt.ylabel('{}'.format(name))
    plt.savefig('{}_{}.png'.format(opt.save_path, name))
    plt.close()


if __name__ == '__main__':
    error = 1.2
    rate = 0.08
    train_loss_list = []
    val_loss_list = []
    error_rate_list = []
    for epoch in range(1, opt.nEpochs + 1):
        adjust_learning_rate(optimizer, epoch)

        train_loss = train(epoch)
        train_loss_list.append(train_loss)
        plot_curve('train_loss', train_loss_list)

        is_best = False

        loss, erate = val()
        val_loss_list.append(loss)
        plot_curve('val_loss', train_loss_list)
        error_rate_list.append(erate)
        plot_curve('error_rate', error_rate_list)

        if loss < error or erate < rate:
            error=loss
            rate=erate
            is_best = True

        if opt.kitti or opt.kitti2015:
            if epoch % 50 == 0 or is_best:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)
        else:
            if epoch % 5 == 0 or is_best:
                save_checkpoint(opt.save_path, epoch, {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, is_best)

    save_checkpoint(opt.save_path, opt.nEpochs, {
        'epoch': opt.nEpochs,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best)
