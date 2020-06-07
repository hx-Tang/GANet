from __future__ import print_function
import argparse

import os
import time
import torch
import torch.nn.parallel
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from models.GANet_deep import GANet
from dataloader.data import get_test_set

# Training settings
parser = argparse.ArgumentParser(description='PyTorch GANet Example')
parser.add_argument('--max_disp', type=int, default=192, help="max disp")
parser.add_argument('--resume', type=str, default='', help="resume from saved model")
parser.add_argument('--left_right', type=int, default=0, help="use right view for training. Default=False")
parser.add_argument('--testBatchSize', type=int, default=3, help='testing batch size')
parser.add_argument('--cuda', type=int, default=1, help='use cuda? Default=True')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--kitti', type=int, default=0, help='kitti dataset? Default=False')
parser.add_argument('--kitti2015', type=int, default=0, help='kitti 2015? Default=False')
parser.add_argument('--data_path', type=str, default='/ssd1/zhangfeihu/data/stereo/', help="data root")
parser.add_argument('--val_list', type=str, default='./lists/sceneflow_test_select.list', help="validation list")
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
elif opt.model == 'MyGANet9':
    from models.MyGANet9 import GANet
else:
    raise Exception("No suitable model found ...")

cuda = opt.cuda
# cuda = True
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

print('===> Loading datasets')
test_set = get_test_set(opt.data_path, opt.val_list, [384, 768], opt.left_right, opt.kitti, opt.kitti2015)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
model = GANet(opt.max_disp)

if cuda:
    model = torch.nn.DataParallel(model).cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))

def val():
    epoch_error2 = 0
    epoch_rate2 = 0
    epoch_time = 0
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
        start_full_time = time.time()
        if valid>0:
            with torch.no_grad():
                disp2 = model(input1,input2)
                t=time.time() - start_full_time
                disp2 = disp2.cpu()
                disp2 = disp2.detach()
                error2 = torch.mean(torch.abs(disp2[mask] - target[mask]))
                rate2 = torch.sum(torch.abs(disp2[mask] - target[mask]) > 3.0).numpy() / torch.sum(mask).numpy()
                valid_iteration += 1
                epoch_time += t
                epoch_error2 += error2.item()
                epoch_rate2 += rate2.item()
                print("===> Test({}/{}): EPE Error: ({:.4f}),Error Rate: {:.4f}, Time:{:.2f}s ".format(iteration, len(testing_data_loader), error2.item(), rate2.item(),t))

    print("===> Test: Avg. EPE Error: ({:.4f}),AVG Error Rate: {:.4f}, Average Time:{:.2f}s".format(epoch_error2 / valid_iteration, epoch_rate2 / valid_iteration,epoch_time/(valid_iteration)))
    return epoch_error2 / valid_iteration, epoch_rate2 / valid_iteration

if __name__ == '__main__':
    val()
