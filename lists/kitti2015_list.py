from PIL import Image
import numpy as np
import random
from struct import unpack
import sys
import re
# /homes/ht314/dataset2/frames_cleanpass/35mm_focallength/scene_backwards/fast

x = 200

train_list = open('kitti_train.list', 'w')
test_list = open('kitti_test.list', 'w')
val_list = open('kitti_val10.list', 'w')

file_list = []
for i in range(0, x):
    rgbl = "{:0>6}_10.png\n".format(i)
    file_list.append(rgbl)

tmp = file_list[:]
file_path = '/homes/ht314/dataset/training/'

for current_file in file_list:
    try:
        filename = file_path + 'image_2/' + current_file[0: len(current_file) - 1]
        left = Image.open(filename)
        filename = file_path + 'image_3/' + current_file[0: len(current_file) - 1]
        right = Image.open(filename)
        filename = file_path + 'disp_occ_0/' + current_file[0: len(current_file) - 1]
    except:
        print(filename)
        tmp.remove(current_file)
file_list = tmp[:]

length = len(file_list)

random.shuffle(file_list)

train= [file_list[i] for i in range(int(length * 0.85))]
test = [file_list[i] for i in range(int(length * 0.85), length - 10)]
val = [file_list[i] for i in range(length - 10, length)]

train_list.write(''.join(train))
train_list.close()

test_list.write(''.join(test))
test_list.close()

val_list.write(''.join(val))
val_list.close()

