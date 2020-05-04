from PIL import Image
import numpy as np
import random
from struct import unpack
import sys
import re
# /homes/ht314/dataset2/frames_cleanpass/35mm_focallength/scene_backwards/fast
def readPFM(file):
    with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
    return img, height, width


X = [('backwards',300),('forwards',300)]

train_list = open('sceneflow_train.list', 'w')
test_list = open('sceneflow_test.list', 'w')
val_list = open('sceneflow_val24.list', 'w')

file_list = []
for x in X:
        for i in range(1, x[1]):
            rgbl = "35mm_focallength/scene_{}/fast/left/{:0>4}.png\n".format(x[0], i)
            file_list.append(rgbl)

tmp = file_list[:]
data_path = '/homes/ht314/dataset2/'

for current_file in file_list:
    try:
        A = current_file
        filename = data_path + 'frames_cleanpass/' + A[0: len(A) - 1]
        left = Image.open(filename)
        filename = data_path + 'frames_cleanpass/' + A[0: len(A) - 14] + 'right/' + A[len(A) - 9:len(A) - 1]
        right = Image.open(filename)
        filename = data_path + 'disparity/' + A[0: len(A) - 4] + 'pfm'
        disp_left, height, width = readPFM(filename)
    except:
        print(filename)
        tmp.remove(current_file)
file_list = tmp[:]

length = len(file_list)

random.shuffle(file_list)

train= [file_list[i] for i in range(int(length * 0.8))]
test = [file_list[i] for i in range(int(length * 0.8), length - 24)]
val = [file_list[i] for i in range(length - 24, length)]

train_list.write(''.join(train))
train_list.close()

test_list.write(''.join(test))
test_list.close()

val_list.write(''.join(val))
val_list.close()

