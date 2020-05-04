CUDA_VISIBLE_DEVICES=2 python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/homes/ht314/dataset/training/' \
                  --test_list='lists/kitti_test.list' \
                  --save_path='./result/GANet11/' \
                  --kitti2015=1 \
                  --model='GANet11'\
                  --resume='./checkpoint/GANet11/kitti2015_final.pth'
exit
CUDA_VISIBLE_DEVICES=1 python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/homes/ht314/dataset/training/' \
                  --test_list='lists/kitti_test.list' \
                  --save_path='./result/GANet11/2015/' \
                  --kitti2015=1 \
                  --model='GANet11'\
                  --resume='./checkpoint/GANet11/kitti2015_final.pth'
exit
###############
python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/ssd1/zhangfeihu/data/kitti/2015//testing/' \
                  --test_list='lists/kitti2015_test.list' \
                  --save_path='./result/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/kitti2015_final.pth'
exit

python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/media/feihu/Storage/stereo/kitti/testing/' \
                  --test_list='lists/kitti2012_test.list' \
                  --save_path='./result/' \
                  --kitti=1 \
                  --resume='./checkpoint/kitti2012_final.pth'



