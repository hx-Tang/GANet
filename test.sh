CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --max_disp=192 --val_list='lists/sceneflow_test.list' --kitti2015=0 --data_path='/homes/ht314/dataset2/' --save_path='/homes/ht314/my-GANet/result/MyGANet9_test_layer/sceneflow/' --resume='./checkpoint/MyGANet9/sceneflow_t2_best.pth' --model='MyGANet9'