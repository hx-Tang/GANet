# MyGANet9 2015
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=16 \
                --crop_height=256 \
                --crop_width=896 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti_train.list' \
                --val_list='lists/kitti_val10.list'\
                --save_path='./checkpoint/MyGANet9/2015_final' \
                --kitti2015=1 \
                --shift=3 \
                --lr=0.001 \
                --resume='' \
                --model='MyGANet9'\
                --nEpochs=300 2>&1 |tee logs/MyGANet9_noresm_e300.txt
exit
# GANet11 sceneflow
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=8 \
                --crop_height=144 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset2/' \
                --training_list='lists/sceneflow_train.list' \
                --val_list='lists/sceneflow_val24.list'\
                --save_path='./checkpoint/GANet11/sceneflow_r2' \
                --resume='./checkpoint/GANet11/sceneflow_epoch_300.pth' \
                --model='GANet11' \
                --nEpochs=300 2>&1 |tee logs/GANet11_sceneflow_e600.txt
exit
# MyGANet9 sceneflow
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=16 \
                --crop_height=256 \
                --crop_width=896 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti_train.list' \
                --val_list='lists/kitti_val10.list'\
                --save_path='./checkpoint/MyGANet9/kitti2015_final' \
                --shift=3 \
                --kitti2015=1 \
                --lr=0.001 \
                --resume='./checkpoint/MyGANet9/sceneflow_t2_best.pth' \
                --model='MyGANet9'\
                --nEpochs=150 2>&1 |tee logs/MyGANet9_kitti2015_e150.txt
exit
# MyGANet4_8_rf 2015
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=16 \
                --crop_height=256 \
                --crop_width=896 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2012_train170.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/MyGANet4_8_rf/2015_rf_t22' \
                --kitti2015=1 \
                --shift=0 \
                --lr=0.0005 \
                --resume='./checkpoint/MyGANet4_8_rf/2015_rf_t21_epoch_300.pth' \
                --model='MyGANet4_8_rf'\
                --nEpochs=300 2>&1 |tee logs/MyGANet4_8_rf_2015_t22_e300.txt
exit
# GANet11 2015
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --batchSize=6 \
                --crop_height=144 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=6 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2015_train.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/GANet11/2015' \
                --kitti2015=1 \
                --shift=3 \
                --resume='./checkpoint/GANet11/2015_epoch_150.pth' \
                --model='GANet11'\
                --nEpochs=150 2>&1 |tee logs/GANet11_2015_e300.txt

exit
# MyGANet5 2015
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=16 \
                --crop_height=256 \
                --crop_width=896 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2015_train.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/MyGANet5/2015_rf_round2' \
                --kitti2015=1 \
                --shift=0 \
                --lr=0.00005 \
                --resume='./checkpoint/MyGANet5/2015_rf_epoch_150.pth' \
                --model='MyGANet5'\
                --nEpochs=150 2>&1 |tee logs/MyGANet5_2015_rf_round2_e150.txt
exit
# ./checkpoint/GANet11/sceneflow_epoch_10.pth
# shift = 0 for finetune
#./checkpoint/MyGANet5/2015_epoch_50.pth
# MyGANet4_8 sceneflow
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=16 \
                --crop_height=256 \
                --crop_width=896 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/homes/ht314/dataset2/' \
                --training_list='lists/sceneflow_train.list' \
                --val_list='lists/sceneflow_val24.list'\
                --save_path='./checkpoint/MyGANet4_8/sf' \
                --resume='./checkpoint/MyGANet4_8/sf_epoch_23.pth' \
                --model='MyGANet4_8_t1'\
                --nEpochs=150 2>&1 |tee logs/MyGANet4_8_sf_e150.txt
exit
#./checkpoint/MyGANet4_8/2015_ef_epoch_1000.pth
# MyGANet4_8 2015 finetune ef
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=24 \
                --crop_height=256 \
                --crop_width=896 \
                --max_disp=192 \
                --thread=24 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2015_train.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/MyGANet4_8/sf_ef' \
                --kitti2015=1 \
                --shift=0 \
                --resume='./checkpoint/MyGANet4_8/2015_ef_epoch_1000.pth' \
                --model='MyGANet4_8'\
                --nEpochs=400 2>&1 |tee logs/MyGANet4_8_2015_rf_e1000.txt
exit
# shift = 0 for finetune
#./checkpoint/MyGANet4_8/2015_epoch_50.pth
# MyGANet4 2015
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --batchSize=9 \
                --crop_height=192 \
                --crop_width=576 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2015_train.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/MyGANet4/2015+ef' \
                --kitti2015=1 \
                --shift=3 \
                --resume='./checkpoint/MyGANet4/2015_epoch_50.pth' \
                --model='MyGANet4'\
                --nEpochs=300 2>&1 |tee logs/MyGANet4_2015_rf_e50.txt
exit
#./checkpoint/MyGANet4/2015_epoch_50.pth

# MyGANet3 2015 finetune ef
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=16 \
                --crop_height=192 \
                --crop_width=576 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2015_train.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/MyGANet3/2015+ef_round2' \
                --kitti2015=1 \
                --shift=3 \
                --resume='./checkpoint/MyGANet3/2015+ef_epoch_50.pth' \
                --model='MyGANet3'\
                --nEpochs=300 2>&1 |tee logs/MyGANet3_2015_rf_e50_ed2.txt
exit
#./checkpoint/MyGANet3/2015_epoch_50.pth

# MyGANet2 2015
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=8 \
                --crop_height=144 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2015_train.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/MyGANet2/2015' \
                --kitti2015=1 \
                --shift=3 \
                --resume='' \
                --model='MyGANet2'\
                --nEpochs=40 2>&1 |tee logs/MyGANet2_2015_e50.txt
exit
#./checkpoint/MyGANet2/2015_epoch_10.pth
# MyGANet 2015
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=8 \
                --crop_height=144 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2015_train.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/MyGANet/2015' \
                --kitti2015=1 \
                --shift=3 \
                --resume='./checkpoint/MyGANet/2015_epoch_100.pth' \
                --model='MyGANet'\
                --nEpochs=10 2>&1 |tee logs/MyGANet_2015_base-e100_ft-ER_e10.txt
exit
#####


# GANet11 2015
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=8 \
                --crop_height=144 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=8 \
                --data_path='/homes/ht314/dataset/training/' \
                --training_list='lists/kitti2015_train.list' \
                --val_list='lists/kitti2012_val24.list'\
                --save_path='./checkpoint/GANet11/2015_base_e22' \
                --kitti2015=1 \
                --shift=3 \
                --resume='./checkpoint/GANet11/2015_apexbn_test_epoch_22.pth' \
                --model='GANet11'\
                --nEpochs=28 2>&1 |tee logs/GANet11_2015_e50.txt
exit
# ./checkpoint/GANet11/sceneflow_epoch_10.pth

#####################

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
                --crop_height=240 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/ssd1/zhangfeihu/data/stereo/' \
                --training_list='lists/sceneflow_train.list' \
                --save_path='./checkpoint/sceneflow' \
                --resume='' \
                --model='GANet_deep' \
                --nEpochs=11 2>&1 |tee logs/log_train_sceneflow.txt

exit
#Fine tuning for kitti 2015
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
                --crop_height=240 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/media/feihu/Storage/stereo/data_scene_flow/training/' \
                --training_list='lists/kitti2015_train.list' \
                --save_path='./checkpoint/finetune_kitti2015' \
                --kitti2015=1 \
                --shift=3 \
                --resume='./checkpoint/sceneflow_epoch_10.pth' \
                --nEpochs=800 2>&1 |tee logs/log_finetune_kitti2015.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=8 \
                --crop_height=240 \
                --crop_width=1248 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/media/feihu/Storage/stereo/data_scene_flow/training/' \
                --training_list='lists/kitti2015_train.list' \
                --save_path='./checkpoint/finetune2_kitti2015' \
                --kitti2015=1 \
                --shift=3 \
                --lr=0.0001 \
                --resume='./checkpoint/finetune_kitti2015_epoch_800.pth' \
                --nEpochs=8 2>&1 |tee logs/log_finetune_kitti2015.txt

#Fine tuning for kitti 2012

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 \
                --crop_height=240 \
                --crop_width=528 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/media/feihu/Storage/stereo/kitti/training/' \
                --training_list='lists/kitti2012_train.list' \
                --save_path='./checkpoint/finetune_kitti' \
                --kitti=1 \
                --shift=3 \
                --resume='./checkpoint/sceneflow_epoch_10.pth' \
                --nEpochs=800 2>&1 |tee logs/log_finetune2_kitti.txt

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=8 \
                --crop_height=240 \
                --crop_width=1248 \
                --max_disp=192 \
                --thread=16 \
                --data_path='/media/feihu/Storage/stereo/kitti/training/' \
                --training_list='lists/kitti2012_train.list' \
                --save_path='./checkpoint/finetune2_kitti' \
                --kitti=1 \
                --shift=3 \
                --lr=0.0001 \
                --resume='./checkpoint/finetune_kitti_epoch_800.pth' \
                --nEpochs=8 2>&1 |tee logs/log_finetune2_kitti.txt




