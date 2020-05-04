export LD_INCLUDE_PATH="/homes/ht314/MLproject/include:$LD_INCLUDE_PATH"
export CUDA_HOME="$CUDA_ROOT"
export CUDNN_INCLUDE_DIR="$CPLUS_INCLUDE_PATH"
export CUDNN_LIB_DIR="$LIBRARY_PATH"

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
#echo $TORCH
cd libs/GANet
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib

cd ../sync_bn
python setup.py clean
rm -rf build
python setup.py build
cp -r build/lib* build/lib
