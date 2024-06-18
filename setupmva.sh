export CUDNN_PATH="/home/helsens/miniconda3/envs/mva_training_gpu/lib/python3.12/site-packages/nvidia/cudnn/"

export LD_LIBRARY_PATH="$CUDNN_PATH/lib":"/usr/local/cuda/lib64":"$LD_LIBRARY_PATH"
export PATH="$PATH":"/usr/local/cuda/bin"