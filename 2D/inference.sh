#!/bin/bash

# default configuration
# KMP_BLOCKTIME=0 w/o numactl
# KMP_BLOCKTIME=0 w numactl
# KMP_BLOCKTIME=1 w/o numactl
# KMP_BLOCKTIME=1 w numactl

if [[ $# -ge 1 ]] && [[ $1 == 'opt' ]]; then
    echo "Set environment variables"
    export OMP_NUM_THREADS=26
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,compact
    export KMP_SETTINGS=0
    export intra_op_parallelism_threads=26
    export inter_op_parallelism_threads=1
fi

source /root/tf_unet2d/venv_tf_unet2d_py37/bin/activate
python 04_Inference.py --batch_size 128
