#!/bin/bash

# default configuration w/o numactl
# KMP_BLOCKTIME=1 w/o numactl
# default configuration w numactl
# KMP_BLOCKTIME=1 w numactl
# KMP_BLOCKTIME=1 w numactl 2 instances
# KMP_BLOCKTIME=1 w numactl 4 instances

if [[ $# -ge 1 ]] && [[ $1 == 'opt' ]]; then
    echo "Set environment variables"
    export OMP_NUM_THREADS=28
    export KMP_BLOCKTIME=1
    export KMP_AFFINITY=granularity=fine,compact
    export KMP_SETTINGS=0
    export intra_op_parallelism_threads=28
    export inter_op_parallelism_threads=1
fi

source /mnt/sdb/jingxu1/framework/tensorflow/venvs/venv_intel_pip_py37/bin/activate
python 04_Inference.py --batch_size 1
