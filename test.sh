#!/bin/bash
#SBATCH --job-name=Hello_twcc    
#SBATCH --nodes=2                
#SBATCH --ntasks-per-node=8      
#SBATCH --cpus-per-task=4        
#SBATCH --gres=gpu:8             
#SBATCH --account="ACD110018"   
#SBATCH --partition=gtest
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=ss110062207@gapp.nthu.edu.tw

SIF=env.sif
SINGULARITY="singularity run --nv $SIF"

# pytorch horovod benchmark script from
# wget https://raw.githubusercontent.com/horovod/horovod/v0.20.3/examples/pytorch/pytorch_synthetic_benchmark.py
HOROVOD="python deep_tf.py -m swin_eg"

# enable NCCL log
export NCCL_DEBUG=INFO
export PMIX_MCA_gds=hash

srun --mpi=pmi2 $SINGULARITY $HOROVOD
