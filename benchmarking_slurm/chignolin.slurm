#!/bin/bash
#SBATCH -A m4229
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH --job-name=bba
#SBATCH --output=slurm.out

export SLURM_CPU_BIND="cores"
export OMP_NUM_THREADS=64

source ./benchmarking_slurm/env_mpi_nersc.sh

cd ..
srun ./run_gen_benchmark_westpa.sh chignolin 50000

