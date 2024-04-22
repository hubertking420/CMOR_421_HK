#!/bin/bash
#SBATCH --job-name=CMOR-421-521
#SBATCH --partition=scavenge
#SBATCH --reservation=cmor421
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
echo "My job ran on:"
echo $SLURM_NODELIST
if [ 0 -eq 1 ]; then
  srun stencil_1
fi

if [ 0 -eq 1 ]; then
  srun stencil_2
fi

if [ 0 -eq 1 ]; then
  srun reduction_0
fi

if [ 0 -eq 1 ]; then
  srun reduction_1
fi

if [ 1 -eq 1 ]; then
  srun reduction_2
fi 