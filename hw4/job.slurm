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
srun ./a.out