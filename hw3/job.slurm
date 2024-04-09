#!/bin/bash 
#SBATCH --job-name=CMOR-421-521
#SBATCH --partition=scavenge
#SBATCH --reservation=cmor421
#SBATCH --ntasks=4 
#SBATCH --mem-per-cpu=1G 
#SBATCH --time=00:30:00 
echo "My job ran on:" 
echo $SLURM_NODELIST 
srun -n 4 cannon 1024
srun -n 4 summa 1024
