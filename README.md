CMOR 421: High Performance Computing
3/21/2024
Creating interactive node with 24 CPUs:
srun --partition=interactive --pty --export=ALL --cpus-per-task=24 --time=00:30:00 /bin/bash

3/28/2024
OpenMPI on NOTS:

Load module via:
module load GCC OpenMPI

Compile:
mpic++ hello_world_mpi.cpp -o hello

Create the file job.slurm
#!/bin/bash 
#SBATCH --job-name=CMOR-421-521
#SBATCH --partition=scavenge
#SBATCH --ntasks=4 
#SBATCH --mem-per-cpu=1G 
#SBATCH --time=00:30:00 
echo "My job ran on:" 
echo $SLURM_NODELIST 
srun -n 2 ./<name of executable>

Partition options:
Commons, scavenge, interactive

Run the job:
sbatch myMPIjob.slurm

Check status:
squeue -u hpk1

Check job queue:
squeue

View output:
cat slurm<jobnumber>.out

1/28/2024
NOTS Login: 
ssh -Y hpk1@nots.rice.edu
password
rm -rf cmor-421-521-s24
git clone https://github.com/jlchan/cmor-421-521-s24.git
module load GCC/13.1.0
g++ -O3 file.cpp
./a.out

Julia on NOTS:
module load Julia/version
import Pkg; Pkg.add("BenchmarkTools"); Pkg.add("LinearAlgebra"); Pkg.add("LoopVectorization")
exit
