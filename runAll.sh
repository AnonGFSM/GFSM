#!/bin/bash

#SBATCH --nodes=1
#SBATCH -A ecsstaff
#SBATCH --time=00:10:00

module load cuda/11.1
module load cmake/3.22.0
module load gnumake/4.2
module load gcc/8.5.0

which nvcc
export CPATH=/local/software/cuda/11.1/targets/x86_64-linux/include:$CPATH

cd build
cmake ..
make
./queryRun.sh Enron.g
./queryRun.sh Gowalla.g
./queryRun.sh roadNet-PA.g
