#!/bin/bash

#SBATCH --nodes=1
#SBATCH -A ecsstaff
#SBATCH --time=00:03:00

module load cuda/11.7
module load cmake/3.22.0
module load gnumake/4.2
module load gcc/8.5.0

nvcc --version
gcc --version
cmake --version
make --version

export CPATH=/local/software/cuda/11.7/targets/x86_64-linux/include:$CPATH

echo $CPATH
echo $LD_LIBRARY_PATH
echo $PATH

which nvcc

