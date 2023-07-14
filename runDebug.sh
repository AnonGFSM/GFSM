#!/bin/bash

#SBATCH --nodes=1
#SBATCH -A ecsstaff
#SBATCH --time=01:20:00

module load cuda/11.7
module load cmake/3.22.0
module load gnumake/4.2
module load gcc/8.5.0

which nvcc
export CPATH=/local/software/cuda/11.7/targets/x86_64-linux/include:$CPATH

cd build
cmake ..
make

./debuggingRun1.sh watdiv.g _4_query_0.g watdiv-Queries
