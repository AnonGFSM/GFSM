#!/bin/bash

#SBATCH --nodes=1
#SBATCH -A ecsstaff
#SBATCH --time=03:30:00

module load cuda/11.1
module load cmake/3.22.0
module load gnumake/4.2
module load gcc/8.5.0

which nvcc
export CPATH=/local/software/cuda/11.1/targets/x86_64-linux/include:$CPATH

cd build
cmake ..
make

echo "Query Graph, Data Graph, Preallocate Time, Query Parse Time, Data Parse Time, Kernel Time, Reallocate Time, ReKernel Time, Function Count, Solution Write Time, GPU Time, End to End Time,"

# ./labelRun.sh roadNet-PA-Labels.g roadNet-PA-Queries
./labelRun.sh Gowalla-Labels.g Gowalla-Queries
./labelRun.sh Enron-Labels.g Enron-Queries
