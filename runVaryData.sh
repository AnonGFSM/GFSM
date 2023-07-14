#!/bin/bash

#SBATCH --nodes=1
#SBATCH -A ecsstaff
#SBATCH --time=02:50:00

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

./labelRun.sh DataGraph_64000_32_1_1_0_9.g VaryingDataQueries
./labelRun.sh DataGraph_128000_32_1_1_0_9.g VaryingDataQueries
./labelRun.sh DataGraph_256000_32_1_1_0_9.g VaryingDataQueries
./labelRun.sh DataGraph_512000_32_1_1_0_9.g VaryingDataQueries
./labelRun.sh DataGraph_1024000_32_1_1_0_9.g VaryingDataQueries
./labelRun.sh DataGraph_2048000_32_1_1_0_9.g VaryingDataQueries
./labelRun.sh DataGraph_4096000_32_1_1_0_9.g VaryingDataQueries
