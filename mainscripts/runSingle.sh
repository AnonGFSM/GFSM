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
for i in {1..4}
        do
                ./GFSM labelled-graphs/Gowalla-Queries/_4_query_19.g labelled-graphs/Gowalla-Labels.g
        done
