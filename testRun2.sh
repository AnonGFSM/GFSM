#!/bin/bash

#SBATCH --nodes=1
#SBATCH -A ecsstaff
#SBATCH --time=00:03:00

module load cuda/11.1
module load cmake/3.22.0
module load gnumake/4.2
module load gcc/8.5.0

which nvcc
export CPATH=/local/software/cuda/11.1/targets/x86_64-linux/include:$CPATH

cd build
cmake ..
make
./GFSM query/_2_road2.g data/Enron.g
./GFSM query/_3_road3.g data/Enron.g
./GFSM query/_3_triangle.g data/Enron.g
