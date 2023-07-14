module load cuda/11.1
module load cmake/3.22.0
module load gnumake/4.2
module load gcc/8.5.0
export CPATH=/local/software/cuda/11.1/targets/x86_64-linux/include:$CPATH
cmake ..
make
