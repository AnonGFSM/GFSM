#!/bin/bash

#SBATCH --nodes=1
#SBATCH -A ecsstaff
#SBATCH --time=00:02:00

cp -r build/query /dev/shm/GFSM_temp
cp -r build/data/roadNet-PA.g /dev/shm/GFSM_temp
./GFSM /dev/shm/GFSM_temp/_2_road2.g /dev/shm/GFSM_temp/roadNet-PA.g
./GFSM /dev/shm/GFSM_temp/_3_road3.g /dev/shm/GFSM_temp/roadNet-PA.g
./GFSM /dev/shm/GFSM_temp/_3_triangle.g /dev/shm/GFSM_temp/roadNet-PA.g
./GFSM /dev/shm/GFSM_temp/_4_road4.g /dev/shm/GFSM_temp/roadNet-PA.g
./GFSM /dev/shm/GFSM_temp/_4_square.g /dev/shm/GFSM_temp/roadNet-PA.g
./GFSM /dev/shm/GFSM_temp/_4_roadY.g /dev/shm/GFSM_temp/roadNet-PA.g
./GFSM /dev/shm/GFSM_temp/_5_plus.g /dev/shm/GFSM_temp/roadNet-PA.g
./GFSM /dev/shm/GFSM_temp/_5_road5.g /dev/shm/GFSM_temp/roadNet-PA.g
cd /dev/shm/
rm -r GFSM_temp


