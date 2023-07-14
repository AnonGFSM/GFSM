cd /dev/shm
mkdir GFSM_temp
cd GFSM_temp
mkdir data
mkdir query
cd ~/GFSM/build

cp -a labelled-graphs/$3/$2 /dev/shm/GFSM_temp/query/$2
cp -r labelled-graphs/$1 /dev/shm/GFSM_temp/data

./GFSM /dev/shm/GFSM_temp/query/$2 /dev/shm/GFSM_temp/data/$1

cd /dev/shm/
rm -r GFSM_temp
