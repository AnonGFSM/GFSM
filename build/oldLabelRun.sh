cd /dev/shm
mkdir GFSM_temp
cd GFSM_temp
mkdir data
mkdir query
cd ~/GFSM/build

cp -a labelled-graphs/$2/. /dev/shm/GFSM_temp/query
cp -r labelled-graphs/$1 /dev/shm/GFSM_temp/data

for QUERY in /dev/shm/GFSM_temp/query/*.g
do
	./GFSM $QUERY /dev/shm/GFSM_temp/data/$1
done

cd /dev/shm/
cd GFSM_temp
dir
cd query
dir
cd /dev/shm/
rm -r GFSM_temp
