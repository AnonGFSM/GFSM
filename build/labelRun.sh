cd /dev/shm
rm -r GFSM_temp
mkdir GFSM_temp
cd GFSM_temp
mkdir data
mkdir query
cd ~/GFSM/build

cp -a labelled-graphs/$2/. /dev/shm/GFSM_temp/query
cp -r labelled-graphs/$1 /dev/shm/GFSM_temp/data

for QUERY in /dev/shm/GFSM_temp/query/*.g
do
	for i in {1..4}
	do	
		timeout --foreground -k 10 2m ./GFSM $QUERY /dev/shm/GFSM_temp/data/$1 >> output.csv
	done
done

cd /dev/shm/
rm -r GFSM_temp
