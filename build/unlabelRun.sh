cd /dev/shm
mkdir GFSM_temp
cd GFSM_temp
mkdir data
mkdir query
cd ~/GFSM/build

cp -a query/. /dev/shm/GFSM_temp/query
cp -r data/$1 /dev/shm/GFSM_temp/data

for QUERY in /dev/shm/GFSM_temp/query/*.g
do
	for i in {1..5}
	do	
		./GFSM $QUERY /dev/shm/GFSM_temp/data/$1 >> $1.csv
	done
done

cd /dev/shm/
rm -r GFSM_temp
