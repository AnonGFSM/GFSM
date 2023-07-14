cd /dev/shm
rm -r GFSM_temp
mkdir GFSM_temp
cd GFSM_temp
mkdir data
mkdir query
cd ~/GFSM/build

cp -a labelled-graphs/$2/. /dev/shm/GFSM_temp/query
cp -a labelled-graphs/$1/. /dev/shm/GFSM_temp/data

for DATA in /dev/shm/GFSM_temp/data/*.g
do
	for QUERY in /dev/shm/GFSM_temp/query/*.g
	do
		for i in {1..4}
		do	
			echo $DATA "->" $QUERY "\n"
			timeout --foreground -k 10 3m ./GFSM $QUERY $DATA >> output.csv
		done
	done
done

cd /dev/shm/
rm -r GFSM_temp
