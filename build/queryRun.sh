cp -r query /dev/shm/GFSM_temp
cp -r data/$1 /dev/shm/GFSM_temp
for QUERY in _2_road2.g _3_road3.g _3_triangle.g _4_road4.g _4_square.g _5_road5.g _5_plus.g
do
	for i in {1..2}
        do
		./GFSM /dev/shm/GFSM_temp/$QUERY /dev/shm/GFSM_temp/$1
	done
done
cd /dev/shm/
rm -r GFSM_temp
