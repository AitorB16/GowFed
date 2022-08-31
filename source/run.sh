#!/bin/bash

'''CNL'''

#Matx creation CNL
for i in {1..6}
do
    python3 create_matrix_cnl.py $i
done

#Experiments
for i in {1..6}
do
    python3 netw_cnl.py $i
done

for i in {1..6}
do
    python3 netw_cnl_AE.py $i
done

'''FL'''

#Matx creation FL
for i in {1..6}
do
    python3 create_matrix_fl.py $i
done

#Experiments
for i in {1..6}
do
    python3 netw_fl.py $i
done

for i in {1..6}
do
    python3 netw_fl_AE.py $i
done

for i in {1..6}
do
    python3 netw_fl_AM.py $i
done
