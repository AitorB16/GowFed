#!/bin/sh
python3 create_matrix_fl.py $1
python3 netw_fl_AE.py $1
