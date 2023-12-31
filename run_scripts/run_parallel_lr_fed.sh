#!/bin/bash

python fed_train_proposed.py --epoch_per_round 2 --num_fed_loops 50 --init_lr 1e-3 > out_lr_3.out &
python fed_train_proposed.py --epoch_per_round 2 --num_fed_loops 50 --init_lr 1e-4 > out_lr_4.out &
python fed_train_proposed.py --epoch_per_round 2 --num_fed_loops 50 --init_lr 1e-5 > out_lr_5.out &
python fed_train_proposed.py --epoch_per_round 2 --num_fed_loops 50 --init_lr 1e-6 > out_lr_6.out &