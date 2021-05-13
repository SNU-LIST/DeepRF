#!/bin/bash
RANDOM=`date "+%N"`
for i in {1..50}
do

  python ../envs/generation.py --tag "exc_generation" --env "Exc-v51" --seed $RANDOM --gpu "0"

done