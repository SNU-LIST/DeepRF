#!/bin/bash

python ../envs/refinement.py --tag "exc_refinement" --env "Exc-v51" --gpu "0" --samples 256 --preset "../logs/exc_generation/seed_rfs.mat"