#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J TS2Vec_HAR
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=32GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 00:30
### added outputs and errors to files
#BSUB -o logs/Output_TS2Vec_HAR_%J.out
#BSUB -e logs/Error_TS2Vec_HAR_%J.err

module load python3/3.9.11

python3 main.py --data_path /work3/theb/timeseries/HAR/ --epochs 40
