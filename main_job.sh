#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J train_selfsupervised_no_classifier
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 02:00
### added outputs and errors to files
#BSUB -o logs/Output_selfsuper_no_class_%J.out
#BSUB -e logs/Error_selfsuper_no_class_%J.err

module load python3/3.9.11

python3 main.py --data_path /work3/theb/timeseries/ECG/ --epochs 50 --train_classifier False --finetune_path /work3/theb/timeseries/EMG/ 
