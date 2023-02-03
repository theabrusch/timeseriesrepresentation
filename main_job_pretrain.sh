#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_selfsupervised_pretrain_HAR_mult
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=32GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 24:00
### added outputs and errors to files
#BSUB -o logs/Output_selfsuper_pretrain_HAR_mult_%J.out
#BSUB -e logs/Error_selfsuper_pretrain_HAR_mult_%J.err

module load python3/3.9.11

python3 main.py --data_path /work3/theb/timeseries/HAR/ --epochs 40 --train_classifier False --finetune_path /work3/theb/timeseries/Gesture/ --finetune_epochs 40 --batch_size 32 --pretrain True --delta 0.5 --stride 1 --abs_budget False --loss poly --finetune True --learning_rate 3e-4 --sample_channel True
