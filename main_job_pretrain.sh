#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J pretrain_HAR_multtrain_before
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
#BSUB -o logs/Output_pretrain_HAR_multtrain_avgbefore_%J.out
#BSUB -e logs/Error_pretrain_HAR_multtrain_avgbefore_%J.err

module load python3/3.9.11

python3 main.py --data_path /work3/theb/timeseries/HAR/ --epochs 40 --train_classifier False --finetune_path /work3/theb/timeseries/Gesture/ --finetune_epochs 10 --batch_size 16 --pretrain True --delta 0.5 --stride 1 --abs_budget True --loss poly --finetune True --learning_rate 1e-3 --evaluate_latent_space False --finetune_latentspace False --optimize_encoder True --avg_channels 'before' --sample_channel False
