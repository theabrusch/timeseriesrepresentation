#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J ssl_eeg_pretrain_all
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=128GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 24:00
### added outputs and errors to files
#BSUB -o logs/Output_ssl_eeg_pretrain_all_%J.out
#BSUB -e logs/Error_ssl_eeg_pretrain_all_%J.err

module load python3/3.9.11

python3 main_eeg.py --config_path sleepeeg.yml --epochs 40 --train_classifier False --finetune_epochs 40 --batch_size 32 --pretrain True --delta 0.8 --stride 1 --abs_budget False --finetune False --learning_rate 3e-4 --sample_channel False --sample_pretrain_subjs False --optimize_encoder False --target_batch_size 32 --sample_test_subjs 20 --normalize True --sample_finetune_subjs 30 --balanced_sampling True
