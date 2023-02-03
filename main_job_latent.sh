#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_selfsupervised_evallatent
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=32GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 01:00
### added outputs and errors to files
#BSUB -o logs/Output_selfsuper_evallatent_%J.out
#BSUB -e logs/Error_selfsuper_evallatent_%J.err

module load python3/3.9.11

python3 main.py --data_path /work3/theb/timeseries/ECG/ --epochs 100 --train_classifier True --finetune_path /work3/theb/timeseries/EMG/ --finetune_epochs 100 --finetune True --batch_size 64 --pretrain False --delta 0.5 --stride 1 --learning_rate 1e-3 --pretrained_model_path outputs/pretrained_model_classifier_False_TFC_True_stride_1_loss_poly.pt --finetune False --evaluate_latent_space True --overwrite True --abs_budget True
