#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J ssl_eeg_bal_finetune
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=128GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 02:00
### added outputs and errors to files
#BSUB -o logs/Output_ssl_eeg_bal_finetune_%J.out
#BSUB -e logs/Error_ssl_eeg_samp_subjects_norm_%J.err

module load python3/3.9.11

python3 main_eeg.py --config_path sleepeeg.yml --epochs 100 --train_classifier False --finetune_epochs 100 --batch_size 32 --pretrain False --delta 1 --stride 1 --abs_budget False --finetune True --learning_rate 1e-3 --sample_channel False --sample_pretrain_subjs 60 --optimize_encoder True --target_batch_size 32 --sample_test_subjs 20 --normalize False --sample_finetune_subjs 20 --balanced_sampling True --pretrained_model_path /zhome/89/a/117273/Desktop/timeseriesrepresentation/outputs/TFC_True_multchannel_None_sleepeeg_v_3/pretrained_model.pt
