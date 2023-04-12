#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J mult_channel
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
#BSUB -o logs/Output_multchannel_%J.out
#BSUB -e logs/Error_multchannel_%J.err

module load python3/3.9.11

python3 main.py --data_path sleepeeg.yml --finetune_path 'same' --epochs 20 --finetune_epochs 40 --batch_size 80 --load_model True --save_model True --pretrain False --finetune True --optimize_encoder True --learning_rate 1e-3 --pool 'max' --choose_best True --evaluate_latent_space False --multi_channel_setup 'sample_channel' --sample_pretrain_subjs 60 --sample_finetune_train_subjs 16 --sample_finetune_val_subjs 4 --pretrained_model_path /zhome/89/a/117273/Desktop/timeseriesrepresentation/outputs/ts2vec_sleepeeg_v_46/pretrained_model.pt --target_batch_size 16 --encoder 'ts2vec' --ft_learning_rate 1e-4 --sample_test_subjs False --target_batch_size 32
