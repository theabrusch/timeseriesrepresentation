#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J multiview_scratch_sleepedf_lowreg
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=256GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 05:00
### added outputs and errors to files
#BSUB -o logs/Output_multiview_scratch_sleepedf_%J.out
#BSUB -e logs/Error_multiview_scratch_sleepedf_%J.err

module load python3/3.9.11

python3 main_multiview.py --data_path sleepeeg.yml --finetune_path sleepedf.yml --pretrain_epochs 20 --finetune_epochs 40 --batchsize 32 --load_model False --save_model False --pretrain False --finetune True --optimize_encoder False --learning_rate 1e-3 --pool 'max' --choose_best True --evaluate_latent_space False --multi_channel_setup 'sample_channel' --sample_pretrain_subjects False --sample_finetune_train_subjects 16 --sample_finetune_val_subjects 4 --target_batchsize 32 --encoder 'wave2vec' --ft_learning_rate 1e-3 --sample_test_subjects False --flatten False --weight_decay 1e-4 --balanced_sampling 'finetune' --feat_do 0.1 --hidden_channels 256 --layers 2 --num_message_passing_rounds 2 --output_path 'pretrained_outputs' --pretrained_model_path 'pretrained_outputs/MultiView_sleepeeg_pretrain_True_pretrain_subjs_False_multi_channel_setup_sample_channel_v_5/pretrained_model.pt' --seed_generator True
