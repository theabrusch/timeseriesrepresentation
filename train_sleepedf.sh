#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J ft_sleepedf_COCOA_contr
### number of core
#BSUB -n 5
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=8GB]"
### Number of hours needed
#BSUB -N
#BSUB -W 10:00
### added outputs and errors to files
#BSUB -o logs/Output_multiview_ft_sleepedf_GNN_time_%J.out
#BSUB -e logs/Error_multiview_ft_sleepedf_GNN_time_%J.err

module load python3/3.9.11
source timeseries-env/bin/activate

python3 main_multiview.py --data_path sleepeeg.yml --finetune_path sleepedf.yml --finetune_epochs 40 --batchsize 32 --load_model True --save_model False --pretrain False --finetune True --optimize_encoder True --early_stopping_criterion 'loss' --evaluate_latent_space False --sample_finetune_train_subjects False --sample_finetune_val_subjects False --target_batchsize 32 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --feat_do 0.1 --hidden_channels 256 --layers 6 --num_message_passing_rounds 3 --conv_do 0.1 --job_id $LSB_JOBID --output_path 'ft_checkpoints' --pretraining_setup 'COCOA' --loss 'contrastive' --ft_learning_rate 1e-3 --sample_generator 10 50 100 --seed 41 --readout_layer True --projection_head True

python3 main_multiview.py --data_path sleepeeg.yml --finetune_path sleepedf.yml --finetune_epochs 40 --batchsize 32 --load_model True --save_model False --pretrain False --finetune True --optimize_encoder True --early_stopping_criterion 'loss' --evaluate_latent_space False --sample_finetune_train_subjects False --sample_finetune_val_subjects False --target_batchsize 32 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --feat_do 0.1 --hidden_channels 256 --layers 6 --num_message_passing_rounds 3 --conv_do 0.1 --job_id $LSB_JOBID --output_path 'ft_checkpoints' --pretraining_setup 'COCOA' --loss 'contrastive' --ft_learning_rate 1e-3 --sample_generator 10 50 100 --seed 141 --readout_layer True --projection_head True

python3 main_multiview.py --data_path sleepeeg.yml --finetune_path sleepedf.yml --finetune_epochs 40 --batchsize 32 --load_model True --save_model False --pretrain False --finetune True --optimize_encoder True --early_stopping_criterion 'loss' --evaluate_latent_space False --sample_finetune_train_subjects False --sample_finetune_val_subjects False --target_batchsize 32 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --feat_do 0.1 --hidden_channels 256 --layers 6 --num_message_passing_rounds 3 --conv_do 0.1 --job_id $LSB_JOBID --output_path 'ft_checkpoints' --pretraining_setup 'COCOA' --loss 'contrastive' --ft_learning_rate 1e-3 --sample_generator 10 50 100 --seed 241 --readout_layer True --projection_head True
