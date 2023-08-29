#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J ft_sleepedf_seqclr_time_lr1e4_41
### number of core
#BSUB -n 5
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=8GB]"
### Number of hours needed
#BSUB -N
#BSUB -W 02:00
### added outputs and errors to files
#BSUB -o logs/Output_ft_sleepedf_seqclr_%J.out
#BSUB -e logs/Error_ft_sleepedf_seqclr_%J.err

module load python3/3.9.11
source timeseries-env/bin/activate

python3 main_seqclr.py --data_path sleepeeg.yml --finetune_path sleepedf.yml --finetune_epochs 40 --batchsize 32 --load_model True --save_model False --pretrain False --finetune True --optimize_encoder True --early_stopping_criterion 'acc' --sample_finetune_train_subjects False --sample_finetune_val_subjects False --job_id $LSB_JOBID --target_batchsize 32 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --output_path 'ft_checkpoints' --ft_learning_rate 1e-4 --sample_generator 10 50 100 --seed 41 --encoder 'SeqCLR_W' --loss 'time_loss' --pretraining_length 30 




