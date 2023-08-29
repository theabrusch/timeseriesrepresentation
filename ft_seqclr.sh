#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J seqclr_nooptenc_s[1-5]
### number of core
#BSUB -n 5
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=8GB]"
### Number of hours needed
#BSUB -N
#BSUB -W 05:00
### added outputs and errors to files
#BSUB -o logs/Output_ft_sleepedf_seqclr_%J_%I.out
#BSUB -e logs/Error_ft_sleepedf_seqclr_%J_%I.err

module load python3/3.9.11
source timeseries-env/bin/activate

losses=('contrastive')
learning_rates=(5e-4 1e-3 3e-3)
seeds=(41 141 241 341 441)

python3 main_seqclr.py --data_path sleepeeg.yml --finetune_path sleepedf.yml --finetune_epochs 40 --batchsize 32 --load_model False --save_model False --pretrain False --finetune True --optimize_encoder False --early_stopping_criterion 'acc' --sample_finetune_train_subjects False --sample_finetune_val_subjects False --job_id $LSB_JOBID --target_batchsize 32 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --output_path 'ft_checkpoints' --ft_learning_rate 3e-3 --sample_generator 10 50 100 --seed ${seeds[(($LSB_JOBINDEX-1)%5)]} --encoder 'SeqCLR_R' --loss ${losses[(($LSB_JOBINDEX-1) % 1)]} --pretraining_length 30 --suffix '_R'


