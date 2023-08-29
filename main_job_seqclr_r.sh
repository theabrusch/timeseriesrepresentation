#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J seqclr_all_W_contr
### number of core
#BSUB -n 12
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=15GB]"
### Number of hours needed
#BSUB -N
#BSUB -W 24:00
### added outputs and errors to files
#BSUB -o logs/Output_seqclr_%J.out
#BSUB -e logs/Error_seqclr_%J.err

module load python3/3.9.11
source timeseries-env/bin/activate

python3 main_seqclr.py --data_path sleepeeg.yml --finetune_path 'sleepedf.yml' --pretrain_epochs 30 --batchsize 100 --load_model False --save_model True --pretrain True --finetune False --learning_rate 1e-3 --sample_pretrain_subjects False --sample_finetune_train_subjects 16 --sample_finetune_val_subjects 4 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --output_path 'pretrained_outputs' --encoder 'SeqCLR_R' --loss 'contrastive' --temperature 0.05 --pretraining_length 20
