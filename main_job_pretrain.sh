#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J TS2Vec_pretrain_large
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=64GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 24:00
### added outputs and errors to files
#BSUB -o logs/Output_TS2Vec_pretrain_large_%J.out
#BSUB -e logs/Error_TS2Vec_pretrain_large_%J.err

module load python3/3.9.11

python3 main.py --data_path sleepeeg.yml --epochs 40 --sample_pretrain_subjs 200 --sample_test_subjs False --batch_size 64 --load_model False --pretrained_model_path outputs/ts2vec_sleepeeg_v_2/pretrained_model.pt --sample_finetune_train_subjs 40 --sample_finetune_val_subjs 40
