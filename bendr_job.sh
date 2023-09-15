#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J ft_bendr[5-15]
### number of core
#BSUB -n 8
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=5GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 05:00
### added outputs and errors to files
#BSUB -o logs/Output_bendr_lr5e4_%J.out
#BSUB -e logs/Error_bendr_lr5e4_%J.err

module load python3/3.9.11
source timeseries-env/bin/activate

learning_rates=(5e-4 1e-3 3e-3)
seeds=(41 141 241 341 441)

python3 main_bendr.py --data_path 'sleepeeg.yml' --finetune_path 'sleepedf.yml' --sample_pretrain_subjects 4 --epochs 2 --pretrain False --load_model True --finetune True --sample_finetune_train_subjects False --sample_finetune_val_subjects False --sample_test_subjects False --ft_learning_rate ${learning_rates[(($LSB_JOBINDEX-1)/5%3)]} --optimize_encoder True --target_batchsize 32 --upsample_bendr False --pretrained_model_path '/zhome/89/a/117273/Desktop/timeseriesrepresentation/pretrained_models/bendr_original/encoder.pt' --sample_generator 10 50 100 1000 --seed ${seeds[(($LSB_JOBINDEX-1)%5)]} --load_original_bendr True

