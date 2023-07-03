#!/bin/sh
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J ft_bendr
### number of core
#BSUB -n 1
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=40GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 24:00
### added outputs and errors to files
#BSUB -o logs/Output_bendr_lr5e4_%J.out
#BSUB -e logs/Error_bendr_lr5e4_%J.err

module load python3/3.9.11
source timeseries-env/bin/activate

python3 main_bendr.py --data_path 'sleepeeg.yml' --finetune_path 'sleepedf.yml' --sample_pretrain_subjects False --epochs 2 --pretrain False --load_model True --finetune True --sample_finetune_train_subjects False --sample_finetune_val_subjects False --sample_test_subjects False --ft_learning_rate 5e-4 --optimize_encoder True --target_batchsize 32 --upsample_bendr False --pretrained_model_path '/zhome/89/a/117273/Desktop/timeseriesrepresentation/bendr_checkpoints/encoder_best_val.pt' --sample_generator 10 25 50 100 200 500 1000 --seed 41 

python3 main_bendr.py --data_path 'sleepeeg.yml' --finetune_path 'sleepedf.yml' --sample_pretrain_subjects False --epochs 2 --pretrain False --load_model True --finetune True --sample_finetune_train_subjects False --sample_finetune_val_subjects False --sample_test_subjects False --ft_learning_rate 5e-4 --optimize_encoder True --target_batchsize 32 --upsample_bendr False --pretrained_model_path '/zhome/89/a/117273/Desktop/timeseriesrepresentation/bendr_checkpoints/encoder_best_val.pt' --sample_generator 10 25 50 100 200 500 1000 --seed 141

python3 main_bendr.py --data_path 'sleepeeg.yml' --finetune_path 'sleepedf.yml' --sample_pretrain_subjects False --epochs 2 --pretrain False --load_model True --finetune True --sample_finetune_train_subjects False --sample_finetune_val_subjects False --sample_test_subjects False --ft_learning_rate 5e-4 --optimize_encoder True --target_batchsize 32 --upsample_bendr False --pretrained_model_path '/zhome/89/a/117273/Desktop/timeseriesrepresentation/bendr_checkpoints/encoder_best_val.pt' --sample_generator 10 25 50 100 200 500 1000 --seed 241

python3 main_bendr.py --data_path 'sleepeeg.yml' --finetune_path 'sleepedf.yml' --sample_pretrain_subjects False --epochs 2 --pretrain False --load_model True --finetune True --sample_finetune_train_subjects False --sample_finetune_val_subjects False --sample_test_subjects False --ft_learning_rate 5e-4 --optimize_encoder True --target_batchsize 32 --upsample_bendr False --pretrained_model_path '/zhome/89/a/117273/Desktop/timeseriesrepresentation/bendr_checkpoints/encoder_best_val.pt' --sample_generator 10 25 50 100 200 500 1000 --seed 341

python3 main_bendr.py --data_path 'sleepeeg.yml' --finetune_path 'sleepedf.yml' --sample_pretrain_subjects False --epochs 2 --pretrain False --load_model True --finetune True --sample_finetune_train_subjects False --sample_finetune_val_subjects False --sample_test_subjects False --ft_learning_rate 5e-4 --optimize_encoder True --target_batchsize 32 --upsample_bendr False --pretrained_model_path '/zhome/89/a/117273/Desktop/timeseriesrepresentation/bendr_checkpoints/encoder_best_val.pt' --sample_generator 10 25 50 100 200 500 1000 --seed 441
