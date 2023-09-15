#!/bin/sh
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J wave2vec_proj_optenc[1-30]
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

models=('COCOA' 'GNN')
learning_rates=(5e-4 1e-3 3e-3)
seeds=(41 141 241 341 441)

python3 main_multiview.py --data_path sleepeeg.yml --finetune_path sleepedf.yml --finetune_epochs 40 --batchsize 32 --load_model True --save_model False --pretrain False --finetune True --optimize_encoder True --early_stopping_criterion 'loss' --evaluate_latent_space False --sample_finetune_train_subjects False --sample_finetune_val_subjects False --target_batchsize 32 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --feat_do 0.1 --hidden_channels 256 --layers 6 --num_message_passing_rounds 3 --conv_do 0.1 --job_id $LSB_JOBID --output_path 'ft_checkpoints' --pretraining_setup ${models[(($LSB_JOBINDEX-1))%2]} --loss 'COCOA' --ft_learning_rate ${learning_rates[(($LSB_JOBINDEX-1)/10%3)]} --sample_generator 25 200 --seed ${seeds[(($LSB_JOBINDEX-1)/2%5)]} --readout_layer True --projection_head True --track_similarity False
