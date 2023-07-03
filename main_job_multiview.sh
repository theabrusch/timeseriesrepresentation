#!/bin/sh
#BSUB -q gpua10
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J multiview_COCOA_COCOA
### number of core
#BSUB -n 1
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=180GB]"
### Number of hours needed
#BSUB -N
#BSUB -W 12:00
### added outputs and errors to files
#BSUB -o logs/Output_multiview_COCOA_COCOA_%J.out
#BSUB -e logs/Error_multiview_COCOA_COCOA_%J.err

module load python3/3.9.11
source timeseries-env/bin/activate

python3 main_multiview.py --data_path sleepeeg.yml --finetune_path 'sleepedf.yml' --pretrain_epochs 10 --finetune_epochs 40 --batchsize 64 --load_model False --save_model True --pretrain True --finetune False --optimize_encoder True --learning_rate 1e-3 --evaluate_latent_space False --multi_channel_setup 'sample_channel' --sample_pretrain_subjects False --sample_finetune_train_subjects 16 --sample_finetune_val_subjects 4 --target_batchsize 32 --encoder 'wave2vec' --ft_learning_rate 1e-4 --sample_test_subjects False --weight_decay 1e-2 --balanced_sampling 'finetune' --feat_do 0.1 --hidden_channels 256 --output_path 'pretrained_outputs' --loss 'COCOA' --pretraining_setup 'COCOA' --num_message_passing_rounds 3 --readout_layer True
