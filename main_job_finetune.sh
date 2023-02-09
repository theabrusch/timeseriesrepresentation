#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_selfsupervised_finetune
### number of core
#BSUB -n 1 
### specify that all cores should be on the same host
#BSUB -R "span[hosts=1]"
### specify the memory needed
#BSUB -R "rusage[mem=10GB]"
### Number of hours needed
#BSUB -N 
#BSUB -W 01:00
### added outputs and errors to files
#BSUB -o logs/Output_selfsuper_finetune_%J.out
#BSUB -e logs/Error_selfsuper_finetune_%J.err

module load python3/3.9.11

python3 main.py --data_path /work3/theb/timeseries/HAR/ --epochs 100 --train_classifier False --finetune_path /work3/theb/timeseries/Gesture/ --finetune_epochs 20 --finetune True --batch_size 32 --pretrain False --delta 0.5 --stride 1 --learning_rate 1e-3 --abs_budget False --evaluate_latent_space False --optimize_encoder True --finetune_latentspace False --sample_channel True --loss poly --pretrained_model_path outputs/classifier_False_TFC_True_abs_budget_False_stride_1_loss_poly_HAR_v_4/pretrained_model.pt
