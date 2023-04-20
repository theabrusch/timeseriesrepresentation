import torch
import argparse
from utils.bendr import BendingCollegeWav2Vec, ConvEncoderBENDR, BENDRContextualizer
from eegdataset import construct_eeg_datasets
from torch.optim import AdamW
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import os
import datetime
import wandb
from dn3.transforms.batch import RandomTemporalCrop

def check_output_path(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        i = 1
        while os.path.exists(output_path + f'_v_{i}'):
            i+=1
        output_path = output_path + f'_v_{i}'
        os.makedirs(output_path, exist_ok=True)
    return output_path

mul_channel_explanations = {
     'None': 'Multi channel setup is set to None.',
     'sample_channel': 'Multi channel setup is set to sample_channel. This means that sampled channels will be used as each others augmented versions.',
     'avg_ch': 'Multi channel setup is set to ch_avg. This means that the channels are averaged before convolutions.'
}
def main(args):
    dset = args.data_path.split('/')[-1].strip('.yml')
    output_path = f'{args.output_path}/MultiView_{dset}_pretrain_{args.pretrain}_pretrain_subjs_{args.sample_pretrain_subjects}'
    
    output_path = check_output_path(output_path)
    args.outputh_path = output_path
        
    print('Saving outputs in', output_path)

    args.train_mode = 'pretrain' 
    args.standardize_epochs = 'channelwise'
    pretrain_loader, pretrain_val_loader, finetune_loader, finetune_val_loader, test_loader, (channels, time_length, num_classes) = construct_eeg_datasets(**vars(args))
    
    encoder = ConvEncoderBENDR(channels, encoder_h=args.hidden_size, out_dim=args.out_dim)
    contextualizer = BENDRContextualizer(args.out_dim, layer_drop=0.01)

    # add arguments from BENDR config
    bending_college_args = {
        "mask_rate": 0.065,
        "mask_span": 10,
        "layer_drop": 0.01,
        "multi_gpu": True,
        "temp": 0.1,
        "encoder_grad_frac": 0.1,
        "num_negatives": 20,
        "enc_feat_l2": 1.0
    }
    optimizer_params = {
        "lr": 0.00002,
        "weight_decay": 0.01,
        "betas": [0.9, 0.98]
    }
    augmentation_params = {
        "upsample_crop": 32,
        "batch_crop_frac": 0.05
    }
    training_params = {
        "epochs": args.epochs,
        "validation_interval": 100,
        "train_log_interval": 100,
        "batch_size": 64,
        "warmup_frac": 0.05
    }
    process = BendingCollegeWav2Vec(encoder, contextualizer, **bending_college_args)

    # Slower learning rate for the encoder
    process.set_optimizer(torch.optim.Adam(process.parameters(), **optimizer_params))

   
    process.add_batch_transform(RandomTemporalCrop(max_crop_frac=augmentation_params["batch_crop_frac"]))


    def epoch_checkpoint(metrics):
        if not args.no_save:
            encoder.save('checkpoints/encoder_epoch_{}.pt'.format(metrics['epoch']))
            contextualizer.save('checkpoints/contextualizer_epoch_{}.pt'.format(metrics['epoch']))

    #def simple_checkpoint(metrics):
    ##    if not args.no_save:
    ##        torch.save(process.best)
    #       encoder.save('checkpoints/encoder.pt')
    #        contextualizer.save('checkpoints/contextualizer.pt')

    process.fit(pretrain_loader, epoch_callback=epoch_checkpoint, num_workers=args.num_workers,
                validation_dataset=pretrain_val_loader, resume_epoch=args.resume, log_callback=simple_checkpoint,
                **training_params)

    print(process.evaluate(pretrain_val_loader))

    if not args.no_save:
        encoder.save('checkpoints/encoder_best_val.pt')
        contextualizer.save('checkpoints/contextualizer_best_val.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--data_path', type = str, default = 'sleepeeg_local.yml')
    parser.add_argument('--finetune_path', type = str, default = 'sleepedf_local.yml')
    parser.add_argument('--no_save', type = eval, default = False)
    parser.add_argument('--load_model', type = eval, default = False)
    parser.add_argument('--num-workers', default=6, type=int)
    parser.add_argument('--resume', default=None, type=int)

    parser.add_argument('--pretrain', type = eval, default = True)
    parser.add_argument('--evaluate_latent_space', type = eval, default = False)
    parser.add_argument('--finetune', type = eval, default = False)
    parser.add_argument('--optimize_encoder', type = eval, default = True)
    parser.add_argument('--pretrained_model_path', type = str, default = None)
    parser.add_argument('--output_path', type = str, default = 'outputs')
    parser.add_argument('--pretraining_setup', type = str, default = 'GNN')

    # data arguments
    parser.add_argument('--balanced_sampling', type = str, default = 'finetune')
    parser.add_argument('--seed_generator', type = eval, default = '[10,20]')

    # model arguments
    parser.add_argument('--hidden_size', type = int, default = 256)
    parser.add_argument('--out_dim', type = int, default = 64)
    parser.add_argument('--epochs', type = int, default = 10)

    # eeg arguments
    parser.add_argument('--sample_pretrain_subjects', type = eval, default = 3)
    parser.add_argument('--sample_finetune_train_subjects', type = eval, default = 3)
    parser.add_argument('--sample_finetune_val_subjects', type = eval, default = 1)
    parser.add_argument('--sample_test_subjects', type = eval, default = 1)

    # optimizer arguments
    parser.add_argument('--batchsize', type = int, default = 64)
    parser.add_argument('--target_batchsize', type = int, default = 64)
    args = parser.parse_args()
    main(args)

