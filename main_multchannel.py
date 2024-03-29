import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from utils.trainer import TFC_trainer, train_classifier, evaluate_latent_space, finetune_model, evaluate_model
from utils.models import TFC_encoder, ContrastiveLoss, ContrastiveLoss2, ClassifierModule
from utils.dataset import TFC_Dataset
from torch.optim import Adam
from utils.plot_functions import plot_contrastive_losses
import pickle
import os
import datetime


def main(args):
    train = torch.load(args.data_path + 'train.pt')
    dset = args.data_path.split('/')[-2]
    TFC_dset = TFC_Dataset(train['samples'], train['labels'], dset, abs_budget=args.abs_budget)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = f'{args.output_path}/classifier_{args.train_classifier}_TFC_{args.train_TFC}_abs_budget_{args.abs_budget}_stride_{args.stride}_loss_{args.loss}_{dset}'
    
    if not args.pretrain and (args.evaluate_latent_space or args.finetune):
        args.overwrite = True
    
    if not args.overwrite:
        if not os.path.exists(output_path):
                os.makedirs(output_path)
        elif len(os.listdir(output_path)) == 0:
            output_path = output_path
        else:
            i = 1
            while os.path.exists(output_path + f'_v_{i}') and not len(os.listdir(output_path + f'_v_{i}')) == 0:
                i+=1
            output_path = output_path + f'_v_{i}'
            os.makedirs(output_path, exist_ok=True)
    print('Saving outputs in', output_path)
    if args.pretrain:
        val = torch.load(args.data_path + 'val.pt')
        test = torch.load(args.data_path + 'test.pt')
        
        print('Loading data')
        time = datetime.datetime.now()
        train_loader = DataLoader(TFC_dset, batch_size = args.batch_size, shuffle = True, drop_last=True)

        val_dset = TFC_Dataset(val['samples'], val['labels'], dset = dset, abs_budget=args.abs_budget)
        test_dset = TFC_Dataset(test['samples'], test['labels'], dset = dset, test_mode = True)
        val_loader = DataLoader(val_dset, batch_size = args.batch_size, drop_last=True)
        test_loader = DataLoader(test_dset, batch_size = args.batch_size)
        time2 = datetime.datetime.now()     
        print('Loading the data took', time2-time, 's.')
        
        print('Initializing model')
        model = TFC_encoder(in_channels = TFC_dset.channels, input_size = TFC_dset.time_length, 
                            num_classes = TFC_dset.num_classes, stride = args.stride, classify = False)
        model.to(device)

        optimizer = Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

        if args.loss == 'poly':
            loss_fn = ContrastiveLoss(tau = 0.2, device = device)
        else:
            loss_fn = ContrastiveLoss2(tau = 0.2, device = device)

        print('Training model')
        if args.train_TFC:
            time = datetime.datetime.now()
            if args.train_classifier:
                Classifier = ClassifierModule(TFC_dset.num_classes)
                Classifier.to(device)
                class_optimizer = Adam(Classifier.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
            else:
                Classifier = None
                class_optimizer = None

            model, losses = TFC_trainer(model = model, 
                                        train_loader = train_loader, 
                                        optimizer = optimizer, 
                                        loss_fn = loss_fn, 
                                        epochs = args.epochs, 
                                        val_loader = val_loader, 
                                        device = device, 
                                        classifier = Classifier,
                                        class_optimizer = class_optimizer)
            time2 = datetime.datetime.now()    
            print('Pre-training for', args.epochs,'epochs took', time2-time, 's.')
        else:
            args.train_classifier = True
            time = datetime.datetime.now()
            model, losses = train_classifier(model=model,
                                            train_loader=train_loader,
                                            optimizer=optimizer,
                                            epochs=args.epochs,
                                            val_loader=val_loader,
                                            device= device)
            time2 = datetime.datetime.now()   
            print('Pre-training for',args.epochs,'epochs took', time2-time, 's.')
            time = time2

        if args.save_model:
            model.eval()
            path = f'{output_path}/pretrained_model.pt'
            torch.save(model.state_dict(), path)
        plot_contrastive_losses(losses['train'], f'{output_path}/pretrain_train_losses.png')
        plot_contrastive_losses(losses['val'], f'{output_path}/pretrain_val_losses.png')

        with open(f'{output_path}/pretrain_losses.pickle', 'wb') as file:
                pickle.dump(losses, file)
    else:
        if args.pretrained_model_path is not None:
            pretrained_path = args.pretrained_model_path
        else:
            pretrained_path = f'{output_path}/pretrained_model.pt'

        model = TFC_encoder(in_channels = TFC_dset.channels, input_size = TFC_dset.time_length, 
                            num_classes = TFC_dset.num_classes, stride = args.stride, classify = args.train_classifier)
        model.load_state_dict(torch.load(pretrained_path))
        model.to(device=device)
        if args.loss == 'poly':
            loss_fn = ContrastiveLoss(tau = 0.2, device = device)
        else:
            loss_fn = ContrastiveLoss2(tau = 0.2, device = device)

    if args.evaluate_latent_space:
            time = datetime.datetime.now() 
            val = torch.load(args.data_path + 'val.pt')
            val_dset = TFC_Dataset(val['samples'], val['labels'], dset = dset, abs_budget=args.abs_budget)
            val_loader = DataLoader(val_dset, batch_size = args.batch_size, drop_last=False)
            outputs = evaluate_latent_space(model = model, data_loader = val_loader, device = device, classifier = args.train_classifier, save_h = False)

            time2 = datetime.datetime.now()   
            print('Evaluating the latent space took', time2-time, 's.')
            
            with open(f'{output_path}/pretrain_latent_variables.pickle', 'wb') as file:
                pickle.dump(outputs, file)

    if not args.optimize_encoder:
        args.finetune_latentspace = False

    if args.finetune:
        time = datetime.datetime.now()   
        ft_train = torch.load(args.finetune_path + 'train.pt')
        ft_val = torch.load(args.finetune_path + 'val.pt')
        ft_test = torch.load(args.finetune_path + 'test.pt')
        finetune_dset = args.finetune_path.split('/')[-2]

        ft_TFC_dset = TFC_Dataset(ft_train['samples'], ft_train['labels'], dset = finetune_dset, fine_tune_mode=True)
        ft_train_loader = DataLoader(ft_TFC_dset, batch_size = args.target_batch_size, shuffle = True, drop_last=False)
        ft_val_dset = TFC_Dataset(ft_val['samples'], ft_val['labels'], dset = finetune_dset, fine_tune_mode=True)
        ft_val_loader = DataLoader(ft_val_dset, batch_size = args.target_batch_size, shuffle = True, drop_last=False)
        ft_test_dset = TFC_Dataset(ft_test['samples'], ft_test['labels'], dset = finetune_dset, test_mode = True)
        ft_test_loader = DataLoader(ft_test_dset, batch_size = args.target_batch_size)
        
        # evaluate latent space prior to finetune
        if args.finetune_latentspace:
            outputs_val = evaluate_latent_space(model = model, data_loader = ft_val_loader, device = device, classifier = args.train_classifier, save_h = False)
            ft_test_lat_dset = TFC_Dataset(ft_test['samples'], ft_test['labels'], dset = finetune_dset, test_mode = False, fine_tune_mode=True)
            ft_test_lat_loader = DataLoader(ft_test_lat_dset, batch_size = args.target_batch_size)
            outputs_test = evaluate_latent_space(model = model, data_loader = ft_test_lat_loader, device = device, classifier = args.train_classifier, save_h = False)
            outputs_train = evaluate_latent_space(model = model, data_loader = ft_train_loader, device = device, classifier = args.train_classifier, save_h = False)
            with open(f'{output_path}/prior_finetune_val_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_val, file)
            with open(f'{output_path}/prior_finetune_test_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_test, file)
            with open(f'{output_path}/prior_finetune_train_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_train, file)
        
        time2 = datetime.datetime.now()     
        print('Loading the finetuning data took', time2-time, 's.')

        Classifier = ClassifierModule(ft_TFC_dset.num_classes)

        Classifier.to(device)
        class_optimizer = Adam(Classifier.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

        time = datetime.datetime.now()  
        if args.optimize_encoder:
            optimizer = Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
        else:
            optimizer = None

        model, losses = finetune_model(model = model, 
                                        classifier = Classifier, 
                                        data_loader = ft_train_loader, 
                                        val_loader = ft_val_loader,
                                        loss_fn = loss_fn, 
                                        optimizer = optimizer, 
                                        class_optimizer = class_optimizer, 
                                        epochs = args.finetune_epochs, 
                                        device = device,
                                        lambda_ = 0.2, 
                                        delta = args.delta)
        time2 = datetime.datetime.now()     
        print('Finetuning the model for', args.finetune_epochs,'epochs took', time2-time, 's.')
        # evaluate latent space prior to finetune
        if args.finetune_latentspace:
            outputs_val = evaluate_latent_space(model = model, data_loader = ft_val_loader, device = device, classifier = args.train_classifier, save_h = False)
            outputs_test = evaluate_latent_space(model = model, data_loader = ft_test_lat_loader, device = device, classifier = args.train_classifier, save_h = False)
            outputs_train = evaluate_latent_space(model = model, data_loader = ft_train_loader, device = device, classifier = args.train_classifier, save_h = False)
            with open(f'{output_path}/post_finetune_val_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_val, file)
            with open(f'{output_path}/post_finetune_test_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_test, file)
            with open(f'{output_path}/post_finetune_train_latent_variables_{finetune_dset}.pickle', 'wb') as file:
                    pickle.dump(outputs_train, file)
        time = time2
        results = evaluate_model(model = model,
                                classifier = Classifier, 
                                test_loader = ft_test_loader,
                                device = device)
        time2 = datetime.datetime.now()     
        print('Evaluating the finetuned model took', time2-time, 's.')
        plot_contrastive_losses(losses['train'], f'{output_path}/finetune_train_loss_optenc_{args.optimize_encoder}_{finetune_dset}.png')
        plot_contrastive_losses(losses['val'], f'{output_path}/finetune_val_loss_optenc_{args.optimize_encoder}_{finetune_dset}.png')

        with open(f'{output_path}/finetune_results_optenc_{args.optimize_encoder}_{finetune_dset}.pickle', 'wb') as file:
            pickle.dump(results, file)
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training arguments
    parser.add_argument('--train_TFC', type = eval, default = True)
    parser.add_argument('--train_classifier', type = eval, default = True)
    parser.add_argument('--evaluate_latent_space', type = eval, default = True)
    parser.add_argument('--save_model', type = eval, default = True)
    parser.add_argument('--finetune', type = eval, default = True)
    parser.add_argument('--finetune_latentspace', type = eval, default = True)
    parser.add_argument('--optimize_encoder', type = eval, default = True)
    parser.add_argument('--pretrain', type = eval, default = True)
    parser.add_argument('--pretrained_model_path', type = str, default = None)
    # data arguments
    parser.add_argument('--data_path', type = str, default = 'datasets/Gesture/')
    parser.add_argument('--finetune_path', type = str, default = 'datasets/EMG/')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--target_batch_size', type = int, default = 22)
    parser.add_argument('--output_path', type = str, default = 'outputs')
    parser.add_argument('--overwrite', type = eval, default = False)

    # augmentation arguments
    parser.add_argument('--abs_budget', type = eval, default = False)
    parser.add_argument('--stride', type = int, default = 4)
    # optimizer arguments
    parser.add_argument('--loss', type = str, default = 'poly')
    parser.add_argument('--delta', type = float, default = 0.5)
    parser.add_argument('--learning_rate', type = float, default = 3e-6)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)
    parser.add_argument('--epochs', type = int, default = 0)
    parser.add_argument('--finetune_epochs', type = int, default = 1)
    args = parser.parse_args()
    main(args)

