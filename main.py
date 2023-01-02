import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from utils.trainer import TFC_trainer, train_classifier, evaluate_latent_space, finetune_model, evaluate_model
from utils.models import TFC_encoder, ContrastiveLoss, ClassifierModule
from utils.dataset import TFC_Dataset
from torch.optim import Adam
from utils.plot_functions import plot_contrastive_losses
import pickle
import datetime


def main(args):

    train = torch.load(args.data_path + 'train.pt')
    val = torch.load(args.data_path + 'val.pt')
    test = torch.load(args.data_path + 'test.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Loading data')
    TFC_dset = TFC_Dataset(train['samples'], train['labels'])
    train_loader = DataLoader(TFC_dset, batch_size = args.batch_size, shuffle = True, drop_last=True)

    val_dset = TFC_Dataset(val['samples'], val['labels'])
    test_dset = TensorDataset(test['samples'], test['labels'])
    val_loader = DataLoader(val_dset, batch_size = args.batch_size, drop_last=True)
    test_loader = DataLoader(test_dset, batch_size = args.batch_size)

    if args.finetune:
        ft_train = torch.load(args.finetune_path + 'train.pt')
        ft_val = torch.load(args.finetune_path + 'val.pt')
        ft_test = torch.load(args.finetune_path + 'test.pt')
        ft_TFC_dset = TFC_Dataset(ft_train['samples'], ft_train['labels'])
        ft_train_loader = DataLoader(ft_TFC_dset, batch_size = args.batch_size, shuffle = True, drop_last=False)
        ft_val_dset = TFC_Dataset(ft_val['samples'], ft_val['labels'])
        ft_test_dset = TensorDataset(ft_test['samples'], ft_test['labels'])
        ft_val_loader = DataLoader(ft_val_dset, batch_size = args.batch_size, drop_last=False)
        ft_test_loader = DataLoader(ft_test_dset, batch_size = args.batch_size)

        Classifier = ClassifierModule(ft_TFC_dset.num_classes)
    
    print('Initializing model')
    model = TFC_encoder(in_channels = TFC_dset.channels, input_size = TFC_dset.time_length, 
                        num_classes = TFC_dset.num_classes, classify = args.train_classifier)
    model.to(device)

    optimizer = Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

    loss_fn = ContrastiveLoss(tau = 0.2, device = device)

    print('Training model')
    if args.train_TFC:
        model, losses = TFC_trainer(model = model, 
                                    train_loader = train_loader, 
                                    optimizer = optimizer, 
                                    loss_fn = loss_fn, 
                                    epochs = args.epochs, 
                                    val_loader = val_loader, 
                                    device = device, 
                                    train_classifier = args.train_classifier)
    else:
        args.train_classifier = True
        model, losses = train_classifier(model=model,
                                         train_loader=train_loader,
                                         optimizer=optimizer,
                                         epochs=args.epochs,
                                         val_loader=val_loader,
                                         device= device)

    plot_contrastive_losses(losses['train'], 'outputs/training_outputs_train_classifier_{}_TFC_{}.png'.format(args.train_classifier, args.train_TFC))
    plot_contrastive_losses(losses['val'], 'outputs/validation_outputs_train_classifier_{}_TFC_{}.png'.format(args.train_classifier, args.train_TFC))
    
    with open('outputs/losses_train_classifier_{}_TFC_{}.pickle'.format(args.train_classifier, args.train_TFC), 'wb') as file:
            pickle.dump(losses, file)

    if args.finetune:
        Classifier.to(device)
        optimizer = Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)
        class_optimizer = Adam(Classifier.parameters(), lr = args.learning_rate, weight_decay=args.weight_decay)

        model = finetune_model(model = model, 
                                classifier = Classifier, 
                                data_loader = ft_train_loader, 
                                loss_fn = loss_fn, 
                                optimizer = optimizer, 
                                class_optimizer = class_optimizer, 
                                epochs = args.finetune_epochs, 
                                device = device)
        
        results = evaluate_model(model = model,
                                 classifier = Classifier, 
                                 test_loader = ft_test_loader,
                                 device = device)
        with open('outputs/results_train_{}_test{}.pickle'.format(args.data_path.split('/')[-2], args.finetune_path.split('/')[-2]), 'wb') as file:
            pickle.dump(results, file)
    else:
        outputs = evaluate_latent_space(model = model, data_loader = val_loader, device = device, classifier = args.train_classifier)
        
        with open('outputs/latents_train_classifier_{}_TFC_{}.pickle'.format(args.train_classifier, args.train_TFC), 'wb') as file:
            pickle.dump(outputs, file)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str, default = 'datasets/ECG/')
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--train_TFC', type = eval, default = True)
    parser.add_argument('--train_classifier', type = eval, default = False)
    parser.add_argument('--learning_rate', type = float, default = 3e-6)
    parser.add_argument('--weight_decay', type = float, default = 5e-4)

    parser.add_argument('--finetune', type = eval, default = True)
    parser.add_argument('--finetune_path', type = str, default = 'datasets/EMG/')
    parser.add_argument('--finetune_epochs', type = int, default = 40)
    parser.add_argument('--epochs', type = int, default = 100)
    args = parser.parse_args()
    main(args)

