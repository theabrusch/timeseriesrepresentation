import torch
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score

def TFC_trainer(model, 
                train_loader, 
                optimizer, 
                loss_fn, 
                epochs, 
                val_loader, 
                device, 
                train_classifier, 
                delta_ = 1, 
                lambda_ = 0.5, 
                eta_ = 0.5):
    """Function for training the time frequency contrastive model for time series. 

    Args:
        model (torch.Module): The model to train
        train_loader (torch.utils.data.DataLoader): Dataloader containing the train data on which to train the model
        optimizer (torch.optim.Optimizer): Optimizer with which to optimize the model. 
        loss_fn (torch.Module): Function implementing the contrastive loss function to use for 
                                optimizing the self-supervised part of the model.
        epochs (int): Number of epochs to train the model for. 
        val_loader (torch.utils.data.DataLoader): Dataloader containing the validation data on which to validate the model
        device (torch.device): CPU or GPU depending on availability
        train_classifier (bool): Whether to train the classifier along with the contrastive loss. 
        delta_ (int, optional): Parameter to add in the time frequency consistency loss. Defaults to 1.
        lambda_ (float, optional): Parameter weighing the time and frequency loss vs the time-frequency consistency loss. Defaults to 0.5.
        eta_ (float, optional): Parameter weighing the contrastive loss vs the classifier. Defaults to 0.5.

    Returns:
    torch.Module: Final model after training
    dict: Dictionary containing all of the losses
    """    
    time_loss_total = []
    freq_loss_total = []
    time_freq_loss_total = []
    loss_total = []
    val_time_loss_total = []
    val_freq_loss_total = []
    val_time_freq_loss_total = []
    val_loss_total = []

    if train_classifier:
        class_loss_fn = torch.nn.CrossEntropyLoss()
        class_loss_total = []
        val_class_loss_total = []

    for epoch in range(epochs):
        print('\n', epoch + 1 , 'of', epochs)
        epoch_time, epoch_freq, epoch_time_freq, epoch_class, epoch_loss = [0, 0, 0, 0, 0]
        val_epoch_time, val_epoch_freq, val_epoch_time_freq, val_epoch_class, val_epoch_loss, val_epoch_acc = [0, 0, 0, 0, 0, 0]
        model.train()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)

            if train_classifier:
                h_t, z_t, h_f, z_f, out = model(x_t, x_f)
                h_t_aug, z_t_aug, h_f_aug, z_f_aug, _ = model(x_t_aug, x_f_aug)
            else:
                h_t, z_t, h_f, z_f = model(x_t, x_f)
                h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(x_t_aug, x_f_aug)

            time_loss = loss_fn(h_t, h_t_aug)
            freq_loss = loss_fn(h_f, h_f_aug)

            time_freq_pos = loss_fn(z_t, z_f)
            time_freq_neg  = loss_fn(z_t, z_f_aug), loss_fn(z_t_aug, z_f), loss_fn(z_t_aug, z_f_aug)
            loss_TFC = (time_freq_pos - time_freq_neg[0] + delta_) + (time_freq_pos - time_freq_neg[1] + delta_) + (time_freq_pos - time_freq_neg[2] + delta_)

            loss = lambda_*(time_loss + freq_loss) + (1-lambda_)*loss_TFC

            if train_classifier:
                class_loss = class_loss_fn(out, y)
                loss *= (1-eta_)
                loss += eta_*class_loss
                epoch_class += class_loss.detach().cpu()

            epoch_time += time_loss.detach().cpu()
            epoch_freq += freq_loss.detach().cpu()
            epoch_time_freq += loss_TFC.detach().cpu()
            epoch_loss += loss.detach().cpu()

            loss.backward()
            optimizer.step()
        
        print('\nTraining losses:')
        print('Time consistency loss:', epoch_time/(i+1))
        print('Frequency consistency loss:', epoch_freq/(i+1))
        print('Time-freq consistency loss:', epoch_time_freq/(i+1))
        print('Total loss:', epoch_loss/(i+1))
            
        time_loss_total.append(epoch_time/(i+1))
        freq_loss_total.append(epoch_freq/(i+1))
        time_freq_loss_total.append(epoch_time_freq/(i+1))
        loss_total.append(epoch_loss/(i+1))
        if train_classifier:
            class_loss_total.append(epoch_class/(i+1))
            print('Classification loss:', epoch_class/(i+1))

        # evaluate on validation set
        model.eval()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(val_loader):
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)
            if train_classifier:
                h_t, z_t, h_f, z_f, out = model(x_t, x_f)
                h_t_aug, z_t_aug, h_f_aug, z_f_aug, _ = model(x_t_aug, x_f_aug)
            else:
                h_t, z_t, h_f, z_f = model(x_t, x_f)
                h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(x_t_aug, x_f_aug)

            time_loss = loss_fn(h_t, h_t_aug)
            freq_loss = loss_fn(h_f, h_f_aug)

            time_freq_pos = loss_fn(z_t, z_f)
            time_freq_neg  = loss_fn(z_t, z_f_aug), loss_fn(z_t_aug, z_f), loss_fn(z_t_aug, z_f_aug)
            loss_TFC = 3*time_freq_pos - time_freq_neg[0] - time_freq_neg[1] - time_freq_neg[2] + 3*delta_

            loss = lambda_*(time_loss + freq_loss) + (1-lambda_)*loss_TFC

            if train_classifier:
                class_loss = class_loss_fn(out, y)
                loss += class_loss
                val_epoch_class += class_loss.detach().cpu()

                if i == 0:
                    y_pred = torch.argmax(out.detach().cpu(), dim = 1)
                    y_true = y.detach().cpu()
                else:
                    y_pred = torch.cat((y_pred, torch.argmax(out.detach().cpu(), dim = 1)), dim = 0)
                    y_true = torch.cat((y_true, y.detach().cpu()), dim = 0)
                

            val_epoch_time += time_loss.detach().cpu()
            val_epoch_freq += freq_loss.detach().cpu()
            val_epoch_time_freq += loss_TFC.detach().cpu()
            val_epoch_loss += loss.detach().cpu()
        
        print('\nValidation losses')
        print('Time consistency loss:', val_epoch_time/(i+1))
        print('Frequency consistency loss:', val_epoch_freq/(i+1))
        print('Time-freq consistency loss:', val_epoch_time_freq/(i+1))
        print('Total loss:', val_epoch_loss/(i+1))
        
        val_time_loss_total.append(val_epoch_time/(i+1))
        val_freq_loss_total.append(val_epoch_freq/(i+1))
        val_time_freq_loss_total.append(val_epoch_time_freq/(i+1))
        val_loss_total.append(val_epoch_loss/(i+1))
        if train_classifier:
            val_class_loss_total.append(val_epoch_class/(i+1))
            acc = accuracy_score(y_true, y_pred)
            print('Accuracy:', acc)
            print('Classification loss:', val_epoch_class/(i+1))



    losses = {
        'train': {
            'time_loss': time_loss_total,
            'freq_loss': freq_loss_total,
            'time_freq_loss': time_freq_loss_total,
            'loss': loss_total},
        'val': {
            'time_loss': val_time_loss_total,
            'freq_loss': val_freq_loss_total,
            'time_freq_loss': val_time_freq_loss_total,
            'loss': val_loss_total}
        }
    if train_classifier:
        losses['train']['class_loss'] = class_loss_total
        losses['val']['class_loss'] = val_class_loss_total

    return model, losses

def train_classifier(model, 
                    train_loader, 
                    optimizer, 
                    epochs, 
                    val_loader, 
                    device):
    """Function for training only the classifier part of the TFC model. 

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        optimizer (_type_): _description_
        epochs (_type_): _description_
        val_loader (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """    
    
    
    loss_total = []
    val_loss_total = []

    class_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print('\n', epoch + 1 , 'of', epochs)
        epoch_loss = 0
        epoch_acc = 0 
        val_epoch_loss = 0
        val_epoch_acc = 0 
        y_pred = []
        y_true = []
        model.train()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)

            _, _, _, _, out = model(x_t, x_f)

            class_loss = class_loss_fn(out, y)
                
            epoch_loss += class_loss.detach().cpu()

            if i == 0:
                y_pred = torch.argmax(out.detach().cpu(), dim = 1)
                y_true = y.detach().cpu()
            else:
                y_pred = torch.cat((y_pred, torch.argmax(out.detach().cpu(), dim = 1)), dim = 0)
                y_true = torch.cat((y_true, y.detach().cpu()), dim = 0)

            class_loss.backward()
            optimizer.step()
        
        epoch_acc += accuracy_score(y_true, y_pred)
        print('\nTraining losses:')
        print('Accuracy', epoch_acc)
        print('Total loss:', epoch_loss/(i+1))

        loss_total.append(epoch_loss/(i+1))

        # evaluate on validation set
        model.eval()
        y_pred = []
        y_true = []
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(val_loader):
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long()
            _, _, _, _, out = model(x_t, x_f)

            class_loss = class_loss_fn(out.detach().cpu(), y)

            if i == 0:
                y_pred = torch.argmax(out.detach().cpu(), dim = 1)
                y_true = y.detach().cpu()
            else:
                y_pred = torch.cat((y_pred, torch.argmax(out.detach().cpu(), dim = 1)), dim = 0)
                y_true = torch.cat((y_true, y.detach().cpu()), dim = 0)
            val_epoch_loss += class_loss.detach().cpu()
        
        val_epoch_acc += accuracy_score(y_true, y_pred)
        
        print('\nValidation losses')
        print('Accuracy:', val_epoch_acc)
        print('Total loss:', val_epoch_loss/(i+1))
    
        val_loss_total.append(val_epoch_loss/(i+1))



    losses = {
        'train': {
            'loss': loss_total},
        'val': {
            'loss': val_loss_total}
        }
    return model, losses

def finetune_model(model, 
                  classifier, 
                  data_loader, 
                  loss_fn, 
                  optimizer, 
                  class_optimizer, 
                  epochs, 
                  device, 
                  lambda_ = 0.2):

    model.train()
    classifier.train()
    class_loss_fn = torch.nn.CrossEntropyLoss()

    collect_class_loss = torch.zeros(epochs)
    collect_loss = torch.zeros(epochs)
    collect_time_loss = torch.zeros(epochs)
    collect_freq_loss = torch.zeros(epochs)
    collect_time_freq_loss = torch.zeros(epochs)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_class_loss = 0
        epoch_time_loss = 0
        epoch_freq_loss = 0
        epoch_time_freq_loss = 0

        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(data_loader):
            optimizer.zero_grad()
            class_optimizer.zero_grad()
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.long().to(device)
            h_t, z_t, h_f, z_f = model(x_t, x_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(x_t_aug, x_f_aug)

            time_loss = loss_fn(h_t, h_t_aug)
            freq_loss = loss_fn(h_f, h_f_aug)

            time_freq_pos = loss_fn(z_t, z_f)
            time_freq_neg  = loss_fn(z_t, z_f_aug), loss_fn(z_t_aug, z_f), loss_fn(z_t_aug, z_f_aug)
            loss_TFC = (time_freq_pos - time_freq_neg[0] + 1) + (time_freq_pos - time_freq_neg[1] + 1) + (time_freq_pos - time_freq_neg[2] + 1)

            y_out = classifier(torch.cat([z_t, z_f], dim = -1))
            class_loss = class_loss_fn(y_out, y)
            loss = class_loss + lambda_*(time_loss + freq_loss) + (1-lambda_)*loss_TFC
            loss.backward()
            optimizer.step()
            class_optimizer.step()
            epoch_loss += loss.detach().cpu()
            epoch_class_loss += class_loss.detach().cpu()
            epoch_time_loss += time_loss.detach().cpu()
            epoch_freq_loss += freq_loss.detach().cpu()
            epoch_time_freq_loss += loss_TFC.detach().cpu()
        
        collect_class_loss[epoch] = epoch_class_loss / (i+1)
        collect_loss[epoch] = epoch_loss / (i+1)
        collect_time_loss[epoch] = epoch_time_loss / (i+1)
        collect_freq_loss[epoch] = epoch_freq_loss / (i+1)
        collect_time_freq_loss[epoch] = epoch_time_freq_loss / (i+1)

        print('Epoch loss:', epoch_loss/(i+1))
        print('Class. loss:', epoch_class_loss/(i+1))
    
    losses = {
        'Loss': collect_loss,
        'Class loss': collect_class_loss,
        'Time loss': collect_time_loss,
        'Freq loss': collect_freq_loss,
        'Time freq loss': collect_time_freq_loss
    }
    return model, losses

    
def evaluate_model(model,
                   classifier,
                   test_loader,
                   device):
    
    model.eval()
    classifier.eval()
    for i, (x_t, x_f, y) in enumerate(test_loader):
        x_t, x_f, y = x_t.float().to(device), x_f.float().to(device), y.long()
        _, z_t, _, z_f = model(x_t, x_f)
        y_out = classifier(torch.cat([z_t, z_f], dim = -1))

        if i == 0:
            y_pred = torch.argmax(y_out, dim = -1).detach().cpu()
            y_true = y
        else:
            y_pred = torch.cat([y_pred, torch.argmax(y_out, dim = -1).detach().cpu()], dim = 0)
            y_true = torch.cat([y_true, y], dim = 0)
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f, _ = precision_recall_fscore_support(y_true, y_pred)
    #auroc = roc_auc_score(y_true, y_pred)
    #auprc = average_precision_score(y_true, y_pred)

    results = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 score': f
        #'AUROC': auroc, 
        #'AUPRC': auprc
    }

    for res in results.keys():
        print(res, ':', results[res])
    
    return results






def evaluate_latent_space(model, data_loader, device, classifier):
    model.eval()
    for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(data_loader):
        x_t, x_f, x_t_aug, x_f_aug = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device)
        normal_outputs = model(x_t, x_f)
        augmented_outputs = model(x_t_aug, x_f_aug)

        normal_outputs = [out.detach().cpu().numpy() for out in normal_outputs]
        augmented_outputs = [out.detach().cpu().numpy() for out in augmented_outputs[:-1]]

        h_latent_space = np.concatenate((normal_outputs[0][np.newaxis,:, :], normal_outputs[2][np.newaxis, :, :], augmented_outputs[0][np.newaxis, :, :], augmented_outputs[2][np.newaxis, :, :]), axis = 0)
        z_latent_space = np.concatenate((normal_outputs[1][np.newaxis, :, :], normal_outputs[3][np.newaxis, :, :], augmented_outputs[1][np.newaxis, :, :], augmented_outputs[3][np.newaxis, :, :]), axis = 0)

        if i == 0:
            collect_h_latent_space = h_latent_space
            collect_z_latent_space = z_latent_space
            if classifier:
                collect_y_out = normal_outputs[-1]
                collect_y = y 
        else:
            collect_h_latent_space = np.concatenate((collect_h_latent_space, h_latent_space), axis = 1)
            collect_z_latent_space = np.concatenate((collect_z_latent_space, z_latent_space), axis = 1)
            if classifier:
                collect_y_out = np.concatenate((collect_y_out, normal_outputs[-1]), axis = 0)
                collect_y = np.concatenate((collect_y, y), axis = 0)
    
    columns_h = ['h_t', 'h_f', 'h_t_aug', 'h_f_aug'] 
    columns_z = ['z_t', 'z_f', 'z_t_aug', 'z_f_aug'] 
    outputs = dict()

    for latent in [zip(columns_h, collect_h_latent_space), zip(columns_z, collect_z_latent_space)]:
        for i, (name, var) in enumerate(latent):
            outputs[name] = var
    outputs['y'] = collect_y
    outputs['y_pred'] = collect_y_out

    return outputs


    