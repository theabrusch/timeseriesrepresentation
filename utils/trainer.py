import torch

def TFC_trainer(model, train_loader, optimizer, loss_fn, epochs, val_loader, device, train_classifier, delta_ = 1, lambda_ = 0.5):
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
        val_epoch_time, val_epoch_freq, val_epoch_time_freq, val_epoch_class, val_epoch_loss = [0, 0, 0, 0, 0]
        model.train()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(train_loader):
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.float().to_device()

            h_t, z_t, h_f, z_f, out = model(x_t, x_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug, _ = model(x_t_aug, x_f_aug)

            time_loss = loss_fn(h_t, h_t_aug)
            freq_loss = loss_fn(h_f, h_f_aug)

            time_freq_pos = loss_fn(z_t, z_f)
            time_freq_neg  = loss_fn(z_t, z_f_aug), loss_fn(z_t_aug, z_f), loss_fn(z_t_aug, z_f_aug)
            loss_TFC = 3*time_freq_pos - time_freq_neg[0] - time_freq_neg[1] - time_freq_neg[2] + 3*delta_

            loss = lambda_*(time_loss + freq_loss) + (1-lambda_)*loss_TFC

            if train_classifier:
                class_loss = class_loss_fn(out, y)
                loss += class_loss
                epoch_class += class_loss.detach().cpu()/len(x_t)

            epoch_time += time_loss.detach().cpu()/len(x_t)
            epoch_freq += freq_loss.detach().cpu()/len(x_t)
            epoch_time_freq += loss_TFC.detach().cpu()/len(x_t)
            epoch_loss += loss.detach().cpu()/len(x_t)

            loss.backward()
            optimizer.step()
        
        print('\nTraining losses:')
        print('Time consistency loss:', epoch_time)
        print('Frequency consistency loss:', epoch_freq)
        print('Time-freq consistency loss:', epoch_time_freq)
        print('Total loss:', epoch_loss)
        print('Classification loss:', epoch_class)
            
        time_loss_total.append(epoch_time)
        freq_loss_total.append(epoch_freq)
        time_freq_loss_total.append(epoch_time_freq)
        loss_total.append(epoch_loss)
        class_loss_total.append(epoch_class)

        # evaluate on validation set
        model.eval()
        for i, (x_t, x_f, x_t_aug, x_f_aug, y) in enumerate(val_loader):
            x_t, x_f, x_t_aug, x_f_aug, y = x_t.float().to(device), x_f.float().to(device), x_t_aug.float().to(device), x_f_aug.float().to(device), y.float().to_device()
            h_t, z_t, h_f, z_f, out = model(x_t, x_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug, _ = model(x_t_aug, x_f_aug)
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
            val_epoch_time += time_loss.detach().cpu()/len(x_t)
            val_epoch_freq += freq_loss.detach().cpu()/len(x_t)
            val_epoch_time_freq += loss_TFC.detach().cpu()/len(x_t)
            val_epoch_loss += loss.detach().cpu()/len(x_t)
        
        print('\nValidation losses')
        print('Time consistency loss:', epoch_time)
        print('Frequency consistency loss:', epoch_freq)
        print('Time-freq consistency loss:', epoch_time_freq)
        print('Total loss:', epoch_loss)
        print('Classification loss:', epoch_class)
        val_time_loss_total.append(epoch_time)
        val_freq_loss_total.append(epoch_freq)
        val_time_freq_loss_total.append(epoch_time_freq)
        val_loss_total.append(epoch_loss)
        val_class_loss_total.append(epoch_class)



    losses = {
        'train': {
            'time_loss': time_loss_total,
            'freq_loss': freq_loss_total,
            'time_freq_loss': time_freq_loss_total,
            'class_loss': class_loss_total,
            'loss': loss_total},
        'val': {
            'time_loss': val_time_loss_total,
            'freq_loss': val_freq_loss_total,
            'time_freq_loss': val_time_freq_loss_total,
            'class_loss': val_class_loss_total,
            'loss': val_loss_total}
        }
    return model, losses

