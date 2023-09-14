import torch
import time

def fit(model, epochs, train_loader, val_loader, optimizer, loss_fn, patience, model_save_dir, device=None):
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        start_time = time.time()
        train_loss_epoch = train(model, train_loader, optimizer, loss_fn, device=device) 
        val_loss_epoch, target_log, pred_log = validate(model, val_loader, loss_fn, device=device)    
    
        train_loss.append(train_loss_epoch)
        val_loss.append(val_loss_epoch)
        
        print(f'Epoch No.{epoch} || Training loss: {str(round(train_loss_epoch, 6))} || Validation loss: {str(round(val_loss_epoch, 6))} || Time: {str(round(time.time() - start_time, 3))} sec')
    
        # early stop
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping at {epoch} epochs')
                break
        
    return train_loss, val_loss
        

def train(model, loader, opt, loss_fn, device=None):
    model.train()
    train_loss_batch = []
    
    for seq, target in loader: 
        if len(seq.shape) == 2: # for testing dataloader: batch_size = 1
            seq, target = seq.unsqueeze(0), target.unsqueeze(0)
    
        seq, target = seq.to(device), target[:, :, -1].squeeze(-1).to(device)
        opt.zero_grad()
        pred = model(seq)
        
        loss = loss_fn(pred, target)
        loss.backward()
        opt.step()
        train_loss_batch.append(loss.item())
        
    train_loss_epoch = sum(train_loss_batch) / len(train_loss_batch)
    
    return train_loss_epoch


def validate(model, loader, criterion, device=None):
    model.eval()
    val_loss_batch = []
    target_log = torch.zeros(0).to(device)
    pred_log = torch.zeros(0).to(device)
    
    with torch.no_grad():
        for seq, target in loader: 
            if len(seq.shape) == 2: # for testing dataloader: batch_size = 1
                seq, target = seq.unsqueeze(0), target.unsqueeze(0)
            seq, target = seq.to(device), target[:, :, -1].squeeze(-1).to(device)
            pred = model(seq)
            loss = criterion(pred, target)
            
            val_loss_batch.append(loss.item())
            target_log = torch.cat([target_log, target])
            pred_log = torch.cat([pred_log, pred])
            
        val_loss_epoch = sum(val_loss_batch) / len(val_loss_batch)

    return val_loss_epoch, target_log, pred_log