import torch
import time
import torch.nn as nn
import os,sys
import numpy as np


def fit(model, epochs, train_loader, val_loader, optimizer, loss_fn, es_patience, model_save_dir, device=None):
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=False)
    train_loss = []
    val_loss = []
    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        
        start_time = time.time()
        train_loss_epoch = train(model, train_loader, optimizer, loss_fn, device=device) 
        val_loss_epoch, target_log, pred_log = validate(model, val_loader, loss_fn, device=device)    
        scheduler.step(train_loss_epoch)
        train_loss.append(np.sqrt(train_loss_epoch))
        val_loss.append(np.sqrt(val_loss_epoch))
        
        print(f'Epoch No.{epoch} || Training loss: {str(round(train_loss[-1], 6))} || Validation loss: {str(round(val_loss[-1], 6))} || Time: {str(round(time.time() - start_time, 3))} sec')
    
        # early stop
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_save_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= es_patience:
                print(f'Early stopping at {epoch} epochs')
                break
        
    return train_loss, val_loss
        

def train(model, loader, opt, loss_fn, device=None):
    model.train()
    train_loss_batch = []
    for i,(seq, target) in enumerate(loader): 
      
      
        seq = seq.to(device) # [B,S,D]
        target = target.to(device) # [B,P,D]
        pred, attns = model(seq) # [B,P,1]
        
        tgt_y = target[:, :, -1] # [B,P,1]
        
        loss = loss_fn(pred, tgt_y)
   
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
        for i,(seq, target) in enumerate(loader):            
            seq = seq.to(device)
            target = target.to(device)
            pred, attns = model(seq) ###[B,P,1]
            tgt_y = target[:, :, -1]##[B,P,1]
            loss = criterion(pred, tgt_y)
            val_loss_batch.append(loss.item())
            target_log = torch.cat([target_log, tgt_y])
            pred_log = torch.cat([pred_log, pred])
            
        val_loss_epoch = sum(val_loss_batch) / len(val_loss_batch)

    return val_loss_epoch, target_log, pred_log
