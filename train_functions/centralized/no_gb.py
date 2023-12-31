import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import argparse
import numpy as np

def validate_model(model:nn.Module,
                   valid_loader:DataLoader,
                   args:argparse.Namespace,
                   device:torch.device,
                   criterion:nn.modules.loss._Loss,
                   val_acc:list[float], 
                   val_loss:list[float],
                   valid_loss_min:float):
    
    '''
    Validate a model

    Inputs:
        model:nn.Module                           -> original model being trained,
        valid_loader:DataLoader                   -> dataloader of validation dataset,
        args:argparse.Namespace                   -> arguments including acc_used and steps_per_decay,
        device:torch.device                       -> GPU/CPU where the computation takes place,
        criterion:nn.modules.loss._Loss           -> loss function,
        optimizer:optim.Optimizer                 -> optimizer object,
        scheduler:optim.lr_scheduler._LRScheduler -> learning rate scheduler object,
        valid_acc:list[float]                     -> train accuracy list, 
        valid_loss:list[float]                    -> train loss list,
        valid_loss_min:float                      -> latest minimum validation loss

    Returns:
        network_learned:bool                      -> indicates whether the network has improved (validation loss has decreased)
    '''
    
    batch_loss = 0
    total_t = 0
    correct_t = 0

    with torch.no_grad():
        model.eval()
        for data_t, target_t in (valid_loader):

            # data_t = data_t.unsqueeze(1)

            ### To device
            if args.acc_used:
                data_t = data_t.to(device)
                target_t = target_t.to(device)

            ### Fwd pass
            outputs_t = model(data_t)

            ### Print Stats
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _, pred_t = torch.max(outputs_t, dim=1)
            _, target_t_label = torch.max(target_t, dim=1)
            correct_t += torch.sum(pred_t==target_t_label).item()
            total_t += target_t.size(0)
        
        val_acc.append(100*correct_t/total_t)
        val_loss.append(batch_loss/len(valid_loader))
        network_learned = batch_loss < valid_loss_min
    
    return network_learned


def train_model(model:nn.Module,
                   train_loader:DataLoader,
                   args:argparse.Namespace,
                   device:torch.device,
                   criterion:nn.modules.loss._Loss,
                   optimizer:optim.Optimizer,
                   scheduler:optim.lr_scheduler._LRScheduler,
                   train_acc:list[float], 
                   train_loss:list[float],
                   total_steps_taken:int):

    '''
    Train a model for one epoch iterating through the complete train dataset.

    Inputs:
        model:nn.Module                           -> original model being trained,
        train_loader:DataLoader                   -> dataloader of train dataset,
        args:argparse.Namespace                   -> arguments including acc_used and steps_per_decay,
        device:torch.device                       -> GPU/CPU where the computation takes place,
        criterion:nn.modules.loss._Loss           -> loss function,
        optimizer:optim.Optimizer                 -> optimizer object,
        scheduler:optim.lr_scheduler._LRScheduler -> learning rate scheduler object,
        train_acc:list[float]                     -> train accuracy list, 
        train_loss:list[float]                    -> train loss list,
        total_steps_taken:int                     -> total steps taken (for scheduler activation)

    Returns:
        nothing
    '''

    correct = 0
    total = 0
    running_loss = 0.0
    model.train()
    
    for batch_idx, (data_, target_) in enumerate(train_loader):

        if args.acc_used:
            data_ = data_.to(device)
            target_ = target_.to(device)
        
        ### Zeroing gradients
        optimizer.zero_grad()

        ### Fwd pass
        # outputs = model(data_)
        outputs = model(data_)

        # print(outputs)
        # print(target_)
        # raise "kirekhar"
        
        ### Gradient calc
        loss = criterion(outputs, target_)
        loss.backward()
        optimizer.step()

        ### Acquire metrics
        running_loss += loss.item()
        _, pred = torch.max(outputs, dim=1)
        _, target_label = torch.max(target_, dim=1)
        correct += torch.sum(pred==target_label).item()
        total += target_.size(0)
        
        ### Decrease the learning rate based on the number of steps taken
        # if total_steps_taken%args.steps_per_decay==0:
        if (total_steps_taken+1)%len(train_loader)==0:
            scheduler.step()

        ### Zero gradients
        total_steps_taken += 1
                
    train_acc.append(100*correct/total)
    train_loss.append(running_loss/len(train_loader))

    return total_steps_taken

        
