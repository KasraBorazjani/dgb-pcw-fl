import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from collections import defaultdict


def validate_model_gb(model:nn.Module,
                   valid_loader:DataLoader,
                   args:argparse.Namespace,
                   device:torch.device,
                   criterion:nn.modules.loss._Loss,
                   val_acc:list[float], 
                   val_loss:list[float],
                   valid_loss_min:float,
                   modality:str):
    
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
        modality:str                              -> modality over which validation is taking place

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
            outputs_t = model(data_t, modality)

            ### Print Stats
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            _, pred_t = torch.max(outputs_t, dim=1)
            _, target_t_label = torch.max(target_t, dim=1)
            correct_t += torch.sum(pred_t==target_t_label).item()
            total_t += target_t.size(0)
        
        val_acc.append(100*correct_t/total_t)
        val_loss.append(batch_loss/len(valid_loader))
        # network_learned = batch_loss < valid_loss_min


def train_model_gb(model:nn.Module,
                   train_loader:DataLoader,
                   args:argparse.Namespace,
                   device:torch.device,
                   criterion:nn.modules.loss._Loss,
                   optimizer:optim.Optimizer,
                   scheduler:optim.lr_scheduler._LRScheduler,
                   train_acc:list[float], 
                   train_loss:list[float],
                   total_steps_taken:int,
                   modality:str,
                   train_mode:str="train"):

    '''
    Train a model for one epoch iterating through the complete train dataset.

    Inputs:
        model:nn.Module                           -> original model being trained,
        train_loader:DataLoader                   -> dataloader of train dataset,
        args:argparse.Namespace                   -> arguments including acc_used and steps_per_decay,
        device:torch.device                       -> GPU/CPU where the computation takes place,
        criterion:nn.modules.loss._Loss           -> loss function,
        optimizer:optim.Optimizer                 -> optimizer object,
        scheduler:optim.lr_scheduler._LRScheduler -> learning rate scheduler object <not active>,
        train_acc:list[float]                     -> train accuracy list, 
        train_loss:list[float]                    -> train loss list,
        total_steps_taken:int                     -> total steps taken (for scheduler activation) <not active>
        modality:str                              -> modality classifier to train with
        train_mode:str <deactive atm>             -> whether the model is being trained or the weights are being estimated (to have scheduler or not)

    Returns:
        total_steps_taken: int                    -> total steps taken after training (for scheduler control)
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
        outputs = model(data_, modality)

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
        
        ### Decrease the learning rate based on the number of steps taken (deactivated atm)
        # if total_steps_taken%args.steps_per_decay==0:
        # if (total_steps_taken+1)%(len(model.modalities)*len(train_loader))==0 and train_mode=="train":
        #     scheduler.step()

        ### Zero gradients
        # total_steps_taken += 1
                
    train_acc.append(100*correct/total)
    train_loss.append(running_loss/len(train_loader))

    ## remove model hooks

    # return total_steps_taken

def train_and_validate_gb(model:nn.Module,
                       train_loader:DataLoader,
                       valid_loader:DataLoader,
                       args:argparse.Namespace,
                       device:torch.device,
                       criterion:nn.modules.loss._Loss,
                       optimizer:optim.Optimizer,
                       scheduler:optim.lr_scheduler._LRScheduler,
                       train_acc:list[float], 
                       train_loss:dict,
                       val_acc:list[float],
                       val_loss:dict,
                       total_steps_taken:int,
                       valid_loss_min:float,
                       modality_weights:dict,
                       train_mode:str="train"):

    '''
    Train a model for multiple epochs and validate the performance.

    Inputs:
        model:nn.Module                           -> original model being trained,
        train_loader:DataLoader                   -> dataloader of train dataset,
        args:argparse.Namespace                   -> arguments including acc_used and steps_per_decay,
        device:torch.device                       -> GPU/CPU where the computation takes place,
        criterion:nn.modules.loss._Loss           -> loss function,
        optimizer:optim.Optimizer                 -> optimizer object,
        scheduler:optim.lr_scheduler._LRScheduler -> learning rate scheduler object <not active>,
        train_acc:list[float]                     -> train accuracy list, 
        train_loss:list[float]                    -> train loss list,
        total_steps_taken:int                     -> total steps taken (for scheduler activation) <not active>
        modality:str                              -> modality classifier to train with
        train_mode:str <deactive atm>             -> whether the model is being trained or the weights are being estimated

    Returns:
        total_steps_taken: int                    -> total steps taken after training (for scheduler control)
    '''
    if train_mode=="train":
        num_train_epochs = args.super_epoch_len
    elif train_mode=="we":
        num_train_epochs = args.we_epoch_len

    for modality in model.modalities+["multimodal"]:
        
        for epoch in range(num_train_epochs):

            train_model_gb(model, train_loader, args, device,
                        criterion, optimizer, scheduler, train_acc,
                        train_loss[modality], total_steps_taken, modality, train_mode)
        
        validate_model_gb(model, valid_loader, args, device,
                        criterion, val_acc, val_loss[modality],
                        valid_loss_min, modality)
    

def encoder_hooker(gradient_weight):

    def hook(module, grad_input, grad_output):
        for key in module.state_dict().keys():
            layer_name, layer_attr = key.split(".")
            getattr(getattr(module, layer_name), layer_attr).grad *= gradient_weight
    
    return hook


def module_hooker(gradient_weight):

    def hook(module, grad_input, grad_output):
        for key in module.state_dict().keys():
            encoder_name, layer_name, layer_attr = key.split(".")
            getattr(getattr(getattr(module, encoder_name), layer_name), layer_attr).grad *= gradient_weight
        
    return hook


def hook_initializer(model:nn.Module, weight_dict:dict):

    handle_dict = {}

    for modality in model.modalities:
        modality_hook = encoder_hooker(weight_dict[modality])
        modality_handle = getattr(model, modality+"_encoder").register_full_backward_hook(modality_hook)
        handle_dict[modality] = modality_handle
    
    return handle_dict


def network_hook_initializer(model:nn.Module, modality_weight:float):

    model_hook = module_hooker(modality_weight)
    handle = model.register_full_backward_hook(model_hook)

    return handle


def hook_remover(handle_dict:dict):

    for modality_handle in handle_dict.values():
        modality_handle.remove()


def calculate_overfitting(train_loss:list[float],
                          train_gb_loss:list[float],
                          val_loss:list[float],
                          val_gb_loss:list[float]):

    return (val_gb_loss[-1] - train_gb_loss[-1]) - (val_loss[-1] - train_loss[-1])


def calculate_generalization(val_loss:list[float],val_gb_loss:list[float]):

    return val_gb_loss[-1] - val_loss[-1]

def calculate_ogr2(train_loss:list[float],
                   train_gb_loss:list[float],
                   val_loss:list[float],
                   val_gb_loss:list[float]):

    overfitting = calculate_overfitting(train_loss, train_gb_loss, val_loss, val_gb_loss)
    generalization = calculate_generalization(val_loss, val_gb_loss)

    return abs(generalization)/(abs(overfitting) ** 2)

def calculate_weights(train_loss_dict, train_gb_loss_dict, val_loss_dict, val_gb_loss_dict, modalities, modality_weights):

    modality_ogrs = {}

    for modality in modalities+["multimodal"]:
        modality_ogrs[modality] = calculate_ogr2(train_loss_dict[modality],
                                                 train_gb_loss_dict[modality],
                                                 val_loss_dict[modality],
                                                 val_gb_loss_dict[modality])
    # modality_weights = {}
    sum_ogrs = np.sum(np.asarray(list(modality_ogrs.values())))
    for modality in modalities+["multimodal"]:
        modality_weights[modality].append(modality_ogrs[modality] / sum_ogrs)


def initialize_loss_dicts(modalities):
    train_gb_loss_dict = defaultdict(list)
    val_gb_loss_dict = defaultdict(list)
    train_loss_dict = defaultdict(list)
    val_loss_dict = defaultdict(list)

    for modality in modalities+["multimodal"]:
        val_loss_dict[modality] = []
        train_loss_dict[modality] = []
        val_gb_loss_dict[modality] = []
        train_gb_loss_dict[modality] = []
    
    return train_gb_loss_dict, val_gb_loss_dict, train_loss_dict, val_loss_dict
    
    
def initialize_weight_dict(modalities):

    modality_weights = defaultdict(list)

    for modality in modalities+["multimodal"]:
        modality_weights[modality] = [1.0]
    
    return modality_weights
        
    
def train_simul_model_gb(model:nn.Module,
                   train_loader:DataLoader,
                   args:argparse.Namespace,
                   device:torch.device,
                   criterion:nn.modules.loss._Loss,
                   optimizer:optim.Optimizer,
                   scheduler:optim.lr_scheduler._LRScheduler,
                   train_acc:list[float], 
                   train_loss_dict:dict,
                   total_steps_taken:int,
                   weight_dict:dict,
                   train_mode:str="train"):

    '''
    Train a model for one epoch iterating through the complete train dataset.

    Inputs:
        model:nn.Module                           -> original model being trained,
        train_loader:DataLoader                   -> dataloader of train dataset,
        args:argparse.Namespace                   -> arguments including acc_used and steps_per_decay,
        device:torch.device                       -> GPU/CPU where the computation takes place,
        criterion:nn.modules.loss._Loss           -> loss function,
        optimizer:optim.Optimizer                 -> optimizer object,
        scheduler:optim.lr_scheduler._LRScheduler -> learning rate scheduler object <not active>,
        train_acc:list[float]                     -> train accuracy list, 
        train_loss:list[float]                    -> train loss list,
        total_steps_taken:int                     -> total steps taken (for scheduler activation) <not active>
        modality:str                              -> modality classifier to train with
        train_mode:str <deactive atm>             -> whether the model is being trained or the weights are being estimated (to have scheduler or not)

    Returns:
        total_steps_taken: int                    -> total steps taken after training (for scheduler control)
    '''

    correct = 0
    total = 0

    running_mm_loss = 0.0
    running_mrna_loss = 0.0
    running_image_loss = 0.0
    running_clinical_loss = 0.0

    model.train()
    
    for batch_idx, (data_, target_) in enumerate(train_loader):

        if args.acc_used:
            data_ = data_.to(device)
            target_ = target_.to(device)
        
        ### Zeroing gradients
        optimizer.zero_grad() 


        ### Fwd pass
        # outputs = model(data_)
        mm_output, mrna_output, image_output, clinical_output = model(data_)

        # print(outputs)
        # print(target_)
        # raise "kirekhar"

        ### Calc loss
        loss_mm = criterion(mm_output, target_)
        loss_mrna = criterion(mrna_output, target_)
        loss_image = criterion(image_output, target_)
        loss_clinical = criterion(clinical_output, target_)

        ### Acquire losses
        running_mm_loss += loss_mm.item()
        running_mrna_loss += loss_mrna.item()
        running_image_loss += loss_image.item()
        running_clinical_loss += loss_clinical.item()

        ### Weigh losses 
        loss_mm *= weight_dict["multimodal"][-1]
        loss_mrna *= weight_dict["mrna"][-1]
        loss_image *= weight_dict["image"][-1]
        loss_clinical *= weight_dict["clinical"][-1]

        ### Backprop weighted losses
        loss_mm.backward(retain_graph=True)
        loss_mrna.backward(retain_graph=True)
        loss_image.backward(retain_graph=True)
        loss_clinical.backward()

        #### Take a step with the learning rate given
        optimizer.step()

        ### Calculate Accuracy
        _, pred = torch.max(mm_output, dim=1)
        _, target_label = torch.max(target_, dim=1)
        correct += torch.sum(pred==target_label).item()
        total += target_.size(0)
        
        ### Decrease the learning rate based on the number of steps taken (deactivated atm)
        # if total_steps_taken%args.steps_per_decay==0:
        if (total_steps_taken+1)%(len(model.modalities)*len(train_loader))==0 and train_mode=="we":
            scheduler.step()

        ### Zero gradients
        # total_steps_taken += 1

    if train_mode=="train":      
        train_acc.append(100*correct/total)
        
    train_loss_dict["multimodal"].append(running_mm_loss/len(train_loader))
    train_loss_dict["mrna"].append(running_mrna_loss/len(train_loader))
    train_loss_dict["image"].append(running_image_loss/len(train_loader))
    train_loss_dict["clinical"].append(running_clinical_loss/len(train_loader))

def validate_simul_model_gb(model:nn.Module,
                   valid_loader:DataLoader,
                   args:argparse.Namespace,
                   device:torch.device,
                   criterion:nn.modules.loss._Loss,
                   val_acc:list[float], 
                   val_loss_dict:dict,
                   valid_loss_min:float,
                   train_mode:str):
    
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
        modality:str                              -> modality over which validation is taking place

    Returns:
        network_learned:bool                      -> indicates whether the network has improved (validation loss has decreased)
    '''
    
    total_t = 0
    correct_t = 0

    mm_batch_loss = 0
    mrna_batch_loss = 0
    image_batch_loss = 0
    clinical_batch_loss = 0

    with torch.no_grad():
        model.eval()
        for data_t, target_t in (valid_loader):

            # data_t = data_t.unsqueeze(1)

            ### To device
            if args.acc_used:
                data_t = data_t.to(device)
                target_t = target_t.to(device)

            ### Fwd pass
            mm_output, mrna_output, image_output, clinical_output = model(data_t)

            ### Print Stats
            mm_loss = criterion(mm_output, target_t)
            mrna_loss = criterion(mrna_output, target_t)
            image_loss = criterion(image_output, target_t)
            clinical_loss = criterion(clinical_output, target_t)

            mm_batch_loss += mm_loss.item()
            mrna_batch_loss += mrna_loss.item()
            image_batch_loss += image_loss.item()
            clinical_batch_loss += clinical_loss.item()

            _, pred_t = torch.max(mm_output, dim=1)
            _, target_t_label = torch.max(target_t, dim=1)
            correct_t += torch.sum(pred_t==target_t_label).item()
            total_t += target_t.size(0)
        
        if train_mode=="train":
            val_acc.append(100*correct_t/total_t)

        val_loss_dict["multimodal"].append(mm_batch_loss/len(valid_loader))
        val_loss_dict["mrna"].append(mrna_batch_loss/len(valid_loader))
        val_loss_dict["image"].append(image_batch_loss/len(valid_loader))
        val_loss_dict["clinical"].append(clinical_batch_loss/len(valid_loader))
        
        # network_learned = batch_loss < valid_loss_min


def train_and_validate_simul_gb(model:nn.Module,
                       train_loader:DataLoader,
                       valid_loader:DataLoader,
                       args:argparse.Namespace,
                       device:torch.device,
                       criterion:nn.modules.loss._Loss,
                       optimizer:optim.Optimizer,
                       scheduler:optim.lr_scheduler._LRScheduler,
                       train_acc:list[float], 
                       train_loss:dict,
                       val_acc:list[float],
                       val_loss:dict,
                       total_steps_taken:int,
                       valid_loss_min:float,
                       modality_weights:dict,
                       train_mode:str="train"):

    '''
    Train a model for multiple epochs and validate the performance.

    Inputs:
        model:nn.Module                           -> original model being trained,
        train_loader:DataLoader                   -> dataloader of train dataset,
        args:argparse.Namespace                   -> arguments including acc_used and steps_per_decay,
        device:torch.device                       -> GPU/CPU where the computation takes place,
        criterion:nn.modules.loss._Loss           -> loss function,
        optimizer:optim.Optimizer                 -> optimizer object,
        scheduler:optim.lr_scheduler._LRScheduler -> learning rate scheduler object <not active>,
        train_acc:list[float]                     -> train accuracy list, 
        train_loss:list[float]                    -> train loss list,
        total_steps_taken:int                     -> total steps taken (for scheduler activation) <not active>
        modality:str                              -> modality classifier to train with
        train_mode:str <deactive atm>             -> whether the model is being trained or the weights are being estimated

    Returns:
        total_steps_taken: int                    -> total steps taken after training (for scheduler control)
    '''
    if train_mode=="train":
        num_train_epochs = args.super_epoch_len
    elif train_mode=="we":
        num_train_epochs = args.we_epoch_len
        
    for epoch in range(num_train_epochs):

        train_simul_model_gb(model, train_loader, args, device,
                    criterion, optimizer, scheduler, train_acc,
                    train_loss, total_steps_taken, modality_weights, train_mode)

        validate_simul_model_gb(model, valid_loader, args, device,
                        criterion, val_acc, val_loss, valid_loss_min, train_mode)


