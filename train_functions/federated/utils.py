import torch
import torch.nn as nn
import os
from models import CustomRNAModel
from copy import deepcopy
import numpy as np
from torch.utils.data import DataLoader
from models.multimodal import *
from torch import optim


def model_assigner(modalities):

    if modality_to_classifier_mapper(modalities) == 'mrna_image_clinical':
        return CustomMultiModalModel()
    elif modality_to_classifier_mapper(modalities) == 'mrna_image':
        return CustomRNAImgModel()
    elif modality_to_classifier_mapper(modalities) == 'mrna_clinical':
        return CustomRNAClinModel()
    elif modality_to_classifier_mapper(modalities) == 'image_clinical':
        return CustomImgClinModel()
    elif modality_to_classifier_mapper(modalities) == 'mrna':
        return CustomRNAModel()
    elif modality_to_classifier_mapper(modalities) == 'image':
        return CustomImgModel()
    elif modality_to_classifier_mapper(modalities) == 'clinical':
        return CustomClinicalModel()
    else:
        print("Wrong modalities. Model cannot be created.")
        raise KeyError


def average_encoder_weights(global_model_encoder, encoder_dict, dataset_sizes, modality, device):
    with torch.no_grad():
        for key in global_model_encoder.state_dict().keys():
            # print(f"aggregating {key} layer")
            if not 'num_batches_tracked' in key:
                sum_of_sizes = 0
                temp = torch.zeros_like(global_model_encoder.state_dict()[key]).to(device=device)
                # global_model_encoder.state_dict()[key] = torch.zeros(global_model_encoder.state_dict()[key].size())
                for (cohort_id, cohort_weights) in encoder_dict.items():
                    
                    # print(cohort_weights.state_dict()[key])
                    # global_model_encoder.state_dict()[key] += cohort_weights.state_dict()[key].data.clone() * dataset_sizes[cohort_id]
                    # print(temp.get_device())
                    # print(cohort_weights.state_dict()[key].get_device())
                    temp += cohort_weights.state_dict()[key] * dataset_sizes[cohort_id]
                    sum_of_sizes += dataset_sizes[cohort_id]
                    # print(dataset_sizes[cohort_id])
                # print(sum_of_sizes)
                # if modality == 'mrna':
                #     if key == 'fc1.weight':
                #         print(global_model_encoder.state_dict()[key]/sum_of_sizes)
                temp /= sum_of_sizes
                global_model_encoder.state_dict()[key].data.copy_(temp)
                # if modality == 'mrna':
                #     if key == 'fc1.weight':
                #         print(global_model_encoder.state_dict()[key])
    
    return global_model_encoder

def modality_to_classifier_mapper(modalities):
    if("mrna" in modalities):
        if("image" in modalities):
            if("clinical" in modalities):
                return "mrna_image_clinical"
            else:
                return "mrna_image"
        elif("clinical" in modalities):
            return "mrna_clinical"
        else:
            return "mrna"
    elif("image" in modalities):
        if("clinical" in modalities):
            return "image_clinical"
        else:
            return "image"
    elif("clinical" in modalities):
        return "clinical"
    else:
        raise ValueError


def encoder_to_classifier_mapper(modalities):
    if("mrna_encoder" in modalities):
        if("image_encoder" in modalities):
            if("clinical_encoder" in modalities):
                return "mrna_image_clinical"
            else:
                return "mrna_image"
        elif("clinical_encoder" in modalities):
            return "mrna_clinical"
        else:
            return "mrna"
    elif("image_encoder" in modalities):
        if("clinical_encoder" in modalities):
            return "image_clinical"
        else:
            return "image"
    elif("clinical_encoder" in modalities):
        return "clinical"
    else:
        raise ValueError


def classifier_to_modality_mapper(classifier):
    mapping_dict = {'mrna':["mrna"], 'image':['image'], 'clinical':['clinical'], 'mrna_image':['mrna', 'image'], 'mrna_clinical':['mrna', 'clinical'],
                    'image_clinical':['image', 'clinical'], 'mrna_image_clinical':['mrna', 'image', 'clinical']}
    return mapping_dict[classifier]


def create_client(cohort, clientbuildnum, dataset, args, device):
    print(f"Creating client {clientbuildnum} for cohort {cohort}")
    client = {}
    client['cohort_id'] = (cohort, clientbuildnum)
    client['dataset'] = dataset
    print(f"len dataset = {len(dataset)}")
    client['modalities'] = dataset.modalities
    client['dataloader'] = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
    client['train_loss_memory'] = []
    client['valid_loss_memory'] = []
    client['train_acc_memory'] = []
    client['valid_acc_memory'] = []
    client['valid_loss_min'] = np.inf
    client['total_steps_taken'] = 0
    client['dataset_size'] = len(dataset)
    client['model'] = CustomFederatedModel(dataset.modalities, dataset.column_map)
    if args.acc_used:
        client['model'].to(device)
    if args.lr_decay_rate!=None:
        client['optimizer'] = optim.SGD(client['model'].parameters(), lr=args.init_lr)
        client['scheduler'] = optim.lr_scheduler.ExponentialLR(client['optimizer'], gamma=args.lr_decay_rate)
    else:
        client['optimizer'] = optim.Adam(client['model'].parameters(), lr=args.init_lr)
    
    return client
    


def init_weight_dict(modalities):
    weight_dict = {}
    if len(modalities)==1:
        weight_dict[modalities[0]] = [1.0]
    else:
        for modality in modalities + [modality_to_classifier_mapper(modalities)]:
            weight_dict[modality] = [1.0]
    
    return weight_dict


def create_client_gb(cohort, clientbuildnum, dataset, args, device):

    print(f"Creating client {clientbuildnum} for cohort {cohort}")
    client = {}
    client['cohort_id'] = (cohort, clientbuildnum)
    client['dataset'] = dataset
    print(f"len dataset = {len(dataset)}")
    client['modalities'] = dataset.modalities
    client['dataloader'] = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
    client['train_loss_memory'] = []
    client['valid_loss_memory'] = []
    client['train_acc_memory'] = []
    client['valid_acc_memory'] = []
    client['valid_loss_min'] = np.inf
    client['total_steps_taken'] = 0
    client['dataset_size'] = len(dataset)
    # client['model'] = CustomFederatedModel(dataset.modalities, dataset.column_map)
    client['model'] = model_assigner(client['modalities'])
    client['column_map'] = dataset.column_map

    if args.acc_used:
        client['model'].to(device)
    
    if args.lr_decay_rate!=None:
        client['optimizer'] = optim.SGD(client['model'].parameters(), lr=args.init_lr)
        client['scheduler'] = optim.lr_scheduler.ExponentialLR(client['optimizer'], gamma=args.lr_decay_rate)
    else:
        client['optimizer'] = optim.Adam(client['model'].parameters(), lr=args.init_lr)
    
    client['weight_dict'] = init_weight_dict(client['modalities'])
    
    return client


def initialize_classifiers(args, global_classifiers_dict):

    for classifier_pair in global_classifiers_dict.keys():
        classifier_model_path = os.path.join(args.saved_model_path, 'federated_'+classifier_pair+'_start_model.pt')
        dummy_model = CustomFederatedModel(modalities=classifier_to_modality_mapper(classifier_pair))
        dummy_model.load_state_dict(torch.load(classifier_model_path))
        global_classifiers_dict[classifier_pair] = deepcopy(dummy_model.classifier)
    
    return global_classifiers_dict


def sync_client_with_global(client, global_model, global_classifiers_dict):
    with torch.no_grad():
        for modality in client['modalities']:
            getattr(client['model'], modality+"_encoder").load_state_dict(getattr(global_model, modality+"_encoder").state_dict())
        
        client['model'].classifier.load_state_dict(global_classifiers_dict[modality_to_classifier_mapper(client['modalities'])].state_dict())


def train_one_epoch(client, args, device, criterion):

    running_loss = 0.0
    correct = 0
    total = 0
    client['model'].train()
    client['optimizer'] = optim.Adam(client['model'].parameters(), client['optimizer'].state_dict()['param_groups'][0]['lr'])

    for batch_idx, (data_, target_) in enumerate(client['dataloader']):
        
        if batch_idx < args.max_sgd_per_epoch:
            if args.acc_used:
                data_ = data_.to(device)
                target_ = target_.to(device)
            
            ### Zero Our Parameter Gradients
            client['optimizer'].zero_grad()

            ### FWD + BKWD + OPTIM
            outputs = client['model'](data_)
            # print(outputs)
            # raise ValueError
            # print(outputs)
            loss = criterion(outputs, target_)
            loss.backward()
            # print(getattr(client['model'].mrna_encoder, '1').weight.grad)
            client['optimizer'].step()

            ### Print Stats
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            _, target_label = torch.max(target_, dim=1)
            correct += torch.sum(pred==target_label).item()
            total += target_.size(0)
            # if(batch_idx) == 3:
            #     print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch, NUM_EPOCHS, batch_idx, total_steps, loss.item()))
            
            client['total_steps_taken'] += 1

            ## Scheduler step
            # if((args.lr_decay_rate is not None) and(client['total_steps_taken'] % args.steps_per_decay == 0)):
            if((args.lr_decay_rate is not None) and(client['total_steps_taken'] % len(client['dataloader']) == 0)):
                client['scheduler'].step()
        
    client['train_acc_memory'].append(100*correct/total)
    client['train_loss_memory'].append(running_loss/len(client['dataloader']))


def validate_model(client, validation_dataloaders, args, device, criterion):
    with torch.no_grad():
        batch_loss = 0
        total_t = 0
        correct_t = 0
        client['model'].eval()
        for val_loader in validation_dataloaders:
            for val_batch_idx, (data_t, target_t) in enumerate(val_loader):

                if args.acc_used:
                    data_t = data_t.to(device)  
                    target_t = target_t.to(device)
                outputs_t = client['model'](data_t)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                _, target_t_label = torch.max(target_t, dim=1)
                correct_t += torch.sum(pred_t==target_t_label).item()
                total_t += target_t.size(0)
            
        client['valid_acc_memory'].append(100*correct_t/total_t)
        client['valid_loss_memory'].append(batch_loss/np.sum([len(val_ldr) for val_ldr in validation_dataloaders]))
        network_learned = client['valid_loss_memory'][-1] < client['valid_loss_min']
        
    return network_learned


def append_model_to_network_cache(client, trained_encoders_dict, trained_classifiers_dict):

    for modality in client['modalities']:
        trained_encoders_dict[modality][client['cohort_id']] = getattr(client['model'], modality+"_encoder")
    # network[modality_to_classifier_mapper(client['modalities'])].append({'classifier':client['model'].classifier, 'dataset_size':client['dataset_size']})
    
    trained_classifiers_dict[modality_to_classifier_mapper(client['modalities'])][client['cohort_id']] = client['model'].classifier


def aggregate_model_parts(global_model, trained_encoders_dict, trained_classifiers_dict, global_classifiers_dict, dataset_sizes_dict, device):
    with torch.no_grad():
        ## Aggregating encoders
        for modality in trained_encoders_dict.keys():
            print(f"aggregating {modality} encoders")
            getattr(global_model, modality+"_encoder").load_state_dict(average_encoder_weights(getattr(global_model, modality+"_encoder"), trained_encoders_dict[modality], dataset_sizes_dict, modality, device).state_dict())
        
        ## Aggregating classifiers
        for classifier_pair in trained_classifiers_dict.keys():
            global_classifiers_dict[classifier_pair].load_state_dict(average_encoder_weights(global_classifiers_dict[classifier_pair], trained_classifiers_dict[classifier_pair], dataset_sizes_dict, classifier_pair, device).state_dict())
        
        ## Sync with global model
        global_model.classifier.load_state_dict(global_classifiers_dict[modality_to_classifier_mapper(list(trained_encoders_dict.keys()))].state_dict())


def validate_global_model(global_model,
                          validation_dataloaders,
                          args,
                          device,
                          criterion,
                          global_valid_acc_memory,
                          global_valid_loss_memory):
    eval_batch_loss = 0
    eval_total_t = 0
    eval_correct_t = 0
    with torch.no_grad():
        global_model.eval()
        for val_loader in validation_dataloaders:
            for val_batch_idx, (data_t, target_t) in enumerate(val_loader):

                if args.acc_used:
                    data_t = data_t.to(device)
                    target_t = target_t.to(device)
                outputs_t = global_model(data_t)
                loss_t = criterion(outputs_t, target_t)
                eval_batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                _, target_t_label = torch.max(target_t, dim=1)
                eval_correct_t += torch.sum(pred_t==target_t_label).item()
                eval_total_t += target_t.size(0)
            
        global_valid_acc_memory.append(100*eval_correct_t/eval_total_t)
        global_valid_loss_memory.append(eval_batch_loss/np.sum([len(val_ldr) for val_ldr in validation_dataloaders]))


##########
### GB ###
##########

def train_one_epoch_gb(client, args, device, criterion):

    running_loss = 0.0
    correct = 0
    total = 0
    client['model'].train()
    client['optimizer'] = optim.Adam(client['model'].parameters(), lr=client['optimizer'].state_dict()['param_groups'][0]['lr'])

    for batch_idx, (data_, target_) in enumerate(client['dataloader']):
        
        if batch_idx < args.max_sgd_per_epoch:
            if args.acc_used:
                data_ = data_.to(device)
                target_ = target_.to(device)
            
            data_unpacked_ = unpack_data(data_, client['modalities'], client['column_map'], unpack_mode="train")
            
            ### Zero Our Parameter Gradients
            client['optimizer'].zero_grad()

            ### FWD + BKWD + OPTIM
            outputs = client['model'](data_unpacked_)
            # print(outputs)
            # raise ValueError
            # print(outputs)
            loss = criterion(outputs, target_)
            loss.backward()
            scale_gradients_bckwd(client['model'], client['weight_dict'], client['modalities'])
            # print(getattr(client['model'].mrna_encoder, '1').weight.grad)
            client['optimizer'].step()

            ### Print Stats
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            _, target_label = torch.max(target_, dim=1)
            correct += torch.sum(pred==target_label).item()
            total += target_.size(0)
            # if(batch_idx) == 3:
            #     print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch, NUM_EPOCHS, batch_idx, total_steps, loss.item()))
            
            client['total_steps_taken'] += 1

            ## Scheduler step
            # if((args.lr_decay_rate is not None) and(client['total_steps_taken'] % args.steps_per_decay == 0)):
            if((args.lr_decay_rate is not None) and(client['total_steps_taken'] % len(client['dataloader']) == 0)):
                client['scheduler'].step()
        
    client['train_acc_memory'].append(100*correct/total)
    client['train_loss_memory'].append(running_loss/len(client['dataloader']))


def validate_model_gb(client, validation_dataloaders, args, device, criterion):
    with torch.no_grad():
        batch_loss = 0
        total_t = 0
        correct_t = 0
        client['model'].eval()
        for val_loader in validation_dataloaders:
            for val_batch_idx, (data_t, target_t) in enumerate(val_loader):

                if args.acc_used:
                    data_t = data_t.to(device)  
                    target_t = target_t.to(device)
                
                data_t_unpacked = unpack_data(data_t, client['modalities'], client['column_map'], unpack_mode="valid")

                outputs_t = client['model'](data_t_unpacked)
                loss_t = criterion(outputs_t, target_t)
                batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                _, target_t_label = torch.max(target_t, dim=1)
                correct_t += torch.sum(pred_t==target_t_label).item()
                total_t += target_t.size(0)
            
        client['valid_acc_memory'].append(100*correct_t/total_t)
        client['valid_loss_memory'].append(batch_loss/np.sum([len(val_ldr) for val_ldr in validation_dataloaders]))
        network_learned = client['valid_loss_memory'][-1] < client['valid_loss_min']
        
    return network_learned


def validate_global_model_gb(global_model,
                             validation_dataloaders,
                             args,
                             device,
                             criterion,
                             global_valid_acc_memory,
                             global_valid_loss_memory,
                             network_modalities,
                             network_column_map):

    eval_batch_loss = 0
    eval_total_t = 0
    eval_correct_t = 0

    with torch.no_grad():

        global_model.eval()

        for val_loader in validation_dataloaders:

            for val_batch_idx, (data_t, target_t) in enumerate(val_loader):

                if args.acc_used:
                    data_t = data_t.to(device)
                    target_t = target_t.to(device)

                data_t_unpacked = unpack_data(data_t, network_modalities, network_column_map, unpack_mode="valid")

                outputs_t = global_model(data_t_unpacked)
                loss_t = criterion(outputs_t, target_t)
                eval_batch_loss += loss_t.item()
                _, pred_t = torch.max(outputs_t, dim=1)
                _, target_t_label = torch.max(target_t, dim=1)
                eval_correct_t += torch.sum(pred_t==target_t_label).item()
                eval_total_t += target_t.size(0)
            
        global_valid_acc_memory.append(100*eval_correct_t/eval_total_t)
        global_valid_loss_memory.append(eval_batch_loss/np.sum([len(val_ldr) for val_ldr in validation_dataloaders]))


def scale_gradients_bckwd(model, weight_dict, modalities):
    client_combination_weight = weight_dict[modality_to_classifier_mapper(modalities)][-1]
    for modality in modalities:
        for layer_str in getattr(model, modality+"_encoder").state_dict().keys():
            layer_name, layer_type = layer_str.split(".")
            # print(weight_dict[modality])
            # print(client_combination_weight)
            getattr(getattr(getattr(model, modality+"_encoder"), layer_name), layer_type).grad *= weight_dict[modality][-1] * client_combination_weight
    for layer_str in model.classifier.state_dict().keys():
        layer_name, layer_type = layer_str.split(".")
        getattr(getattr(model.classifier, layer_name), layer_type).grad *= client_combination_weight


def unpack_data(x, modalities, column_map, unpack_mode):
    
    if(unpack_mode == "train"):

        if len(modalities) > 1:

            final_pack = []
            cursor = 0
            for modality in modalities:
                # print(column_map[modality])
                final_pack.append(x[:, cursor:cursor+len(column_map[modality])])
                cursor += len(column_map[modality])
            
            return final_pack

        else:
            return x
        
    elif(unpack_mode=="valid"):
        if len(modalities) > 1:
            final_pack = []
            for modality in modalities:
                if modality=="mrna":
                    final_pack.append(x[:, :20531])
                
                elif modality=="image":
                    final_pack.append(x[:, 20531:20681])
                
                elif modality=="clinical":
                    final_pack.append(x[:, 20681:])
            
            return final_pack
        
        else:
            if modalities[0]=="mrna":
                return x[:, :20531]
                
            elif modalities[0]=="image":
                return x[:, 20531:20681]
            
            elif modalities[0]=="clinical":
                return x[:, 20681:]