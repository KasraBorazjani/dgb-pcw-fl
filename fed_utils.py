import torch
import torch.nn as nn
from models import CustomRNAModel
from copy import deepcopy
import numpy as np

def return_scaled_weights(model, client_batch_num:int, total_batch_num:int):

    coeff = client_batch_num/total_batch_num
    final_scaled_weights = {}
    layer_keys = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']
    # layer_keys = ['fc2', 'fc7', 'fc8', 'fc9', 'fc10',
    #               'fc11', 'fc12', 'fc13', 'fc14', 'fc15',
    #               'fc16', 'fc17', 'fc18', 'fc19']

    for layer_key in model.layer_keys:
        final_scaled_weights[layer_key] = {}

    with torch.no_grad():
        for layer_name in layer_keys:
            final_scaled_weights[layer_name] = {}
            # print(layer_name)
            # print(model.track_layers[layer_name])
            final_scaled_weights[layer_name]['weight'] = coeff * model.track_layers[layer_name].weight.data
            final_scaled_weights[layer_name]['bias'] = coeff * model.track_layers[layer_name].bias.data

    return final_scaled_weights


def sum_scaled_weights(local_scaled_weights, layer_keys):
    final_parameters = {}
    # layer_keys = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']
    # layer_keys = ['fc2', 'fc7', 'fc8', 'fc9', 'fc10',
    #               'fc11', 'fc12', 'fc13', 'fc14', 'fc15',
    #               'fc16', 'fc17', 'fc18', 'fc19']

    for layer_key in layer_keys:
        final_parameters[layer_key] = {}
    
    for layer_key in layer_keys:
        final_parameters[layer_key]['weight'] = local_scaled_weights[0][layer_key]['weight'] + local_scaled_weights[1][layer_key]['weight'] + local_scaled_weights[2][layer_key]['weight']
        final_parameters[layer_key]['bias'] = local_scaled_weights[0][layer_key]['bias'] + local_scaled_weights[1][layer_key]['bias'] + local_scaled_weights[2][layer_key]['bias']
    
    return final_parameters

def set_layer_weights(model, parameters):
    # layer_keys = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5']
    layer_keys = ['fc2', 'fc7', 'fc8', 'fc9', 'fc10',
                  'fc11', 'fc12', 'fc13', 'fc14', 'fc15',
                  'fc16', 'fc17', 'fc18', 'fc19']
    with torch.no_grad():
        for layer_name in layer_keys:
            model.track_layers[layer_name].weight.data *= 0
            model.track_layers[layer_name].bias.data *= 0
            model.track_layers[layer_name].weight.data += parameters[layer_name]['weight']
            model.track_layers[layer_name].bias.data += parameters[layer_name]['bias']

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

def classifier_to_modality_mapper(classifier):
    mapping_dict = {'mrna':["mrna"], 'image':['image'], 'clinical':['clinical'], 'mrna_image':['mrna', 'image'], 'mrna_clinical':['mrna', 'clinical'],
                    'image_clinical':['image', 'clinical'], 'mrna_image_clinical':['mrna', 'image', 'clinical']}
    return mapping_dict[classifier]

def is_full_modal(modalities):
    if(("mrna" in modalities) and ("image" in modalities) and ("clinical" in modalities)):
        return True
    else:
        return False

def is_bi_modal(modalities):
    if(("mrna" in modalities) and ("image" in modalities)):
        return True
    else:
        return False

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

def random_average_attention_weights(global_model_encoder, encoder_dict, dataset_sizes, modality, device):
    selected_layer = "fc1" if np.random.random(size=1)>0.5 else "fc2"
    with torch.no_grad():
        for key in global_model_encoder.state_dict().keys():
            # print(f"aggregating {key} layer")
            if selected_layer in key:
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

def average_encoder_weights_regularized(global_model_encoder, encoder_dict, dataset_sizes, modality, gamma):
    with torch.no_grad():
        for key in global_model_encoder.state_dict().keys():
            sum_of_sizes = 0
            global_model_encoder.state_dict()[key] *= gamma
            aggregation_result = torch.zeros(global_model_encoder.state_dict()[key].size())
            for (cohort_id, cohort_weights) in encoder_dict.items():
                
                # print(cohort_weights.state_dict()[key])
                aggregation_result += cohort_weights.state_dict()[key].data.clone() * dataset_sizes[cohort_id]
                sum_of_sizes += dataset_sizes[cohort_id]
                # print(dataset_sizes[cohort_id])
            # print(sum_of_sizes)
            # if modality == 'mrna':
            #     if key == 'fc1.weight':
            #         print(global_model_encoder.state_dict()[key]/sum_of_sizes)
            aggregation_result /= sum_of_sizes
            global_model_encoder.state_dict()[key] += (1-gamma) * aggregation_result

            # if modality == 'mrna':
            #     if key == 'fc1.weight':
            #         print(global_model_encoder.state_dict()[key])
    
    return global_model_encoder

def average_classifier_weights(global_model_classifier, classifier_dict, dataset_sizes, modality):

    for key in global_model_classifier.state_dict().keys():
        sum_of_sizes = 0
        global_model_classifier.state_dict()[key] = torch.zeros(global_model_classifier.state_dict()[key].size())
        for (cohort_id, cohort_weights) in classifier_dict.items():
            
            # torch.save(f'{cohort_id}_{modality}_weights.csv')
            global_model_classifier.state_dict()[key] += cohort_weights.state_dict()[key] * dataset_sizes[cohort_id]
            sum_of_sizes += dataset_sizes[cohort_id]
            # print(dataset_sizes[cohort_id])
        # print(sum_of_sizes)
        # print(global_model_encoder.state_dict()[key]/sum_of_sizes)
        global_model_classifier.state_dict()[key] = global_model_classifier.state_dict()[key]/sum_of_sizes
    
    return global_model_classifier