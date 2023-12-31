import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
from collections import defaultdict
from .utils import modality_to_classifier_mapper, create_client, encoder_to_classifier_mapper
from itertools import chain, combinations


def calc_dotproduct(gradient_1, gradient_2):
    return torch.mm(gradient_1.view(1, -1), gradient_2.view(-1, 1)).item()


def calc_submodel_dotproduct(local_submodel, new_global_submodel, old_global_submodel):

    with torch.no_grad():

        submodel_dp = 0

        for layer_key in local_submodel.state_dict().keys():

            local_sm_key_grad = local_submodel.state_dict()[layer_key] - old_global_submodel.state_dict()[layer_key]
            global_sm_key_grad = new_global_submodel.state_dict()[layer_key] - old_global_submodel.state_dict()[layer_key]

            submodel_dp += torch.mm(local_sm_key_grad.view(1, -1), global_sm_key_grad.view(-1, 1)).item()
            # submodel_dp += calc_dotproduct(local_sm_key_grad, global_sm_key_grad)
        
        return submodel_dp


def calc_model_dotproduct(client,
                          new_global_model:nn.Module,
                          old_global_model:nn.Module,
                          new_global_classifiers,
                          old_global_classifiers):
    
    model_dp = 0
    global_model_children = [name for name, module in old_global_model.named_children()]
    iterated_children_list = []

    for name, module in client['model'].named_children():

        if 'classifier' in name:
            classifier_str = encoder_to_classifier_mapper(iterated_children_list)
            model_dp += calc_submodel_dotproduct(module, new_global_classifiers[classifier_str], old_global_classifiers[classifier_str])

        if name in global_model_children:
            model_dp += calc_submodel_dotproduct(module, new_global_model.get_submodule(name), old_global_model.get_submodule(name))
        
        iterated_children_list.append(name)
    
    return model_dp