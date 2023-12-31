import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random
import math
from collections import defaultdict

from dataset_utils import collect_labels_from_df, remove_unwanted_labels, map_to_one_hot, map_to_one_hot_binary
from datasets.dataset_classes import CustomClinicalDatasetStratified


def build_clin_train_test_strat(fraction, random_state=None):
    if random_state:
        random.seed(random_state)
    
    final_df = pd.read_csv('complete_clinical_plus_stage_balanced.csv', delimiter=",")
    final_df = remove_unwanted_labels(final_df)
    final_df['stage'] = final_df.stage.map(lambda x: map_to_one_hot_binary(x))
    features = final_df.drop(columns=['stage', 'pid']).astype(np.float32).values
    labels = final_df.stage.values

    indices_per_label = defaultdict(list)

    for index, label in enumerate(labels):
        # print(np.argmax(label))
        indices_per_label[np.argmax(label)].append(index)
    
    first_set_indices, second_set_indices = list(), list()

    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    
    first_set_inputs = list(map(features.__getitem__, first_set_indices))
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = list(map(features.__getitem__, second_set_indices))
    second_set_labels = list(map(labels.__getitem__, second_set_indices))

    train_dataset = CustomClinicalDatasetStratified(first_set_inputs, first_set_labels)
    test_dataset = CustomClinicalDatasetStratified(second_set_inputs, second_set_labels)

    return train_dataset, test_dataset