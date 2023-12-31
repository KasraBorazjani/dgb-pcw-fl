import os
import torch
import numpy as np
import pandas as pd
# from torchvision import transforms as transforms
from torch.utils.data import Dataset
import random
import math
from collections import defaultdict

from dataset_utils import collect_labels_from_df, remove_unwanted_labels, map_to_one_hot, map_to_one_hot_binary
from datasets.dataset_classes import CustomRNADatasetStratified



def create_mrna_train_test_strat(fraction, data_path, random_state=None):
    genes_path = os.path.join(data_path, "Complete_mRNAseq.csv")
    print(genes_path)
    gene_table = pd.read_table(genes_path, delimiter=',', low_memory=False).apply(lambda x: x.astype(str).str.lower())
    gene_table = gene_table.transpose()
    gene_table.reset_index(inplace=True)
    gene_table.rename(columns={'index':'pid'}, inplace=True)
    gene_table['pid'] = gene_table.pid.map(lambda x: x.lower())
    gene_table['pid'] = gene_table.pid.map(lambda x: x.replace('-', '_'))
    gene_table = gene_table.drop(columns=0)
    gene_table.drop(index=0, inplace=True)
    gene_table['seq_mode'] = gene_table.pid.map(lambda x: x[13:15])
    remove_ids = gene_table.loc[gene_table['seq_mode']=='11'].index
    gene_table.drop(index=remove_ids.values, inplace=True)

    stages_path = os.path.join(data_path, "Complete_stages.csv")
    stages_table = pd.read_table(stages_path, delimiter=",")
    stages_table = stages_table.drop(columns='Unnamed: 0')
    stages_table.head()
    stages_table['pid'] = stages_table.pid.map(lambda x: x.replace('-', '_'))

    for i in gene_table.index.values:
        if gene_table.pid.loc[i][:12] not in stages_table.pid.values:
            gene_table.drop(index=i, inplace=True)

    gene_table['headless'] = gene_table.pid.map(lambda x: x[:12])

    for stage_id in stages_table.index.values:
        if stages_table.pid.loc[stage_id] not in gene_table.headless.values:
            stages_table.drop(index=stage_id, inplace=True)
    
    final_df = pd.merge(stages_table, gene_table, left_on='pid', right_on='headless')
    final_df.drop(columns=['pid_x', 'pid_y', 'seq_mode', 'headless'], inplace=True)

    final_df = remove_unwanted_labels(final_df)
    final_df['stage'] = final_df['stage'].map(lambda x: map_to_one_hot_binary(x))

    features = final_df.drop(columns=['stage', 'Unnamed: 0.1']).astype(np.float32).values
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

    train_dataset = CustomRNADatasetStratified(first_set_inputs, first_set_labels)
    test_dataset = CustomRNADatasetStratified(second_set_inputs, second_set_labels)

    return train_dataset, test_dataset

