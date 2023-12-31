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
from datasets.dataset_classes import CustomMultiModalDatasetStratified

def split_mm_centralized_strat_gb(features, labels, fraction, modalities, column_map, dataset_name, random_state=None):
    
    # data_path = os.path.join(".", "multi_modal_features", time_label)

    print(modalities)

    if random_state:
        random.seed(random_state)

    indices_per_label = defaultdict(list)
    
    print(labels['stage'].unique())
    
    for stage in labels['stage'].unique():
        indices_per_label[stage] = labels[labels['stage']==stage].index.values
        # print(f"len indices for label {stage}: {len(indices_per_label[stage])}")
    
    first_set_indices, second_set_indices = list(), list()

    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices.tolist(), n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices.tolist()) - set(random_indices_sample))
        # print(f"first set indices number for stage {label}: {len(first_set_indices)}")
        # print(f"second set indices number for stage {label}: {len(second_set_indices)}")
    
    
    first_set_inputs = features.loc[first_set_indices]
    first_set_labels = labels.loc[first_set_indices]
    second_set_inputs = features.loc[second_set_indices]
    second_set_labels = labels.loc[second_set_indices]

    test_dataset = CustomMultiModalDatasetStratified(first_set_inputs, first_set_labels, modalities, column_map)
    
    return test_dataset, second_set_inputs, second_set_labels


def create_datasets_mm_centralized_gb(data_path, validation_split, random_state=None):

    # Load mRNA data
    mrna_table_brca = pd.read_csv(os.path.join(data_path, 'brca_mrna.csv'))
    mrna_table_lusc = pd.read_csv(os.path.join(data_path, 'lusc_mrna.csv'))
    mrna_table_lihc = pd.read_csv(os.path.join(data_path, 'lihc_mrna.csv'))
    complete_mrna = pd.concat([mrna_table_brca, mrna_table_lusc, mrna_table_lihc], axis=0, join='inner')
    mrna_duplicates = complete_mrna.pid.duplicated()
    complete_mrna = complete_mrna.loc[mrna_duplicates != True]

    # Load image data
    image_table_brca = pd.read_csv(os.path.join(data_path, 'brca_image.csv'))
    image_table_lusc = pd.read_csv(os.path.join(data_path, 'lusc_image.csv'))
    image_table_lihc = pd.read_csv(os.path.join(data_path, 'lihc_image.csv'))
    complete_image = pd.concat([image_table_brca, image_table_lusc, image_table_lihc], axis=0, join='inner')
    image_duplicates = complete_image.pid.duplicated()
    complete_image = complete_image.loc[image_duplicates != True]

    # Load clinical data
    clinical_table_brca = pd.read_csv(os.path.join(data_path, 'brca_clinical.csv'))
    clinical_table_lusc = pd.read_csv(os.path.join(data_path, 'lusc_clinical.csv'))
    clinical_table_lihc = pd.read_csv(os.path.join(data_path, 'lihc_clinical.csv'))
    complete_clinical = pd.concat([clinical_table_brca, clinical_table_lusc, clinical_table_lihc], axis=0, join='inner')
    clinical_duplicates = complete_clinical.pid.duplicated()
    complete_clinical = complete_clinical.loc[clinical_duplicates != True]

    # Concatenate the feature datasets to form the final features dataset
    complete_dataset = complete_mrna.merge(complete_image, how='inner', on='pid').merge(complete_clinical, how='inner', on='pid')

    # Forming the column_map dict
    mrna_columns = complete_mrna.drop(columns=["pid"]).columns.values
    image_columns = complete_image.drop(columns=["pid"]).columns.values
    clinical_columns = complete_clinical.drop(columns=["pid"]).columns.values
    column_map = {"mrna":mrna_columns, "image":image_columns, "clinical":clinical_columns}

    # Load stage data
    stages_table_brca = pd.read_csv(os.path.join(data_path, 'brca_stages.csv'))
    stages_table_lusc = pd.read_csv(os.path.join(data_path, 'lusc_stages.csv'))
    stages_table_lihc = pd.read_csv(os.path.join(data_path, 'lihc_stages.csv'))
    complete_stages = pd.concat([stages_table_brca, stages_table_lusc, stages_table_lihc], axis=0, join='inner')
    complete_stages.reset_index(drop=True, inplace=True)
    stages_duplicates = complete_stages.pid.duplicated()
    complete_stages = complete_stages.loc[stages_duplicates != True]
    complete_stages.drop(columns=["Unnamed: 0"], inplace=True)

    # Sorting features and stages based on "pid" values
    complete_dataset.sort_values(by="pid", axis=0, inplace=True)
    complete_stages.sort_values(by="pid", axis=0, inplace=True)

    # dropping pid column to create separatable feature and label sets
    features = complete_dataset.drop(columns=["pid"]).astype(np.float32)
    labels = complete_stages.drop(columns=["pid"])

    # Splitting test and train datasets
    test_dataset, rest_inputs, rest_labels = split_mm_centralized_strat_gb(features, labels, 0.2, ["mrna", "image", "clinical"], column_map, "all", random_state)
    train_gb_dataset, rest_inputs, rest_labels = split_mm_centralized_strat_gb(rest_inputs, rest_labels, 0.2, ["mrna", "image", "clinical"], column_map, "all", random_state)
    train_dataset = CustomMultiModalDatasetStratified(rest_inputs, rest_labels, ["mrna", "image", "clinical"], column_map)

    return test_dataset, train_gb_dataset, train_dataset


def create_mm_centralized_strat(features, labels, fraction, modalities, column_map, dataset_name, random_state=None):
    
    # data_path = os.path.join(".", "multi_modal_features", time_label)

    print(modalities)

    if random_state:
        random.seed(random_state)

    indices_per_label = defaultdict(list)
    
    print(labels['stage'].unique())
    
    for stage in labels['stage'].unique():
        indices_per_label[stage] = labels[labels['stage']==stage].index.values
        print(f"len indices for label {stage}: {len(indices_per_label[stage])}")
    
    first_set_indices, second_set_indices = list(), list()

    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices.tolist(), n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices.tolist()) - set(random_indices_sample))
        print(f"first set indices number for stage {label}: {len(first_set_indices)}")
        print(f"second set indices number for stage {label}: {len(second_set_indices)}")
    
    
    first_set_inputs = features.loc[first_set_indices]
    first_set_labels = labels.loc[first_set_indices]
    second_set_inputs = features.loc[second_set_indices]
    second_set_labels = labels.loc[second_set_indices]

    test_dataset = CustomMultiModalDatasetStratified(first_set_inputs, first_set_labels, modalities, column_map)
    train_dataset = CustomMultiModalDatasetStratified(second_set_inputs, second_set_labels, modalities, column_map)
    
    return test_dataset, train_dataset



def create_datasets_mm_centralized(data_path, validation_split, random_state=None):

    # Load mRNA data
    mrna_table_brca = pd.read_csv(os.path.join(data_path, 'brca_mrna.csv'))
    mrna_table_lusc = pd.read_csv(os.path.join(data_path, 'lusc_mrna.csv'))
    mrna_table_lihc = pd.read_csv(os.path.join(data_path, 'lihc_mrna.csv'))
    complete_mrna = pd.concat([mrna_table_brca, mrna_table_lusc, mrna_table_lihc], axis=0, join='inner')
    mrna_duplicates = complete_mrna.pid.duplicated()
    complete_mrna = complete_mrna.loc[mrna_duplicates != True]

    # Load image data
    image_table_brca = pd.read_csv(os.path.join(data_path, 'brca_image.csv'))
    image_table_lusc = pd.read_csv(os.path.join(data_path, 'lusc_image.csv'))
    image_table_lihc = pd.read_csv(os.path.join(data_path, 'lihc_image.csv'))
    complete_image = pd.concat([image_table_brca, image_table_lusc, image_table_lihc], axis=0, join='inner')
    image_duplicates = complete_image.pid.duplicated()
    complete_image = complete_image.loc[image_duplicates != True]

    # Load clinical data
    clinical_table_brca = pd.read_csv(os.path.join(data_path, 'brca_clinical.csv'))
    clinical_table_lusc = pd.read_csv(os.path.join(data_path, 'lusc_clinical.csv'))
    clinical_table_lihc = pd.read_csv(os.path.join(data_path, 'lihc_clinical.csv'))
    complete_clinical = pd.concat([clinical_table_brca, clinical_table_lusc, clinical_table_lihc], axis=0, join='inner')
    clinical_duplicates = complete_clinical.pid.duplicated()
    complete_clinical = complete_clinical.loc[clinical_duplicates != True]

    # Concatenate the feature datasets to form the final features dataset
    complete_dataset = complete_mrna.merge(complete_image, how='inner', on='pid').merge(complete_clinical, how='inner', on='pid')

    # Forming the column_map dict
    mrna_columns = complete_mrna.drop(columns=["pid"]).columns.values
    image_columns = complete_image.drop(columns=["pid"]).columns.values
    clinical_columns = complete_clinical.drop(columns=["pid"]).columns.values
    column_map = {"mrna":mrna_columns, "image":image_columns, "clinical":clinical_columns}

    # Load stage data
    stages_table_brca = pd.read_csv(os.path.join(data_path, 'brca_stages.csv'))
    stages_table_lusc = pd.read_csv(os.path.join(data_path, 'lusc_stages.csv'))
    stages_table_lihc = pd.read_csv(os.path.join(data_path, 'lihc_stages.csv'))
    complete_stages = pd.concat([stages_table_brca, stages_table_lusc, stages_table_lihc], axis=0, join='inner')
    complete_stages.reset_index(drop=True, inplace=True)
    stages_duplicates = complete_stages.pid.duplicated()
    complete_stages = complete_stages.loc[stages_duplicates != True]
    complete_stages.drop(columns=["Unnamed: 0"], inplace=True)

    # Sorting features and stages based on "pid" values
    complete_dataset.sort_values(by="pid", axis=0, inplace=True)
    complete_stages.sort_values(by="pid", axis=0, inplace=True)

    # dropping pid column to create separatable feature and label sets
    features = complete_dataset.drop(columns=["pid"]).astype(np.float32)
    labels = complete_stages.drop(columns=["pid"])

    # Splitting test and train datasets
    test_dataset, train_dataset = create_mm_centralized_strat(features, labels, validation_split, ["mrna", "image", "clinical"], column_map, "all", random_state)

    return test_dataset, train_dataset