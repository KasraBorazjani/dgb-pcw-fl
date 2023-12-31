import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random
import math
from collections import defaultdict

from datasets.dataset_classes import CustomMultiModalDatasetStratified


def create_mm_fed_strat(features, labels, fraction, modalities, column_map, dataset_name, random_state=None):
    
    # data_path = os.path.join(".", "multi_modal_features", time_label)

    print(modalities)

    if random_state:
        random.seed(random_state)

    indices_per_label = defaultdict(list)
    
    for stage in labels['stage'].unique():
        indices_per_label[stage] = labels[labels['stage']==stage].index.values
    
    first_set_indices, second_set_indices = list(), list()

    first_set_indices, second_set_indices = list(), list()

    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices.tolist(), n_samples_for_label)
        if ((dataset_name == 'brca') and (label == "stage ii")):
            appending_indices_sample = random.sample(random_indices_sample, round(len(random_indices_sample) * 0.33))
        else:
            appending_indices_sample = random_indices_sample
        first_set_indices.extend(appending_indices_sample)
        second_set_indices.extend(set(indices.tolist()) - set(random_indices_sample))
    
    first_set_inputs = features.loc[first_set_indices]
    first_set_labels = labels.loc[first_set_indices]
    second_set_inputs = features.loc[second_set_indices]
    second_set_labels = labels.loc[second_set_indices]

    test_dataset = CustomMultiModalDatasetStratified(first_set_inputs, first_set_labels, modalities, column_map)
    
    return test_dataset, second_set_inputs, second_set_labels


def create_datasets_fed_stratified(args, client_name, data_path, random_state):
    if args.num_fold in [0, 1]:
        if client_name == 'brca':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values

            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            # print(mrna_table.pid)
            # print(image_table.pid)
            # print(clinical_table.pid)
            multi_modal_frame = mrna_table.merge(image_table, how="right", on="pid").merge(clinical_table, how="right", on="pid")
            
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == 'tri_modal':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["mrna", "image"], column_map, "brca", random_state)
                mrna_dataset, _, _ = create_mm_fed_strat(rest_features, rest_labels, 1, ["mrna"], column_map, "brca", random_state)

                return [test_dataset, mm_dataset, mrna_img_dataset, mrna_dataset]

            elif args.mode == 'bi_modal':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image"], column_map, "brca", random_state)
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna", "image"], column_map, "brca", random_state)
                bm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["mrna"], column_map, "brca", random_state)
                um_dataset, _, _ = create_mm_fed_strat(rest_features, rest_labels, 1, ["image"], column_map, "brca", random_state)

                return [test_dataset, mm_dataset, bm_dataset, um_dataset]
            
            elif args.mode == 'upper_bound':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                bm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                um_dataset, _, _ = create_mm_fed_strat(rest_features, rest_labels, 1, ["mrna", "image", "clinical"], column_map, "brca", random_state)

                return [test_dataset, mm_dataset, bm_dataset, um_dataset]
            

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image"], column_map, "brca", random_state)
            # mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna", "image"], column_map, "brca", random_state)
            # mm_2_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.195, ["mrna", "image"], column_map, "brca", random_state)
            # mm_3_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.24, ["mrna", "image"], column_map, "brca", random_state)
            # img_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.32, ["image"], column_map, "brca", random_state)
            # mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # img_2_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["image"], column_map, "brca", random_state)
            # mrna_2_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1, ["mrna"], column_map, "brca", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna"], column_map, "brca", random_state)
            # mrna_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.195, ["mrna"], column_map, "brca", random_state)
            # mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.24, ["mrna"], column_map, "brca", random_state)
            # img_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.32, ["mrna"], column_map, "brca", random_state)
            # mrna_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # clin_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
            
            # return [test_dataset, mm_1_dataset, mm_2_dataset, mm_3_dataset, img_1_dataset, mrna_1_dataset, img_2_dataset, mrna_2_dataset]
        
        elif client_name == 'lusc':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values
            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            multi_modal_frame = mrna_table.merge(image_table, how="right").merge(clinical_table, how="right")
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == 'tri_modal':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                img_clinical_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["image", "clinical"], column_map, "lusc", random_state)
                img_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["image"], column_map)

                return [test_dataset, mm_dataset, img_clinical_dataset, img_dataset]

            elif args.mode == 'bi_modal':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image"], column_map, "lusc", random_state)
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["mrna", "image"], column_map, "lusc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["mrna"], column_map, "lusc", random_state)
                img_1_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["image"], column_map)

                return [test_dataset, mm_1_dataset, mrna_1_dataset, img_1_dataset]
            

            elif args.mode == 'upper_bound':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                img_1_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [test_dataset, mm_1_dataset, mrna_1_dataset, img_1_dataset]

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.09, ["mrna"], column_map, "lusc", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.317, ["mrna"], column_map, "lusc", random_state)
            # mrna_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.464, ["mrna"], column_map, "lusc", random_state)
            # mrna_img_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
        
        elif client_name == 'lihc':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values

            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            multi_modal_frame = mrna_table.merge(image_table, how="right").merge(clinical_table, how="right")
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == 'tri_modal':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                mrna_clinical_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["mrna", "clinical"], column_map, "lihc", random_state)
                clinical_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["clinical"], column_map)

                return [test_dataset, mm_dataset, mrna_clinical_dataset, clinical_dataset]

            elif args.mode == 'bi_modal':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image"], column_map, "lihc", random_state)
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna", "image"], column_map, "lihc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["mrna"], column_map, "lihc", random_state)
                img_1_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["image"], column_map)

                return [test_dataset, mm_1_dataset, mrna_1_dataset, img_1_dataset]
            
            elif args.mode == 'upper_bound':
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                img_1_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [test_dataset, mm_1_dataset, mrna_1_dataset, img_1_dataset]

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1, ["mrna"], column_map, "lihc", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna"], column_map, "lihc", random_state)
            # mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.588, ["mrna"], column_map, "lihc", random_state)
            # img_clin_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
    
    elif args.num_fold == 2:
        if client_name == 'brca':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values

            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            # print(mrna_table.pid)
            # print(image_table.pid)
            # print(clinical_table.pid)
            multi_modal_frame = mrna_table.merge(image_table, how="right", on="pid").merge(clinical_table, how="right", on="pid")
            
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == 'tri_modal':
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna", "image"], column_map, "brca", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                mrna_dataset, _, _ = create_mm_fed_strat(rest_features, rest_labels, 1, ["mrna"], column_map, "brca", random_state)
                return [mm_dataset, mrna_img_dataset, test_dataset, mrna_dataset]

            elif args.mode == 'bi_modal':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image"], column_map, "brca", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna"], column_map, "brca", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["mrna", "image"], column_map, "brca", random_state)
                image_1_dataset, _, _ = create_mm_fed_strat(rest_features, rest_labels, 1, ["image"], column_map, "brca", random_state)
                
                return [mm_1_dataset, mrna_1_dataset, test_dataset, image_1_dataset]
            
            elif args.mode == 'upper_bound':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["mrna", "image", "clincial"], column_map, "brca", random_state)
                image_1_dataset, _, _ = create_mm_fed_strat(rest_features, rest_labels, 1, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                
                return [mm_1_dataset, mrna_1_dataset, test_dataset, image_1_dataset]

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image"], column_map, "brca", random_state)
            # mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna", "image"], column_map, "brca", random_state)
            # mm_2_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.195, ["mrna", "image"], column_map, "brca", random_state)
            # mm_3_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.24, ["mrna", "image"], column_map, "brca", random_state)
            # img_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.32, ["image"], column_map, "brca", random_state)
            # mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # img_2_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["image"], column_map, "brca", random_state)
            # mrna_2_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1, ["mrna"], column_map, "brca", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna"], column_map, "brca", random_state)
            # mrna_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.195, ["mrna"], column_map, "brca", random_state)
            # mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.24, ["mrna"], column_map, "brca", random_state)
            # img_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.32, ["mrna"], column_map, "brca", random_state)
            # mrna_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # clin_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
            
            # return [test_dataset, mm_1_dataset, mm_2_dataset, mm_3_dataset, img_1_dataset, mrna_1_dataset, img_2_dataset, mrna_2_dataset]
        
        elif client_name == 'lusc':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values
            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            multi_modal_frame = mrna_table.merge(image_table, how="right").merge(clinical_table, how="right")
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == 'tri_modal':
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                img_clinical_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["image", "clinical"], column_map, "lusc", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                img_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["image"], column_map)

                return [mm_dataset, img_clinical_dataset, test_dataset, img_dataset]

            elif args.mode == 'bi_modal':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image"], column_map, "lusc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["mrna"], column_map, "lusc", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["mrna", "image"], column_map, "lusc", random_state)
                img_1_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["image"], column_map)

                return [mm_1_dataset, mrna_1_dataset, test_dataset, img_1_dataset]
            
            elif args.mode == 'upper_bound':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                img_1_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [mm_1_dataset, mrna_1_dataset, test_dataset, img_1_dataset]

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.09, ["mrna"], column_map, "lusc", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.317, ["mrna"], column_map, "lusc", random_state)
            # mrna_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.464, ["mrna"], column_map, "lusc", random_state)
            # mrna_img_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
            
        
        elif client_name == 'lihc':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values

            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            multi_modal_frame = mrna_table.merge(image_table, how="right").merge(clinical_table, how="right")
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == 'tri_modal':
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                mrna_clinical_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna", "clinical"], column_map, "lihc", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                clinical_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["clinical"], column_map)

                return [mm_dataset, mrna_clinical_dataset, test_dataset, clinical_dataset]

            elif args.mode == 'bi_modal':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image"], column_map, "lihc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna"], column_map, "lihc", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["mrna", "image"], column_map, "lihc", random_state)
                img_1_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["image"], column_map)

                return [mm_1_dataset, mrna_1_dataset, test_dataset, img_1_dataset]
            
            elif args.mode == 'upper_bound':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                test_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                img_1_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [mm_1_dataset, mrna_1_dataset, test_dataset, img_1_dataset]

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1, ["mrna"], column_map, "lihc", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna"], column_map, "lihc", random_state)
            # mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.588, ["mrna"], column_map, "lihc", random_state)
            # img_clin_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            
            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
    
    elif args.num_fold == 3:
        if client_name == 'brca':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values

            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            # print(mrna_table.pid)
            # print(image_table.pid)
            # print(clinical_table.pid)
            multi_modal_frame = mrna_table.merge(image_table, how="right", on="pid").merge(clinical_table, how="right", on="pid")
            
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == 'tri_modal':
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna", "image"], column_map, "brca", random_state)
                mrna_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["mrna"], column_map, "brca", random_state)
                test_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [mm_dataset, mrna_img_dataset, mrna_dataset, test_dataset]

            elif args.mode == 'bi_modal':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image"], column_map, "brca", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna"], column_map, "brca", random_state)
                image_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["image"], column_map, "brca", random_state)
                test_dataset, _, _ = create_mm_fed_strat(rest_features, rest_labels, 1, ["mrna", "image"], column_map, "brca", random_state)

                return [mm_1_dataset, mrna_1_dataset, image_1_dataset, test_dataset]
            
            elif args.mode == 'upper_bound':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2016, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3368, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                image_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5079, ["mrna", "image", "clinical"], column_map, "brca", random_state)
                test_dataset, _, _ = create_mm_fed_strat(rest_features, rest_labels, 1, ["mrna", "image", "clinical"], column_map, "brca", random_state)

                return [mm_1_dataset, mrna_1_dataset, image_1_dataset, test_dataset]

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image"], column_map, "brca", random_state)
            # mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna", "image"], column_map, "brca", random_state)
            # mm_2_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.195, ["mrna", "image"], column_map, "brca", random_state)
            # mm_3_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.24, ["mrna", "image"], column_map, "brca", random_state)
            # img_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.32, ["image"], column_map, "brca", random_state)
            # mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # img_2_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["image"], column_map, "brca", random_state)
            # mrna_2_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1, ["mrna"], column_map, "brca", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna"], column_map, "brca", random_state)
            # mrna_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.195, ["mrna"], column_map, "brca", random_state)
            # mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.24, ["mrna"], column_map, "brca", random_state)
            # img_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.32, ["mrna"], column_map, "brca", random_state)
            # mrna_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.465, ["mrna"], column_map, "brca", random_state)
            # clin_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
            
            # return [test_dataset, mm_1_dataset, mm_2_dataset, mm_3_dataset, img_1_dataset, mrna_1_dataset, img_2_dataset, mrna_2_dataset]
        
        elif client_name == 'lusc':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values
            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            multi_modal_frame = mrna_table.merge(image_table, how="right").merge(clinical_table, how="right")
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == 'tri_modal':
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                img_clinical_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["image", "clinical"], column_map, "lusc", random_state)
                img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["image"], column_map, "lusc", random_state)
                test_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [mm_dataset, img_clinical_dataset, img_dataset, test_dataset]

            elif args.mode == 'bi_modal':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image"], column_map, "lusc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["mrna"], column_map, "lusc", random_state)
                image_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["image"], column_map, "lusc", random_state)
                test_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image"], column_map)

                return [mm_1_dataset, mrna_1_dataset, image_1_dataset, test_dataset]
            
            elif args.mode == 'upper_bound':
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1981, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.3370, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                image_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5084, ["mrna", "image", "clinical"], column_map, "lusc", random_state)
                test_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [mm_1_dataset, mrna_1_dataset, image_1_dataset, test_dataset]

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.09, ["mrna"], column_map, "lusc", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.317, ["mrna"], column_map, "lusc", random_state)
            # mrna_clin_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.464, ["mrna"], column_map, "lusc", random_state)
            # mrna_img_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
            
        
        elif client_name == 'lihc':
            # print(client_name)
            mrna_file_name = client_name + '_mrna.csv'
            mrna_file_path = os.path.join(data_path, mrna_file_name)
            mrna_table = pd.read_csv(mrna_file_path, delimiter=",")

            stage_file_name = client_name + "_stages.csv"
            stage_file_path = os.path.join(data_path, stage_file_name)
            stage_table = pd.read_csv(stage_file_path, delimiter=",")

            image_file_name = client_name + '_image.csv'
            image_file_path = os.path.join(data_path, image_file_name)
            image_table = pd.read_csv(image_file_path, delimiter=",")

            clinical_file_name = client_name + '_clinical.csv'
            clinical_file_path = os.path.join(data_path, clinical_file_name)
            clinical_table = pd.read_csv(clinical_file_path, delimiter=",")

            mrna_table.sort_values(by="pid", axis=0, inplace=True)
            stage_table.sort_values(by="pid", axis=0, inplace=True)
            image_table.sort_values(by="pid", axis=0, inplace=True)
            clinical_table.sort_values(by="pid", axis=0, inplace=True)

            mrna_columns = mrna_table.drop(columns=["pid"]).columns.values
            image_columns = image_table.drop(columns=["pid"]).columns.values
            clinical_columns = clinical_table.drop(columns=["pid"]).columns.values

            column_map = {"mrna": mrna_columns, "image":image_columns, "clinical":clinical_columns}

            multi_modal_frame = mrna_table.merge(image_table, how="right").merge(clinical_table, how="right")
            # stage_table['stage'] = stage_table.stage.map(lambda x: map_to_one_hot_binary(x))


            
            features = multi_modal_frame.drop(columns=["pid"]).astype(np.float32)
            labels = stage_table.drop(columns=["pid"])

            if args.mode == "tri_modal":
                mm_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                mrna_clinical_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna", "clinical"], column_map, "lihc", random_state)
                clinical_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["clinical"], column_map, "lihc", random_state)
                test_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [mm_dataset, mrna_clinical_dataset, clinical_dataset, test_dataset]

            elif args.mode == "bi_modal":
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image"], column_map, "lihc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna"], column_map, "lihc", random_state)
                image_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["image"], column_map, "lihc", random_state)
                test_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image"], column_map)

                return [mm_1_dataset, mrna_1_dataset, image_1_dataset, test_dataset]
            
            elif args.mode == "upper_bound":
                mm_1_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.2, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                mrna_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.333, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                image_1_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.5, ["mrna", "image", "clinical"], column_map, "lihc", random_state)
                test_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna", "image", "clinical"], column_map)

                return [mm_1_dataset, mrna_1_dataset, image_1_dataset, test_dataset]

            # test_dataset, rest_features, rest_labels = create_mm_fed_strat(features, labels, 0.1, ["mrna"], column_map, "lihc", random_state)
            # mm_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.37, ["mrna"], column_map, "lihc", random_state)
            # mrna_img_dataset, rest_features, rest_labels = create_mm_fed_strat(rest_features, rest_labels, 0.588, ["mrna"], column_map, "lihc", random_state)
            # img_clin_dataset = CustomMultiModalDatasetStratified(rest_features, rest_labels, ["mrna"], column_map)

            
            # return [test_dataset, mm_dataset, bm_dataset, um_dataset]
    
    else:
        print("no data available for cohort")
        raise ValueError

