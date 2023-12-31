import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random
import math
from collections import defaultdict

CLIENT_LIST = {"client_1": "LUSC", "client_2": "BRCA", "client_3": "LIHC", "client_central":"Complete"}
from dataset_utils import collect_labels_from_df, remove_unwanted_labels, map_to_one_hot, map_to_one_hot_binary


class CustomRNADataset(Dataset):
    def __init__(self, client_number, index_list):
        super(CustomRNADataset, self).__init__()

        client_name = CLIENT_LIST["client_"+str(client_number)]
        filename = "rnaEigen_"+client_name+".csv"
        features_path = os.path.join(".", "mRNA_features", filename)
        labels_filename = client_name+"_stages.csv"
        clinical_table = pd.read_csv(labels_filename, delimiter=",")
        features_table = pd.read_csv(features_path, delimiter=",").apply(lambda x: x.astype(str).str.lower())
        features_table['pid'] = features_table.pid.apply(lambda x: x[:-3])
        clinical_table['pid'] = clinical_table.pid.apply(lambda x: x.replace("-", "_"))
        final_df = clinical_table.merge(features_table, on="pid")
        final_df = remove_unwanted_labels(final_df)
        final_df['stage'] = final_df['stage'].map(lambda x: map_to_one_hot_binary(x))
        self.features = final_df.drop(columns=['Unnamed: 0', 'stage', 'pid']).astype(np.float32).values
        # self.features = self.features[index_list]
        self.labels = final_df.stage.values
        # self.labels = self.labels[index_list]
        
        
        self.num_samples = len(self.labels)
        self.num_features = self.features.shape[1]

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        feature_set = self.features[index]
        label = self.labels[index]
        index_feature = torch.Tensor(feature_set.tolist())
        index_label = torch.Tensor(label)
        return index_feature, index_label

class CustomRNADatasetBIG(Dataset):

    def __init__(self, client_number) -> None:
        super(CustomRNADatasetBIG, self).__init__()


        


        genes_filename = "Complete_mRNAseq.csv"
        genes_path = os.path.join(".", "mRNA_features", genes_filename)
        gene_table = pd.read_table(genes_path, delimiter=',', low_memory=False).apply(lambda x: x.astype(str).str.lower())
        gene_table = gene_table.transpose()
        # print(len(gene_table))
        gene_table.reset_index(inplace=True)
        gene_table.rename(columns={'index':'pid'}, inplace=True)
        gene_table['pid'] = gene_table.pid.map(lambda x: x.lower())
        gene_table['pid'] = gene_table.pid.map(lambda x: x.replace('-', '_'))
        gene_table = gene_table.drop(columns=0)
        gene_table.drop(index=0, inplace=True)
        gene_table['seq_mode'] = gene_table.pid.map(lambda x: x[13:15])
        remove_ids = gene_table.loc[gene_table['seq_mode']=='11'].index
        # print("remove ids len: ", len(remove_ids))
        gene_table.drop(index=remove_ids.values, inplace=True)

        stages_path = "Complete_stages.csv"
        stages_table = pd.read_table(stages_path, delimiter=",")
        stages_table = stages_table.drop(columns='Unnamed: 0')
        stages_table.head()
        stages_table['pid'] = stages_table.pid.map(lambda x: x.replace('-', '_'))

        # print(len(stages_table))


        # print("before: ", len(gene_table))

        for i in gene_table.index.values:
            if gene_table.pid.loc[i][:12] not in stages_table.pid.values:
                gene_table.drop(index=i, inplace=True)

        gene_table['headless'] = gene_table.pid.map(lambda x: x[:12])

        # print("after: ", len(gene_table))

        for stage_id in stages_table.index.values:
            if stages_table.pid.loc[stage_id] not in gene_table.headless.values:
                stages_table.drop(index=stage_id, inplace=True)
        
        final_df = pd.merge(stages_table, gene_table, left_on='pid', right_on='headless')
        final_df.drop(columns=['pid_x', 'pid_y', 'seq_mode', 'headless'], inplace=True)

        final_df = remove_unwanted_labels(final_df)
        final_df['stage'] = final_df['stage'].map(lambda x: map_to_one_hot_binary(x))

        print(final_df.stage.value_counts())

        # print(final_df.columns)

        self.features = final_df.drop(columns=['stage', 'Unnamed: 0.1']).astype(np.float32).values
        self.labels = final_df.stage.values
        

        self.num_samples = len(self.labels)
        self.num_features = self.features.shape[1]

        # print(self.num_features)

    
    def reduce_to(self, index_list):
        self.features = self.features[index_list]
        self.labels = self.labels[index_list]
        self.num_samples = len(self.labels)
        self.num_features = self.features.shape[1]
        # print(self.num_samples)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        feature_set = self.features[index]
        label = self.labels[index]
        index_feature = torch.Tensor(feature_set.tolist())
        index_label = torch.Tensor(label)
        return index_feature, index_label


class CustomRNADatasetCentral(Dataset):

    def __init__(self) -> None:
        super(CustomRNADatasetCentral, self).__init__()

        
        # genes_filename = "Complete_mRNAseq.csv"
        # genes_path = os.path.join(".", "mRNA_features", genes_filename)
        # gene_table = pd.read_csv(genes_path, low_memory=False).apply(lambda x: x.astype(str).str.lower())
        # gene_table = gene_table.transpose()
        # gene_table.reset_index(inplace=True)
        # gene_table.rename(columns={'index':'pid'}, inplace=True)
        # gene_table['pid'] = gene_table.pid.map(lambda x: x.lower())
        # gene_table['pid'] = gene_table.pid.map(lambda x: x.replace('-', '_'))
        # gene_table = gene_table.drop(columns=0)
        # gene_table.drop(index=0, inplace=True)
        # gene_table['seq_mode'] = gene_table.pid.map(lambda x: x[13:15])
        # remove_ids = gene_table.loc[gene_table['seq_mode']=='11'].index
        # gene_table.drop(index=remove_ids.values, inplace=True)

        # stages_path = "Complete_stages.csv"
        # stages_table = pd.read_table(stages_path, delimiter=",")
        # stages_table = stages_table.drop(columns='Unnamed: 0')
        # stages_table.head()
        # stages_table['pid'] = stages_table.pid.map(lambda x: x.replace('-', '_'))


        # for i in gene_table.index.values:
        #     if gene_table.pid.loc[i][:12] not in stages_table.pid.values:
        #         gene_table.drop(index=i, inplace=True)

        # gene_table['headless'] = gene_table.pid.map(lambda x: x[:12])

        # for stage_id in stages_table.index.values:
        #     if stages_table.pid.loc[stage_id] not in gene_table.headless.values:
        #         stages_table.drop(index=stage_id, inplace=True)
        
        # final_df = pd.merge(stages_table, gene_table, left_on='pid', right_on='headless')
        # final_df.drop(columns=['pid_x', 'pid_y', 'seq_mode', 'headless', 'Unnamed: 0.1'], inplace=True)

        # final_df = remove_unwanted_labels(final_df)
        # final_df.to_csv("Centralized_dataset.csv")
        final_df = pd.read_csv("Centralized_dataset.csv")
        final_df['stage'] = final_df['stage'].map(lambda x: map_to_one_hot(x))
        

        # for col in final_df.columns:
        #     if col != 'stage':
        #         final_df[col] = final_df[col].astype(np.float32)
        #         mean = final_df[col].mean()
        #         std_div = final_df[col].std()
        #         final_df[col] = (final_df[col] - mean) / std_div

        self.features = final_df.drop(columns=['stage', 'Unnamed: 0']).astype(np.float32).values
        # self.features = final_df.drop(columns=['stage']).astype(np.float32).values
        self.labels = final_df.stage.values
        

        self.num_samples = len(self.labels)
        self.num_features = self.features.shape[1]

    
    def reduce_to(self, index_list):
        self.features = self.features[index_list]
        self.labels = self.labels[index_list]
        self.num_samples = len(self.labels)
        self.num_features = self.features.shape[1]


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        feature_set = self.features[index]
        label = self.labels[index]
        index_feature = torch.Tensor(feature_set.tolist())
        index_label = torch.Tensor(label)
        return index_feature, index_label


class CustomImageDataset(Dataset):

    def __init__(self) -> None:

        super(CustomImageDataset, self).__init__()

        final_df = pd.read_csv('Complete_image_plus_stage.csv', delimiter=",")
        final_df = remove_unwanted_labels(final_df)
        final_df['stage'] = final_df.stage.map(lambda x: map_to_one_hot_binary(x))
        self.features = final_df.drop(columns=['stage', 'pid']).astype(np.float32).values
        self.labels = final_df.stage.values

        self.num_samples = len(self.labels)
        self.num_features = self.features.shape[1]


    def reduce_to(self, index_list):
        self.features = self.features[index_list]
        self.labels = self.labels[index_list]
        self.num_samples = len(self.labels)
        self.num_features = self.features.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        feature_set = self.features[index]
        label = self.labels[index]
        index_feature = torch.Tensor(feature_set.tolist())
        index_label = torch.Tensor(label)
        return index_feature, index_label