import torch
from torch.utils.data import Dataset

from dataset_utils import collect_labels_from_df, remove_unwanted_labels, map_to_one_hot, map_to_one_hot_binary

### Multimodal Dataset ###
class CustomMultiModalDatasetStratified(Dataset):
    def __init__(self, features, labels, modalities, column_map) -> None:
        super(CustomMultiModalDatasetStratified, self).__init__()

        labels['stage'] = labels['stage'].map(lambda x: map_to_one_hot_binary(x))

        self.features = {}

        for modality in modalities:
            self.features[modality] = features[column_map[modality]].values

        self.labels = labels.stage.values

        self.num_samples = len(self.labels)
        self.modalities = modalities
        self.column_map = column_map
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):

        feature_set = []
        # print(self.features['clinical'][index])
        for modality in self.features.keys():
            # print(len(self.features[modality][index]))
            feature_set.extend(self.features[modality][index])
        label = self.labels[index]
        # print(len(feature_set))
        feature_tensor = torch.Tensor(feature_set)
        label_tensor = torch.Tensor(label)

        return feature_tensor, label_tensor


### mRNA Dataset ###
class CustomRNADatasetStratified(Dataset):
    def __init__(self, features, labels) -> None:
        super(CustomRNADatasetStratified, self).__init__()

        self.features = features
        self.labels = labels

        self.num_samples = len(self.labels)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):

        feature_Set = self.features[index]
        label = self.labels[index]
        feature_tensor = torch.Tensor(feature_Set.tolist())
        label_tensor = torch.Tensor(label)

        return feature_tensor, label_tensor


### Image Dataset ###
class CustomImageDatasetStratified(Dataset):
    def __init__(self, features, labels) -> None:
        super(CustomImageDatasetStratified, self).__init__()

        self.features = features
        self.labels = labels

        self.num_samples = len(self.labels)
        # self.num_features = self.features.shape[1]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):

        feature_Set = self.features[index]
        label = self.labels[index]
        feature_tensor = torch.Tensor(feature_Set.tolist())
        label_tensor = torch.Tensor(label)

        return feature_tensor, label_tensor


### Clinical Dataset ###
class CustomClinicalDatasetStratified(Dataset):

    def __init__(self, features, labels) -> None:
        super(CustomClinicalDatasetStratified, self).__init__()

        self.features = features
        self.labels = labels

        self.num_samples = len(self.labels)
        # self.num_features = self.features.shape[1]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):

        feature_Set = self.features[index]
        label = self.labels[index]
        feature_tensor = torch.Tensor(feature_Set.tolist())
        label_tensor = torch.Tensor(label)

        return feature_tensor, label_tensor