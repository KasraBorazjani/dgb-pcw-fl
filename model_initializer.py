import torch
from models import *
from train_functions.federated.utils import *


###############
#  Federated  #
###############

### Instantiating federated models based on MODE ### -- uncomment if needed

modality_list = [["mrna"], ["image"], ["clinical"], ["mrna", "image"], ["mrna", "clinical"], ["image", "clinical"], ["mrna", "image", "clinical"]]

MODE = 'no_attention'

if MODE == 'no_attention':
    for modalities in modality_list:
        model = model_assigner(modalities)
        torch.save(model.state_dict(), f"../saved_models/federated_{modality_to_classifier_mapper(modalities)}_start_model.pt")
        print(modality_to_classifier_mapper(modalities))

elif MODE == 'vanilla_attention':
    for modalities in modality_list:
        model = CustomFederatedDistributedAttentionModel(modalities=modalities)
        torch.save(model.state_dict(), f"../saved_models/federated_attention_{modality_to_classifier_mapper(modalities)}_start_model.pt")
        print(modality_to_classifier_mapper(modalities))



###############
# Centralized #
###############

### Instantiating mRNA centralized models ### -- uncomment if needed
# model = CustomRNAModelBIG(feature_size=20531, num_classes=2)
# torch.save(model.state_dict(), "../saved_models/mrna_start_model_large.pt")
# model = CustomRNAModelMedium(feature_size=20531, num_classes=2)
# torch.save(model.state_dict(), "../saved_models/mrna_start_model_medium.pt")
# model = CustomRNAModelSmall(feature_size=20531, num_classes=2)
# torch.save(model.state_dict(), "../saved_models/mrna_start_model_small.pt")


# model = CustomGBFederatedModelSimulOut(["mrna", "image", "clinical"])
# torch.save(model.state_dict(), "../saved_models/central_simul_gb_start_model.pt")

