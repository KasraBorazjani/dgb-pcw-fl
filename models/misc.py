import torch
from torch import nn
from torch.nn import functional as F

class CustomSVMLike(nn.Module):

    def __init__(self, feature_num, class_num) -> None:

        super(CustomSVMLike, self).__init__()

        self.fc1 = nn.Linear(feature_num, 21000)
        self.fc2 = nn.Linear(21000, class_num)
        self.lgsm = nn.LogSoftmax(dim=1)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        output = self.lgsm(self.fc2(x))

        return output


class CustomRNAModel(nn.Module):

    def __init__(self, feature_size, num_classes) -> None:

        super(CustomRNAModel, self).__init__()

        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)

        self.track_layers = {'fc1':self.fc1, 'fc2':self.fc2, 'fc3':self.fc3, 'fc4':self.fc4, 'fc5':self.fc5}

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x), dim=1)

        return x


class MMmRNAEncoder(nn.Module):
    
    def __init__(self) -> None:
        super(MMmRNAEncoder, self).__init__()
        self.fc1 = nn.Linear(20531, 2048)
        # self.fc2 = nn.Linear(16384, 8192)
        # self.fc3 = nn.Linear(8192, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 64)
        # self.fc6 = nn.Linear(512, 128)
        # self.fc7 = nn.Linear(512, 64)
        self.fc8 = nn.Linear(64,16)

        # self.bn1 = nn.BatchNorm1d(2048)
        # self.bn2 = nn.BatchNorm1d(1024)
        # self.bn3 = nn.BatchNorm1d(64)

        # self.indropout = nn.Dropout(0.5)
        self.dropout = nn.Dropout(0.6)

        self.layer_keys = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7']
    
    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc7(x))
        x = self.dropout(x)
        x = F.relu(self.fc8(x))

        return x

class MMImageEncoder(nn.Module):

    def __init__(self) -> None:
        super(MMImageEncoder, self).__init__()
        self.fc1 = nn.Linear(150, 64)
        self.fc2 = nn.Linear(64,16)
        self.dropout = nn.Dropout(0.2)

        self.layer_keys = ['fc1']
    
    def forward(self, x):
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc3(x))

        return x

class MMClinEncoder(nn.Module):

    def __init__(self) -> None:
        super(MMClinEncoder, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64,16)

        self.layer_keys = ['fc1']
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x



class CustomGBFederatedModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self, modalities, column_map=None) -> None:
        super(CustomGBFederatedModel, self).__init__()
        self.column_map = column_map
        self.modalities = modalities

        if "mrna" in self.modalities:
            self.mrna_encoder = nn.Sequential(*self._create_mrna_encoder())
            self.mrna_classifier = nn.Sequential(*self._create_single_modal_classifier())
        if "image" in self.modalities:
            self.image_encoder = nn.Sequential(*self._create_image_encoder())
            self.image_classifier = nn.Sequential(*self._create_single_modal_classifier())
        if "clinical" in self.modalities:
            self.clinical_encoder = nn.Sequential(*self._create_clinical_encoder())
            self.clinical_classifier = nn.Sequential(*self._create_single_modal_classifier())

        # self.encoders = {}
        # if "mrna" in self.modalities:
        #     self.encoders["mrna"] = MMmRNAEncoder()
        
        # if "image" in self.modalities:
        #     self.encoders["image"] = MMImageEncoder()
        
        # if "clinical" in self.modalities:
        #     self.encoders["clinical"] = MMClinEncoder()
        

        # self.fc1 = nn.Linear(16*len(self.modalities), 16)
        # self.fc2 = nn.Linear(16, 8)
        # self.classifier = nn.Linear(8,2)

        ## Setting 1 - More compatible with clinical data ##
        # self.classifier = nn.Linear(64*len(self.modalities), 2)

        
        self.classifier = nn.Sequential(*self._create_classifier())
    
    def forward(self, x, fwd_mode):
        if fwd_mode=="multimodal":
            encoder_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
            cursor = 0
            for modality in self.modalities:
                encoder_outputs = torch.cat((encoder_outputs, getattr(self, modality+"_encoder")(x[:, cursor:cursor+len(self.column_map[modality])])), dim=1)
                cursor += len(self.column_map[modality])
            x = self.classifier(encoder_outputs)
            
            # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

            return x
        
        elif fwd_mode=="mrna":
            encoder_output = self.mrna_encoder(x[:,0:len(self.column_map["mrna"])])
            output = self.mrna_classifier(encoder_output)

            return output
        
        elif fwd_mode=="image":
            encoder_output = self.image_encoder(x[:,len(self.column_map["mrna"]):len(self.column_map["image"])])
            output = self.image_classifier(encoder_output)

            return output
        
        elif fwd_mode=="clinical":
            encoder_output = self.clinical_encoder(x[:,len(self.column_map["image"])+len(self.column_map["mrna"]):len(self.column_map["clinical"])])
            output = self.clinical_classifier(encoder_output)

            return output

    
    def _create_classifier(self):

        fc = []
        fc.append(nn.Linear(16*len(self.modalities), 16))
        
        fc.append(nn.ReLU())
        # fc.append(nn.Linear(16, 8))
        # fc.append(nn.ReLU())
        fc.append(nn.Linear(16,2))
        # fc.append(nn.LogSoftmax(dim=1))

        return fc

    def _create_single_modal_classifier(self):

        fc = []
        fc.append(nn.Linear(16, 16))
        
        fc.append(nn.ReLU())
        # fc.append(nn.Linear(16, 8))
        # fc.append(nn.ReLU())
        fc.append(nn.Linear(16,2))
        # fc.append(nn.LogSoftmax(dim=1))

        return fc

    def _create_mrna_encoder(self):

        encoder_stack = []

        encoder_stack.append(nn.Dropout(0.2))
        encoder_stack.append(nn.Linear(20531, 16384))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Dropout(0.2))
        encoder_stack.append(nn.Linear(16384, 8192))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Dropout(0.2))
        encoder_stack.append(nn.Linear(8192, 4096))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Linear(4096, 2048))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Linear(2048, 512))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Dropout(0.2))
        encoder_stack.append(nn.Linear(512, 128))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Dropout(0.2))
        encoder_stack.append(nn.Linear(128, 64))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Dropout(0.2))
        encoder_stack.append(nn.Linear(64, 16))
        encoder_stack.append(nn.ReLU())

        return encoder_stack
    
    def _create_image_encoder(self):

        encoder_stack = []
        
        # encoder_stack.append(nn.Dropout(0.2))
        encoder_stack.append(nn.Linear(150, 64))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Linear(64, 64))
        encoder_stack.append(nn.ReLU())
        encoder_stack.append(nn.Dropout(0.2))
        encoder_stack.append(nn.Linear(64, 16))
        encoder_stack.append(nn.ReLU())

        return encoder_stack

    def _create_clinical_encoder(self):

        encoder_stack = []
        
        encoder_stack.append(nn.Linear(11, 16))
        encoder_stack.append(nn.ReLU())
        # encoder_stack.append(nn.Dropout(0.2))
        # encoder_stack.append(nn.Linear(64, 16))
        # encoder_stack.append(nn.ReLU())

        return encoder_stack