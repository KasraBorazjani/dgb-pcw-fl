import torch
from torch import nn
from torch.nn import functional as F



class CustomFederatedModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self, modalities, column_map=None) -> None:
        super(CustomFederatedModel, self).__init__()
        self.column_map = column_map
        self.modalities = modalities

        if "mrna" in self.modalities:
            self.mrna_encoder = nn.Sequential(*self._create_mrna_encoder())
        if "image" in self.modalities:
            self.image_encoder = nn.Sequential(*self._create_image_encoder())
        if "clinical" in self.modalities:
            self.clinical_encoder = nn.Sequential(*self._create_clinical_encoder())

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
    
    def forward(self, x):
        encoder_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
        cursor = 0
        for modality in self.modalities:
            # print(f"forwarding {modality} modality")
            # encoder_outputs = torch.cat((encoder_outputs, getattr(self, modality+"_encoder")(x[:, cursor:cursor+len(self.column_map[modality])])), dim=1)
            # cursor += len(self.column_map[modality])

            if modality=="mrna":
                mrna_encoder_output = self.mrna_encoder(x[:, cursor:cursor+len(self.column_map[modality])])
                encoder_outputs = torch.cat((encoder_outputs, mrna_encoder_output), dim=1)
            
            if modality=="image":
                image_encoder_output = self.image_encoder(x[:, cursor:cursor+len(self.column_map[modality])])
                encoder_outputs = torch.cat((encoder_outputs, image_encoder_output), dim=1)
            
            if modality=="clinical":
                clinical_encoder_output = self.clinical_encoder(x[:, cursor:cursor+len(self.column_map[modality])])
                encoder_outputs = torch.cat((encoder_outputs, clinical_encoder_output), dim=1)
            
            cursor += len(self.column_map[modality])

        output = self.classifier(encoder_outputs)
        # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

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


class AttentionModuleBiModal(nn.Module):

    def __init__(self, in_dim) -> None:
        super(AttentionModuleBiModal, self).__init__()
        self.fc0 = nn.Linear(1, 1)
        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(1, 1)
        self.fc4 = nn.Linear(1, 1)
    
    def forward(self, m1_t, m2_t):

        # m1_t -> batch_size x 16 x 1
        # m2_t -> batch_size x 16 x 1

        translated_m2 = self.fc0(m2_t) # batch_size x 16 x 1
        # print(translated_m2.shape)
        # print(m1_t.shape)
        m_matrix = torch.tanh(torch.matmul(m1_t, translated_m2.permute(0, 2, 1))) # batch_size x 16 x 16
        # print(m_matrix.shape)

        
        w_m1 = self.fc1(m1_t).permute(0, 2, 1) # batch_size x 1 x 16
        # print(w_m1.shape)
        w_m2 = self.fc2(m2_t).permute(0, 2, 1) # batch_size x 1 x 16
        # print(w_m2.shape)

        a_m1 = torch.tanh(w_m1 + torch.matmul(w_m2 , m_matrix.permute(0, 2, 1))) # batch_size x 1 x 16
        a_m2 = torch.tanh(w_m2 + torch.matmul(w_m1 , m_matrix)) # batch_size x 1 x 16

        # print("a_m1: ", a_m1.shape)
        # print("a_m2: ", a_m2.shape)

        a_1 = nn.Softmax(dim=1)(self.fc3(a_m1.permute(0, 2, 1))) # batch_size x 16 x 1
        a_2 = nn.Softmax(dim=1)(self.fc4(a_m2.permute(0, 2, 1))) # batch_size x 16 x 1
        # print("a_1: ", a_1.shape)

        attention_1 = torch.diag_embed(torch.squeeze(a_1)) # batch_size x 16 x 16
        attention_2 = torch.diag_embed(torch.squeeze(a_2)) # batch_size x 16 x 16
        # print("attention_1: ", attention_1)

        attended_x1 = torch.matmul(attention_1, m1_t).squeeze(2)
        attended_x2 = torch.matmul(attention_2, m2_t).squeeze(2)

        # print(m1_t)
        # print("attended_x1: ", attended_x1) # batch_size x 16 x 1

        return attended_x1, attended_x2


class CustomFederatedDistributedAttentionModel(nn.Module):

    def __init__(self, modalities, column_map=None) -> None:
        super(CustomFederatedDistributedAttentionModel, self).__init__()

        self.column_map = column_map
        self.modalities = modalities

        if "mrna" in self.modalities:
            self.mrna_encoder = nn.Sequential(*self._create_mrna_encoder())
        if "image" in self.modalities:
            self.image_encoder = nn.Sequential(*self._create_image_encoder())
        if "clinical" in self.modalities:
            self.clinical_encoder = nn.Sequential(*self._create_clinical_encoder())
        
        if (len(self.modalities) == 2):
            setattr(self, self.modalities[0]+"_"+self.modalities[1]+"_attention", AttentionModuleBiModal(16))
        
        if(len(self.modalities) == 3):
            self.mrna_image_attention = AttentionModuleBiModal(16)
            self.mrna_clinical_attention = AttentionModuleBiModal(16)
            self.image_clinical_attention = AttentionModuleBiModal(16)
        

        
        self.classifier = nn.Sequential(*self._create_classifier())

        # self.fc1 = nn.Linear(16*len(self.modalities), 16)
        # self.fc2 = nn.Linear(16, 8)
        # self.classifier = nn.Linear(8,2)

        ## Setting 1 - More compatible with clinical data ##
        # self.classifier = nn.Linear(64*len(self.modalities), 2)
    
    
    def forward(self, x):
        cursor = 0
        encoder_output_list = []
        for modality in self.modalities:
            encoder_output_list.append(getattr(self, modality+"_encoder")(x[:, cursor:cursor+len(self.column_map[modality])]))
            cursor += len(self.column_map[modality])
        
        if len(self.modalities) == 3:
            attention_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
            # print(encoder_output_list[0].unsqueeze_(2).permute(0, 2, 1).shape)
            mrna_mid_1, image_mid_1 = self.mrna_image_attention(encoder_output_list[0].unsqueeze(2).permute(0, 2, 1), encoder_output_list[1].unsqueeze(2).permute(0, 2, 1))
            image_mid_2, clinical_mid_1 = self.image_clinical_attention(encoder_output_list[1].unsqueeze(2).permute(0, 2, 1), encoder_output_list[2].unsqueeze(2).permute(0, 2, 1))
            mrna_mid_2, clinical_mid_2 = self.mrna_clinical_attention(encoder_output_list[0].unsqueeze(2).permute(0, 2, 1), encoder_output_list[2].unsqueeze(2).permute(0, 2, 1))
            attention_outputs = torch.cat((mrna_mid_1, mrna_mid_2, image_mid_1, image_mid_2, clinical_mid_1, clinical_mid_2), dim=1)
            # mrna_clinical_input = torch.cat((encoder_output_list[0], encoder_output_list[2]), dim=1)
            # image_clinical_input = torch.cat((encoder_output_list[1], encoder_output_list[2]), dim=1)

            # mrna_image_output = self.attention_dict["mrna_image"](mrna_image_input)
            # mrna_clinical_output = self.attention_dict["mrna_clinical"](mrna_clinical_input)
            # image_clinical_output = self.attention_dict["image_clinical"](image_clinical_input)

            # attention_outputs = torch.cat((mrna_image_output, mrna_clinical_output, image_clinical_output), dim=1)
        
        elif len(self.modalities) == 2:
            # print(encoder_output_list[0].shape)
            # print("encoder output shape: ", encoder_output_list[0].shape)
            # print("encoder output: ", encoder_output_list[0])
            mod1_mid, mod2_mid = getattr(self, self.modalities[0]+"_"+self.modalities[1]+"_attention")(encoder_output_list[0].unsqueeze(2), encoder_output_list[1].unsqueeze(2))
            attention_outputs = torch.cat((mod1_mid, mod2_mid), dim=1)
            # print(attention_outputs.shape)
            # raise
        
        else:
            attention_outputs = encoder_output_list[0]

        output = self.classifier(attention_outputs)


        # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

        return output
    
    def _create_classifier(self):

        fc = []

        if len(self.modalities) == 3:
            fc.append(nn.Linear(2*16*len(self.modalities), 16))
        else:
            fc.append(nn.Linear(16*len(self.modalities), 16))
        
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


### Not Used Yet ###
class AttentionModuleTriModal(nn.Module):

    def __init__(self, in_dim) -> None:
        super(AttentionModuleTriModal, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.fc2 = nn.Linear(in_dim, 1)
    
    def forward(self, x1, x2, x3):
        scores = torch.matmul(torch.tanh(self.fc1(x1)), self.fc2(x2).transpose(0, 1))
        scores = torch.matmul(torch.tanh(self.fc1(x3)), scores.transpose(0, 1))

        weights = F.softmax(scores, dim=1)

        attention_x1 = torch.matmul(weights, x1)
        attention_x2 = torch.matmul(weights.transpose(0,1), x2)
        attention_x3 = torch.matmul(weights.transpose(0,1), x3)

        return attention_x1, attention_x2, attention_x3
    


class CustomGBFederatedModelSimulOut(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self, modalities, column_map=None) -> None:
        super(CustomGBFederatedModelSimulOut, self).__init__()
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
    
    def forward(self, x):

        encoder_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
        cursor = 0
        for modality in self.modalities:
            if modality=="mrna":
                mrna_encoder_output = getattr(self, modality+"_encoder")(x[:, cursor:cursor+len(self.column_map[modality])])
                mrna_classification_output = self.mrna_classifier(mrna_encoder_output)
                encoder_outputs = torch.cat((encoder_outputs, mrna_encoder_output), dim=1)
            
            if modality=="image":
                image_encoder_output = getattr(self, modality+"_encoder")(x[:, cursor:cursor+len(self.column_map[modality])])
                image_classification_output = self.image_classifier(image_encoder_output)
                encoder_outputs = torch.cat((encoder_outputs, image_encoder_output), dim=1)
            
            if modality=="clinical":
                clinical_encoder_output = getattr(self, modality+"_encoder")(x[:, cursor:cursor+len(self.column_map[modality])])
                clinical_classification_output = self.clinical_classifier(clinical_encoder_output)
                encoder_outputs = torch.cat((encoder_outputs, clinical_encoder_output), dim=1)
            
            cursor += len(self.column_map[modality])
        
        x = self.classifier(encoder_outputs)

        return x, mrna_classification_output, image_classification_output, clinical_classification_output

    
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


#####################################
#####################################
### Multi-modal pair-based models ###
#####################################
#####################################


## Defs ##
def create_mrna_encoder():

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

def create_image_encoder():

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

def create_clinical_encoder():

    encoder_stack = []
    
    encoder_stack.append(nn.Linear(11, 16))
    encoder_stack.append(nn.ReLU())
    # encoder_stack.append(nn.Dropout(0.2))
    # encoder_stack.append(nn.Linear(64, 16))
    # encoder_stack.append(nn.ReLU())

    return encoder_stack

def create_classifier(modality_len):

        fc = []
        fc.append(nn.Linear(16*modality_len, 16))
        
        fc.append(nn.ReLU())
        # fc.append(nn.Linear(16, 8))
        # fc.append(nn.ReLU())
        fc.append(nn.Linear(16,2))
        # fc.append(nn.LogSoftmax(dim=1))

        return fc


# class CustomMultiModalModel(nn.Module):

#     '''
#     Multi-modal model to be used in federated and centralized simulations
#         - Has len(modalities) number of encoders and a classifier 
#     '''

#     def __init__(self) -> None:
#         super(CustomMultiModalModel, self).__init__()

#         self.mrna_encoder = nn.Sequential(*create_mrna_encoder())
#         self.image_encoder = nn.Sequential(*create_image_encoder())
#         self.clinical_encoder = nn.Sequential(*create_clinical_encoder())
#         self.classifier = nn.Sequential(*create_classifier(3))
    
#     def forward(self, x):
#         # encoder_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
        
#         mrna_encoder_output = self.mrna_encoder(x[0])
#         # print(mrna_encoder_output.shape)
    
#         image_encoder_output = self.image_encoder(x[1])
#         # print(image_encoder_output.shape)
    
#         clinical_encoder_output = self.clinical_encoder(x[2])
#         # print(clinical_encoder_output.shape)

#         encoder_outputs = torch.cat([mrna_encoder_output, image_encoder_output, clinical_encoder_output], dim=1)

#         x = self.classifier(encoder_outputs)
#         # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

#         return x


class CustomMultiModalModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self) -> None:
        super(CustomMultiModalModel, self).__init__()

        self.mrna_encoder = nn.Sequential(*create_mrna_encoder())
        self.image_encoder = nn.Sequential(*create_image_encoder())
        self.clinical_encoder = nn.Sequential(*create_clinical_encoder())
        self.classifier = nn.Sequential(*create_classifier(3))
    
    def forward(self, x_mrna, x_img, x_clin):
        # encoder_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
        
        mrna_encoder_output = self.mrna_encoder(x_mrna)
        # print(mrna_encoder_output.shape)
    
        image_encoder_output = self.image_encoder(x_img)
        # print(image_encoder_output.shape)
    
        clinical_encoder_output = self.clinical_encoder(x_clin)
        # print(clinical_encoder_output.shape)

        encoder_outputs = torch.cat([mrna_encoder_output, image_encoder_output, clinical_encoder_output], dim=3)
        # print(encoder_outputs.shape)

        x = self.classifier(encoder_outputs)
        # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

        return x



class CustomRNAImgModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self) -> None:
        super(CustomRNAImgModel, self).__init__()

        self.mrna_encoder = nn.Sequential(*create_mrna_encoder())
        self.image_encoder = nn.Sequential(*create_image_encoder())
        self.classifier = nn.Sequential(*create_classifier(2))
        
        for param in self.mrna_encoder.parameters():
            param.requires_grad_(True)
        for param in self.image_encoder.parameters():
            param.requires_grad_(True)
        for param in self.classifier.parameters():
            param.requires_grad_(True)
    
    def forward(self, x):
        
        mrna_encoder_output = self.mrna_encoder(x[0])
        # print(mrna_encoder_output.shape)
        # print(mrna_encoder_output.shape)

        image_encoder_output = self.image_encoder(x[1])
        # print(image_encoder_output.shape)
        # print(image_encoder_output.shape)

        encoder_outputs = torch.cat([mrna_encoder_output, image_encoder_output], dim=1)

        output = self.classifier(encoder_outputs)
        # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

        return output



class CustomRNAClinModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self) -> None:
        super(CustomRNAClinModel, self).__init__()

        self.mrna_encoder = nn.Sequential(*create_mrna_encoder())
        self.clinical_encoder = nn.Sequential(*create_clinical_encoder())
        self.classifier = nn.Sequential(*create_classifier(2))
    
    def forward(self, x):
        # encoder_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
        
        mrna_encoder_output = self.mrna_encoder(x[0])
        clinical_encoder_output = self.clinical_encoder(x[1])
        
        encoder_outputs = torch.cat([mrna_encoder_output, clinical_encoder_output], dim=1)

        x = self.classifier(encoder_outputs)
        # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

        return x



class CustomImgClinModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self) -> None:
        super().__init__()

        self.image_encoder = nn.Sequential(*create_image_encoder())
        self.clinical_encoder = nn.Sequential(*create_clinical_encoder())
        self.classifier = nn.Sequential(*create_classifier(2))
    
    
    def forward(self, x):
        
        image_encoder_output = self.image_encoder(x[0])
        clinical_encoder_output = self.clinical_encoder(x[1])

        encoder_outputs = torch.cat([image_encoder_output, clinical_encoder_output], dim=1)

        x = self.classifier(encoder_outputs)
        # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

        return x


class CustomRNAModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self) -> None:
        super(CustomRNAModel, self).__init__()

        self.mrna_encoder = nn.Sequential(*create_mrna_encoder())
        self.classifier = nn.Sequential(*create_classifier(1))
    
    def forward(self, x):
        
        mrna_encoder_output = self.mrna_encoder(x)
        # print("encoder output shape: ", mrna_encoder_output.shape)

        output = self.classifier(mrna_encoder_output)
        # # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

        return output



class CustomImgModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self) -> None:
        super(CustomImgModel, self).__init__()

        self.image_encoder = nn.Sequential(*create_image_encoder())
        self.classifier = nn.Sequential(*create_classifier(1))
    
    def forward(self, x):
        # encoder_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
        # print(x.shape)
        
        image_encoder_output = self.image_encoder(x)
        # print(image_encoder_output.shape)
    
        output = self.classifier(image_encoder_output)
        # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

        return output

class CustomClinicalModel(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self) -> None:
        super(CustomClinicalModel, self).__init__()

        self.clinical_encoder = nn.Sequential(*create_clinical_encoder())
        self.classifier = nn.Sequential(*create_classifier(1))
    
    def forward(self, x):
        # encoder_outputs = torch.empty((0)).to(device=next(self.parameters()).device)
        
        # print(x.shape)

        clinical_encoder_output = self.clinical_encoder(x)
        # print(clinical_encoder_output.shape)

        output = self.classifier(clinical_encoder_output)
        # x = F.log_softmax(self.classifier(encoder_outputs), dim=1)

        return output
    
    
    

class CustomMMGBFederatedModelSimulOut(nn.Module):

    '''
    Multi-modal model to be used in federated and centralized simulations
        - Has len(modalities) number of encoders and a classifier 
    '''

    def __init__(self) -> None:
        super(CustomMMGBFederatedModelSimulOut, self).__init__()

        self.mrna_encoder = nn.Sequential(*self._create_mrna_encoder())
        self.mrna_classifier = nn.Sequential(*self._create_single_modal_classifier())
        self.image_encoder = nn.Sequential(*self._create_image_encoder())
        self.image_classifier = nn.Sequential(*self._create_single_modal_classifier())
        self.clinical_encoder = nn.Sequential(*self._create_clinical_encoder())
        self.clinical_classifier = nn.Sequential(*self._create_single_modal_classifier())

        self.classifier = nn.Sequential(*self._create_classifier())
    
    def forward(self, x_mrna, x_img, x_clinical):

        mrna_encoder_output = self.mrna_encoder(x_mrna)
        mrna_classification_output = self.mrna_classifier(mrna_encoder_output)
        # encoder_outputs = torch.cat((encoder_outputs, mrna_encoder_output), dim=1)
            
        image_encoder_output = self.image_encoder(x_img)
        image_classification_output = self.image_classifier(image_encoder_output)
        # encoder_outputs = torch.cat((encoder_outputs, image_encoder_output), dim=1)
        
        clinical_encoder_output = self.clinical_encoder(x_clinical)
        clinical_classification_output = self.clinical_classifier(clinical_encoder_output)
        # encoder_outputs = torch.cat((encoder_outputs, clinical_encoder_output), dim=1)
            
        # cursor += len(self.column_map[modality])

        encoder_outputs = torch.cat((mrna_encoder_output, image_encoder_output, clinical_encoder_output), dim=3)
        
        x = self.classifier(encoder_outputs)

        return x, mrna_classification_output, image_classification_output, clinical_classification_output

    
    def _create_classifier(self):

        fc = []
        fc.append(nn.Linear(16*3, 16))
        
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