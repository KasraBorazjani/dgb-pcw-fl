import torch
from torch import nn
from torch.nn import functional as F

class CustomClinicalModel(nn.Module):

    def __init__(self, num_classes) -> None:

        super(CustomClinicalModel, self).__init__()

        self.fc1 = nn.Linear(10, 64)
        self.fc5 = nn.Linear(64, num_classes)


        # self.track_layers = {'fc1':self.fc1, 'fc2':self.fc2, 'fc3':self.fc3, 'fc4': self.fc4,
        #                      'fc5': self.fc5, 'fc6': self.fc6, 'fc7': self.fc7, 'fc8': self.fc8,
        #                      'fc9': self.fc9, 'fc10': self.fc10, 'fc11': self.fc11, 'fc12': self.fc12,
        #                      'fc13':self.fc13, 'fc14':self.fc14, 'fc15':self.fc15, 'fc16':self.fc16,
        #                      'fc17':self.fc17}

    def forward(self, x):

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = F.log_softmax(self.fc5(x), dim=1)

        return x