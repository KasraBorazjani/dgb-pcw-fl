import torch
from torch import nn
from torch.nn import functional as F

class CustomMnistModel(nn.Module):

    def __init__(self, feature_size, num_classes) -> None:

        super(CustomMnistModel, self).__init__()

        self.fc1 = nn.Linear(feature_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, num_classes)
        # self.dropout = nn.Dropout(0.3)

        self.track_layers = {'fc1':self.fc1, 'fc2':self.fc2, 'fc3':self.fc3, 'fc4':self.fc4}

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x