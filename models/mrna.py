import torch
from torch import nn
from torch.nn import functional as F

class CustomRNAModelSmall(nn.Module):

    def __init__(self, feature_size, num_classes) -> None:

        super(CustomRNAModelSmall, self).__init__()

        self.fc1 = nn.Linear(feature_size, 4096)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc15 = nn.Linear(16384, 16384)
        # torch.nn.init.xavier_uniform_(self.fc15.weight)
        # self.fc2 = nn.Linear(16384, 8192)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc16 = nn.Linear(8192, 8192)
        # torch.nn.init.xavier_uniform_(self.fc16.weight)
        # self.fc3 = nn.Linear(8192, 4096)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)
        # self.fc17 = nn.Linear(4096, 4096)
        # torch.nn.init.xavier_uniform_(self.fc17.weight)
        self.fc4 = nn.Linear(4096, 2048)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        # self.fc5 = nn.Linear(2048, 1024)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        self.fc6 = nn.Linear(2048, 512)
        # torch.nn.init.xavier_uniform_(self.fc6.weight)
        # self.fc7 = nn.Linear(512, 256)
        # torch.nn.init.xavier_uniform_(self.fc7.weight)
        self.fc8 = nn.Linear(512, 128)
        # torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.fc9 = nn.Linear(128, 64)
        # torch.nn.init.xavier_uniform_(self.fc9.weight)
        self.fc10 = nn.Linear(64, 32)
        # torch.nn.init.xavier_uniform_(self.fc10.weight)
        # self.fc11 = nn.Linear(32, 16)
        # torch.nn.init.xavier_uniform_(self.fc11.weight)
        # self.fc12 = nn.Linear(16, 8)
        # torch.nn.init.xavier_uniform_(self.fc12.weight)
        # self.fc13 = nn.Linear(8, 4)
        # torch.nn.init.xavier_uniform_(self.fc13.weight)
        self.fc14 = nn.Linear(32, num_classes)
        # torch.nn.init.xavier_uniform_(self.fc14.weight)
        self.dropout = nn.Dropout(0.3)


        # self.track_layers = {'fc1':self.fc1, 'fc2':self.fc2, 'fc3':self.fc3, 'fc4': self.fc4,
        #                      'fc5': self.fc5, 'fc6': self.fc6, 'fc7': self.fc7, 'fc8': self.fc8,
        #                      'fc9': self.fc9, 'fc10': self.fc10, 'fc11': self.fc11, 'fc12': self.fc12,
        #                      'fc13':self.fc13, 'fc14':self.fc14, 'fc15':self.fc15, 'fc16':self.fc16,
        #                      'fc17':self.fc17}

    def forward(self, x):

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc15(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc16(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc17(x))
        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        # x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))
        # x = F.relu(self.fc13(x))
        x = F.log_softmax(self.fc14(x), dim=1)

        return x

class CustomRNAModelBIG(nn.Module):

    def __init__(self, feature_size, num_classes) -> None:

        super(CustomRNAModelBIG, self).__init__()

        self.fc1 = nn.Linear(feature_size, 16384)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc15 = nn.Linear(16384, 16384)
        # torch.nn.init.xavier_uniform_(self.fc15.weight)
        self.fc2 = nn.Linear(16384, 8192)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc16 = nn.Linear(8192, 8192)
        # torch.nn.init.xavier_uniform_(self.fc16.weight)
        self.fc3 = nn.Linear(8192, 4096)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)
        # self.fc17 = nn.Linear(4096, 4096)
        # torch.nn.init.xavier_uniform_(self.fc17.weight)
        self.fc4 = nn.Linear(4096, 2048)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        # self.fc5 = nn.Linear(2048, 1024)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        self.fc6 = nn.Linear(2048, 512)
        # torch.nn.init.xavier_uniform_(self.fc6.weight)
        # self.fc7 = nn.Linear(512, 256)
        # torch.nn.init.xavier_uniform_(self.fc7.weight)
        self.fc8 = nn.Linear(512, 128)
        # torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.fc9 = nn.Linear(128, 64)
        # torch.nn.init.xavier_uniform_(self.fc9.weight)
        self.fc10 = nn.Linear(64, 32)
        # torch.nn.init.xavier_uniform_(self.fc10.weight)
        # self.fc11 = nn.Linear(32, 16)
        # torch.nn.init.xavier_uniform_(self.fc11.weight)
        # self.fc12 = nn.Linear(16, 8)
        # torch.nn.init.xavier_uniform_(self.fc12.weight)
        # self.fc13 = nn.Linear(8, 4)
        # torch.nn.init.xavier_uniform_(self.fc13.weight)
        self.fc14 = nn.Linear(32, num_classes)
        # torch.nn.init.xavier_uniform_(self.fc14.weight)
        self.dropout = nn.Dropout(0.3)


        # self.track_layers = {'fc1':self.fc1, 'fc2':self.fc2, 'fc3':self.fc3, 'fc4': self.fc4,
        #                      'fc5': self.fc5, 'fc6': self.fc6, 'fc7': self.fc7, 'fc8': self.fc8,
        #                      'fc9': self.fc9, 'fc10': self.fc10, 'fc11': self.fc11, 'fc12': self.fc12,
        #                      'fc13':self.fc13, 'fc14':self.fc14, 'fc15':self.fc15, 'fc16':self.fc16,
        #                      'fc17':self.fc17}

    def forward(self, x):

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc15(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc16(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc17(x))
        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        # x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))
        # x = F.relu(self.fc13(x))
        x = F.log_softmax(self.fc14(x), dim=1)

        return x

class CustomRNAModelMedium(nn.Module):

    def __init__(self, feature_size, num_classes) -> None:

        super(CustomRNAModelMedium, self).__init__()

        self.fc1 = nn.Linear(feature_size, 8192)
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # self.fc15 = nn.Linear(16384, 16384)
        # torch.nn.init.xavier_uniform_(self.fc15.weight)
        # self.fc2 = nn.Linear(16384, 8192)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # self.fc16 = nn.Linear(8192, 8192)
        # torch.nn.init.xavier_uniform_(self.fc16.weight)
        self.fc3 = nn.Linear(8192, 4096)
        # torch.nn.init.xavier_uniform_(self.fc3.weight)
        # self.fc17 = nn.Linear(4096, 4096)
        # torch.nn.init.xavier_uniform_(self.fc17.weight)
        self.fc4 = nn.Linear(4096, 2048)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        # self.fc5 = nn.Linear(2048, 1024)
        # torch.nn.init.xavier_uniform_(self.fc4.weight)
        self.fc6 = nn.Linear(2048, 512)
        # torch.nn.init.xavier_uniform_(self.fc6.weight)
        # self.fc7 = nn.Linear(512, 256)
        # torch.nn.init.xavier_uniform_(self.fc7.weight)
        self.fc8 = nn.Linear(512, 128)
        # torch.nn.init.xavier_uniform_(self.fc8.weight)
        self.fc9 = nn.Linear(128, 64)
        # torch.nn.init.xavier_uniform_(self.fc9.weight)
        self.fc10 = nn.Linear(64, 32)
        # torch.nn.init.xavier_uniform_(self.fc10.weight)
        # self.fc11 = nn.Linear(32, 16)
        # torch.nn.init.xavier_uniform_(self.fc11.weight)
        # self.fc12 = nn.Linear(16, 8)
        # torch.nn.init.xavier_uniform_(self.fc12.weight)
        # self.fc13 = nn.Linear(8, 4)
        # torch.nn.init.xavier_uniform_(self.fc13.weight)
        self.fc14 = nn.Linear(32, num_classes)
        # torch.nn.init.xavier_uniform_(self.fc14.weight)
        self.dropout = nn.Dropout(0.3)


        # self.track_layers = {'fc1':self.fc1, 'fc2':self.fc2, 'fc3':self.fc3, 'fc4': self.fc4,
        #                      'fc5': self.fc5, 'fc6': self.fc6, 'fc7': self.fc7, 'fc8': self.fc8,
        #                      'fc9': self.fc9, 'fc10': self.fc10, 'fc11': self.fc11, 'fc12': self.fc12,
        #                      'fc13':self.fc13, 'fc14':self.fc14, 'fc15':self.fc15, 'fc16':self.fc16,
        #                      'fc17':self.fc17}

    def forward(self, x):

        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc15(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc16(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc17(x))
        x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        # x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        # x = F.relu(self.fc11(x))
        # x = F.relu(self.fc12(x))
        # x = F.relu(self.fc13(x))
        x = F.log_softmax(self.fc14(x), dim=1)

        return x

class CustomRNAModelConv(nn.Module):

    def __init__(self, input_features, class_num) -> None:

        super(CustomRNAModelConv, self).__init__()

        self.conv1 = nn.Conv1d(1, 24, kernel_size=input_features, padding="same")
        # torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.conv1.bias)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=10)
        
        self.conv2 = nn.Conv1d(24, 24, kernel_size=input_features, padding="same")
        # torch.nn.init.xavier_uniform(self.conv2.weight)
        torch.nn.init.xavier_uniform(self.conv2.bias)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)

        # self.conv3 = nn.Conv1d(64, 64, kernel_size=input_features, padding="same")
        torch.nn.init.xavier_uniform(self.conv3.weight)
        torch.nn.init.xavier_uniform(self.conv3.bias)
        # self.relu3 = nn.ReLU()
        # self.maxpool3 = nn.MaxPool1d(kernel_size=10)

        self.fc1 = nn.Linear(24624, 10000)
        self.relu4 = nn.ReLU()

        self.fc2 = nn.Linear(10000, 128)
        self.relu5 = nn.ReLU()

        # self.fc3 = nn.Linear(4096, 10000)
        # self.relu6 = nn.ReLU()

        # self.fc4 = nn.Linear(10000, 4096)
        # self.relu7 = nn.ReLU()

        # self.fc5 = nn.Linear(4096, 2048)
        # self.relu8 = nn.ReLU()

        # self.fc6 = nn.Linear(2048, 1024)
        # self.relu9 = nn.ReLU()

        # self.fc7 = nn.Linear(1024, 512)
        # self.relu10 = nn.ReLU()

        # self.fc8 = nn.Linear(512, 256)
        # self.relu11 = nn.ReLU()

        # self.fc9 = nn.Linear(256, 128)
        # self.relu12 = nn.ReLU()

        self.fc10 = nn.Linear(128, 64)
        self.relu13 = nn.ReLU()

        self.fc11 = nn.Linear(64, 32)
        self.relu14 = nn.ReLU()

        self.fc12 = nn.Linear(32, 16)
        self.relu15 = nn.ReLU()

        self.fc13 = nn.Linear(16, 8)
        self.relu16 = nn.ReLU()

        self.fc14 = nn.Linear(8, class_num)
        self.relu17 = nn.ReLU()

        self.logsm = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)


        # x = self.conv3(x)
        # x = self.relu3(x)
        # x = self.maxpool3(x)

        x = nn.Flatten()(x)
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        # x = self.fc3(x)
        # x = self.relu6(x)

        # x = self.fc4(x)
        # x = self.relu7(x)

        # x = self.fc5(x)
        # x = self.relu8(x)

        # x = self.fc6(x)
        # x = self.relu9(x)

        # x = self.fc7(x)
        # x = self.relu10(x)

        # x = self.fc8(x)
        # x = self.relu11(x)

        # x = self.fc9(x)
        # x = self.relu12(x)

        x = self.fc10(x)
        x = self.relu13(x)

        x = self.fc11(x)
        x = self.relu14(x)

        x = self.fc12(x)
        x = self.relu15(x)

        x = self.fc13(x)
        x = self.relu16(x)

        x = self.fc14(x)
        output = self.logsm(x)

        return output