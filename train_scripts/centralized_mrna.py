import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

from datasets.misc import CustomRNADatasetBIG
from models import CustomRNAModel, CustomRNAModelBIG, CustomRNAModelConv, CustomSVMLike
from fed_utils import return_scaled_weights, sum_scaled_weights, set_layer_weights

BATCH_SIZE = 64
NUM_WORKERS = 2
NUM_EPOCHS = 10
NUM_FEATURES = 20531
NUM_CLASSES = 2
VALIDATION_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 42
MPS_USED = True
LEARNING_RATE = 5e-5
DECAY_RATE = 0.9
DECAY_STEPS = 7

##### Centralized Training Procedure #####

def main():

    # Setting the device to Apple Silicon
    if MPS_USED:
        device = torch.device("mps")


    # Create dataset
    train_dataset = CustomRNADatasetBIG("central")
    valid_dataset = CustomRNADatasetBIG("central")
    dataset_size = len(train_dataset)
    print("Initial datasets created")

    # Create train and test datasetold
    indices = list(range(dataset_size))
    split = int(np.floor(VALIDATION_SPLIT * dataset_size))
    if SHUFFLE_DATASET:
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_dataset.reduce_to(train_indices)
    valid_dataset.reduce_to(val_indices)
    print("Individual datasets created")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataloaders created")
    
    # Create model
    model = CustomRNAModelBIG(NUM_FEATURES, NUM_CLASSES)
    if MPS_USED:
        model.to(device)
    print("Model created")
    model.load_state_dict(torch.load(f'initial_model_{NUM_FEATURES}_{NUM_CLASSES}.pt')) 

    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_steps_taken = 0

    print("Train loop variables created")

    for epoch in range(NUM_EPOCHS):

        running_loss = 0.0
        

        print(f'Epoch {epoch+1}\n')
        
        for batch_idx, (data_, target_) in enumerate(train_loader):

            model.train()

            correct = 0
            total = 0

            # data_ /= 1e3
            # print(data_[:,2:8])
            # print(target_)
            # data_ = data_.unsqueeze(1)

            if MPS_USED:
                data_ = data_.to(device)
                target_ = target_.to(device)

            ### Fwd pass
            outputs = model(data_)

            # print(outputs)
            # print(target_)
            # raise "kirekhar"
            
            ### Gradient calc
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            ### Print stats
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            _, target_label = torch.max(target_, dim=1)
            correct += torch.sum(pred==target_label).item()
            total += target_.size(0)
            
            
            if total_steps_taken%DECAY_STEPS==0:
                scheduler.step()
                    
            train_acc.append(100*correct/total)
            train_loss.append(loss.item())


            ## Validate Model Accuracy

            batch_loss = 0
            total_t = 0
            correct_t = 0


            with torch.no_grad():
                model.eval()
                for data_t, target_t in (valid_loader):

                    # data_t = data_t.unsqueeze(1)

                    ### To device
                    if MPS_USED:
                        data_t = data_t.to(device)
                        target_t = target_t.to(device)

                    ### Fwd pass
                    outputs_t = model(data_t)

                    ### Print Stats
                    loss_t = criterion(outputs_t, target_t)
                    # print(loss_t.item())
                    batch_loss += loss_t.item()
                    _, pred_t = torch.max(outputs_t, dim=1)
                    _, target_t_label = torch.max(target_t, dim=1)
                    correct_t += torch.sum(pred_t==target_t_label).item()
                    total_t += target_t.size(0)
                
                val_acc.append(100*correct_t/total_t)
                val_loss.append(batch_loss/len(valid_loader))
                network_learned = batch_loss < valid_loss_min

                if (batch_idx % 2 == 0):
                    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx}/{total_steps}],\ntrain loss: {train_loss[-1]:.4f}, train acc: {(100 * correct / total):.4f} \
                        \nvalidation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

                    

                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(model.state_dict(), 'classification_model_centralized.pt')
                    print('Saving current model due to improvement')
            
            # model.train()

            ### Zero gradients
            optimizer.zero_grad()
            total_steps_taken += 1
    
    
    plt.figure()
    plt.plot(np.array([x for x in range(len(train_loss))]), train_loss, color='r', label='Train')
    plt.plot(np.array([x for x in range(len(val_loss))]), val_loss, color='g', label='Validation')
    plt.title("mRNA Loss for Centralized")
    plt.xlabel('Iterations')
    plt.ylabel('Crossentropy Loss')
    plt.legend()
    plt.savefig(f"results/centralized/loss_lr_{LEARNING_RATE}_gamma_{DECAY_RATE}_every{DECAY_STEPS}steps_mRNA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    plt.figure()
    plt.plot(np.array([x for x in range(len(train_acc))]), train_acc, color='r', label='Train')
    plt.plot(np.array([x for x in range(len(val_acc))]), val_acc, color='g', label='Validation')
    plt.title("mRNA Accuracy for Centralized")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f"results/centralized/acc_lr_{LEARNING_RATE}_gamma_{DECAY_RATE}_every{DECAY_STEPS}steps_mRNA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    with torch.no_grad():
        cm_pred = []
        cm_target = []
        for data_cm, target_cm in (valid_loader):

            # data_t = data_t.unsqueeze(1)

            ### To device
            if MPS_USED:
                data_cm = data_cm.to(device)
                target_cm = target_cm.to(device)

            ### Fwd pass
            outputs_cm = model(data_cm)

            _, pred_cm = torch.max(outputs_cm, dim=1)
            _, target_cm_label = torch.max(target_cm, dim=1)
            
            cm_pred.append(pred_cm.numpy(force=True)[:])
            cm_target.append(target_cm_label.numpy(force=True)[:])
        
        print(len(cm_target))
        cm = confusion_matrix(cm_target, cm_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot()
        plt.savefig(f"results/centralized/cm_lr_{LEARNING_RATE}_gamma_{DECAY_RATE}_every{DECAY_STEPS}steps_mRNA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    


        
main()