import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from datasets.misc import CustomRNADataset, CustomRNADatasetBIG
from models import CustomRNAModel, CustomRNAModelBIG
from metrics import accuracy
from fed_utils import return_scaled_weights, sum_scaled_weights, set_layer_weights

BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 1
NUM_FEATURES = 20531
NUM_CLASSES = 2
VALIDATION_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 42
MPS_USED = False
NUM_CLIENTS = 3
NUM_MODALITIES = 1
NUM_FED_LOOPS = 30

def main():

    if MPS_USED:
        device = torch.device("mps")

    # Building Network
    network = {}
    network['clients'] = []
    network['validation_datasets'] = []
    
    # Building Global Model for Network
    
    model = CustomRNAModelBIG(NUM_FEATURES, NUM_CLASSES)
    if MPS_USED:
        model.to(device)
    model.load_state_dict('initial_model.pt')
    network['global_model'] = model
    
    # Building Client Models and Datasets
    for client_num_prep in range(NUM_CLIENTS):
        print(f"preparing client {client_num_prep+1}")
        client = {}
        client['id'] = client_num_prep+1
        client['datasets'] = {}
        client['dataloaders'] = {}
        client['train_loss_memory'] = []
        client['valid_loss_memory'] = []
        client['train_acc_memory'] = []
        client['valid_acc_memory'] = []
        client['total_steps_taken'] = 0
        train_dataset = CustomRNADatasetBIG(client_num_prep+1)
        valid_dataset = CustomRNADatasetBIG(client_num_prep+1)
        dataset_size = len(train_dataset)
        print("Initial datasets created")

        ## Create train and valid splits
        indices = list(range(dataset_size))
        split = int(np.floor(VALIDATION_SPLIT * dataset_size))
        if SHUFFLE_DATASET:
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        
        ## Reduce the train and valid datasets using the splits extracted
        train_dataset.reduce_to(train_indices)
        valid_dataset.reduce_to(val_indices)
        print(f"client {client_num_prep}\nlen for train dataset = {len(train_dataset)}, len for val dataset = {len(valid_dataset)}")
        
        ## Appending dataset to the client's dataset dict
        client['datasets']['train'] = train_dataset
        client['datasets']['valid'] = valid_dataset
        print("Individual datasets created")

        ## Creating dataloaders with shuffling batches
        train_loader = DataLoader(client['datasets']['train'], batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(client['datasets']['valid'], batch_size=BATCH_SIZE, shuffle=True)
        print("Dataloaders created")

        client['dataloaders']['train'] = train_loader
        client['dataloaders']['valid'] = valid_loader

        ## Model-Optimizer-Scheduler initialization
        model = CustomRNAModelBIG(NUM_FEATURES, NUM_CLASSES)
        if MPS_USED:
            model.to(device)
        client['model'] = model
        client['optimizer'] = optim.Adam(client['model'].parameters(), lr=1e-5)
        client['scheduler'] = optim.lr_scheduler.ExponentialLR(client['optimizer'], gamma=0.9)
        network['clients'].append(client)
    
    


    network['total_steps'] = 0
    for client in network['clients']:
        network['total_steps'] += len(client['dataloaders']['train'])

    # Building Training Loop
    for fed_round in range(NUM_FED_LOOPS):
        print(f"\n \n### Start of federated round {fed_round+1} ###")

        print(f"len clients: ", len(network['clients']))

        network['scaled_local_weights'] = []
        for client in network['clients']:

            print(f"training client {client['id']}")
            for model_it in range(len(client['models'])):
            
                ## Loading Global Model Weights in the Beginning of Federated Loop
                model = client['model']

                model.load_state_dict(network['global_model'].state_dict())
                criterion = nn.CrossEntropyLoss()
                optimizer = client['optimizer']
                scheduler = client['scheduler']
                valid_loss_min = np.Inf
                # temp_val_loss = []
                # temp_val_acc = []
                total_steps = len(client['dataloaders']['train'])

                for epoch in range(NUM_EPOCHS):
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    print(f'Epoch {epoch}\n')

                    for batch_idx, (data_, target_) in enumerate(train_loader):

                        if MPS_USED:
                            data_.to(device)
                            target_.to(device)

                        if batch_idx < 12:
                            ### Zero Our Parameter Gradients
                            optimizer.zero_grad()

                            ### FWD + BKWD + OPTIM
                            outputs = model(data_)
                            loss = criterion(outputs, target_)
                            loss.backward()
                            optimizer.step()

                            ### Print Stats
                            running_loss += loss.item()
                            _, pred = torch.max(outputs, dim=1)
                            _, target_label = torch.max(target_, dim=1)
                            correct += torch.sum(pred==target_label).item()
                            total += target_.size(0)
                            if(batch_idx) % 2 == 0:
                                print('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch, NUM_EPOCHS, batch_idx, total_steps, loss.item()))
                            
                            ## Scheduler step
                            if client['total_steps_taken'] % 5 == 0:
                                scheduler.step()
                            
                            client['total_steps_taken'] += 1

                            step_accuracy = 100 * torch.sum(pred==target_label).item() / target_.size(0)
                            client['train_acc_memory'].append(step_accuracy)
                            step_loss = loss.item()
                            client['train_loss_memory'].append(step_loss)
                            
                            if batch_idx%2==1:
                                print(f'Epoch [{epoch}/{NUM_EPOCHS}], Step [{batch_idx}/{total_steps}]: Accuracy = {step_accuracy}, Loss = {step_loss}')
                    
                    
                    # temp_train_acc.append(100*correct/total)
                    # temp_train_loss.append(running_loss/total_steps)
                    # print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')


                    ## Validate Model Accuracy

                    batch_loss = 0
                    total_t = 0
                    correct_t = 0


                    with torch.no_grad():
                        model.eval()
                        for data_t, target_t in (client['dataloaders']['valid'][model_it]):

                            if MPS_USED:
                                data_t = data_t.to(device)
                                target_t.to(device)
                            outputs_t = model(data_t)
                            loss_t = criterion(outputs_t, target_t)
                            batch_loss += loss_t.item()
                            _, pred_t = torch.max(outputs_t, dim=1)
                            _, target_t_label = torch.max(target_t, dim=1)
                            correct_t += torch.sum(pred_t==target_t_label).item()
                            total_t += target_t.size(0)
                        
                        val_acc.append(100*correct_t/total_t)
                        val_loss.append(batch_loss/len(client['dataloaders']['valid'][model_it]))
                        network_learned = batch_loss < valid_loss_min
                        print(f"validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n")

                        if network_learned:
                            valid_loss_min = batch_loss
                            torch.save(model.state_dict(), 'classification_model.pt')
                            print('Saving current model due to improvement')
                    
                    model.train()
                
                for train_acc_sample in train_acc:
                    client['train_acc_memory'][str(model_it+1)].append(train_acc_sample)
                for train_loss_sample in train_loss:
                    client['train_loss_memory'][str(model_it+1)].append(train_loss_sample)
                for val_acc_sample in val_acc:
                    client['valid_acc_memory'][str(model_it+1)].append(val_acc_sample)
                for val_loss_sample in val_loss:
                    client['valid_loss_memory'][str(model_it+1)].append(val_loss_sample)

                scaled_weights = return_scaled_weights(client['models'][model_it], total_steps, network['total_steps'])
                network['scaled_local_weights'][str(model_it+1)].append(scaled_weights)
        
        ## Aggregate models
        print("aggregating models")
        average_weights = sum_scaled_weights(network['scaled_local_weights']['1'])

        set_layer_weights(network['global_models'][0], average_weights)
    
    plt.figure()
    plt.plot(np.array([x for x in range(len(network['clients'][0]['train_loss_memory']['1']))]), network['clients'][0]['train_loss_memory']['1'], color='r')
    plt.plot(np.array([x for x in range(len(network['clients'][0]['valid_loss_memory']['1']))]), network['clients'][0]['valid_loss_memory']['1'], color='g')
    plt.title("loss for client 0")
    plt.savefig("results/client_0.png")

    plt.figure()
    plt.plot(np.array([x for x in range(len(network['clients'][1]['train_loss_memory']['1']))]), network['clients'][1]['train_loss_memory']['1'], color='r')
    plt.plot(np.array([x for x in range(len(network['clients'][1]['valid_loss_memory']['1']))]), network['clients'][1]['valid_loss_memory']['1'], color='g')
    plt.title("loss for client 0")
    plt.savefig("results/client_1.png")

    plt.figure()
    plt.plot(np.array([x for x in range(len(network['clients'][2]['train_loss_memory']['1']))]), network['clients'][2]['train_loss_memory']['1'], color='r')
    plt.plot(np.array([x for x in range(len(network['clients'][2]['valid_loss_memory']['1']))]), network['clients'][2]['valid_loss_memory']['1'], color='g')
    plt.title("loss for client 0")
    plt.savefig("results/client_2.png")


    

        
        
main()

            


                

                        
                    
                        
                    





    

    

    
