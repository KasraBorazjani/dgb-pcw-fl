import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms, datasets
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import pandas as pd

from models.mnist import CustomMnistModel

BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 3
NUM_FEATURES = 28*28
NUM_CLASSES = 10
VALIDATION_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 42
MPS_USED = False

##### Centralized Training Procedure #####

def main():

    # Setting the device to Apple Silicon
    if MPS_USED:
        device = torch.device("mps")


    # Create dataset
    train_set = datasets.MNIST("", train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    valid_set = datasets.MNIST("", train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataloaders created")
    
    # Create model
    model = CustomMnistModel(NUM_FEATURES, NUM_CLASSES)
    if MPS_USED:
        model.to(device)
    print("Model created")
    # model.load_state_dict(torch.load('mnist_initial_model.pt'))

    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_steps_taken = 0

    print("Train loop variables created")

    for epoch in range(NUM_EPOCHS):

        running_loss = 0.0
        epoch_steps_taken = 1
        # correct = 0
        # total = 0

        print(f'Epoch {epoch}\n')
        print("starting epoch")
        for batch_idx, (data_, target_) in enumerate(train_loader):

            correct = 0
            total = 0

            if batch_idx < 800:
                if MPS_USED:
                    data_ = data_.to(device)
                    target_ = target_.to(device)
                
                # print(data_.device)

                ### Fwd pass
                outputs = model(data_.view(-1, 28*28))

                ### Gradient calc
                loss = criterion(outputs, target_)
                loss.backward()
                optimizer.step()

                ### Print stats
                running_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred==target_).item()
                total += target_.size(0)
                
                
                # if total_steps_taken%5==0:
                #     scheduler.step()
                        
                train_acc.append(100*correct/total)
                train_loss.append(loss.item())
                # print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')


                ## Validate Model Accuracy

                batch_loss = 0
                total_t = 0
                correct_t = 0


                with torch.no_grad():
                    model.eval()
                    for data_t, target_t in (valid_loader):

                        ### To device
                        if MPS_USED:
                            data_t = data_t.to(device)
                            target_t = target_t.to(device)

                        ### Fwd pass
                        outputs_t = model(data_t.view(-1, 28*28))

                        ### Print Stats
                        loss_t = criterion(outputs_t, target_t)
                        batch_loss += loss_t.item()
                        _, pred_t = torch.max(outputs_t, dim=1)
                        correct_t += torch.sum(pred_t==target_t).item()
                        total_t += target_t.size(0)
                    
                    val_acc.append(100*correct_t/total_t)
                    val_loss.append(batch_loss/len(valid_loader))
                    network_learned = batch_loss < valid_loss_min
                    # print(f"validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n")

                    if network_learned:
                        valid_loss_min = batch_loss
                        torch.save(model.state_dict(), 'classification_model_mnist.pt')
                        # print('Saving current model due to improvement')

                    if (batch_idx % 10 == 0):
                        print(f'Epoch [{epoch}/{NUM_EPOCHS}], Step [{batch_idx}/{total_steps}],\ntrain loss: {train_loss[-1]:.4f}, train acc: {(100 * correct / total):.4f} \
                            \nvalidation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
                
                model.train()

                ### Zero gradients
                optimizer.zero_grad()
                epoch_steps_taken += 1
                total_steps_taken += 1
    
    plt.figure()
    plt.plot(np.array([x for x in range(len(train_loss))]), train_loss, color='r')
    plt.plot(np.array([x for x in range(len(val_loss))]), val_loss, color='g')
    plt.title("Loss for Centralized")
    # plt.ylim((0, 100))
    plt.savefig("results/mnist_centralized_loss.png")

    plt.figure()
    plt.plot(np.array([x for x in range(len(train_acc))]), train_acc, color='r')
    plt.plot(np.array([x for x in range(len(val_acc))]), val_acc, color='g')
    plt.title("Accuracy for Centralized")
    # plt.ylim((0, 100))
    plt.savefig("results/mnist_centralized_acc.png")

    train_loss_dict = pd.DataFrame({"loss":train_loss})
    train_acc_dict = pd.DataFrame({"loss":train_acc})
    val_loss_dict = pd.DataFrame({"loss":val_loss})
    val_acc_dict = pd.DataFrame({"loss":val_acc})

    train_acc_dict.to_csv("mnist_train_acc.csv")
    train_loss_dict.to_csv("mnist_train_loss.csv")
    val_acc_dict.to_csv("mnist_val_acc.csv")
    val_loss_dict.to_csv("mnist_val_loss.csv")



    

        
        
main()

            


                

                        
                    
                        
                    





    

    

    
