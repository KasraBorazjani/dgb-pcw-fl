import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from datetime import datetime

from datasets.image import build_img_train_test_strat
from models.image import CustomImageModel, CustomImageModelMultiClass

BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_EPOCHS = 200
NUM_FEATURES = 150
VALIDATION_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 42
MPS_USED = True
LEARNING_RATE = 1e-4 
DECAY_RATE = 0.99
DECAY_STEPS = 50
TRAIN_MODE = "binary"

if TRAIN_MODE == "multiclass":
    NUM_CLASSES = 6
    F1_AVERAGE_MODE = "macro"
    CM_LABELS = [0,1,2,3,4,5]
    model_load_dir = '../saved_models/image_start_model_multiclass_1.pt'
elif TRAIN_MODE == "binary":
    NUM_CLASSES = 2
    F1_AVERAGE_MODE = "binary"
    CM_LABELS = [0,1]
    model_load_dir = '../saved_models/image_start_model_2.pt'



##### Centralized Training Procedure #####

def main():

    path_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    SAVE_DIR = os.path.join("..","results","image", path_time)
    os.mkdir(SAVE_DIR)

    # Setting the device to Apple Silicon
    if MPS_USED:
        device = torch.device("mps")


    # Create test and train datasets
    train_dataset, valid_dataset = build_img_train_test_strat(1-VALIDATION_SPLIT, RANDOM_SEED)
    print("Individual datasets created")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataloaders created")
    
    # Create model
    model = CustomImageModelMultiClass(NUM_FEATURES, NUM_CLASSES)
    if MPS_USED:
        model.to(device)
    print("Model created")
    model.load_state_dict(torch.load(model_load_dir)) 

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

                # if (batch_idx % 2 == 0):
                #     print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx}/{total_steps}],\ntrain loss: {train_loss[-1]:.4f}, train acc: {(100 * correct / total):.4f} \
                #         \nvalidation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

                    

                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'classification_model_image_centralized.pt'))
                    # print('Saving current model due to improvement')
            
            # model.train()

            ### Zero gradients
            optimizer.zero_grad()
            total_steps_taken += 1
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx}/{total_steps}],\ntrain loss: {train_loss[-1]:.4f}, train acc: {(100 * correct / total):.4f} \
                  \nvalidation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')
    
    
    

    with torch.no_grad():
        cm_pred = np.array([])
        cm_target = np.array([])
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
            
            # print(pred_cm.numpy(force=True))
            cm_pred = np.append(cm_pred, pred_cm.numpy(force=True))
            cm_target = np.append(cm_target, target_cm_label.numpy(force=True))
        
        final_f1_score = f1_score(cm_target, cm_pred, average=F1_AVERAGE_MODE)
        print(f"Final model f1-score: {final_f1_score}")
        # print(cm_target)
        cm = confusion_matrix(cm_target, cm_pred, labels=CM_LABELS)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CM_LABELS)
        disp.plot()
        plt.savefig(os.path.join(SAVE_DIR, f"cm_f1_{final_f1_score}_lr_{LEARNING_RATE}_gamma_{DECAY_RATE}_every{DECAY_STEPS}steps.png"))

    plt.figure()
    plt.plot(np.array([x for x in range(len(train_acc))]), train_acc, color='r', label='Train')
    plt.plot(np.array([x for x in range(len(val_acc))]), val_acc, color='g', label='Validation')
    plt.title("Image Accuracy for Centralized")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"acc_f1_{final_f1_score}_lr_{LEARNING_RATE}_gamma_{DECAY_RATE}_every{DECAY_STEPS}steps.png"))
    val_acc_save = np.asarray(val_acc)
    np.savetxt(os.path.join(SAVE_DIR, f"acc_f1_{final_f1_score}_lr_{LEARNING_RATE}_gamma_{DECAY_RATE}_every{DECAY_STEPS}steps.csv"), val_acc_save, delimiter=",")

    plt.figure()
    plt.plot(np.array([x for x in range(len(train_loss))]), train_loss, color='r', label='Train')
    plt.plot(np.array([x for x in range(len(val_loss))]), val_loss, color='g', label='Validation')
    plt.title("Image Loss for Centralized")
    plt.xlabel('Iterations')
    plt.ylabel('Crossentropy Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"loss_f1_{final_f1_score}_lr_{LEARNING_RATE}_gamma_{DECAY_RATE}_every{DECAY_STEPS}steps.png"))
    val_loss_save = np.asarray(val_loss)
    np.savetxt(os.path.join(SAVE_DIR, f"loss_f1_{final_f1_score}_lr_{LEARNING_RATE}_gamma_{DECAY_RATE}_every{DECAY_STEPS}steps.csv"), val_loss_save, delimiter=",")



    


        
main()