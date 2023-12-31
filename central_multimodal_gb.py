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
import argparse
from collections import defaultdict

from datasets.multimodal_central import create_datasets_mm_centralized_gb
from models import CustomGBFederatedModelSimulOut
from torchvision import transforms, datasets
from setup_utils import setup_gpu_device
from train_functions.centralized.gb import train_and_validate_simul_gb, initialize_loss_dicts, initialize_weight_dict, calculate_weights
from training_gb_centralized import train_model, validate_model


# BATCH_SIZE = 16
# NUM_WORKERS = 2
# NUM_EPOCHS = 20 
# NUM_FEATURES = 18
# NUM_CLASSES = 2
# VALIDATION_SPLIT = 0.2
# SHUFFLE_DATASET = True
# RANDOM_SEED = 42
# MPS_USED = True
# LEARNING_RATE = 1e-3
# DECAY_RATE = 0.9
# DECAY_STEPS = 20

##### Centralized Training Procedure #####

def parse_arguments():

    # Create an argument parser object
    parser = argparse.ArgumentParser(description='Argparse for learning parameters')

    # Add arguments to the parser

    ## Hyperparameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 5)')
    parser.add_argument('--shuffle_dataset', action='store_true', default=True, help='shuffling dataset (default: True)')
    parser.add_argument('--random_seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--acc_used', action='store_true', default=True, help='whether acceleration (GPU/MPS) is used (default: False)')
    parser.add_argument('--validation_split', type=float, default=0.2, help='split ratio of train and validation datasets (default: 30)')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='Initial learning rate (default: 1e-6)')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='Learning rate decay rate (default: 0.9)')
    parser.add_argument('--steps_per_decay', type=int, default=5, help='Number of steps per learning rate decay (default: 5)')
    
    ## Read and write paths
    parser.add_argument('--data_path', type=str, default="../data/multi_modal_features/may_19_2023", help='directory for data to be loaded from')
    parser.add_argument('--result_path', type=str, default="../results", help='directory for the results to be stored')
    parser.add_argument('--saved_model_path', type=str, default="../saved_models", help='directory for the saved models to be loaded from')

    ## Gradient blending parameters
    parser.add_argument('--num_super_epochs', type=int, default=200, help='Number of local epochs per federated round (default: 1)')
    parser.add_argument('--super_epoch_len', type=int, default=3, help='length of super epochs')
    parser.add_argument('--we_epoch_len', type=int, default=3, help='length of super epochs')


    # Parse the command-line arguments
    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    if "multimodal" not in os.listdir(args.result_path):
        os.mkdir(os.path.join(args.result_path, "multimodal"))
    
    if "gb" not in os.listdir(os.path.join(args.result_path, "multimodal")):
        os.mkdir(os.path.join(args.result_path, "multimodal", "gb"))

    SAVE_DIR = os.path.join(args.result_path,"multimodal","gb",datetime.now().strftime('%Y%m%d_%H%M%S')+"_"+str(args.init_lr)+"_"+str(args.lr_decay_rate)+"_"+str(args.steps_per_decay))
    os.mkdir(SAVE_DIR)

    # DATA_DIR = os.path.join("..", "data", "multi_modal_features", "may_19_2023")

    

    device = setup_gpu_device(args)

    # Create test and train datasets
    valid_dataset, train_gb_dataset, train_dataset = create_datasets_mm_centralized_gb(args.data_path, args.validation_split, args.random_seed)
    print("Individual datasets created")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
    train_gb_loader = DataLoader(train_gb_dataset, batch_size=args.batch_size, shuffle=args.shuffle_dataset)
    
    print("Dataloaders created")
    
    # Create model
    model = CustomGBFederatedModelSimulOut(train_dataset.modalities, train_dataset.column_map)
    if args.acc_used:
        model.to(device)
    print("Model created")
    model.load_state_dict(torch.load(os.path.join(args.saved_model_path, f'central_simul_gb_start_model.pt')) )

    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader)
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    valid_loss_min = np.Inf
    val_acc = []
    train_acc = []
    total_steps_taken = 0
    
    train_gb_loss_dict, val_gb_loss_dict, train_loss_dict, val_loss_dict = initialize_loss_dicts(model.modalities)

    modality_weights = initialize_weight_dict(model.modalities)

    print("Train loop variables created")

    for super_epoch in range(args.num_super_epochs):

        print(f'Super epoch {super_epoch+1}\n') 


        ## Train model for train_epochs epochs and validate
        train_and_validate_simul_gb(model, train_loader, valid_loader,
                              args, device, criterion, optimizer,
                              scheduler, train_acc, train_loss_dict, val_acc,
                              val_loss_dict, total_steps_taken, valid_loss_min,
                              modality_weights, train_mode="train")
        # print(train_loss_dict)
        # print(train_acc)
        # print(val_loss_dict)
        # print(val_acc)
        ## Print Stats
        print(f'Super epoch [{super_epoch}/{args.num_super_epochs}], \
            \ntrain loss: {train_loss_dict["multimodal"][-1]:.4f}, train acc: {train_acc[-1]:.4f} \
            \nvalidation loss: {val_loss_dict["multimodal"][-1]:.4f}, validation acc: {val_acc[-1]:.4f}\n')
        
        ## Train model for weight estimation and validate
        train_and_validate_simul_gb(model, train_gb_loader, valid_loader,
                              args, device, criterion, optimizer,
                              scheduler, train_acc, train_gb_loss_dict, val_acc,
                              val_gb_loss_dict, total_steps_taken, valid_loss_min,
                              modality_weights, train_mode="we")
        
        print(f'Super epoch [{super_epoch}/{args.num_super_epochs}], \
            weight estimation \
            \ntrain loss: {train_gb_loss_dict["multimodal"][-1]:.4f}\
            \nvalidation loss: {val_gb_loss_dict["multimodal"][-1]:.4f}')
        
        calculate_weights(train_loss_dict, train_gb_loss_dict, val_loss_dict, val_gb_loss_dict, model.modalities, modality_weights)
        
        

        ## Save model in case performance has improved
        # if network_learned:
        #     valid_loss_min = val_loss[-1]
        #     torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'classification_model_multimodal_centralized.pt'))
        #     print('Saving current model due to improvement')

            
    
    
    

    with torch.no_grad():
        cm_pred = np.array([])
        cm_target = np.array([])
        for data_cm, target_cm in (valid_loader):

            # data_t = data_t.unsqueeze(1)

            ### To device
            if args.acc_used:
                data_cm = data_cm.to(device)
                target_cm = target_cm.to(device)

            ### Fwd pass
            outputs_cm_mm, outputs_cm_mrna, outputs_cm_image, outputs_cm_clinical = model(data_cm)

            _, pred_cm = torch.max(outputs_cm_mm, dim=1)
            _, target_cm_label = torch.max(target_cm, dim=1)
            
            # print(pred_cm.numpy(force=True))
            cm_pred = np.append(cm_pred, pred_cm.numpy(force=True))
            cm_target = np.append(cm_target, target_cm_label.numpy(force=True))
        
        final_f1_score = f1_score(cm_target, cm_pred)
        print(f"Final model f1-score: {final_f1_score}")
        # print(cm_target)
        cm = confusion_matrix(cm_target, cm_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot()
        plt.savefig(os.path.join(SAVE_DIR, f"cm_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))

    plt.figure()
    plt.plot(np.array([x for x in range(len(train_acc))]), train_acc, color='r', label='Train')
    plt.plot(np.array([x for x in range(len(val_acc))]), val_acc, color='g', label='Validation')
    plt.title("Accuracy for Centralized")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"acc_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
    val_acc_save = np.asarray(val_acc)
    np.savetxt(os.path.join(SAVE_DIR, f"acc_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.csv"), val_acc_save, delimiter=",")

    plt.figure()
    plt.plot(np.array([x for x in range(len(train_loss_dict["multimodal"]))]), train_loss_dict["multimodal"], color='r', label='Train')
    plt.plot(np.array([x for x in range(len(val_loss_dict["multimodal"]))]), val_loss_dict["multimodal"], color='g', label='Validation')
    plt.title("Loss for Centralized")
    plt.xlabel('Iterations')
    plt.ylabel('Crossentropy Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"loss_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
    val_loss_save = np.asarray(val_loss_dict["multimodal"])
    np.savetxt(os.path.join(SAVE_DIR, f"loss_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.csv"), val_loss_save, delimiter=",")

    plt.figure()
    plt.plot(np.array([x for x in range(len(modality_weights["multimodal"]))]), modality_weights["multimodal"], color='r', label="multimodal")
    plt.xlabel("Super Epoch")
    plt.ylabel("Loss Weight")
    plt.title("Multimodal Weight Evolution")
    # plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"mw_mm_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))

    plt.figure()
    plt.plot(np.array([x for x in range(len(modality_weights["mrna"]))]), modality_weights["mrna"], color='g', label="mrna")
    plt.xlabel("Super Epoch")
    plt.ylabel("Loss Weight")
    plt.title("mRNA Weight Evolution")
    # plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"mw_mrna_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))

    plt.figure()
    plt.plot(np.array([x for x in range(len(modality_weights["image"]))]), modality_weights["image"], color='g', label="image")
    plt.xlabel("Super Epoch")
    plt.ylabel("Loss Weight")
    plt.title("Image Weight Evolution")
    # plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"mw_image_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))

    plt.figure()
    plt.plot(np.array([x for x in range(len(modality_weights["clinical"]))]), modality_weights["clinical"], color='g', label="clinical")
    plt.xlabel("Super Epoch")
    plt.ylabel("Loss Weight")
    plt.title("Clinical Weight Evolution")
    # plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"mw_clinical_f1_{final_f1_score}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
    



    


        
main()